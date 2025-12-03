import argparse
import json
import random
from pathlib import Path

from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.audio.io.AudioFileClip import AudioFileClip

# Try MoviePy concatenate; fall back if missing
try:
    from moviepy.video.compositing.concatenate import concatenate_videoclips
except Exception:
    def concatenate_videoclips(clips, method="compose"):
        if not clips:
            raise ValueError("No clips to concatenate")
        W, H = clips[0].w, clips[0].h
        t = 0.0
        placed = []
        for c in clips:
            placed.append(c.with_start(t))  # MoviePy 2.x
            t += c.duration
        return CompositeVideoClip(placed, size=(W, H)).with_duration(t)

# Audio loop (MoviePy 2.x)
try:
    from moviepy.audio.fx.all import audio_loop
except Exception:
    try:
        from moviepy.audio.fx.audio_loop import audio_loop
    except Exception:
        audio_loop = None


def cover_scale_factor(img_w, img_h, target_w, target_h):
    return max(target_w / img_w, target_h / img_h)


def ken_burns_clip(image_path: Path, size=(1920, 1080), duration=5.0,
                   zoom_start=1.0, zoom_end=1.1, pan="auto",
                   start_offset=None, end_offset=None):
    W, H = size
    base = ImageClip(str(image_path)).with_duration(duration)

    s0 = cover_scale_factor(base.w, base.h, W, H)
    base_scaled = base.resized(s0)

    z0, z1 = zoom_start, zoom_end
    # Zoom function
    def z(t):
        return z0 + (z1 - z0) * (t / duration)

    zoomed = base_scaled.resized(z)

    # Compute dynamic excess size over time based on zoom
    def dx_at(t):
        return max(0, int(base.w * s0 * z(t)) - W)
    def dy_at(t):
        return max(0, int(base.h * s0 * z(t)) - H)

    # Choose pan or use provided offsets; derive ratios so we can clamp per-frame
    if pan == "auto" and (start_offset is None or end_offset is None):
        pan = random.choice([
            "left_to_right", "right_to_left",
            "top_to_bottom", "bottom_to_top", "none"
        ])

    # Start/end ratios in [0,1]: 0 means fully left/top (-dx/-dy), 1 means right/bottom (0)
    start_rx = end_rx = start_ry = end_ry = 0.5  # default center
    if start_offset is not None and end_offset is not None:
        # Convert provided pixel offsets to ratios using start zoom bounds
        dx0 = dx_at(0.0)
        dy0 = dy_at(0.0)
        def to_ratio(off, dmax):
            # clamp pixel offsets first; off is in [-dmax, 0]
            off = min(0, max(-dmax, int(off)))
            # map [-dmax, 0] -> [0,1]
            return 0.0 if dmax == 0 else (off + dmax) / dmax
        start_rx = to_ratio(start_offset.get("x", -dx0 // 2), dx0)
        start_ry = to_ratio(start_offset.get("y", -dy0 // 2), dy0)
        end_dx0 = dx_at(duration)
        end_dy0 = dy_at(duration)
        end_rx = to_ratio(end_offset.get("x", -end_dx0 // 2), end_dx0)
        end_ry = to_ratio(end_offset.get("y", -end_dy0 // 2), end_dy0)
    else:
        # Derive ratios from pan choice
        if pan == "left_to_right":
            start_rx, end_rx = 0.0, 1.0
            start_ry = end_ry = 0.5
        elif pan == "right_to_left":
            start_rx, end_rx = 1.0, 0.0
            start_ry = end_ry = 0.5
        elif pan == "top_to_bottom":
            start_ry, end_ry = 0.0, 1.0
            start_rx = end_rx = 0.5
        elif pan == "bottom_to_top":
            start_ry, end_ry = 1.0, 0.0
            start_rx = end_rx = 0.5
        else:  # none/center
            start_rx = end_rx = 0.5
            start_ry = end_ry = 0.5

    # Position function with per-frame clamping using ratios
    def pos(t):
        # interpolate ratios
        rx = start_rx + (end_rx - start_rx) * (t / duration)
        ry = start_ry + (end_ry - start_ry) * (t / duration)
        dx = dx_at(t)
        dy = dy_at(t)
        # map ratios back to pixel offsets in [-dx, 0], [-dy, 0]
        x = -dx + int(rx * dx)
        y = -dy + int(ry * dy)
        return (x, y)

    clip = CompositeVideoClip([zoomed.with_position(pos)], size=size).with_duration(duration)

    # Log ratios and also the start/end pixel offsets that were used at endpoints
    dx_start, dy_start = dx_at(0.0), dy_at(0.0)
    dx_end, dy_end = dx_at(duration), dy_at(duration)
    x0 = -dx_start + int(start_rx * dx_start)
    y0 = -dy_start + int(start_ry * dy_start)
    x1 = -dx_end + int(end_rx * dx_end)
    y1 = -dy_end + int(end_ry * dy_end)

    params = {
        "image": str(image_path),
        "frame_size": {"width": W, "height": H},
        "duration": duration,
        "base_scale_cover": s0,
        "zoom_start": z0,
        "zoom_end": z1,
        "pan": pan,
        "start_offset": {"x": x0, "y": y0},
        "end_offset": {"x": x1, "y": y1},
        "start_ratio": {"x": start_rx, "y": start_ry},
        "end_ratio": {"x": end_rx, "y": end_ry},
    }
    return clip, params


def build_slideshow(image_files, audio_file=None, size=(1920, 1080),
                    per_image=5.0, fps=30, zoom=0.10, reuse_params=None):
    clips = []
    logs = []
    param_map = {}
    if reuse_params and "images" in reuse_params:
        for entry in reuse_params["images"]:
            param_map[Path(entry.get("image", "")).name] = entry

    for idx, img in enumerate(image_files):
        default_zin = (idx % 2 == 0)
        default_z0, default_z1 = (1.0, 1.0 + zoom) if default_zin else (1.0 + zoom, 1.0)

        entry = param_map.get(img.name)
        if entry:
            z0 = float(entry.get("zoom_start", default_z0))
            z1 = float(entry.get("zoom_end", default_z1))
            pan = entry.get("pan", "none")
            # Prefer ratios if present; fall back to offsets
            sr = entry.get("start_ratio")
            er = entry.get("end_ratio")
            if sr and er:
                start_offset = {"x": None, "y": None}  # signal ratios usage
                end_offset = {"x": None, "y": None}
                # pass None offsets; ken_burns_clip will use ratios from entry? we’ll pass them via pan fields
                # We embed ratios into entry by attaching them to start_offset/end_offset special keys
                start_offset = {"rx": float(sr.get("x", 0.5)), "ry": float(sr.get("y", 0.5))}
                end_offset = {"rx": float(er.get("x", 0.5)), "ry": float(er.get("y", 0.5))}
            else:
                start_offset = entry.get("start_offset")
                end_offset = entry.get("end_offset")
            duration = float(entry.get("duration", per_image))
        else:
            z0, z1 = default_z0, default_z1
            pan = "auto"
            start_offset = end_offset = None
            duration = per_image

        # If ratios provided, translate to pixel offsets at endpoints for logging consistency
        if start_offset and "rx" in start_offset:
            # We don’t have dx/dy here; let ken_burns_clip compute with ratios via pan parameter hack
            # Use pan="none" and send ratios by packing them into pan string; simpler approach: attach to pan as tuple
            pan = {"start_ratio": start_offset, "end_ratio": end_offset}

        clip, params = ken_burns_clip(
            img, size=size, duration=duration,
            zoom_start=z0, zoom_end=z1, pan=pan,
            start_offset=None if isinstance(pan, dict) else start_offset,
            end_offset=None if isinstance(pan, dict) else end_offset
        )

        # If we passed ratios via pan dict, params already contain ratios; else keep offsets
        logs.append(params)
        clips.append(clip)

    video = concatenate_videoclips(clips, method="compose")

    if audio_file and Path(audio_file).exists():
        audio = AudioFileClip(str(audio_file))
        if audio.duration >= video.duration:
            audio = audio.subclipped(0, video.duration)
        else:
            if audio_loop:
                audio = audio.fx(audio_loop, duration=video.duration)
        video = video.with_audio(audio)

    return video.with_fps(fps), logs


def main():
    parser = argparse.ArgumentParser(description="Create a Ken Burns slideshow from images and an optional audio track.")
    parser.add_argument("--images", type=str, default="images", help="Folder with images or a text file listing image paths.")
    parser.add_argument("--audio", type=str, default="audio/audio.mp3", help="Path to audio file (optional).")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--per-image", type=float, default=5.0, help="Seconds per image.")
    parser.add_argument("--zoom", type=float, default=0.12,
                       help="Extra zoom amount (e.g., 0.12 = 12%%).")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--out", type=str, default="output/output.mp4")
    parser.add_argument(
        "--codec",
        type=str,
        default="libx264",
        help=("FFmpeg video encoder to use. Examples: "
              "libx264 (CPU H.264), libx265 (CPU H.265), "
              "h264_nvenc / hevc_nvenc (NVIDIA GPU), "
              "h264_qsv / hevc_qsv (Intel QSV), "
              "h264_amf / hevc_amf (AMD AMF).")
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Render a quick preview (lower resolution, shorter duration, faster preset)."
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Optional path to write a JSON log with per-image parameters."
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Path to a JSON file with previously serialized per-image parameters."
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent

    # Normalize inputs relative to main.py unless absolute
    images_arg = Path(args.images)
    images_path = images_arg if images_arg.is_absolute() else (here / images_arg)
    if images_path.suffix.lower() in {".txt", ".lst"}:
        list_text = images_path.read_text(encoding="utf-8")
        image_files = []
        for line in list_text.splitlines():
            s = line.strip()
            if not s:
                continue
            p = Path(s)
            image_files.append(p if p.is_absolute() else (images_path.parent / p))
    else:
        search_dir = images_path
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
        image_files = sorted([p for p in search_dir.glob("*") if p.suffix.lower() in exts])

    if not image_files:
        raise SystemExit(f"No images found in {images_path}")

    # Apply preview settings
    if args.preview:
        image_files = image_files[:min(6, len(image_files))]
        args.width, args.height = 854, 480
        args.per_image = min(args.per_image, 2.0)
        args.fps = min(args.fps, 24)
        # If using default path, switch to a preview filename inside output/
        if args.out == "output/output.mp4":
            args.out = "output/output_preview.mp4"

    audio_arg = Path(args.audio)
    audio_path = audio_arg if audio_arg.is_absolute() else (here / audio_arg)
    out_arg = Path(args.out)
    out_path = out_arg if out_arg.is_absolute() else (here / out_arg)

    # Ensure the output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load params if provided
    reuse_params = None
    if args.params:
        params_arg = Path(args.params)
        params_path = params_arg if params_arg.is_absolute() else (here / params_arg)
        if params_path.exists():
            reuse_params = json.loads(params_path.read_text(encoding="utf-8"))
            # Normalize image entries to absolute for matching and rendering
            norm_images = []
            for entry in reuse_params.get("images", []):
                img_str = entry.get("image", "")
                img_path = Path(img_str)
                if not img_path.is_absolute():
                    img_path = (here / img_path).resolve()
                entry["image"] = str(img_path)
                norm_images.append(entry)
            reuse_params["images"] = norm_images
        else:
            print(f"Warning: params file not found: {params_path}")

    video, logs = build_slideshow(
        image_files,
        audio_file=str(audio_path) if audio_path.exists() else None,
        size=(args.width, args.height),
        per_image=args.per_image, fps=args.fps, zoom=args.zoom, reuse_params=reuse_params
    )

    # Serialize parameters if requested
    if args.log:
        log_arg = Path(args.log)
        log_path = log_arg if log_arg.is_absolute() else (here / log_arg)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Make output and image paths relative to main.py location
        def to_rel(p: str) -> str:
            try:
                return str(Path(p).resolve().relative_to(here))
            except Exception:
                # Fallback: os-style relative (handles non-subpaths)
                from os import path as osp
                return osp.relpath(str(p), start=str(here))

        payload = {
            "output": to_rel(str(out_path)),
            "codec": args.codec,
            "fps": args.fps,
            "frame_size": {"width": args.width, "height": args.height},
            "images": [
                {**entry, "image": to_rel(entry.get("image", ""))}
                for entry in logs
            ],
        }
        log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    write_kwargs = {
        "codec": args.codec,
        "audio_codec": "aac",
        "threads": 4,
        "fps": args.fps,
        "temp_audiofile": str(out_path.with_suffix(".temp-audio.m4a")),
        "remove_temp": True,
    }
    if args.codec in ("libx264", "libx265"):
        write_kwargs["preset"] = "medium"
        if args.preview:
            write_kwargs["preset"] = "ultrafast"

    if args.preview and args.codec.endswith("_nvenc"):
        write_kwargs["ffmpeg_params"] = ["-preset", "p1", "-cq", "28"]
    elif args.preview and args.codec.endswith("_qsv"):
        write_kwargs["ffmpeg_params"] = ["-global_quality", "30"]
    elif args.preview and args.codec.endswith("_amf"):
        write_kwargs["ffmpeg_params"] = ["-quality", "speed"]

    video.write_videofile(str(out_path), **write_kwargs)


if __name__ == "__main__":
    main()