import argparse
import json
import random
import math
import time
from pathlib import Path

from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip, concatenate_videoclips
from moviepy.audio.io.AudioFileClip import AudioFileClip

# Audio loop (MoviePy 2.x)
from moviepy.audio.fx.AudioLoop import AudioLoop as audio_loop

from moviepy.video.fx.FadeIn import FadeIn as vfx_fadein

from moviepy.video.fx.FadeOut import FadeOut as vfx_fadeout
from moviepy.video.fx.CrossFadeIn import CrossFadeIn as vfx_crossfadein

def cover_scale_factor(img_w, img_h, target_w, target_h):
    return max(target_w / img_w, target_h / img_h) * 1.002


def ken_burns_clip(image_path: Path, size=(1920, 1080), duration=5.0,
                   zoom_start=1.0, zoom_end=1.1, pan="auto",
                   start_offset=None, end_offset=None, max_pan_speed=None):
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
        w_t = math.ceil(base.w * s0 * z(t))
        return max(0, w_t - W)
    def dy_at(t):
        h_t = math.ceil(base.h * s0 * z(t))
        return max(0, h_t - H)

    # Choose pan or use provided offsets; derive ratios so we can clamp per-frame
    if pan == "auto" and (start_offset is None or end_offset is None):
        # Use per-image random for pan selection if a global seed is set
        import inspect
        frame = inspect.currentframe()
        caller_locals = frame.f_back.f_locals if frame and frame.f_back else {}
        idx = caller_locals.get('idx', None)
        global_seed = caller_locals.get('args', {}).get('seed', None) if 'args' in caller_locals else None
        if global_seed is not None and idx is not None:
            rng = random.Random(global_seed + idx)
            pan = rng.choice([
                "left_to_right", "right_to_left",
                "top_to_bottom", "bottom_to_top", "none"
            ])
        else:
            pan = random.choice([
                "left_to_right", "right_to_left",
                "top_to_bottom", "bottom_to_top", "none"
            ])
        print(f"Auto-selected pan: {pan}")

    # Derive safe linear endpoints in pixel space using intersection ranges
    min_dx, min_dy = min(dx_at(0.0), dx_at(duration)), min(dy_at(0.0), dy_at(duration))
    cx = -min_dx // 2
    cy = -min_dy // 2

    def clamp_xy(x, y):
        return (min(0, max(-min_dx, int(x))), min(0, max(-min_dy, int(y))))

    # Endpoints: prefer provided offsets, then ratios in a dict, else from pan direction
    if isinstance(pan, dict):
        sr = pan.get("start_ratio", {})
        er = pan.get("end_ratio", {})
        def rget(d, k, default):
            try:
                return float(d.get(k, default))
            except Exception:
                return default
        rx0 = rget(sr, "rx", rget(sr, "x", 0.5))
        ry0 = rget(sr, "ry", rget(sr, "y", 0.5))
        rx1 = rget(er, "rx", rget(er, "x", 0.5))
        ry1 = rget(er, "ry", rget(er, "y", 0.5))
        px0, py0 = -min_dx + rx0 * min_dx, -min_dy + ry0 * min_dy
        px1, py1 = -min_dx + rx1 * min_dx, -min_dy + ry1 * min_dy
        px0, py0 = clamp_xy(px0, py0)
        px1, py1 = clamp_xy(px1, py1)
    elif start_offset is not None and end_offset is not None:
        px0, py0 = clamp_xy(start_offset.get("x", cx), start_offset.get("y", cy))
        px1, py1 = clamp_xy(end_offset.get("x", cx), end_offset.get("y", cy))
    else:
        print(f"Using pan direction: {pan}")
        if pan == "left_to_right":
            px0, px1 = -min_dx, 0
            py0 = py1 = cy
        elif pan == "right_to_left":
            px0, px1 = 0, -min_dx
            py0 = py1 = cy
        elif pan == "top_to_bottom":
            py0, py1 = -min_dy, 0
            px0 = px1 = cx
        elif pan == "bottom_to_top":
            py0, py1 = 0, -min_dy
            px0 = px1 = cx
        else:
            px0 = px1 = cx
            py0 = py1 = cy

    # Optionally cap overall pan speed (Euclidean) in pixels/second
    if max_pan_speed and duration > 0:
        dist = math.hypot(px1 - px0, py1 - py0)
        max_dist = float(max_pan_speed) * float(duration)
        if dist > max_dist and dist > 0:
            scale = max_dist / dist
            px1 = px0 + (px1 - px0) * scale
            py1 = py0 + (py1 - py0) * scale
            # Re-clamp to intersection bounds
            px1, py1 = clamp_xy(px1, py1)

    # Linear interpolation in pixel space with per-frame safety clamp
    def pos(t):
        a = t / duration
        x = int(px0 + (px1 - px0) * a)
        y = int(py0 + (py1 - py0) * a)
        dx = dx_at(t)
        dy = dy_at(t)
        x = min(0, max(-dx, x))
        y = min(0, max(-dy, y))
        return (x, y)

    clip = CompositeVideoClip([zoomed.with_position(pos)], size=size).with_duration(duration)

    # Log ratios and also the start/end pixel offsets that were used at endpoints
    dx_start, dy_start = dx_at(0.0), dy_at(0.0)
    dx_end, dy_end = dx_at(duration), dy_at(duration)

    params = {
        "image": str(image_path),
        "frame_size": {"width": W, "height": H},
        "duration": duration,
        "base_scale_cover": s0,
        "zoom_start": z0,
        "zoom_end": z1,
        "pan": pan,
        "start_offset": {"x": int(px0), "y": int(py0)},
        "end_offset": {"x": int(px1), "y": int(py1)},
        "start_ratio": {"x": 0.0 if min_dx == 0 else (px0 + min_dx) / max(1, min_dx),
                          "y": 0.0 if min_dy == 0 else (py0 + min_dy) / max(1, min_dy)},
        "end_ratio": {"x": 0.0 if min_dx == 0 else (px1 + min_dx) / max(1, min_dx),
                "y": 0.0 if min_dy == 0 else (py1 + min_dy) / max(1, min_dy)},
        "max_pan_speed": max_pan_speed if max_pan_speed is not None else None,
        "pan_distance": float(math.hypot(px1 - px0, py1 - py0)),
        "pan_speed": float(0.0 if duration == 0 else math.hypot(px1 - px0, py1 - py0) / duration),
    }
    return clip, params


def build_slideshow(image_files, audio_file=None, size=(1920, 1080),
                    per_image=5.0, fps=30, zoom=0.10, reuse_params=None,
                    random_pan=False, fade=0.0, max_pan_speed=None):
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
            # Force random pan: always use pan="auto" and no offsets
            if random_pan:
                print(f"Randomizing pan for image: {img.name}")
                pan = "auto"
                start_offset = end_offset = None
            else:
                pan = entry.get("pan", "none")
                sr = entry.get("start_ratio")
                er = entry.get("end_ratio")
                if sr and er:
                    start_offset = {"rx": float(sr.get("x", 0.5)), "ry": float(sr.get("y", 0.5))}
                    end_offset = {"rx": float(er.get("x", 0.5)), "ry": float(er.get("y", 0.5))}
                else:
                    start_offset = entry.get("start_offset")
                    end_offset = entry.get("end_offset")
                print(f"Reusing pan for image: {img.name} -> {pan}")
            duration = float(entry.get("duration", per_image))
        else:
            print(f"Using default params for image: {img.name}")
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
            end_offset=None if isinstance(pan, dict) else end_offset,
            max_pan_speed=max_pan_speed
        )

        # If we passed ratios via pan dict, params already contain ratios; else keep offsets
        logs.append(params)
        clips.append(clip)

    # Build final video with true crossfade using CrossFadeIn and CompositeVideoClip
    fade = max(0.0, float(fade or 0.0))
    if fade > 0.0:
        composite_clips = []
        t = 0.0
        for idx, clip in enumerate(clips):
            if idx == 0:
                composite_clips.append(clip.with_start(0))
            else:
                # Each subsequent clip starts before the previous ends, overlapping by fade seconds
                start_time = idx * (clip.duration - fade)
                composite_clips.append(vfx_crossfadein(fade).apply(clip.with_start(start_time)))
        total_duration = len(clips) * clips[0].duration - (len(clips) - 1) * fade
        video = CompositeVideoClip(composite_clips, size=size).with_duration(total_duration)
    else:
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
    parser.add_argument(
        "--random-pan",
        action="store_true",
        help="Ignore pan info in params and choose a random pan per image."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible pan selection."
    )
    parser.add_argument(
        "--fade",
        type=float,
        default=0.0,
        help="Seconds to crossfade between images (0 disables)."
    )
    parser.add_argument(
        "--max-pan-speed",
        type=float,
        default=None,
        help="Maximum panning speed in pixels per second (caps movement length)."
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent

    # Start timing
    _render_start = time.perf_counter()

    # Optional reproducible randomness
    if args.seed is not None:
        random.seed(args.seed)

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

    _build_start = time.perf_counter()
    video, logs = build_slideshow(
        image_files,
        audio_file=str(audio_path) if audio_path.exists() else None,
        size=(args.width, args.height),
        per_image=args.per_image, fps=args.fps, zoom=args.zoom, reuse_params=reuse_params,
        random_pan=args.random_pan, fade=args.fade, max_pan_speed=args.max_pan_speed
    )
    _build_end = time.perf_counter()
    _build_time = _build_end - _build_start

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

    _encode_start = time.perf_counter()
    video.write_videofile(str(out_path), **write_kwargs)

    # End timing
    _render_end = time.perf_counter()
    _elapsed = _render_end - _render_start
    _encode_time = _render_end - _encode_start
    print(f"Render completed in {_elapsed:.2f} seconds -> {out_path}")
    print(f" - Codec: {args.codec}")
    print(f" - Build (slideshow generation): {_build_time:.2f}s")
    print(f" - Encode (write_videofile): {_encode_time:.2f}s")


if __name__ == "__main__":
    main()