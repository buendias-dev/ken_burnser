# Ken Burns Video Project

Combine a list of images into a video with a Ken Burns effect and an optional audio track. Supports parameter serialization, reusing saved effect params, preview rendering, and configurable codecs (including GPU encoders via FFmpeg).

## Project Structure

```
kenburns-video-project
├── src
│   ├── main.py          # Main script
│   ├── images           # Input images
│   ├── audio            # Input audio (e.g., audio.mp3)
│   └── utils            # (optional) helpers
├── requirements.txt     # Dependencies
└── README.md            # Documentation
```

## Requirements

Install dependencies in your virtual environment:

```
pip install -r requirements.txt
```

## Usage

Basic render (paths are resolved relative to `src/main.py` by default):
```
python src/main.py --images images --audio audio/audio.mp3 --out output.mp4
```

Common options:
- `--images`: folder of images or a text file listing image paths (relative or absolute).
- `--audio`: audio file path (optional).
- `--width --height`: target frame size (default 1920x1080).
- `--per-image`: seconds per image (default 5.0).
- `--zoom`: extra zoom proportion, e.g., 0.12 = 12%.
- `--fps`: frames per second.
- `--out`: output filename.
- `--codec`: FFmpeg encoder. Examples:
  - CPU: `libx264`, `libx265`
  - NVIDIA: `h264_nvenc`, `hevc_nvenc`
  - Intel: `h264_qsv`, `hevc_qsv`
  - AMD: `h264_amf`, `hevc_amf`

Show help:
```
python src/main.py --help
```

### Preview mode

Render a faster, shorter, lower-resolution preview:
```
python src/main.py --preview --codec h264_nvenc --out output_preview.mp4
```
Preview applies:
- Limits to a few images
- 854x480 resolution
- Shorter per-image duration
- Faster presets/ffmpeg params

### Serialize and reuse effect parameters

Render and save per-image params to JSON (paths serialized relative to `src/main.py`):
```
python src/main.py --images images --audio audio/audio.mp3 --log params.json
```

Re-render using saved params:
```
python src/main.py --images images --audio audio/audio.mp3 --params params.json --out output.mp4
```

Notes:
- When loading `--params`, image paths in the JSON can be relative or absolute; they’re normalized.
- The renderer logs zoom, pan, clamped start/end positions, ratios, and duration per image.

## Ken Burns Effect details

- Images are scaled to cover the target frame.
- Zoom and pan are animated over time.
- Position is clamped each frame to avoid empty borders as zoom changes.
- Pan direction can be auto, or derived from saved params.

## GPU encoding

If your FFmpeg build supports NVENC/QSV/AMF, you can use a GPU encoder via `--codec` (e.g., `h264_nvenc`). MoviePy’s writer doesn’t place input options like `-hwaccel`; GPU decoding is not configured here. Encoding will still use the selected GPU codec.

Verify encoder availability:
```
ffmpeg -hide_banner -encoders | findstr nvenc
ffmpeg -hide_banner -encoders | findstr qsv
ffmpeg -hide_banner -encoders | findstr amf
```

## Tips

- Place source images in `src/images` and audio in `src/audio`.
- Use a text file for a custom image order; paths inside can be relative to the list file location.
- For reproducible renders, serialize params and reuse with `--params`.

## License

This project is open-source. See the license file for details.