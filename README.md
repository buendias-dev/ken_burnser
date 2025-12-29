# Ken Burns Video

Combine a list of images into a video with a Ken Burns effect and an optional audio track. Supports parameter serialization, preview rendering, configurable codecs (including GPU encoders via FFmpeg), crossfades, linear panning, and max pan speed.

## Project Structure

```
ken_burnser
├── main.py             # Main script
├── requirements.txt    # Dependencies
├── README.md           # Documentation
├── images/             # Input images
├── audio/              # Input audio (e.g., audio.mp3)
└── output/             # Rendered videos and logs
```

## Requirements

Install dependencies in your environment:

```
pip install -r requirements.txt
```

## Usage

Basic render (paths are resolved relative to `main.py` by default):
```
python main.py --images images --audio audio/audio.mp3 --out output/output.mp4
```

Common options:
- `--images`: folder of images or a text file listing image paths.
- `--audio`: audio file path (optional).
- `--width --height`: target frame size (default 1920x1080).
- `--per-image`: seconds per image (default 5.0).
- `--zoom`: extra zoom proportion, e.g., 0.12 = 12%.
- `--fps`: frames per second.
- `--out`: output filename (default `output/output.mp4`).
- `--codec`: FFmpeg encoder. Examples:
  - CPU: `libx264`, `libx265`
  - NVIDIA: `h264_nvenc`, `hevc_nvenc`
  - Intel: `h264_qsv`, `hevc_qsv`
  - AMD: `h264_amf`, `hevc_amf`

Show help:
```
python main.py --help
```

### Preview mode

Render a faster, shorter, lower-resolution preview:
```
python main.py --preview --codec h264_nvenc --out output/output_preview.mp4
```
Preview applies:
- Limits to a few images
- 854x480 resolution
- Shorter per-image duration
- Faster presets/ffmpeg params

### Serialize and reuse parameters

Render and save per-image params to JSON (paths serialized relative to `main.py`):
```
python main.py --images images --audio audio/audio.mp3 --log output/params.json
```

Re-render using saved params:
```
python main.py --images images --audio audio/audio.mp3 --params output/params.json --out output/output.mp4
```

Notes:
- When loading `--params`, image paths in the JSON can be relative or absolute; they’re normalized.
- The renderer logs zoom, pan, start/end offsets, ratios, duration, and pan speed per image.

## Command-Line Options

- `--images PATH`: Folder of images or a text file listing image paths. Default: `images`.
- `--audio PATH`: Optional audio file to mix into the video. Default: `audio/audio.mp3`.
- `--width INT`: Output width in pixels. Default: `1920`.
- `--height INT`: Output height in pixels. Default: `1080`.
- `--per-image FLOAT`: Seconds each image stays on screen. Default: `5.0`.
- `--zoom FLOAT`: Extra zoom proportion applied across the clip (e.g., `0.12` = 12%). Default: `0.12`.
- `--fps INT`: Output frames per second. Default: `24`.
- `--out PATH`: Output video path. Default: `output/output.mp4`.
- `--codec NAME`: FFmpeg encoder. Default: `libx264`.
  - Examples: CPU `libx264`, `libx265`; NVIDIA `h264_nvenc`, `hevc_nvenc`; Intel `h264_qsv`, `hevc_qsv`; AMD `h264_amf`, `hevc_amf`.
- `--preview`: Faster, shorter, lower-resolution preview (limits images, sets 854x480, reduces durations, speeds up presets).
- `--log PATH`: Write a JSON log with per-image parameters used during render.
- `--params PATH`: Read per-image parameters from a previous JSON log to reproduce a render.
- `--random-pan`: Ignore pan info from `--params` and choose a random pan per image.
- `--seed INT`: Random seed for reproducible panning when using `--random-pan`.
- `--fade FLOAT`: Crossfade duration in seconds between images (0 disables). Default: `0.0`.
- `--max-pan-speed FLOAT`: Maximum panning speed in pixels per second. Caps movement distance while preserving direction. Default: unset.

## Transitions and Panning

- `--fade SECONDS`: Crossfade between clips for smoother transitions.
  - Fades are applied with safe overlap (limited by clip durations).
- `--random-pan`: Ignore pan info in `--params` and choose a random pan per image.
- `--seed INT`: Seed for reproducible random pan selection.
- Linear panning: Movement is interpolated linearly in pixel space between safe endpoints, preserving direction.
- No black borders: Images are scaled to cover and clamped per-frame; a small overscan and ceiling math prevent edge gaps.
- `--max-pan-speed PIXELS_PER_SECOND`: Caps panning speed; keeps direction by shortening the endpoint distance.

Examples:
```
# Preview with random pan, 0.7s crossfade, and speed cap
python main.py --images images --audio audio/audio.mp3 \
  --preview --out output/output_preview.mp4 --random-pan --seed 21 --fade 0.7 --max-pan-speed 150

# Full render with linear pan and fades
python main.py --images images --audio audio/audio.mp3 --out output/output.mp4 --fade 0.7
```

## GPU encoding

If your FFmpeg build supports NVENC/QSV/AMF, you can use a GPU encoder via `--codec` (e.g., `h264_nvenc`). MoviePy’s writer doesn’t place input options like `-hwaccel`; GPU decoding is not configured here. Encoding will still use the selected GPU codec.

Verify encoder availability:
```
ffmpeg -hide_banner -encoders | findstr nvenc
ffmpeg -hide_banner -encoders | findstr qsv
ffmpeg -hide_banner -encoders | findstr amf
```

## Tips

- Place source images in `images/` and audio in `audio/`.
- Use a text file for a custom image order; paths inside can be relative to the list file location.
- For reproducible renders, serialize params and reuse with `--params`.

## License

This project is open-source. See the license file for details.