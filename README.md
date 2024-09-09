# Whisper Subtitles

This script uses Whisper to transcribe an audio stream in real-time, providing subtitles for any stream that ffmpeg can play.

## Requirements

- Whisper
- ffmpeg
- numpy

## Usage

```bash
python subtitles.py [-h] 
  [-m MODEL] 
  [-d DEVICE] 
  [-l CHUNK_LENGTH] 
  [-n NUM_CHUNKS]
  [-t NUM_LINES] 
  [-r] 
  [-c]
  source
```

### Arguments:

- `source`: File or URL for ffmpeg to play (required)
- `-m, --model`: Whisper model to use. Available options: `tiny`, `base`, `small`, `medium`, `large`, `large_v1`, `large_v2`, `large_v3`, `tiny.en`, `base.en`, `small.en`, `medium.en` (default: "base")
- `-d, --device`: Compute device type. Available options: `cuda`, `cpu` (default: "cuda")
- `-l, --chunk_length`: Length of chunks in seconds (default: 3)
- `-n, --num_chunks`: Number of chunks to process at once (default: 2)
- `-t, --num_lines`: Number of lines to output per subtitle refresh
- `-r, --realtime`: Process in real-time or as fast as possible. Use for files, not realtime streams.
- `-c, --dont_clear`: Don't clear the screen between transcribed lines.

### Example:

```bash
python subtitles.py https://example.com/audio_stream.mp3 -m medium.en -d cpu -l 5 -n 3 -t 4 -r
```

This example uses the 'medium.en' model on CPU, processing 5-second chunks and processes the last 3 chunks. It also displays 4 lines of subtitles per refresh.

For more information on available options, use:

```bash
python subtitles.py -h
```
