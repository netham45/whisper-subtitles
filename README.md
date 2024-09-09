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
  source
```

### Arguments:

- `source`: File or URL for ffmpeg to play (required)
- `-m, --model`: Whisper model to use. Available options: `tiny`, `base`, `small`, `medium`, `large`, `large_v1`, `large_v2`, `large_v3`, `tiny.en`, `base.en`, `small.en`, `medium.en` (default: "base")
- `-d, --device`: Compute device type. Available options: `cuda`, `cpu` (default: "cuda")
- `-l, --chunk_length`: Length of chunks in seconds (default: 3)
- `-n, --num_chunks`: Number of chunks to process at once (default: 2)

### Example:

```bash
python subtitles.py https://example.com/audio_stream.mp3 -m medium.en -d cpu -l 5 -n 3
```

This example uses the 'medium.en' model on CPU, processing 5-second chunks and processes the last 3 chunks.

For more information on available options, use:

```bash
python subtitles.py -h
```