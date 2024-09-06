# Whisper Subtitles

This script uses Whisper to transcribe an audio stream in real-time, providing subtitles for any stream that ffmpeg can play.

## Requirements

- Whisper
- ffmpeg
- numpy

## Usage

```
python subtitles.py [-h] [-m MODEL] [-d DEVICE] [-l CHUNK_LENGTH] [-n NUM_CHUNKS] source
```

### Arguments:

- `source`: File or URL for ffmpeg to play (required)
- `-m, --model`: Whisper model to use (default: "base.en")
- `-d, --device`: 'cuda' for GPU or 'cpu' for CPU (default: "cuda")
- `-l, --chunk_length`: Length of chunks in seconds (default: 3)
- `-n, --num_chunks`: Number of chunks to process (default: 2)

### Example:

```
python subtitles.py https://example.com/audio_stream.mp3 -m medium -d cpu -l 5 -n 3
```

This example uses the 'medium' model on CPU, processing 5-second chunks and processes the last 3 chunks.

For more information on available options, use:

```
python subtitles.py -h
```