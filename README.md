# Whisper Subtitles

This script uses Whisper to transcribe an audio stream in real-time, providing subtitles for any stream that ffmpeg can play.

## Requirements

- Whisper
- ffmpeg
- numpy
- pydantic
- websockets (for WebSocket server mode)

## Usage - CLI Mode

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

## Usage - WebSocket Server Mode

The WebSocket server mode allows you to run a server that provides real-time transcription over WebSockets. When clients connect to a specific endpoint, the server starts transcribing the audio stream from the corresponding IP address and broadcasts the transcription to all connected clients.

```bash
python ws_server.py
```

### WebSocket Endpoints:

The WebSocket server provides endpoints in the following format:

```
ws://server-address:8080/transcribe/<ip>/
```

Where `<ip>` is the IP address of the stream to be transcribed. The server will fetch the audio stream from:

```
https://screamrouter.netham45.org/stream/<ip>/
```

### Example:

To transcribe a stream from IP address 192.168.1.100:

```
ws://localhost:8080/transcribe/192.168.1.100/
```

When a client connects to this endpoint, the server will:
1. Start transcribing the audio stream from https://screamrouter.netham45.org/stream/192.168.1.100/
2. Send transcription updates to all connected clients
3. Automatically stop transcription when all clients disconnect

### JavaScript Client Example:

```javascript
const socket = new WebSocket('ws://localhost:8080/transcribe/192.168.1.100/');

socket.onopen = function(e) {
  console.log('Connection established');
};

socket.onmessage = function(event) {
  console.log('Transcription received:', event.data);
  document.getElementById('subtitles').innerText = event.data;
};

socket.onclose = function(event) {
  if (event.wasClean) {
    console.log(`Connection closed cleanly, code=${event.code} reason=${event.reason}`);
  } else {
    console.log('Connection died');
  }
};

socket.onerror = function(error) {
  console.log(`WebSocket error: ${error.message}`);
};
```

This client will connect to the WebSocket server, receive real-time transcriptions, and display them in an HTML element with the ID 'subtitles'.
