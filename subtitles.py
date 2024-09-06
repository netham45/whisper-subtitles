#!/usr/bin/python3
"""This class transcribes an MP3 stream as subtitles"""
import os
import sys
import subprocess
import re
import tempfile
import threading
from typing import Optional
import numpy as np
import whisper

MODEL_NAME="base.en"  # Any whisper model works here
DEVICE_NAME="cuda"  # "cuda" or "cpu"
CLEAR: str = "\033[2J\033[H"  # ANSI clear code
TEMP_DIR = tempfile.mkdtemp(prefix="subtitles_")
LOADING_MODEL_MESSAGE: str = "\n".join([f"Model '{MODEL_NAME}'",
                                        f"Device: '{DEVICE_NAME}'",
                                        f"Temp Dir Path: '{TEMP_DIR}'"])
LOADING_AUDIO_MESSAGE: str = "Receiving Initial Audio"
CHUNK_LENGTH: int = 3  # Seconds
NUM_CHUNKS: int = 2  # Number of chunks to keep

class Subtitles(threading.Thread):
    """Reads an MP3 stream and does subtitles for it"""
    loaded_model: str = ""
    device_mode: str = ""
    model: Optional[whisper.Whisper] = None
    file_list: list[str] = []
    url: str = ""

    def __init__(self, url: str) -> None:
        super().__init__()
        self.url = url
        self.load_model()
        self.start()

    def load_combined_audio(self) -> np.ndarray:
        """Loads a list of files into a single numpy array."""
        cmd: list[str] = ["ffmpeg",
                          "-nostdin",
                          "-threads", "0",
                          "-i"]
        input_cmd: str = "concat:" # List of files to merge for processing
        for file in self.file_list:
            input_cmd = f"{input_cmd}{file}|"
        cmd.append(input_cmd)
        cmd.extend(["-f", "s16le",  # 16-bit
                    "-ac", "1",     # mono
                    "-acodec", "pcm_s16le", # pcm
                    "-ar", "16000", # 16khz sample
                    "-"])
        try:
            out = subprocess.run(cmd, capture_output=True, check=True).stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    def load_model(self) -> None:
        """Loads a model"""
        print(f"{CLEAR}{LOADING_MODEL_MESSAGE}")
        self.model = whisper.load_model(MODEL_NAME, DEVICE_NAME)

    def process_file(self, file_name) -> str:
        """Adds a mp3 segment to the queue for processing"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        if file_name in self.file_list:
            return f"{CLEAR}{LOADING_AUDIO_MESSAGE}"
        self.file_list.append(file_name)
        if len(self.file_list) > NUM_CHUNKS:
            try:
                os.unlink(self.file_list[0])
            except OSError:
                pass
            self.file_list = self.file_list[-NUM_CHUNKS:]
        if len(self.file_list) >= NUM_CHUNKS:
            audio = self.load_combined_audio()
            result = whisper.transcribe(self.model, audio)
            return f"{CLEAR}{result['text']}"
        return f"{CLEAR}{LOADING_AUDIO_MESSAGE}"

    def run(self) -> None:
        """Starts ffmpeg and listens for new files from it"""
        cmd = [
            "ffmpeg",
            "-i", self.url,
            "-f", "segment",
            "-segment_time", str(CHUNK_LENGTH),
            "-strftime", "1",
            f"{TEMP_DIR}/%H-%M-%S.mp3",
            "-v", "verbose"
        ]
        with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            ) as process:
            prev_file = None
            if process.stdout is None:
                raise RuntimeError("stdout is none")
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                match = re.search(r"Opening '([^']*)'", line)
                if match:
                    file = match.group(1)
                    if prev_file:
                        print(self.process_file(prev_file))
                    prev_file = file

def main() -> None:
    """Main function"""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <url>")
        sys.exit(1)

    url = sys.argv[1]
    Subtitles(url)

if __name__ == "__main__":
    main()
