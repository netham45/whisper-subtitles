#!/usr/bin/python3
"""This class transcribes an MP3 stream as subtitles"""
import os
import subprocess
import re
import tempfile
import threading
import argparse
from typing import Optional
import numpy as np
import whisper

CLEAR: str = "\033[2J\033[H"  # ANSI clear code

class Subtitles(threading.Thread):
    """Reads an MP3 stream and does subtitles for it"""
    __model: Optional[whisper.Whisper] = None
    __running: bool = True
    __file_list: list[str] = []
    __url: str = ""
    __model_name: str = ""
    __device_type: str = ""
    __temp_dir: str = ""
    __chunk_length: int = 0
    __num_chunks: int = 0

    def __init__(self,
                 url: str,
                 model_name: str,
                 device_type: str,
                 chunk_length: int,
                 num_chunks: int) -> None:
        super().__init__()
        self.__url = url
        self.__model_name = model_name
        self.__device_type = device_type
        self.__temp_dir = tempfile.mkdtemp(prefix="subtitles_")
        self.__chunk_length = int(chunk_length)
        self.__num_chunks = int(num_chunks)
        print(f"Model '{self.__model_name}'")
        print(f"Device: '{self.__device_type}'")
        print(f"Temp Dir Path: '{self.__temp_dir}'")
        print(f"Chunk Length: {self.__chunk_length}")
        print(f"Number of Chunks: {self.__num_chunks}")
        print("Loading Model")
        self.__model = whisper.load_model(self.__model_name, self.__device_type)
        print("Loaded Model")
        self.start()

    def __load_combined_audio(self) -> np.ndarray:
        """Loads a list of files into a single numpy array."""
        input_cmd: str = "concat:" # List of files to merge for processing
        for file in self.__file_list:
            input_cmd += f"{file}|"
        cmd: list[str] = ["ffmpeg",
                          "-nostdin",
                          "-threads", "0",
                          "-i", input_cmd,
                          "-f", "s16le",  # 16-bit
                          "-ac", "1",     # mono
                          "-acodec", "pcm_s16le", # pcm
                          "-ar", "16000", # 16khz sample
                          "pipe:"]
        try:
            out = subprocess.run(cmd, capture_output=True, check=True).stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    def __process_file(self, file_name) -> None:
        """Adds a mp3 segment to the queue for processing"""
        if self.__model is None:
            raise RuntimeError("Model not loaded")
        if file_name in self.__file_list:
            return
        self.__file_list.append(file_name)

        if len(self.__file_list) > self.__num_chunks:
            try:
                os.unlink(self.__file_list[0])
            except OSError:
                pass
            self.__file_list = self.__file_list[-self.__num_chunks:]
        if len(self.__file_list) >= self.__num_chunks:
            audio = self.__load_combined_audio()
            result = whisper.transcribe(self.__model, audio)
            print(f"{CLEAR}{result['text']}", end="", flush=True)
        else:
            print(f"{CLEAR}Receiving Initial Audio", end="", flush=True)

    def run(self) -> None:
        """Starts ffmpeg and listens for new files from it"""
        cmd: list[str] = ["ffmpeg",
                          "-i", self.__url,
                          "-f", "segment",
                          "-segment_time", str(self.__chunk_length),
                          "-strftime", "1",
                         f"{self.__temp_dir}/%H-%M-%S.mp3",
                          "-v", "verbose"]
        with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            ) as process:
            prev_file: str = ""
            if process.stdout is None:
                raise RuntimeError("stdout is none")
            while self.__running:
                line: str = process.stdout.readline()
                if not line:
                    break
                match: Optional[re.Match[str]] = re.search(r"Opening '([^']*)'", line)
                if match is not None:
                    file = match.group(1)
                    if prev_file:
                        self.__process_file(prev_file)
                    prev_file = file

    def stop(self) -> None:
        """Stops the thread"""
        self.__running = False

def main() -> None:
    """Main function"""
    parser = argparse.ArgumentParser(
                    prog='Whisper Subtitles',
                    description='Plays a stream using ffmpeg and shows subtitles for the stream')
    parser.add_argument(
                    'source',
                    help="File or URL for ffmpeg to play")
    parser.add_argument(
                    '-m', '--model',
                    default="base.en",
                    help="Whisper model to use")
    parser.add_argument(
                    '-d', '--device',
                    default="cuda",
                    help="'cuda' for GPU or 'cpu' for CPU")
    parser.add_argument(
                    '-l', '--chunk_length',
                    default="3",
                    help="Length of chunks")
    parser.add_argument(
                    '-n', '--num_chunks',
                    default="2",
                    help="Number of chunks to keep/process")
    args = parser.parse_args()
    Subtitles(args.source, args.model, args.device, args.chunk_length, args.num_chunks)

if __name__ == "__main__":
    main()
