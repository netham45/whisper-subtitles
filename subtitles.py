#!/usr/bin/python3
"""This class transcribes a live ffmpeg stream as subtitles"""
import argparse
import os
import re
import subprocess
import tempfile
import threading
from enum import Enum
from typing import Annotated, List, Optional, Union

import numpy as np
import whisper
from pydantic import AnyUrl, BaseModel, Field, FilePath, ValidationError

CLEAR: str = "\033[2J\033[H"  # ANSI clear code

class WhisperDevice(str, Enum):
    """Available Whisper Devices"""
    CUDA = 'cuda'
    CPU = 'cpu'

class WhisperModel(str, Enum):
    """Available Whisper Models"""
    TINY = 'tiny'
    BASE = 'base'
    SMALL = 'small'
    MEDIUM = 'medium'
    LARGE = 'large'
    LARGE_V1 = 'large_v1'
    LARGE_V2 = 'large_v2'
    LARGE_V3 = 'large_v3'
    TINY_EN = 'tiny.en'
    BASE_EN = 'base.en'
    SMALL_EN = 'small.en'
    MEDIUM_EN = 'medium.en'

WhisperDeviceAnnotation = Annotated[WhisperDevice, "Compute device type."]

WhisperModelAnnotation = Annotated[WhisperModel, "Whisper model to run."]

ChunkLengthAnnotation = Annotated[int,
            "Chunk length in seconds for audio to be segmented into.",
            Field(strict=True, ge=0, le=10)]

NumChunksAnnotation = Annotated[int,
            "Number of chunk segments to be transcribed at once.",
            Field(strict=True, ge=0, le=10)]

URLFileAnnotation = Annotated[Union[AnyUrl, FilePath] , "URL or File to be streamed."]

class SubtitleStreamProperties(BaseModel):
    """Subtitle Stream Properties"""
    device_type: WhisperDeviceAnnotation
    whisper_model: WhisperModelAnnotation
    chunk_length: ChunkLengthAnnotation
    num_chunks: NumChunksAnnotation

class Subtitles(threading.Thread):
    """Reads an ffmpeg stream and does subtitles for it"""
    __model: whisper.Whisper
    __running: bool = True
    __file_list: List[str] = []
    __stream_properties: SubtitleStreamProperties
    __url: URLFileAnnotation
    __temp_dir: str

    def __init__(self,
                 url: URLFileAnnotation,
                 stream_properties: SubtitleStreamProperties) -> None:
        super().__init__()
        self.__url = url
        self.__stream_properties = stream_properties
        self.__temp_dir = tempfile.mkdtemp(prefix="subtitles_")
        self.__chunk_length = int(self.__stream_properties.chunk_length)
        self.__stream_properties.num_chunks = int(self.__stream_properties.num_chunks)
        print(f"Model '{self.__stream_properties.whisper_model}'")
        print(f"Device: '{self.__stream_properties.device_type}'")
        print(f"Temp Dir Path: '{self.__temp_dir}'")
        print(f"Chunk Length: {self.__chunk_length}")
        print(f"Number of Chunks: {self.__stream_properties.num_chunks}")
        print("Loading Model")
        self.__model = whisper.load_model(self.__stream_properties.whisper_model,
                                          self.__stream_properties.device_type)
        print("Loaded Model")

    def __load_combined_audio(self) -> np.ndarray:
        """Loads a list of files into a single numpy array."""
        cmd: List[str] = ["ffmpeg",
                          "-nostdin",
                          "-threads", "0",
                          "-i", "concat:" + "|".join(self.__file_list),
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
        if file_name in self.__file_list:
            return
        self.__file_list.append(file_name)

        if len(self.__file_list) > self.__stream_properties.num_chunks:
            try:
                os.unlink(self.__file_list[0])
            except OSError:
                print(f"\nFailed to remove temp file {self.__file_list[0]}")
            self.__file_list = self.__file_list[-self.__stream_properties.num_chunks:]
        if len(self.__file_list) >= self.__stream_properties.num_chunks:
            audio = self.__load_combined_audio()
            result = whisper.transcribe(self.__model, audio)
            print(f"{CLEAR}{result['text']}", end="", flush=True)
        else:
            print(f"{CLEAR}Receiving Initial Audio", end="", flush=True)

    def run(self) -> None:
        """Starts ffmpeg and listens for new files from it"""
        cmd: List[str] = ["ffmpeg",
                          "-i", str(self.__url),
                          "-f", "segment",
                          "-segment_time", str(self.__stream_properties.chunk_length),
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
        for file in self.__file_list:
            try:
                os.unlink(file)
            except OSError:
                pass
        try:
            os.rmdir(self.__temp_dir)
        except OSError:
            pass

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
                    type=str,
                    help=URLFileAnnotation.__metadata__[0]) # pylint: disable=no-member # type: ignore
    parser.add_argument(
                    '-m', '--model',
                    type=WhisperModel,
                    choices=[model.value for model in WhisperModel],
                    default=WhisperModel.BASE_EN,
                    help=WhisperModelAnnotation.__metadata__[0]) # pylint: disable=no-member # type: ignore
    parser.add_argument(
                    '-d', '--device',
                    type=WhisperDevice,
                    choices=[device.value for device in WhisperDevice],
                    default=WhisperDevice.CUDA,
                    help=WhisperDeviceAnnotation.__metadata__[0]) # pylint: disable=no-member # type: ignore
    parser.add_argument(
                    '-l', '--chunk_length',
                    type=int,
                    default=3,
                    help=ChunkLengthAnnotation.__metadata__[0]) # pylint: disable=no-member # type: ignore
    parser.add_argument(
                    '-n', '--num_chunks',
                    type=int,
                    default=2,
                    help=NumChunksAnnotation.__metadata__[0]) # pylint: disable=no-member # type: ignore

    args = parser.parse_args()

    try:
        stream_properties = SubtitleStreamProperties(
            device_type=args.device,
            whisper_model=args.model,
            chunk_length=args.chunk_length,
            num_chunks=args.num_chunks
        )
        subtitles: Subtitles = Subtitles(args.source, stream_properties)
        try:
            subtitles.start()
            subtitles.join()
        except KeyboardInterrupt:
            subtitles.stop()
        except RuntimeError:
            subtitles.stop()
    except ValidationError as e:
        parser.error(str(e))

if __name__ == "__main__":
    main()
