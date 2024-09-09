#!/usr/bin/python3
"""This class transcribes a live ffmpeg stream as subtitles"""
import argparse
import threading
from subprocess import Popen, PIPE
from enum import Enum
from typing import Annotated, List, Union

import numpy as np
import whisper
from pydantic import AnyUrl, BaseModel, Field, FilePath, ValidationError

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
URLFileAnnotation = Annotated[Union[AnyUrl, FilePath] ,
            "URL or File to be streamed."]
RealtimeAnnotation = Annotated[bool,
            "Rather ffmpeg should play in real-time or as fast as it can. Do not use for streams."]
DontclearAnnotation = Annotated[bool,
            "Clear the screen between transcribed lines."]

DEFAULT_MODEL: WhisperModel = WhisperModel.BASE_EN
DEFAULT_DEVICE: WhisperDevice = WhisperDevice.CUDA
DEFAULT_NUM_CHUNKS: int = 2
DEFAULT_CHUNK_LENGTH: int = 3

CLEAR: str = "\033[2J\033[H"  # ANSI clear code
WHISPER_SAMPLE_RATE: int = 16000
FFMPEG_DATA_TYPE: type = np.int16
FFMPEG_DATA_STRING: str = "s16le"
FFMPEG_CHANNELS: int = 1
FFMPEG_LOG_LEVEL: str = "fatal"
FFMPEG_OUTPUT: str = "pipe:"

class SubtitleStreamProperties(BaseModel):
    """Subtitle Stream Properties"""
    device_type: WhisperDeviceAnnotation
    whisper_model: WhisperModelAnnotation
    chunk_duration: ChunkLengthAnnotation
    num_chunks: NumChunksAnnotation
    source: URLFileAnnotation
    ffmpeg_realtime: RealtimeAnnotation
    dont_clear: DontclearAnnotation

class Subtitles(threading.Thread):
    """Reads an ffmpeg stream and does subtitles for it"""
    __model: whisper.Whisper
    __running: bool = True
    __chunks: List[np.ndarray] = []
    __stream_properties: SubtitleStreamProperties
    __chunk_bytes: int
    __process: Popen

    def __init__(self,
                 stream_properties: SubtitleStreamProperties) -> None:
        super().__init__()
        self.__stream_properties = stream_properties
        self.__stream_properties.num_chunks = int(self.__stream_properties.num_chunks)
        self.__chunk_bytes = (self.__stream_properties.chunk_duration *
                              WHISPER_SAMPLE_RATE *
                              np.dtype(FFMPEG_DATA_TYPE).itemsize)
        print(f"Model '{self.__stream_properties.whisper_model}'")
        print(f"Device: '{self.__stream_properties.device_type}'")
        print(f"Chunk Duration: {self.__stream_properties.chunk_duration} seconds")
        print(f"Number of Chunks: {self.__stream_properties.num_chunks}")
        print(f"Source {self.__stream_properties.source}")
        print(f"Chunk Bytes: {self.__chunk_bytes}")
        print("Loading Model")
        self.__model = whisper.load_model(self.__stream_properties.whisper_model,
                                          self.__stream_properties.device_type)
        print("Loaded Model")

    def __write_line(self, line: str):
        """Write a line, clear the screen if configured"""
        if not self.__stream_properties.dont_clear:
            print(f"{CLEAR}{line}", end="", flush=True)
        else:
            print(line, flush=True)

    def __process_chunk(self, data: np.ndarray) -> None:
        """Processes a chunk of audio"""
        self.__chunks.append(data)
        if len(self.__chunks) >= self.__stream_properties.num_chunks:
            combined_audio: np.ndarray = (np.concatenate(self.__chunks).astype(np.float32) /
                                          np.iinfo(FFMPEG_DATA_TYPE).max)
            result = whisper.transcribe(self.__model, combined_audio)
            del self.__chunks[0]
            self.__write_line(str(result['text']))
        else:
            self.__write_line("Receiving Initial Audio")

    def run(self) -> None:
        """Starts ffmpeg and listens for new files from it"""
        cmd: List[str] = ["ffmpeg",
                          "-hide_banner",
                          "-loglevel", FFMPEG_LOG_LEVEL]
        if self.__stream_properties.ffmpeg_realtime:
            cmd.append("-re")
        cmd.extend([      "-i", str(self.__stream_properties.source),
                          "-f", FFMPEG_DATA_STRING,
                          "-ar", str(WHISPER_SAMPLE_RATE),
                          "-ac", str(FFMPEG_CHANNELS),
                          FFMPEG_OUTPUT])
        
        with Popen(cmd, stdout=PIPE, bufsize=self.__chunk_bytes) as self.__process:
            if self.__process.stdout is None:
                raise RuntimeError("stdout is none")
            while self.__running:
                data: bytes = self.__process.stdout.read(self.__chunk_bytes)
                if len(data) == 0:
                    self.__running = False
                    break
                np_data: np.ndarray = np.frombuffer(data, FFMPEG_DATA_TYPE)
                self.__process_chunk(np_data)
            self.__process.wait()

    def stop(self) -> None:
        """Stops the thread"""
        self.__running = False
        if self.__process is not None:
            self.__process.terminate()

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
                    default=DEFAULT_MODEL,
                    help=WhisperModelAnnotation.__metadata__[0]) # pylint: disable=no-member # type: ignore
    parser.add_argument(
                    '-d', '--device',
                    type=WhisperDevice,
                    choices=[device.value for device in WhisperDevice],
                    default=DEFAULT_DEVICE,
                    help=WhisperDeviceAnnotation.__metadata__[0]) # pylint: disable=no-member # type: ignore
    parser.add_argument(
                    '-l', '--chunk_length',
                    type=int,
                    default=DEFAULT_CHUNK_LENGTH,
                    help=ChunkLengthAnnotation.__metadata__[0]) # pylint: disable=no-member # type: ignore
    parser.add_argument(
                    '-n', '--num_chunks',
                    type=int,
                    default=DEFAULT_NUM_CHUNKS,
                    help=NumChunksAnnotation.__metadata__[0]) # pylint: disable=no-member # type: ignore
    parser.add_argument(
                    '-r', '--realtime',
                    help=RealtimeAnnotation.__metadata__[0], # pylint: disable=no-member # type: ignore
                    action='store_true')
    parser.add_argument(
                    '-c', '--dontclear',
                    help=DontclearAnnotation.__metadata__[0], # pylint: disable=no-member # type: ignore
                    action='store_true')

    args = parser.parse_args()

    try:
        stream_properties = SubtitleStreamProperties(
            device_type=args.device,
            whisper_model=args.model,
            chunk_duration=args.chunk_length,
            num_chunks=args.num_chunks,
            source=args.source,
            ffmpeg_realtime=args.realtime,
            dont_clear=args.dontclear
        )
        subtitles: Subtitles = Subtitles(stream_properties)
        try:
            subtitles.start()
            subtitles.join()
        except KeyboardInterrupt:
            pass
        subtitles.stop()
        subtitles.join()
    except ValidationError as e:
        parser.error(str(e))

if __name__ == "__main__":
    main()
