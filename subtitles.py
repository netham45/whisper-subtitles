#!/usr/bin/python3
"""This class transcribes a live ffmpeg stream as subtitles"""
import argparse
import re
import threading
from subprocess import Popen, PIPE
from enum import Enum
from typing import Annotated, List, Optional, Union, Deque, Tuple
import concurrent.futures
from collections import deque

import numpy as np
from faster_whisper import WhisperModel
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
    LARGE_V1 = 'large-v1'
    LARGE_V2 = 'large-v2'
    LARGE_V3 = 'large-v3'
    LARGE_V3_TURBO = 'large-v3-turbo'
    LARGE_V3_TURBO_DISTILL = 'distil-large-v3'
    TURBO = 'turbo'
    TINY_EN = 'tiny.en'
    BASE_EN = 'base.en'
    SMALL_EN = 'small.en'
    MEDIUM_EN = 'medium.en'

WhisperDeviceAnnotation = Annotated[WhisperDevice,
            "Compute device type."]
WhisperModelAnnotation = Annotated[WhisperModel,
            "Whisper model to run."]
ChunkLengthAnnotation = Annotated[float,
            "Chunk length in seconds for audio to be segmented into."]
NumChunksAnnotation = Annotated[int,
            "Number of chunk segments to be transcribed at once."]
NumLinesAnnotation = Annotated[int,
            "Number of lines to output per subtitle refresh"]
HistorySizeAnnotation = Annotated[int,
            "Number of previous segments to use as context for continuity"]
URLFileAnnotation = Annotated[Union[AnyUrl, FilePath],
            "URL or File to be streamed."]
RealtimeAnnotation = Annotated[bool,
            "Process in real-time or as fast as possible. Use for files, not realtime streams."]
DontclearAnnotation = Annotated[bool,
            "Don't clear the screen between transcribed lines."]

class SubtitleStreamProperties(BaseModel):
    """Subtitle Stream Properties"""
    device_type: WhisperDeviceAnnotation
    whisper_model: WhisperModelAnnotation
    chunk_duration: ChunkLengthAnnotation
    num_chunks: NumChunksAnnotation
    source: URLFileAnnotation
    ffmpeg_realtime: RealtimeAnnotation
    dont_clear: DontclearAnnotation
    num_lines: NumLinesAnnotation
    history_size: HistorySizeAnnotation = 3  # Default to 3 previous segments

DEFAULT_MODEL: WhisperModel = WhisperModel.LARGE_V3_TURBO_DISTILL
DEFAULT_DEVICE: WhisperDevice = WhisperDevice.CUDA
DEFAULT_NUM_CHUNKS: int = 15
DEFAULT_NUM_LINES: int = 5
DEFAULT_CHUNK_LENGTH: float = .5
DEFAULT_HISTORY_SIZE: int = 300

CLEAR: str = "\033[2J\033[H"  # ANSI clear code
WHISPER_SAMPLE_RATE: int = 16000
FFMPEG_DATA_TYPE: type = np.int16
FFMPEG_DATA_STRING: str = "s16le"
FFMPEG_CHANNELS: int = 1
FFMPEG_LOG_LEVEL: str = "fatal"
FFMPEG_OUTPUT: str = "pipe:"
MIN_PROBABLY_SPEECH: float = .7

# Rename the imported WhisperModel to avoid class name conflict
from faster_whisper import WhisperModel as FasterWhisperModel

class Subtitles(threading.Thread):
    """Reads an ffmpeg stream and does subtitles for it"""
    __model: FasterWhisperModel
    __running: bool = True
    __chunks: np.ndarray  # 2D numpy array to store audio chunks
    __chunk_count: int = 0  # Counter for current number of chunks
    __max_chunks: int  # Maximum number of chunks to store
    __chunk_size: int  # Size of each chunk in samples
    __stream_properties: SubtitleStreamProperties
    __chunk_bytes: int
    __process: Popen
    __transcript_history: Deque[str] = deque(maxlen=10)  # Store previous transcriptions

    def __init__(self,
                 stream_properties: SubtitleStreamProperties) -> None:
        super().__init__()
        self.__stream_properties = stream_properties
        self.__stream_properties.num_chunks = int(self.__stream_properties.num_chunks)
        self.__max_chunks = self.__stream_properties.num_chunks
        self.__chunk_bytes = round(self.__stream_properties.chunk_duration *
                              WHISPER_SAMPLE_RATE *
                              np.dtype(FFMPEG_DATA_TYPE).itemsize)
        self.__chunk_size = self.__chunk_bytes // np.dtype(FFMPEG_DATA_TYPE).itemsize
        
        # Initialize the 2D numpy array with zeros
        self.__chunks = np.zeros((self.__max_chunks, self.__chunk_size), dtype=FFMPEG_DATA_TYPE)
        
        # Set max length of transcript history based on history_size parameter
        self.__transcript_history = deque(maxlen=self.__stream_properties.history_size)
        
        print(f"Model '{self.__stream_properties.whisper_model}'")
        print(f"Device: '{self.__stream_properties.device_type}'")
        print(f"Chunk Duration: {self.__stream_properties.chunk_duration} seconds")
        print(f"Number of Chunks: {self.__stream_properties.num_chunks}")
        print(f"History Size: {self.__stream_properties.history_size} segments")
        print(f"Source {self.__stream_properties.source}")
        print(f"Chunk Bytes: {self.__chunk_bytes}")
        print("Loading Model")
        # Convert device type to format expected by faster-whisper
        compute_type = "float16" if self.__stream_properties.device_type == WhisperDevice.CUDA else "float32"
        device = "cuda" if self.__stream_properties.device_type == WhisperDevice.CUDA else "cpu"
        # Initialize model using faster-whisper's API
        self.__model = FasterWhisperModel(
            self.__stream_properties.whisper_model.value,  # Model name as a string
            device=device,
            compute_type=compute_type
        )
        print("Loaded Model")

    def __write_line(self, line: str, is_start: bool):
        """Write a line, clear the screen if configured"""
        if is_start and not self.__stream_properties.dont_clear:
            print(f"{CLEAR}{line}", end="", flush=True)
        else:
            print(line, flush=True)

    def __get_previous_context(self) -> str:
        """Get the previous transcript context to use as initial prompt"""
        if not self.__transcript_history:
            return ""
        
        # Join the history with spaces to create a continuous prompt
        retval: str = ""
        for idx, _ in enumerate(self.__transcript_history):
            if idx % self.__stream_properties.num_chunks != 0:
                continue
            retval = self.__transcript_history[(len(self.__transcript_history) - idx) - 1] + retval
        return retval

    def transcribe_with_timeout(self, audio: np.ndarray) -> Optional[List]:
        """Transcribe with a timeout"""
        # Get previous context to use as initial prompt
        initial_prompt = self.__get_previous_context()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Use the proper faster-whisper transcribe method with initial_prompt
            future: concurrent.futures.Future = executor.submit(
                self.__model.transcribe,
                audio,
                beam_size=6,
                language="en",
                initial_prompt=initial_prompt if initial_prompt else None
            )
            try:
                # faster-whisper returns a tuple (segments, info)
                result = future.result(
                    timeout=self.__stream_properties.chunk_duration * .5)
                return result[0]  # Return just the segments
            except concurrent.futures.TimeoutError:
                future.cancel()
                print("\nTranscription timed out")
                return None

    def __process_audio_buffer(self) -> None:
        """Processes the audio buffer when we have enough chunks"""
        # Arrange chunks in chronological order for processing
        # If we've wrapped around the buffer, we need to reorder the chunks
        if self.__chunk_count > self.__max_chunks:
            # Calculate starting index (oldest chunk)
            start_idx = self.__chunk_count % self.__max_chunks
            # Reorder chunks: [oldest_chunk:end, beginning:oldest_chunk]
            ordered_chunks = np.vstack((
                self.__chunks[start_idx:],
                self.__chunks[:start_idx]
            ))
        else:
            # We haven't wrapped around yet, so chunks are already in order
            ordered_chunks = self.__chunks
        
        # Convert to float32 and normalize
        combined_audio = ordered_chunks.flatten().astype(np.float32) / np.iinfo(FFMPEG_DATA_TYPE).max
        
        segments = self.transcribe_with_timeout(combined_audio)
        
        if segments is None:
            return
        
        # Faster-whisper returns a generator, we convert to list to get the last N items
        segments_list = list(segments)
        
        # Update transcript history with new segments
        for segment in segments_list:
            if segment.avg_logprob > -.6:  # Only add confident segments to history
                self.__transcript_history.append(segment.text.strip())
        
        # Take the last N segments based on num_lines
        display_segments = segments_list[-(self.__stream_properties.num_lines - 1):] if segments_list else []
        output: str = ""
        for segment in display_segments:
            # Faster-whisper uses avg_logprob instead of no_speech_prob
            # Higher avg_logprob is better (more confident)
            if segment.avg_logprob > -.6:  # Adjusted threshold, may need tuning
                output += f" {segment.text}"
            output = re.sub("  *", " ", output)
            parts: List[str] = output.split(" ")
            last: int = 0
            second_to_last: int = 0
            third_to_last: int = 0
            for idx, part in enumerate(parts):
                if idx == 0 or len(part) == 0:
                    continue
                if part[0].isupper():
                    if idx == 0 or parts[idx-1][-1:] in [".", "!", "?"]:
                        third_to_last = second_to_last
                        second_to_last = last
                        last = idx
            truncated_output: str = (#(" ".join(parts[third_to_last:second_to_last])).strip() + "\n" +
                                     (" ".join(parts[second_to_last:last])).strip() + "\n" +
                                     (" ".join(parts[last:-1])).strip() + "\n" + 
                                     (" ".join(parts[-1:])).strip())
            self.__write_line(truncated_output, True)

    def run(self) -> None:
        """Starts ffmpeg and listens for new files from it"""
        cmd: List[str] = ["ffmpeg",
                          "-hide_banner",
                          "-loglevel", FFMPEG_LOG_LEVEL]
        if self.__stream_properties.ffmpeg_realtime:
            cmd.append(   "-re")
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
                
                # Calculate the current chunk index in our circular buffer
                chunk_idx = self.__chunk_count % self.__max_chunks
                
                # Calculate samples in this chunk
                samples = len(data) // np.dtype(FFMPEG_DATA_TYPE).itemsize
                
                # Direct read into the 2D array at the current position
                # Create a view of the buffer as 1D array of bytes
                chunk_bytes_view = self.__chunks[chunk_idx, :samples].view(np.uint8)
                
                # Reshape to match the incoming bytes
                bytes_per_sample = np.dtype(FFMPEG_DATA_TYPE).itemsize
                chunk_bytes_view = chunk_bytes_view.reshape(samples * bytes_per_sample)
                
                # Copy the bytes directly into the buffer
                chunk_bytes_view[:len(data)] = np.frombuffer(data, np.uint8)
                
                # Increment chunk counter
                self.__chunk_count += 1
                
                # Process the buffer if we have enough chunks
                if self.__chunk_count >= self.__max_chunks:
                    self.__process_audio_buffer()
                else:
                    self.__write_line(f"Receiving Initial Audio {self.__chunk_count} / {self.__stream_properties.num_chunks}", True)
            
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
                    type=float,
                    default=DEFAULT_CHUNK_LENGTH,
                    help=ChunkLengthAnnotation.__metadata__[0]) # pylint: disable=no-member # type: ignore
    parser.add_argument(
                    '-n', '--num_chunks',
                    type=int,
                    default=DEFAULT_NUM_CHUNKS,
                    help=NumChunksAnnotation.__metadata__[0]) # pylint: disable=no-member # type: ignore
    parser.add_argument(
                    '-t', '--num_lines',
                    type=int,
                    default=DEFAULT_NUM_LINES,
                    help=NumLinesAnnotation.__metadata__[0]) # pylint: disable=no-member # type: ignore
    parser.add_argument(
                    '-i', '--history',
                    type=int,
                    default=DEFAULT_HISTORY_SIZE,
                    help=HistorySizeAnnotation.__metadata__[0]) # pylint: disable=no-member # type: ignore
    parser.add_argument(
                    '-r', '--realtime',
                    help=RealtimeAnnotation.__metadata__[0], # pylint: disable=no-member # type: ignore
                    action='store_true')
    parser.add_argument(
                    '-c', '--dont_clear',
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
            dont_clear=args.dont_clear,
            num_lines=args.num_lines,
            history_size=args.history
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
