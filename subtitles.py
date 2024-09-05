import fastapi
import whisper
import numpy as np
import os
from subprocess import CalledProcessError, run

from fastapi import FastAPI

app = FastAPI()


LoadedModel = ""
DeviceMode = ""
FileList = []

def load_combined_audio(files: list[str], sr: int = 16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i"]

    input = "concat:"
    for file in files:
        input = f"{input}{file}|"
    cmd.append(input)
    cmd.extend([
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ])
    print(cmd)
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") 

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0



@app.get('/loadModel')
def loadModel(requestModel, useGpu):
    global LoadedModel
    global Model
    global DeviceMode

    device = 'cuda'
    if useGpu == 'False': device = 'cpu'

    if LoadedModel != requestModel or DeviceMode != device:
        LoadedModel = requestModel
        DeviceMode = device
        Model = whisper.load_model(LoadedModel, DeviceMode)

@app.get('/processFile')
def processFile(fileName):
    global Model
    global FileList
    FileList.append(fileName)
    if len(FileList) > 10:
        try:
            os.unlink(FileList[0])
        except OSError:
            pass
        FileList.pop(0)
    audio = load_combined_audio(FileList)
    result = whisper.transcribe(Model, audio) 

    # print the recognized text
    print(result['text'])
    return result['text']
