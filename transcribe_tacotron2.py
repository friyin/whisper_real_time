#! python3.7

import argparse
import io
import os
import speech_recognition as sr
import whisper
import torch
import datetime as dt

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
from faster_whisper import WhisperModel
from pydub import AudioSegment

def tstr(time_o):
    return dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def _d(dstr):
    print(f" {tstr(dt.datetime.now())}: {dstr}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="large-v2", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large", "large-v1", "large-v2"])
    parser.add_argument("--non_english", action='store_true', default=True,
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)  
    parser.add_argument("--file", default='/home/freemind/Videos/sample.wav',
                        help="Default microphone name for SpeechRecognition. "
                             "Run this with 'list' to view available Microphones.", type=str)
    parser.add_argument("--outdir", default='output',
                        help="Outputdir (default output). ", type=str)
    parser.add_argument("--frame_rate", default=22050,
                        help="Framerate (default 22050). ", type=int)


    args = parser.parse_args()
    
    # The last time a recording was retreived from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False
    
    # Load / Download model
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    #audio_model = whisper.load_model(model)
    audio_model = WhisperModel(model, device="cuda", compute_type="float16")
    #audio_model = WhisperModel(model, device="cuda")

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    temp_file = NamedTemporaryFile().name
    transcription = ['']

    # Cue the user that we're ready to go.
    _d(f"Model loaded: {model}")

    now = datetime.utcnow()
    # Pull raw recorded audio from the queue.
    
    _d(f"Proceso file: {args.file} frame_rate {args.frame_rate}")
    segments, info = audio_model.transcribe(args.file, beam_size=5)
    af = AudioSegment.from_wav(args.file)
    af = af.set_channels(1).set_frame_rate(args.frame_rate)
    
    _d(f"File: {args.file} length {af.duration_seconds} frame_rate {af.frame_rate}")



    text = ""
    num = 0
    if(not os.path.exists(args.outdir)):
        os.mkdir(args.outdir)
    with open(f'{args.outdir}/filelist.txt', 'w') as f:
        for sg in segments:
            num += 1
            percent = int(sg.end / af.duration_seconds * 100)
            _d(f"Segmento #{num} {percent:02d}% [{sg.start} - {sg.end}]: {sg.text}")
            filename=f'chunk-{num:06d}.wav'
            print(f"{args.outdir}/{filename}|{sg.text}", file=f, flush=True)
            af[sg.start*1000:sg.end*1000].export(f'{args.outdir}/{filename}', format="wav") #Exports to a wav file in the current path.

if __name__ == "__main__":
    main()
