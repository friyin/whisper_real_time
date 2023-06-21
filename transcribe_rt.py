#! python3.7

import argparse
import io
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import speech_recognition as sr
import torch
import time

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
from faster_whisper import WhisperModel



def cls():
    os.system('cls' if os.name=='nt' else 'clear')


def rewrite(path, data):
    if not path:
        return

    with open(path, "w", encoding="utf-8") as f:
        print(f" *** {datetime.now()}: lines {len(data)}", file=f)
        f.write(os.linesep)
        f.writelines(l + os.linesep for l in data)
        f.write(os.linesep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large", "large-v1", "large-v2"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)  
    parser.add_argument("--output_file", default="transcription.txt",
                        help="Output text to file.", type=str)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
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
        
    source = sr.Microphone(sample_rate=16000)
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
    
    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    cls()
    print(f"{datetime.now()}: Model loaded. OK")
    if args.output_file:
        print(f"Output file: {args.output_file}")

    while True:
        try:
            # Infinite loops are bad for processors, must sleep.
            while not data_queue.qsize():
                sleep(0.1)
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now
                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data
                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())
                # Write wav data to the temporary file as bytes.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())
                # Read the transcription.
                #result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
                #segments, info = audio_model.transcribe(temp_file, beam_size=5)
                tstart = time.time()
                segments, info = audio_model.transcribe(temp_file, beam_size=5)
                tend = time.time()
                ttrans = tend - tstart
                #segments, info = audio_model.transcribe(temp_file, beam_size=1, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
                text = ""
                for segment in segments:
                    text += segment.text.strip() + ' '
                # If we detected a pause between recordings, add a new item to our transcripion.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                else:
                    ##transcription[-1] = text
                    transcription[-1] = text
                # Clear the console to reprint the updated transcription.
                cls()
                print(f" *** {datetime.now()}: phrase_complete {phrase_complete} lines {len(transcription)} infer_time {ttrans:.3f}s")

                for line in transcription:
                    print(line)
                rewrite(args.output_file, transcription)
                # Flush stdout.
                print('', end='', flush=True)
                ## Infinite loops are bad for processors, must sleep.
                #while not len(data_queue):
                #    print("Sleeping")
                #    sleep(0.25)
        except KeyboardInterrupt:
            break

    rewrite(args.output_file, transcription)

    print(f" *** {datetime.now()}: transcription end")


if __name__ == "__main__":
    main()
