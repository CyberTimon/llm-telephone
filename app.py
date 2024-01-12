from flask import Flask, request, send_file, render_template, send_from_directory
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from pydantic import BaseModel
import base64
import os
import random
import numpy as np
import torch
import websocket
import json
from TTS.api import TTS
from faster_whisper import WhisperModel
import sys, os
import time
import re

##########

""" ADMIN CONFIG """ 

voice_name = "voice.wav"


os.system("clear")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=False).to("cuda")
print("Loading swiss-finetuned whisper model...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
app = Flask(__name__)


class Item(BaseModel):
    audiofile_path: str

def remove_marks(script):
    return script#re.sub('[!?\\-.]', ',', script)
    
    
def make_prompt(prompt):
    System_String_Start = "<|im_start|>system\n"
    default_input = "<|im_end|>\n<|im_start|>user\n"
    default_output = "<|im_end|>\n<|im_start|>assistant\n"
    final_prompt = default_input + prompt + default_output
    return final_prompt
    
    
def speech_to_text(audiofile_path):
    segments, _ = model.transcribe(
        audiofile_path,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        language="de")
    
    transcribed_text = ""
    for segment in segments:
        transcribed_text += segment.text + " "
    
    return remove_marks(transcribed_text)

def run(audiofile_path):
    llm_prompt = speech_to_text(audiofile_path)
    print("Recieved audio which contains following text:", llm_prompt)
    final_prompt = make_prompt(llm_prompt)
    llm_text = text_to_text(final_prompt, gen_max_new_tokens=8192)
    llm_text = remove_marks(llm_text)
    mp3_path = text_to_speech(llm_text)
    return mp3_path

def text_to_text(gen_prompt="", gen_temp=0.7, gen_top_p=1, gen_top_k=40, gen_repetition_penalty=1.1, gen_max_new_tokens=8192, gen_stop_strings=[]):
    ws = websocket.create_connection("ws://localhost:5005/")
    params = {
        'gen_prompt': gen_prompt,
        'gen_temp': gen_temp,
        'gen_top_p': gen_top_p,
        'gen_top_k': gen_top_k,
        'gen_repetition_penalty': gen_repetition_penalty,
        'gen_max_new_tokens': gen_max_new_tokens,
        'gen_stop_strings': gen_stop_strings,
        'stream': False
    }
    ws.send(json.dumps(params))
    result = ws.recv()
    ws.close()
    return result

def text_to_speech(TEXT_TO_GENERATE, warmup=False):
    reference_voice = voice_name
    print("Generating Speech from :", TEXT_TO_GENERATE)
    if not warmup:
        filename = ("output/voice-ID" + str(random.choice(range(1, 100))) + ".wav")
        tts.tts_to_file(text=TEXT_TO_GENERATE,
                        file_path=filename)
        return filename
    else:
        tts.tts(text="Hallo Welt!")
        
text_to_speech("", True)

@app.route('/process_audio', methods=['POST'])
def process_audio():
    audio = request.files['file']
    audio_segment = AudioSegment.from_file(audio)
    audio_segment.export(voice_name, format="wav")
    mp3_path = run(voice_name)
    return {"message": mp3_path}

@app.route('/get_audio/<path:filename>', methods=['GET'])
def get_audio(filename):
    return send_file(filename, as_attachment=True)

@app.route("/")
def home():
    return render_template("index.html", charset="utf-8")
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=21122)
