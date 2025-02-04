from flask import Flask, render_template, request, jsonify
import sounddevice as sd
import numpy as np
import torch
import time
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# Audio Configuration
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 2

# Load Whisper Model for Speech-to-Text (STT)
stt_model_name = "openai/whisper-small"
stt_processor = WhisperProcessor.from_pretrained(stt_model_name)
stt_model = WhisperForConditionalGeneration.from_pretrained(stt_model_name)
stt_model.config.forced_decoder_ids = None  

# Load Intent Classification Model
intent_model_name = "Serj/intent-classifier"
intent_tokenizer = AutoTokenizer.from_pretrained(intent_model_name)
intent_model = AutoModelForSeq2SeqLM.from_pretrained(intent_model_name)

# Few-Shot Prompt for Better Classification
few_shot_prompt = """\
Customer: I want to buy a new phone.
Intent: Shopping

Customer: How do I pay my electricity bill?
Intent: Bill Payment

Customer: My internet is not working properly.
Intent: Technical Issue

Customer: I need help resetting my bank password.
Intent: Customer Support

Customer: Can you recommend the best laptop for gaming?
Intent: Shopping

Customer: I was trying to make a payment, but I keep getting a 'Payment not authorized' error. 
Intent: Billing Issue

Customer: I received the wrong product and need to return it for a refund.
Intent: Returns & Refunds

Customer: My internet is not working properly and keeps disconnecting frequently.
Intent: Technical Issue

Customer: I am looking to upgrade my phone to the latest model and need advice on the best option.
Intent: Product Inquiry

Customer: I need help resetting my password for my online banking account.
Intent: Customer Support

Customer: I want to track my order that was supposed to be delivered yesterday.
Intent: Delivery / Order Tracking

Customer: I need to change my shipping address for an order I placed yesterday.
Intent: Order Modification

Customer: I am considering subscribing to your premium plan but want to understand the differences between the Basic and Premium plans.
Intent: Subscription Inquiry

Customer: {} 
Intent:"""

# Function to Record Audio Until Silence is Detected
def record_audio_until_silence(sample_rate=SAMPLE_RATE):
    print("🎤 Listening... Speak now")
    audio_buffer = []
    silence_start = None
    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype=np.float32)
    
    with stream:
        while True:
            audio_chunk, _ = stream.read(int(sample_rate * 0.1))
            audio_buffer.append(audio_chunk)
            volume_level = np.max(np.abs(audio_chunk))
            if volume_level < SILENCE_THRESHOLD:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_DURATION:
                    break
            else:
                silence_start = None
    
    recorded_audio = np.concatenate(audio_buffer, axis=0).flatten()
    return recorded_audio

# Function to Convert Speech to Text
def transcribe_audio(audio_data, sample_rate=16000):
    input_features = stt_processor(audio_data, sampling_rate=sample_rate, return_tensors="pt").input_features 
    with torch.no_grad():
        predicted_ids = stt_model.generate(input_features)
    return stt_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

# Function to Classify User Intent
def classify_intent_few_shot(user_input):
    formatted_input = few_shot_prompt.format(user_input)
    inputs = intent_tokenizer(formatted_input, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = intent_model.generate(**inputs)
    return intent_tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    audio_data = record_audio_until_silence()
    transcribed_text = transcribe_audio(audio_data)
    detected_intent = classify_intent_few_shot(transcribed_text)
    return jsonify({"transcription": transcribed_text, "intent": detected_intent})

if __name__ == "__main__":
    app.run(debug=True)
