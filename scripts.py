import sounddevice as sd
import numpy as np
import librosa
import time
import torch
import soundfile as sf
import io
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Audio Configuration
SAMPLE_RATE = 16000  # Standard speech recognition sample rate
SILENCE_THRESHOLD = 0.01  # Silence detection threshold
SILENCE_DURATION = 2  # Stop recording if silence lasts this long (seconds)

# Load Whisper Model for Speech-to-Text (STT)
stt_model_name = "openai/whisper-small"
stt_processor = WhisperProcessor.from_pretrained(stt_model_name)
stt_model = WhisperForConditionalGeneration.from_pretrained(stt_model_name)
stt_model.config.forced_decoder_ids = None  
print("✅ Hugging Face Whisper Model Loaded Successfully")

# Load Intent Classification Model
intent_model_name = "Serj/intent-classifier"
intent_tokenizer = AutoTokenizer.from_pretrained(intent_model_name)
intent_model = AutoModelForSeq2SeqLM.from_pretrained(intent_model_name)
print("✅ 'Serj/intent-classifier' Model Loaded Successfully!")

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
    print("🎤 Listening... Speak now (Stops when silent for 2 seconds)...")

    audio_buffer = []
    silence_start = None
    stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype=np.float32)

    with stream:
        while True:
            audio_chunk, _ = stream.read(int(sample_rate * 0.1))  # Read 100ms chunks
            audio_buffer.append(audio_chunk)

            # Compute volume level
            volume_level = np.max(np.abs(audio_chunk))

            # Check for silence
            if volume_level < SILENCE_THRESHOLD:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_DURATION:
                    print("🛑 Silence detected! Stopping recording.")
                    break
            else:
                silence_start = None  # Reset silence detection if user speaks

    # Convert list of chunks to NumPy array
    recorded_audio = np.concatenate(audio_buffer, axis=0).flatten()
    print("✅ Recording complete.")
    return recorded_audio  # Return raw audio


# Function to Convert Speech to Text using Whisper STT
def transcribe_audio(audio_data, sample_rate=16000):
    print("⏳ Transcribing speech...")

    # Process the NumPy audio array directly (no need to convert to WAV)
    input_features = stt_processor(audio_data, sampling_rate=sample_rate, return_tensors="pt").input_features 

    # Generate transcription
    with torch.no_grad():
        predicted_ids = stt_model.generate(input_features)

    # Decode token IDs into text
    transcription = stt_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription  # Return transcribed text


# Function to Classify User Intent Using Few-Shot Learning
def classify_intent_few_shot(user_input):
    print("🔍 Classifying Intent...")

    # Format the input text with few-shot examples
    formatted_input = few_shot_prompt.format(user_input)

    # Tokenize the input
    inputs = intent_tokenizer(formatted_input, return_tensors="pt", max_length=512, truncation=True)

    # Get model predictions
    with torch.no_grad():
        outputs = intent_model.generate(**inputs)

    # Decode the generated response
    detected_intent = intent_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return detected_intent


# 🔹 Main Function to Run Everything Automatically
def main():
    print("\n🎤 Speak after running this script. It will listen and detect your intent automatically.")
    
    # Step 1: Record Audio
    audio_data = record_audio_until_silence()

    # Step 2: Convert Speech to Text
    transcribed_text = transcribe_audio(audio_data)

    # Step 3: Classify Intent
    detected_intent = classify_intent_few_shot(transcribed_text)

    # Step 4: Display Results
    print("\n📝 Final Transcription:", transcribed_text)
    print("🎯 Detected Intent:", detected_intent)


# Run the script
if __name__ == "__main__":
    main()
