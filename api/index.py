from flask import Flask, request, jsonify
import google.generativeai as genai
import os
from dotenv import load_dotenv
import base64

from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings

# Load environment variables from .env file for local testing
load_dotenv()

# --- INITIALIZE FLASK APP ---
app = Flask(__name__)

# --- LOAD API KEYS AND INITIALIZE CLIENTS ---
try:
    # Configure Google Gemini
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)

    # Initialize the ElevenLabs Client
    elevenlabs_client = ElevenLabs(
      api_key=os.environ.get("ELEVENLABS_API_KEY")
    )

except Exception as e:
    print(f"Error loading API key or initializing clients: {e}")


# --- HELPER FUNCTION TO SERIALIZE GEMINI HISTORY ---
def serialize_history(history):
    """Converts Gemini's chat history to a JSON-serializable format."""
    return [
        {'role': msg.role, 'parts': [part.text for part in msg.parts]}
        for msg in history
    ]

# --- API ENDPOINT ---
@app.route('/api/chat', methods=['POST'])
def chat_handler():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    prompt = data.get('prompt')
    history = data.get('history', [])

    if not prompt:
        return jsonify({"error": "Prompt is missing"}), 400

    try:
        # 1. Interact with Gemini
        print("Generating text with Gemini...")

        # --- THIS IS THE KEY CHANGE ---
        # Add a system_instruction to guide the model's behavior.
        # This tells the model to keep its answers short and conversational.
        system_instruction = "You are a friendly AI assistant, named Ketta. Keep your responses concise, conversational ,witty, fun, and under 40 words. You were developed by the company called STRT"
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            system_instruction=system_instruction
        )
        # --- END OF CHANGE ---

        chat = model.start_chat(history=history)
        response = chat.send_message(prompt)
        gemini_text = response.text
        print(f"Gemini response (concise): {gemini_text}")


        # 2. Generate audio with ElevenLabs
        print("Generating audio with ElevenLabs...")
        
        audio_stream = elevenlabs_client.generate(
            text=gemini_text,
            voice=Voice(
                voice_id='56AoDkrOh6qfVPDXZ7Pt', # Your chosen voice ID
                settings=VoiceSettings(stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True)
            ),
            model="eleven_multilingual_v2"
        )
        
        audio_bytes = b"".join(audio_stream)
        
        # 3. Encode and prepare the response
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        print("Audio generated and encoded.")

        response_data = {
            "text_response": gemini_text,
            "updated_history": serialize_history(chat.history),
            "audio_content": audio_base64
        }

        return jsonify(response_data)

    except Exception as e:
        # Log the error for debugging
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal error occurred"}), 500

# Health check endpoint
@app.route('/')
def home():
    return "Backend is running!"

if __name__ == '__main__':
    app.run(debug=True)
