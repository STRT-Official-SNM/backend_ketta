from flask import Flask, request, jsonify, Response
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

try:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Error loading Google API key: {e}")

@app.route('/api/chat', methods=['POST'])
def chat_handler():
    data = request.get_json()
    prompt = data.get('prompt')
    history = data.get('history', [])

    if not prompt:
        return jsonify({"error": "Prompt is missing"}), 400

    def text_stream_generator():
        """A generator function that streams text from the Gemini API."""
        try:
            print("Starting Gemini text stream...")
            system_instruction = "You are a friendly AI assistant, named Ketta. Keep your responses concise, conversational ,witty, fun, and under 40 words. You were developed by the company called STRT"
            model = genai.GenerativeModel(
                'gemini-1.5-flash',
                system_instruction=system_instruction
            )
            chat = model.start_chat(history=history)

            # Use generate_content with stream=True
            response_stream = chat.send_message(prompt, stream=True)

            # Yield each text chunk as it arrives from the Gemini API
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
            print("Finished Gemini text stream.")
        except Exception as e:
            print(f"An error occurred during Gemini stream: {e}")
            yield "I'm sorry, I encountered an error."

    # Return a Flask Response object with the generator.
    # The mimetype 'text/plain' signals to the client that this is a raw text stream.
    return Response(text_stream_generator(), mimetype='text/plain')

@app.route('/')
def home():
    return "Streaming Gemini Text Backend is running!"
