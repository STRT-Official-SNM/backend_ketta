from flask import Flask, request, jsonify
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file for local testing
load_dotenv()

# --- INITIALIZE FLASK APP ---
app = Flask(__name__)

# --- LOAD API KEYS AND INITIALIZE CLIENTS ---
try:
    # Configure Google Gemini
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)

except Exception as e:
    print(f"Error loading Google API key: {e}")


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

        # Add a system_instruction to guide the model's behavior.
        # This tells the model to keep its answers short and conversational.
        system_instruction = "You are a friendly AI assistant, named Ketta. Keep your responses concise, conversational ,curtious and under 40 words. Try to be funny sometimes, depending on conversation history. Don't intorduce yourself unless asked. Act a little like jarvis. Your responses must cater to the user's query, while being human like. If you feel the need to include a pause when your response is being read aloud, add <break time='0.5s'/> at that place and replace the 0.5s with the amount of time you think will be appropriote. for example 'let me think about it. <break time='1.0s'/> Yes you are right.' You were developed by the company called STRT"
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            system_instruction=system_instruction
        )

        chat = model.start_chat(history=history)
        response = chat.send_message(prompt)
        gemini_text = response.text
        print(f"Gemini response (concise): {gemini_text}")

        # 2. Prepare the response
        response_data = {
            "text_response": gemini_text,
            "updated_history": serialize_history(chat.history),
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
