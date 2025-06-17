from flask import Flask, request, jsonify, Response
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json

load_dotenv()
app = Flask(__name__)

try:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"Error loading Google API key: {e}")

# --- CHAT ENDPOINT (Unchanged) ---
@app.route('/api/chat', methods=['POST'])
def chat_handler():
    # This endpoint's logic remains the same.
    data = request.get_json()
    prompt = data.get('prompt')
    history = data.get('history', [])

    if not prompt:
        return jsonify({"error": "Prompt is missing"}), 400

    def text_stream_generator():
        try:
            system_instruction = "You are a friendly AI assistant, Ketta, expert at intent recognition and chitchat..."
            model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=system_instruction)
            chat = model.start_chat(history=history)
            response_stream = chat.send_message(prompt, stream=True)
            for chunk in response_stream:
                if chunk.text: yield chunk.text
        except Exception as e:
            print(f"Error during Gemini stream: {e}")
            yield "I'm sorry, I encountered an error."

    return Response(text_stream_generator(), mimetype='text/plain')

# --- SUMMARIZATION ENDPOINT (With Enhanced Instructions) ---
@app.route('/api/summarize-history', methods=['POST'])
def summarize_handler():
    data = request.get_json()
    history = data.get('history')

    if not history:
        return jsonify({"error": "History is missing"}), 400

    try:
        print("Starting intelligent history summarization...")

        # --- THIS IS THE KEY IMPROVEMENT ---
        # A much more detailed system instruction for intelligent summarization.
        summarization_system_instruction = """
You are an expert Conversation Summarizer AI. Your job is to condense a conversation history while retaining all critical information.

RULES:
1.  **KEEP all valuable information**: This includes personal facts (e.g., "my name is John", "my favorite color is blue"), user preferences, memories, and any facts or data exchanged.
2.  **DISCARD all conversational fluff**: This includes greetings ("hi", "hello"), pleasantries ("how are you?", "I'm fine"), confirmations ("okay", "got it"), and thank-yous. Also remove turns that add no new information.
3.  **KEEP context**: If a "hello" is followed by "my name is Alex", discard the "hello" turn but ensure the "my name is Alex" turn is kept.
4.  **MAINTAIN FORMAT**: Your output MUST be ONLY a valid JSON list of objects, following the exact 'role' and 'parts' structure of the input.
5.  **NO EXTRA TEXT**: Do not add any explanatory text, comments, or markdown formatting like ```json around your response.
"""
        model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=summarization_system_instruction)
        
        prompt_for_summarizer = f"Please apply your rules to summarize the following conversation history:\n\n{json.dumps(history, indent=2)}"
        
        response = model.generate_content(prompt_for_summarizer)
        
        summarized_history_text = response.text
        
        if summarized_history_text.startswith("```json"):
            summarized_history_text = summarized_history_text[7:].strip()
        if summarized_history_text.endswith("```"):
            summarized_history_text = summarized_history_text[:-3].strip()

        summarized_history = json.loads(summarized_history_text)
        
        print("History summarization complete.")
        return jsonify({"summarized_history": summarized_history})

    except Exception as e:
        print(f"An error occurred during summarization: {e}")
        return jsonify({"summarized_history": history, "error": str(e)}), 500

@app.route('/')
def home():
    return "Streaming Gemini Text Backend is running!"
