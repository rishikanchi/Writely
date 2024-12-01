from flask import Flask, request, jsonify
import logging
import time
import os
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import threading
from openai import OpenAI
import json
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

app = Flask(__name__)

# Create logs directory if it doesn't exist
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler(log_dir / "writely.log"),
                       logging.StreamHandler()
                   ])
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
if not os.getenv('OPENAI_API_KEY'):
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY environment variable is required")

class MessageBuffer:
    def __init__(self):
        self.buffers = {}
        self.lock = threading.Lock()
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
        self.silence_threshold = 120  # 2 minutes silence threshold
        self.min_words_after_silence = 5  # minimum words needed after silence

    def get_buffer(self, session_id):
        current_time = time.time()
        
        # Cleanup old sessions periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            self.cleanup_old_sessions()
        
        with self.lock:
            if session_id not in self.buffers:
                self.buffers[session_id] = {
                    'messages': [],
                    'last_analysis_time': time.time(),
                    'last_activity': current_time,
                    'words_after_silence': 0,
                    'silence_detected': False,
                    'writing_style': 'default'  # Added for writing style preference
                }
            else:
                # Check for silence period
                time_since_activity = current_time - self.buffers[session_id]['last_activity']
                if time_since_activity > self.silence_threshold:
                    self.buffers[session_id]['silence_detected'] = True
                    self.buffers[session_id]['words_after_silence'] = 0
                    self.buffers[session_id]['messages'] = []  # Clear old messages after silence
                
                self.buffers[session_id]['last_activity'] = current_time
                
        return self.buffers[session_id]

    def cleanup_old_sessions(self):
        current_time = time.time()
        with self.lock:
            expired_sessions = [
                session_id for session_id, data in self.buffers.items()
                if current_time - data['last_activity'] > 3600  # Remove sessions older than 1 hour
            ]
            for session_id in expired_sessions:
                del self.buffers[session_id]
            self.last_cleanup = current_time

# Initialize message buffer
message_buffer = MessageBuffer()

ANALYSIS_INTERVAL = 120  # 120 seconds between analyses

def improve_text(text: str, style: str = 'default') -> str:
    """Improve the given text using OpenAI"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are Writely, an AI writing assistant. Improve the provided text while maintaining its core message. Writing style: {style}"},
                {"role": "user", "content": f"Improve this text:\n{text}"}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        improved_text = response.choices[0].message.content.strip()
        logger.info("Text successfully improved")
        return improved_text
    except Exception as e:
        logger.error(f"Error improving text: {str(e)}")
        return text

def create_writing_prompt(messages: list) -> dict:
    """Create writing assistance prompt with template"""
    discussion_text = "\n".join([msg['text'] for msg in messages])
    
    system_prompt = """You are an AI writing assistant. Analyze the text and provide specific improvements. Focus on:
    1. Clarity and coherence
    2. Grammar and style
    3. Impact and engagement
    
    Current text:
    {text}
    
    Provide specific suggestions for improvement.""".format(text=discussion_text)

    return {
        "notification": {
            "prompt": system_prompt,
            "params": ["style", "context"],
            "context": {
                "writing_style": "default"
            }
        }
    }

@app.route('/webhook', methods=['POST'])
def webhook():
    if request.method == 'POST':
        data = request.json
        session_id = data.get('session_id')
        segments = data.get('segments', [])
        
        if not session_id:
            logger.error("No session_id provided in request")
            return jsonify({"message": "No session_id provided"}), 400

        current_time = time.time()
        buffer_data = message_buffer.get_buffer(session_id)

        # Process new messages
        for segment in segments:
            if not segment.get('text'):
                continue

            text = segment['text'].strip()
            if text:
                timestamp = segment.get('start', 0) or current_time
                is_user = segment.get('is_user', False)

                # Count words after silence
                if buffer_data['silence_detected']:
                    words_in_segment = len(text.split())
                    buffer_data['words_after_silence'] += words_in_segment
                    
                    if buffer_data['words_after_silence'] >= message_buffer.min_words_after_silence:
                        buffer_data['silence_detected'] = False
                        buffer_data['last_analysis_time'] = current_time
                        logger.info(f"Silence period ended for session {session_id}")

                can_append = (
                    buffer_data['messages'] and 
                    abs(buffer_data['messages'][-1]['timestamp'] - timestamp) < 2.0 and
                    buffer_data['messages'][-1].get('is_user') == is_user
                )

                if can_append:
                    buffer_data['messages'][-1]['text'] += ' ' + text
                else:
                    buffer_data['messages'].append({
                        'text': text,
                        'timestamp': timestamp,
                        'is_user': is_user
                    })

        # Check if it's time to analyze
        time_since_last_analysis = current_time - buffer_data['last_analysis_time']

        if (time_since_last_analysis >= ANALYSIS_INTERVAL and 
            buffer_data['messages'] and 
            not buffer_data['silence_detected']):
            
            sorted_messages = sorted(buffer_data['messages'], key=lambda x: x['timestamp'])
            
            # Get concatenated text and improve it
            text_to_improve = " ".join([msg['text'] for msg in sorted_messages])
            improved_text = improve_text(text_to_improve, buffer_data.get('writing_style', 'default'))
            
            notification = create_writing_prompt(sorted_messages)
            notification['notification']['improved_text'] = improved_text
            
            buffer_data['last_analysis_time'] = current_time
            buffer_data['messages'] = []

            logger.info(f"Sending writing improvements for session {session_id}")
            
            return jsonify(notification), 200

        return jsonify({}), 202

@app.route('/webhook/setup-status', methods=['GET'])
def setup_status():
    return jsonify({"is_setup_completed": True}), 200

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "active_sessions": len(message_buffer.buffers),
        "uptime": time.time() - start_time
    })

# Add start time tracking
start_time = time.time()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
