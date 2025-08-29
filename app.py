from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch
import os
import logging
import sys
from typing import Generator
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
class Config:
    MODEL_PATH = "llama/models/8B"  # Update this path to where your model is stored
    MAX_LENGTH = 2000
    TEMPERATURE = 0.7
    TOP_P = 0.95
    TOP_K = 40
    REPETITION_PENALTY = 1.1
    MAX_NEW_TOKENS = 500
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LlamaModel:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = Config.DEVICE
        self.streamer = None

    def load_model(self):
        """Load the model and tokenizer"""
        try:
            logger.info(f"Loading model from {Config.MODEL_PATH}")
            logger.info(f"Using device: {self.device}")

            self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
            self.model = AutoModelForCausalLM.from_pretrained(
                Config.MODEL_PATH,
                load_in_8bit=True if self.device == "cuda" else False,
                device_map="auto",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )

            # Enable model evaluation mode
            self.model.eval()
            logger.info("Model loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def generate_response(self, prompt: str) -> Generator[str, None, None]:
        """Generate response using the model with streaming"""
        try:
            # Format the prompt
            formatted_prompt = f"### Human: {prompt.strip()}\n\n### Assistant: "
            
            # Tokenize input
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            
            # Create streamer
            self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
            
            # Generate in a separate thread
            generation_kwargs = dict(
                inputs,
                streamer=self.streamer,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                temperature=Config.TEMPERATURE,
                do_sample=True,
                top_p=Config.TOP_P,
                top_k=Config.TOP_K,
                repetition_penalty=Config.REPETITION_PENALTY,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            # Stream the response
            for text in self.streamer:
                yield text
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            yield f"Error: {str(e)}"

# Initialize model
llama_model = LlamaModel()

@app.before_first_request
def initialize():
    """Initialize the model before first request"""
    if not llama_model.load_model():
        logger.error("Failed to initialize model")
        sys.exit(1)

@app.route('/')
def home():
    """Render the chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    try:
        data = request.json
        if not data or 'prompt' not in data:
            return jsonify({'error': 'No prompt provided'}), 400

        prompt = data['prompt'].strip()
        if not prompt:
            return jsonify({'error': 'Empty prompt'}), 400

        # Log the incoming request
        logger.info(f"Received prompt: {prompt[:50]}...")

        # Generate and stream the response
        def generate():
            for text in llama_model.generate_response(prompt):
                yield text

        return app.response_class(generate(), mimetype='text/plain')

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': llama_model.model is not None,
        'device': llama_model.device
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
