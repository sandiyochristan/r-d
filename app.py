from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

app = Flask(__name__)

# Initialize model and tokenizer
MODEL_PATH = "llama/models/8B"  # Update this path to where your model is stored
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        load_in_8bit=True,  # Enable 8-bit quantization
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print("Model and tokenizer loaded successfully!")

def generate_response(prompt, max_length=500):
    # Ensure model is loaded
    global tokenizer, model
    if tokenizer is None or model is None:
        load_model()
    
    # Format the prompt
    formatted_prompt = f"### Human: {prompt}\n\n### Assistant: "
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            top_k=40,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant's response
    response = response.split("### Assistant:")[-1].strip()
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({'error': 'No prompt provided'}), 400
    
    try:
        response = generate_response(data['prompt'])
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load the model when the server starts
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False)
