from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
from flask_cors import CORS

# Load model dan tokenizer
model_path = './trained_model/v0.3'
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Inisialisasi aplikasi Flask
app = Flask(__name__)

CORS(app)

def generate_response(question):
    input_text = f"question: {question} </s>"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Route untuk menerima pertanyaan dan mengembalikan jawaban
@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    answer = generate_response(question)
    return jsonify({"question": question, "answer": answer})

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
