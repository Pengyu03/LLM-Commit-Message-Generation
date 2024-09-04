# app.py

from flask import Flask, request, jsonify, render_template
import review  # Assuming review.py is in the same directory
import json

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    code_text = request.form['code_text']
    diff_id = request.form['diff_id']
    api_key = request.form['api_key']
    site_key = request.form['site_key']
    train_vectors_file = 'vtrain_java.jsonl'
    diff_msg_file = 'javatrainyuan3.jsonl'

    try:
        commit_message = review.generate_commit_message(code_text, diff_id, train_vectors_file, diff_msg_file, api_key, site_key)
        if commit_message == "Invalid site key":
            return jsonify({"error": "Invalid site key"})
        return jsonify({"commit_message": commit_message})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/records', methods=['POST'])
def records():
    site_key = request.form['site_key']
    records = []
    try:
        with open('commit_messages.jsonl', 'r') as f:
            for line in f:
                record = json.loads(line)
                if record['site_key'] == site_key:
                    records.append(record)
        return jsonify({"records": records})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/progress')
def get_progress():
    return review.get_progress()

if __name__ == '__main__':
    app.run(debug=True, port=8080)
