from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/api/data', methods=['POST'])
def process_data():
    data = request.json
    # Process the data as needed
    return jsonify({"message": "Data processed successfully", "data": data}), 200

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
