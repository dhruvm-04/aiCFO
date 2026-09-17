from flask import Flask, jsonify, render_template, request
from graph import graph

app = Flask(__name__)

def serialize_analysis(value):
    if value is None:
        return None

    return value.model_dump()

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/api/analyze")
def analyze():
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query", "")).strip()

    if not query:
        return jsonify({"error": "Enter an analysis prompt."}), 400

    try:
        result = graph.invoke({"user_query": query})

        return jsonify({
            "user_query": query,
            "automotive_analysis": serialize_analysis(
                result.get("automotive_analysis")
            ),
            "macro_analysis": serialize_analysis(
                result.get("macro_analysis")
            ),
            "cfo_analysis": serialize_analysis(
                result.get("cfo_analysis")
            ),
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )