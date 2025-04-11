from flask import Flask, render_template, request, jsonify
from champion_recommender import ChampionRecommender
import os

app = Flask(__name__)
recommender = ChampionRecommender("data/processed_champion_data.csv")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    query = request.json.get('query', '')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    results = recommender.recommend_champions(query)
    
    if isinstance(results, str):
        return jsonify({'error': results}), 400
    
    return jsonify({'recommendations': results})

if __name__ == '__main__':
    app.run(debug=True) 