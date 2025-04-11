# League of Legends Champion Recommender Web App

A Flask-based web application that recommends League of Legends champions based on natural language descriptions.

## Features

- Modern, responsive web interface
- Natural language processing for champion recommendations
- Example queries for quick testing
- Detailed champion information display
- Visual difficulty rating with stars

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Make sure you have the processed champion data file (`lol_champions_processed.csv`) in the root directory
2. Start the Flask application:
   ```bash
   python app.py
   ```
3. Open your web browser and navigate to `http://localhost:5000`

## Usage

1. Enter your champion preferences in natural language (e.g., "I want a tanky champion who can deal damage")
2. Click "Find Champions" or press Enter
3. View the recommended champions with their details
4. Try the example queries for quick testing

## Example Queries

- "I want to fight and move fast and tank a lot"
- "I want to win easily, I am a beginner"
- "I need a support champion who can heal"
- "I want a ranged champion for mid lane"

## Technical Details

- Frontend: HTML, CSS, JavaScript with Bootstrap 5
- Backend: Flask with Python
- Natural Language Processing: NLTK
- Data Processing: Pandas, NumPy 