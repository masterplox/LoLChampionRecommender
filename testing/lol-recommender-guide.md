# League of Legends Natural Language Champion Recommender
## Complete Implementation Guide

This guide will walk you through creating a natural language champion recommender system for League of Legends, allowing users to describe what they want in plain English (like "I want to fight and move fast and tank a lot") and get appropriate champion recommendations.

## Project Overview

Your recommender will:
1. Process natural language queries about champion preferences
2. Match those preferences to champion attributes
3. Rank champions by how well they match the query
4. Provide recommendations with explanations

## Implementation Steps

### Phase 1: Data Preparation

#### Step 1: Set Up Your Development Environment
```bash
# Create a project directory
mkdir lol-champion-recommender
cd lol-champion-recommender

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install pandas numpy scikit-learn nltk spacy matplotlib seaborn
```

#### Step 2: Load and Preprocess the Dataset
Create a new Python file called `data_processor.py`:

```python
import pandas as pd
import numpy as np
import json
import re

class LoLDataProcessor:
    def __init__(self, data_path):
        """Initialize with path to the LoL champion dataset"""
        self.df = pd.read_csv(data_path)
        self.preprocess_data()
        
    def preprocess_data(self):
        """Preprocess and clean the LoL dataset"""
        # Rename the unnamed column to 'champion'
        if '' in self.df.columns:
            self.df = self.df.rename(columns={'': 'champion'})
        
        # Extract stats from the stats string
        self.df['stats_parsed'] = self.df['stats'].apply(self.parse_stats)
        
        # Extract key stats as direct columns
        self.df['hp'] = self.df['stats_parsed'].apply(lambda x: x.get('hp_base', 0) if x else 0)
        self.df['move_speed'] = self.df['stats_parsed'].apply(lambda x: x.get('ms', 0) if x else 0)
        self.df['attack_damage'] = self.df['stats_parsed'].apply(lambda x: x.get('dam_base', 0) if x else 0)
        self.df['armor'] = self.df['stats_parsed'].apply(lambda x: x.get('arm_base', 0) if x else 0)
        self.df['magic_resist'] = self.df['stats_parsed'].apply(lambda x: x.get('mr_base', 0) if x else 0)
        self.df['attack_range'] = self.df['stats_parsed'].apply(lambda x: x.get('range', 0) if x else 0)
        
        # Clean role information
        self.df['roles'] = self.df['role'].apply(self.parse_role_string)
        
        # Parse positions
        self.df['positions'] = self.df['client_positions'].apply(self.parse_role_string)
        
        # Create derived features
        self.create_derived_features()
        
        # Create a more readable dataset for the recommender
        self.create_recommender_dataset()
        
    def parse_stats(self, stats_str):
        """Parse the stats string into a dictionary"""
        if not isinstance(stats_str, str):
            return None
            
        try:
            # Replace single quotes with double quotes for valid JSON
            json_ready = stats_str.replace("'", '"').replace('None', 'null')
            # Fix keys without quotes
            json_ready = re.sub(r'(\w+):', r'"\1":', json_ready)
            # Parse the JSON
            return json.loads(json_ready)
        except:
            # If parsing fails, return None
            return None
    
    def parse_role_string(self, role_str):
        """Parse role strings like "{'Top'}" into lists"""
        if not isinstance(role_str, str):
            return []
            
        # Remove braces and quotes, split by comma if multiple
        cleaned = role_str.replace("{", "").replace("}", "").replace("'", "")
        return [role.strip() for role in cleaned.split(',')]
    
    def create_derived_features(self):
        """Create useful derived features for recommendations"""
        # Tanky score (combination of health, armor, magic resist, and toughness rating)
        self.df['tankiness'] = (
            self.df['hp'] / self.df['hp'].max() * 0.4 +
            self.df['armor'] / self.df['armor'].max() * 0.3 +
            self.df['magic_resist'] / self.df['magic_resist'].max() * 0.3
        ) * (self.df['toughness'] / 3)  # Scale by toughness rating
        
        # Damage score based on attack damage and damage rating
        self.df['damage_score'] = (
            self.df['attack_damage'] / self.df['attack_damage'].max()
        ) * (self.df['damage'] / 3)  # Scale by damage rating
        
        # Mobility score (based directly on mobility rating and move speed)
        self.df['mobility_score'] = (
            self.df['move_speed'] / self.df['move_speed'].max() * 0.3 +
            self.df['mobility'] / 3 * 0.7  # Mobility rating has bigger impact
        )
        
        # Ease of use (inverted difficulty)
        self.df['beginner_friendly'] = 1 - ((self.df['difficulty'] - 1) / 2)  # Scale to 0-1
        
        # Combat control score
        self.df['combat_control'] = self.df['control'] / 3
        
        # Support/utility capability
        self.df['utility_score'] = self.df['utility'] / 3
        
        # Melee/Ranged flag
        self.df['is_melee'] = self.df['rangetype'] == 'Melee'
        
        # Create role flags
        for role in ['Fighter', 'Tank', 'Mage', 'Assassin', 'Support', 'Marksman']:
            self.df[f'is_{role.lower()}'] = self.df['herotype'].str.contains(role, case=False, na=False) | \
                                            self.df['alttype'].str.contains(role, case=False, na=False)
    
    def create_recommender_dataset(self):
        """Create a clean dataset specifically for the recommender"""
        features = [
            'champion', 'title', 'herotype', 'alttype', 'rangetype', 
            'tankiness', 'damage_score', 'mobility_score', 'beginner_friendly',
            'combat_control', 'utility_score', 'is_melee',
            'is_fighter', 'is_tank', 'is_mage', 'is_assassin', 'is_support', 'is_marksman',
            'roles', 'positions', 'adaptivetype', 'difficulty'
        ]
        self.recommender_df = self.df[features].copy()
        
    def save_processed_data(self, output_path):
        """Save the processed dataset"""
        self.recommender_df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")

# Example usage
if __name__ == "__main__":
    processor = LoLDataProcessor("140325_LoL_champion_data_original.csv")
    processor.save_processed_data("lol_champions_processed.csv")
    print("Sample of processed data:")
    print(processor.recommender_df.head())
```

#### Step 3: Test Data Processing

Create a script to test your data processing:

```python
# test_data_processing.py
from data_processor import LoLDataProcessor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Process the data
processor = LoLDataProcessor("140325_LoL_champion_data_original.csv")
df = processor.recommender_df

# Display basic info
print(f"Dataset contains {len(df)} champions")
print("\nFeature columns:")
print(df.columns.tolist())

# Show distributions of key metrics
metrics = ['tankiness', 'damage_score', 'mobility_score', 
           'beginner_friendly', 'combat_control', 'utility_score']

plt.figure(figsize=(15, 10))
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[metric], kde=True)
    plt.title(f'Distribution of {metric}')
    plt.xlabel(metric)

plt.tight_layout()
plt.savefig('champion_metrics_distribution.png')
print("Generated distribution plot as 'champion_metrics_distribution.png'")

# Champions by primary type
type_counts = df['herotype'].value_counts()
print("\nChampions by primary type:")
print(type_counts)

# Get top champions for each key metric
print("\nTop champions by metric:")
for metric in metrics:
    top_champs = df.sort_values(by=metric, ascending=False).head(5)
    print(f"\nTop 5 champions by {metric}:")
    for _, row in top_champs.iterrows():
        print(f"- {row['champion']}: {row[metric]:.2f}")

# Save processed data
processor.save_processed_data("lol_champions_processed.csv")
```

Run this script to ensure your data processing is working correctly:
```bash
python test_data_processing.py
```

### Phase 2: Building the Recommender System

#### Step 4: Create the Recommender Class

Create a file called `champion_recommender.py`:

```python
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

class ChampionRecommender:
    def __init__(self, data_path):
        """Initialize with path to the processed LoL champion dataset"""
        self.df = pd.read_csv(data_path)
        self.setup_keyword_mappings()
        
    def setup_keyword_mappings(self):
        """Define mappings between natural language keywords and champion attributes"""
        # Core keyword to feature mappings
        self.keyword_mappings = {
            # Movement and mobility
            'fast': ['mobility_score'],
            'mobile': ['mobility_score'],
            'quick': ['mobility_score'],
            'speed': ['mobility_score'],
            'dash': ['mobility_score'],
            'movement': ['mobility_score'],
            
            # Tankiness
            'tank': ['tankiness', 'is_tank'],
            'tanky': ['tankiness', 'is_tank'],
            'durable': ['tankiness'],
            'tough': ['tankiness'],
            'survive': ['tankiness'],
            'health': ['tankiness'],
            'armor': ['tankiness'],
            'resist': ['tankiness'],
            
            # Combat
            'fight': ['damage_score', 'is_fighter'],
            'damage': ['damage_score'],
            'attack': ['damage_score'],
            'kill': ['damage_score'],
            'fighter': ['is_fighter', 'damage_score'],
            'melee': ['is_melee'],
            'ranged': ['is_melee:inverse'],
            
            # Skill level
            'beginner': ['beginner_friendly'],
            'easy': ['beginner_friendly'],
            'simple': ['beginner_friendly'],
            'new': ['beginner_friendly'],
            'novice': ['beginner_friendly'],
            
            # Game phase
            'early': ['damage_score', 'mobility_score'],  # Early game often favors damage and mobility
            'late': ['tankiness', 'utility_score'],       # Late game often favors tanks and utility
            
            # Control
            'control': ['combat_control'],
            'cc': ['combat_control'],
            'stun': ['combat_control'],
            'slow': ['combat_control'],
            
            # Utility
            'utility': ['utility_score'],
            'support': ['utility_score', 'is_support'],
            'heal': ['utility_score', 'is_support'],
            'shield': ['utility_score'],
            'help': ['utility_score', 'is_support'],
            
            # Champion types
            'mage': ['is_mage'],
            'assassin': ['is_assassin'],
            'marksman': ['is_marksman'],
            'adc': ['is_marksman'],
            'carry': ['is_marksman', 'damage_score'],
        }
        
        # Add position-based keywords
        self.position_keywords = {
            'top': 'Top',
            'mid': 'Middle',
            'middle': 'Middle',
            'jungle': 'Jungle',
            'jg': 'Jungle',
            'bot': 'Bottom',
            'bottom': 'Bottom',
            'adc': 'Bottom',
            'supp': 'Support',
            'support': 'Support'
        }
        
    def preprocess_query(self, query):
        """Preprocess a natural language query"""
        # Convert to lowercase
        query = query.lower()
        
        # Remove punctuation
        query = query.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(query)
        
        # Remove stop words (but keep some that might be important like "not")
        stop_words = set(stopwords.words('english')) - {'no', 'not', 'very', 'can', 'don', 'don\'t', 'cant', 'can\'t'}
        tokens = [token for token in tokens if token not in stop_words]
        
        return tokens
    
    def extract_features_from_query(self, tokens):
        """Extract relevant features from query tokens"""
        feature_weights = {}
        position_preference = None
        
        # Check for negations
        negation_indexes = [i for i, token in enumerate(tokens) if token in ['no', 'not', 'don', 'dont', 'cant']]
        
        for i, token in enumerate(tokens):
            # Check if this token is under negation influence
            is_negated = any(neg_idx < i and i - neg_idx <= 2 for neg_idx in negation_indexes)
            
            # Check position keywords
            if token in self.position_keywords:
                position_preference = self.position_keywords[token]
                continue
                
            # Check attribute keywords
            if token in self.keyword_mappings:
                for feature in self.keyword_mappings[token]:
                    if ':inverse' in feature:
                        # Handle inverse features (e.g., 'ranged' → 'is_melee:inverse')
                        base_feature = feature.split(':')[0]
                        if is_negated:  # Double negation
                            if base_feature in feature_weights:
                                feature_weights[base_feature] += 1
                            else:
                                feature_weights[base_feature] = 1
                        else:  # Normal inverse
                            if base_feature in feature_weights:
                                feature_weights[base_feature] -= 1
                            else:
                                feature_weights[base_feature] = -1
                    else:
                        # Regular features
                        if is_negated:
                            if feature in feature_weights:
                                feature_weights[feature] -= 1
                            else:
                                feature_weights[feature] = -1
                        else:
                            if feature in feature_weights:
                                feature_weights[feature] += 1
                            else:
                                feature_weights[feature] = 1
        
        return feature_weights, position_preference
    
    def recommend_champions(self, query, top_n=5):
        """Get champion recommendations based on a natural language query"""
        # Preprocess query
        tokens = self.preprocess_query(query)
        
        # Extract features
        feature_weights, position_preference = self.extract_features_from_query(tokens)
        
        if not feature_weights and not position_preference:
            return "I couldn't extract specific preferences from your query. Try describing what you want more specifically, using terms like 'tank', 'damage', 'support', 'mobile', 'easy', etc."
        
        # Filter by position if specified
        filtered_df = self.df
        if position_preference:
            # Convert from string to list if necessary
            filtered_df = filtered_df[filtered_df['positions'].apply(
                lambda x: position_preference in (x if isinstance(x, list) else eval(x) if isinstance(x, str) else [])
            )]
            
            if len(filtered_df) == 0:
                return f"No champions found for position: {position_preference}. Try another position or remove this filter."
        
        # Calculate scores based on feature weights
        scores = pd.Series(0.0, index=filtered_df.index)
        
        for feature, weight in feature_weights.items():
            if feature in filtered_df.columns:
                # Normalize feature values to 0-1 range if not already
                if filtered_df[feature].max() > 1:
                    normalized_feature = filtered_df[feature] / filtered_df[feature].max()
                else:
                    normalized_feature = filtered_df[feature]
                
                scores += normalized_feature * weight
        
        # Get top recommendations
        filtered_df = filtered_df.copy()
        filtered_df['score'] = scores
        top_champions = filtered_df.sort_values('score', ascending=False).head(top_n)
        
        # Create recommendation results
        recommendations = []
        for _, champion in top_champions.iterrows():
            # Generate reason for recommendation
            reasons = self.generate_recommendation_reason(champion, feature_weights)
            
            recommendations.append({
                'champion': champion['champion'],
                'title': champion['title'],
                'score': champion['score'],
                'reasons': reasons,
                'type': f"{champion['herotype']} / {champion['alttype']}" if pd.notna(champion['alttype']) else champion['herotype'],
                'position': champion['positions'],
                'difficulty': champion['difficulty']
            })
        
        return recommendations
    
    def generate_recommendation_reason(self, champion, feature_weights):
        """Generate reasons why this champion matches the query"""
        reasons = []
        
        # Add reasons based on the champion's strengths that match query preferences
        for feature, weight in feature_weights.items():
            if feature in champion.index and weight > 0:
                if feature == 'tankiness' and champion[feature] > 0.6:
                    reasons.append('good tankiness')
                elif feature == 'damage_score' and champion[feature] > 0.6:
                    reasons.append('high damage output')
                elif feature == 'mobility_score' and champion[feature] > 0.6:
                    reasons.append('excellent mobility')
                elif feature == 'beginner_friendly' and champion[feature] > 0.7:
                    reasons.append('easy to learn')
                elif feature == 'combat_control' and champion[feature] > 0.7:
                    reasons.append('strong crowd control')
                elif feature == 'utility_score' and champion[feature] > 0.6:
                    reasons.append('good utility')
                elif feature.startswith('is_') and champion[feature] > 0:
                    role = feature.replace('is_', '')
                    reasons.append(f'{role} playstyle')
            elif feature in champion.index and weight < 0:
                # Handle negative preferences (e.g., "not a tank")
                if feature == 'is_melee' and champion[feature] < 0.5:
                    reasons.append('ranged champion')
        
        if not reasons:
            return "Good match for your preferences"
        
        return reasons

# Example usage
if __name__ == "__main__":
    recommender = ChampionRecommender("lol_champions_processed.csv")
    query = "I want a champion who is tanky and has good damage"
    results = recommender.recommend_champions(query)
    
    if isinstance(results, str):
        print(results)
    else:
        print(f"Top recommendations for: '{query}'")
        for i, champ in enumerate(results, 1):
            print(f"{i}. {champ['champion']} ({champ['type']})")
            print(f"   Match score: {champ['score']:.2f}")
            print(f"   Reasons: {', '.join(champ['reasons'])}")
            print(f"   Difficulty: {champ['difficulty']}/3")
            print("")
```

#### Step 5: Test the Recommender

Create a script to test your recommender with various queries:

```python
# test_recommender.py
from champion_recommender import ChampionRecommender

# Initialize recommender
recommender = ChampionRecommender("lol_champions_processed.csv")

# Test queries
test_queries = [
    "I want to fight and move fast and tank a lot",
    "I want to win easily, I am a beginner",
    "I should be able to kill champions early",
    "I need a support champion who can heal",
    "I want a ranged champion for mid lane",
    "I want a champion that's good for top lane and can take damage"
]

# Test each query
for query in test_queries:
    print("\n" + "=" * 50)
    print(f"Query: {query}")
    print("=" * 50)
    
    results = recommender.recommend_champions(query)
    
    if isinstance(results, str):
        print(results)
    else:
        print(f"Top recommendations:")
        for i, champ in enumerate(results, 1):
            print(f"{i}. {champ['champion']} ({champ['type']})")
            print(f"   Match score: {champ['score']:.2f}")
            print(f"   Reasons: {', '.join(champ['reasons'])}")
            print(f"   Difficulty: {champ['difficulty']}/3")
            print("")
```

Run the test:
```bash
python test_recommender.py
```

### Phase 3: Creating the User Interface

#### Step 6: Build a Simple Command-Line Interface

Create a command-line interface for your recommender in `recommender_cli.py`:

```python
import sys
from champion_recommender import ChampionRecommender

def main():
    # Check if data file is provided
    data_file = "lol_champions_processed.csv"
    
    # Initialize the recommender
    recommender = ChampionRecommender(data_file)
    
    print("\n======= League of Legends Champion Recommender =======")
    print("Describe what kind of champion you're looking for.")
    print("Example: 'I want a champion who is tanky and has good damage'")
    print("Type 'exit' to quit.")
    print("=" * 55 + "\n")
    
    while True:
        # Get user query
        query = input("What kind of champion are you looking for? > ")
        
        # Check for exit command
        if query.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
        
        # Get recommendations
        results = recommender.recommend_champions(query)
        
        # Display results
        print("\nRecommendations:")
        if isinstance(results, str):
            print(results)
        else:
            for i, champ in enumerate(results, 1):
                print(f"{i}. {champ['champion']} - {champ['title']}")
                print(f"   Type: {champ['type']}")
                print(f"   Position: {', '.join(champ['position']) if isinstance(champ['position'], list) else champ['position']}")
                print(f"   Difficulty: {champ['difficulty']}/3")
                print(f"   Why: {', '.join(champ['reasons'])}")
                print("")
        
        print("-" * 55)

if __name__ == "__main__":
    main()
```

Run the CLI:
```bash
python recommender_cli.py
```

#### Step 7: Build a Simple GUI (Optional but Recommended)

Create a graphical interface using Tkinter in `recommender_gui.py`:

```python
import tkinter as tk
from tkinter import ttk, scrolledtext
import pandas as pd
from champion_recommender import ChampionRecommender

class ChampionRecommenderUI:
    def __init__(self, root, data_path):
        self.root = root
        self.root.title("LoL Champion Recommender")
        self.root.geometry("800x600")
        
        # Initialize the recommender
        self.recommender = ChampionRecommender(data_path)
        
        # Create UI components
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title_label = ttk.Label(
            self.root, 
            text="League of Legends Champion Recommender", 
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=20)
        
        # Description
        desc_text = (
            "Describe what kind of champion you're looking for in natural language.\n"
            "Example: 'I want a fast champion who can tank damage and is good for beginners'"
        )
        desc_label = ttk.Label(self.root, text=desc_text, font=("Arial", 10))
        desc_label.pack(pady=10)
        
        # Query entry
        query_frame = ttk.Frame(self.root)
        query_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.query_entry = ttk.Entry(query_frame, width=70, font=("Arial", 11))
        self.query_entry.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        
        search_button = ttk.Button(query_frame, text="Find Champions", command=self.search)
        search_button.pack(side=tk.RIGHT)
        
        # Results area
        results_frame = ttk.LabelFrame(self.root, text="Recommended Champions")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.results_text = scrolledtext.ScrolledText(
            results_frame, 
            wrap=tk.WORD, 
            width=80, 
            height=15, 
            font=("Arial", 11)
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Example queries
        examples_frame = ttk.LabelFrame(self.root, text="Example Queries")
        examples_frame.pack(fill=tk.X, padx=20, pady=10)
        
        examples = [
            "I want to fight and move fast and tank a lot",
            "I want to win easily, I am a beginner",
            "I should be able to kill champions early",
            "I need a support champion who can heal"
        ]
        
        for example in examples:
            example_button = ttk.Button(
                examples_frame, 
                text=example, 
                command=lambda e=example: self.use_example(e)
            )
            example_button.pack(anchor=tk.W, padx=10, pady=5)
    
    def use_example(self, example):
        """Fill the query field with an example"""
        self.query_entry.delete(0, tk.END)
        self.query_entry.insert(0, example)
    
    def search(self):
        """Search for champions based on the query"""
        query = self.query_entry.get()
        if not query:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Please enter a query describing what you're looking for.")
            return
        
        # Get recommendations
        recommendations = self.recommender.recommend_champions(query)
        
        # Display results
        self.results_text.delete(1.0, tk.END)
        
        if isinstance(recommendations, str):
            # Error message
            self.results_text.insert(tk.END, recommendations)
        else:
            # Valid recommendations
            self.results_text.insert(tk.END, f"Top Recommendations for: '{query}'\n\n")
            
            for i, champ in enumerate(recommendations, 1):
                champ_text = f"{i}. {champ['champion']} - {champ['title']}\n"
                champ_text += f"   Type: {champ['type']}\n"
                
                position_text = champ['position']
                if isinstance(position_text, list):
                    position_text = ', '.join(position_text)
                elif isinstance(position_text, str) and position_text.startswith('{'):
                    # Handle string representation of list
                    position_text = position_text.replace('{', '').replace('}', '').replace("'", '')
                
                champ_text += f"   Position: {position_text}\n"
                champ_text += f"   Difficulty: {champ['difficulty']}/3\n"
                champ_text += f"   Why: {', '.join(champ['reasons'])}\n\n"
                
                self.results_text.insert(tk.END, champ_text)

# Start the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ChampionRecommenderUI(root, "lol_champions_processed.csv")
    root.mainloop()
```

Run the GUI:
```bash
python recommender_gui.py
```

### Phase 4: Evaluation and Refinement

#### Step 8: Evaluate and Refine the Recommender

Create an evaluation script to tune your recommender:

```python
# evaluate_recommender.py
from champion_recommender import ChampionRecommender
import pandas as pd

# Load the recommender
recommender = ChampionRecommender("lol_champions_processed.csv")

# Define test cases with expected champion types
test_cases = [
    {
        "query": "I want a tanky champion", 
        "expected_types": ["Tank", "Fighter"]
    },
    {
        "query": "I want a beginner-friendly champion",
        "expected_difficulty": 1
    },
    {
        "query": "I want a mobile assassin",
        "expected_types": ["Assassin"]
    },
    {
        "query": "I want a support who can heal",
        "expected_types": ["Support"]
    },
    {
        "query": "I want a champion for mid lane",
        "expected_positions": ["Middle"]
    }
]

# Evaluate each test case
print("Recommender Evaluation\n")

for i, test in enumerate(test_cases, 1):
    print(f"Test {i}: {test['query']}")
    results = recommender.recommend_champions(test['query'])
    
    if isinstance(results, str):
        print(f"❌ Error: {results}")
        continue
    
    # Evaluate results
    success = True
    
    # Check champion types
    if 'expected_types' in test:
        type_match_count = 0
        for champ in results:
            champ_types = champ['type'].split(' / ')
            if any(expected in champ_types for expected in test['expected_types']):
                type_match_count += 1
        
        type_match_rate = type_match_count / len(results)
        print(f"Type match rate: {type_match_rate:.0%}")
        if type_match_rate < 0.6:  # At least 60% should match
            success = False
    
    # Check difficulty
    if 'expected_difficulty' in test:
        diff_match_count = sum(1 for champ in results if champ['difficulty'] <= test['expected_difficulty'])
        diff_match_rate = diff_match_count / len(results)
        print(f"Difficulty match rate: {diff_match_rate:.0%}")
        if diff_match_rate < 0.6:  # At least 60% should match
            success = False
    
    # Check positions
    if 'expected_positions' in test:
        pos_match_count = 0
        for champ in results:
            positions = champ['position']
            if isinstance(positions, str):
                positions = positions.replace('{', '').replace('}', '').replace("'", '').split(', ')
            if any(expected in positions for expected in test['expected_positions']):
                pos_match_count += 1
        
        pos_match_rate = pos_match_count / len(results)
        print(f"Position match rate: {pos_match_rate:.0%}")
        if pos_match_rate < 0.6:  # At least 60% should match
            success = False
    
    # Print results summary
    if success:
        print("✅ Test passed\n")
    else:
        print("❌ Test failed\n")
        print("Recommendations:")
        for champ in results[:3]:
            print(f"- {champ['champion']} ({champ['type']})")
        print("")

print("Evaluation complete!")
```

Run the evaluation:
```bash
python evaluate_recommender.py
```

#### Step 9: Improve Keyword Mappings

Based on your evaluation, you may need to refine the keyword mappings in the `ChampionRecommender` class. Here's how to add more mappings:

```python
# Add additional mappings to improve the recommender
additional_mappings = {
    # Add synonyms for existing keywords
    'mobile': ['mobility_score'],
    'speedy': ['mobility_score'],
    'agile': ['mobility_score'],
    
    'powerful': ['damage_score'],
    'strong': ['damage_score'],
    'burst': ['damage_score'],
    
    'durable': ['tankiness'],
    'beefy': ['tankiness'],
    'bulky': ['tankiness'],
    
    'simple': ['beginner_friendly'],
    'starter': ['beginner_friendly'],
    'newbie': ['beginner_friendly'],
    
    # Game-specific terms
    'gank': ['mobility_score', 'combat_control'],
    'engage': ['tankiness', 'mobility_score', 'combat_control'],
    'peel': ['combat_control', 'utility_score'],
    'carry': ['damage_score'],
    'scale': ['tankiness', 'damage_score'],
    'hypercarry': ['damage_score'],
    'splitpush': ['mobility_score', 'damage_score'],
    
    # Champion roles in team fights
    'frontline': ['tankiness', 'is_tank'],
    'backline': ['damage_score', 'is_marksman'],
    'initiator': ['tankiness', 'combat_control'],
    'protector': ['utility_score', 'is_support'],
}

# Add these to your ChampionRecommender class
```

### Phase 5: Deployment and Documentation

#### Step 10: Create Final Package and Documentation

Create a comprehensive README.md file:

```markdown
# League of Legends Champion Recommender

This tool recommends League of Legends champions based on natural language descriptions of what you're looking for in a champion.

## Features

- Process natural language queries like "I want to fight and move fast and tank a lot"
- Match queries to champion attributes and roles
- Recommend champions with explanations of why they match
- Simple GUI interface for easy interaction

## Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install pandas numpy scikit-learn nltk
   ```
3. Download NLTK resources (will be done automatically on first run)

## Usage

### GUI Interface (Recommended)

Run the GUI application:
```
python recommender_gui.py
```

### Command Line Interface

Run the CLI application:
```
python recommender_cli.py
```

## Example Queries

- "I want to fight and move fast and tank a lot"
- "I want to win easily, I am a beginner"
- "I should be able to kill champions early"
- "I need a support champion who can heal"
- "I want a ranged champion for mid lane"

## How It Works

1. **Query Processing**: Tokenizes and processes the natural language query
2. **Feature Extraction**: Maps keywords to champion attributes
3. **Matching**: Scores champions based on extracted features
4. **Ranking**: Sorts champions by score and returns top matches
5. **Explanation**: Generates reasons why each champion matches the query

## Customization

You can customize the keyword mappings in `champion_recommender.py` to improve the matching for specific terms or play styles.

## License

This project is for educational purposes only. League of Legends and all champion data are property of Riot Games.
```

#### Step 11: Package the Project

Create a simple setup script for your project:

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="lol-champion-recommender",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "nltk",
    ],
    entry_points={
        'console_scripts': [
            'lol-recommend-cli=recommender_cli:main',
            'lol-recommend-gui=recommender_gui:main',
        ],
    },
)
```

Install your package:
```bash
pip install -e .
```

## Final Steps

Congratulations! You now have a complete Natural Language Champion Recommender for League of Legends. Here's a quick checklist to make sure everything is working:

1. ✅ Data processing is working properly
2. ✅ Recommender can interpret natural language queries
3. ✅ Recommendations match the query intent
4. ✅ User interface is intuitive and easy to use
5. ✅ Documentation is complete

### Additional Enhancements for the Future

If you want to improve the system further, consider:

1. **Web Interface**: Create a web app using Flask or Django
2. **More Advanced NLP**: Incorporate word embeddings for better semantic understanding
3. **Feedback System**: Add ability for users to rate recommendations
4. **Champion Images**: Display champion portraits with recommendations
5. **Deeper Game Knowledge**: Add more game-specific context like patch performance, counter-picks, etc.

Enjoy your champion recommender project!
