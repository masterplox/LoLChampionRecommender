import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import re
from Levenshtein import ratio as levenshtein_ratio

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

class ChampionRecommender:
    def __init__(self, data_path):
        """Initialize with path to the processed LoL champion dataset"""
        self.df = pd.read_csv(data_path)
        self.setup_nlp()
        self.setup_keyword_mappings()
        
    def setup_nlp(self):
        """Setup NLP features for semantic matching"""
        # Ensure text fields are strings
        self.df['champion'] = self.df['champion'].astype(str)
        self.df['title'] = self.df['title'].astype(str)
        self.df['herotype'] = self.df['herotype'].astype(str)
        self.df['alttype'] = self.df['alttype'].fillna('').astype(str)
        self.df['roles'] = self.df['roles'].astype(str)
        self.df['positions'] = self.df['positions'].astype(str)
        
        # Combine text features for semantic matching
        self.df['combined_text'] = (
            self.df['champion'] + ' ' +
            self.df['title'] + ' ' +
            self.df['herotype'] + ' ' +
            self.df['alttype'] + ' ' +
            self.df['roles'] + ' ' +
            self.df['positions']
        )
        
        # Initialize TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Fit and transform the text features
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['combined_text'])
        
    def get_semantic_similarity(self, query, feature):
        """Get semantic similarity between query and feature using Levenshtein ratio"""
        return levenshtein_ratio(query.lower(), feature.lower())
        
    def extract_features_from_query(self, query):
        """Extract features from query using NLP"""
        # Tokenize query
        tokens = word_tokenize(query.lower())
        
        # Remove stop words and punctuation
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
        
        # Initialize feature weights
        feature_weights = {}
        
        # Define feature categories and their related terms
        feature_categories = {
            'move_speed': ['fast', 'slow', 'speed', 'movement', 'quick', 'mobile'],
            'hp': ['health', 'hp', 'life', 'durable', 'tanky', 'tough'],
            'damage_score': ['damage', 'attack', 'power', 'strong', 'kill'],
            'tankiness': ['tank', 'tanky', 'durable', 'tough', 'survive'],
            'mobility_score': ['mobile', 'dash', 'blink', 'teleport', 'gap closer'],
            'beginner_friendly': ['easy', 'simple', 'beginner', 'new', 'novice'],
            'combat_control': ['cc', 'stun', 'slow', 'root', 'knockup'],
            'utility_score': ['utility', 'support', 'heal', 'shield', 'help'],
            'early_game_strength': ['early', 'lane', 'snowball'],
            'late_game_strength': ['late', 'scaling', 'carry'],
            'teamfight_potential': ['teamfight', 'team', 'fight', 'group'],
            'peel_potential': ['peel', 'protect', 'save', 'help'],
            'engage_potential': ['engage', 'initiate', 'start', 'fight']
        }
        
        # Check each token in the query
        for token in tokens:
            # Check each feature category
            for feature, terms in feature_categories.items():
                # Calculate similarity with each term
                similarities = [self.get_semantic_similarity(token, term) for term in terms]
                max_similarity = max(similarities) if similarities else 0
                
                # If similarity is high enough, add to feature weights
                if max_similarity > 0.6:
                    if feature in feature_weights:
                        feature_weights[feature] += max_similarity
                    else:
                        feature_weights[feature] = max_similarity
                        
        # Check for superlatives and comparatives
        superlatives = ['est', 'most', 'best', 'worst', 'highest', 'lowest']
        for i, token in enumerate(tokens):
            if any(token.endswith(sup) for sup in superlatives):
                # Find the most similar feature
                best_feature = None
                best_similarity = 0
                for feature, terms in feature_categories.items():
                    similarities = [self.get_semantic_similarity(token, term) for term in terms]
                    max_similarity = max(similarities) if similarities else 0
                    if max_similarity > best_similarity:
                        best_similarity = max_similarity
                        best_feature = feature
                        
                if best_feature and best_similarity > 0.6:
                    # Increase weight for superlatives
                    if best_feature in feature_weights:
                        feature_weights[best_feature] += 1.5
                    else:
                        feature_weights[best_feature] = 1.5
                        
        return feature_weights
        
    def setup_keyword_mappings(self):
        """Setup keyword mappings for direct matches"""
        self.keyword_mappings = {
            'fast': ['move_speed'],
            'slow': ['move_speed:inverse'],
            'tank': ['tankiness'],
            'damage': ['damage_score'],
            'support': ['utility_score'],
            'easy': ['beginner_friendly'],
            'hard': ['beginner_friendly:inverse'],
            'cc': ['combat_control'],
            'mobile': ['mobility_score'],
            'early': ['early_game_strength'],
            'late': ['late_game_strength'],
            'teamfight': ['teamfight_potential'],
            'peel': ['peel_potential'],
            'engage': ['engage_potential']
        }
        
        self.position_keywords = {
            'top': 'TOP',
            'jungle': 'JUNGLE',
            'mid': 'MIDDLE',
            'adc': 'BOTTOM',
            'support': 'SUPPORT'
        }
        
    def preprocess_query(self, query):
        """Preprocess the query for matching"""
        # Convert to lowercase
        query = query.lower()
        
        # Remove punctuation
        query = query.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(query)
        
        return tokens
        
    def recommend_champions(self, query, top_n=5):
        """Get champion recommendations based on a natural language query"""
        try:
            print(f"Processing query: {query}")
            
            # Extract features using NLP
            feature_weights = self.extract_features_from_query(query)
            print(f"Extracted features: {feature_weights}")
            
            if not feature_weights:
                return "I couldn't understand your preferences. Try describing what you want more specifically."
            
            # Calculate scores based on feature weights
            scores = pd.Series(0.0, index=self.df.index)
            
            for feature, weight in feature_weights.items():
                if feature in self.df.columns:
                    # Normalize feature values to 0-1 range
                    normalized_feature = self.df[feature] / self.df[feature].max()
                    scores += normalized_feature * weight
            
            # Get top recommendations
            self.df['score'] = scores
            top_champions = self.df.nlargest(top_n, 'score')
            
            # Create recommendation results
            recommendations = []
            for _, champion in top_champions.iterrows():
                # Generate reasons based on highest scoring features
                reasons = []
                for feature, weight in feature_weights.items():
                    if feature in champion.index and weight > 0:
                        value = float(champion[feature])
                        if value > 0.6:  # Only include significant features
                            reasons.append(f"{feature}: {value:.2f}")
                
                recommendations.append({
                    'champion': str(champion['champion']),
                    'title': str(champion['title']),
                    'score': float(champion['score']),
                    'reasons': reasons,
                    'type': f"{champion['herotype']} / {champion['alttype']}" if pd.notna(champion['alttype']) else str(champion['herotype']),
                    'position': str(champion['positions']),
                    'difficulty': int(champion['difficulty']) if pd.notna(champion['difficulty']) else 0,
                    'move_speed': float(champion['move_speed']) if pd.notna(champion['move_speed']) else 0.0,
                    'hp': float(champion['hp']) if pd.notna(champion['hp']) else 0.0,
                    'attack_damage': float(champion['attack_damage']) if pd.notna(champion['attack_damage']) else 0.0,
                    'armor': float(champion['armor']) if pd.notna(champion['armor']) else 0.0,
                    'magic_resist': float(champion['magic_resist']) if pd.notna(champion['magic_resist']) else 0.0,
                    'attack_range': float(champion['attack_range']) if pd.notna(champion['attack_range']) else 0.0
                })
            
            return recommendations
            
        except Exception as e:
            print(f"Error in recommend_champions: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return f"An error occurred while processing your request: {str(e)}"

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

        # Add game phase strengths
        if champion['early_game_strength'] > 0.7:
            reasons.append('strong early game')
        if champion['late_game_strength'] > 0.7:
            reasons.append('strong late game')

        # Add team composition strengths
        if champion['teamfight_potential'] > 0.7:
            reasons.append('excellent teamfighter')
        if champion['peel_potential'] > 0.7:
            reasons.append('good at protecting allies')
        if champion['engage_potential'] > 0.7:
            reasons.append('strong engage potential')

        if not reasons:
            return "Good match for your preferences"

        return reasons


# Example usage
# if __name__ == "__main__":
#     recommender = ChampionRecommender("testing/lol_champions_processed.csv")
#     query = "I want a champion who is tanky and has good damage"
#     results = recommender.recommend_champions(query)

#     if isinstance(results, str):
#         print(results)
#     else:
#         print(f"Top recommendations for: '{query}'")
#         for i, champ in enumerate(results, 1):
#             print(f"{i}. {champ['champion']} ({champ['type']})")
#             print(f"   Match score: {champ['score']:.2f}")
#             print(f"   Reasons: {', '.join(champ['reasons'])}")
#             print(f"   Difficulty: {champ['difficulty']}/3")
#             print("")