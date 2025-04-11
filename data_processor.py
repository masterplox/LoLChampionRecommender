import pandas as pd
import numpy as np
import json
import re

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        
    def load_data(self):
        """Load the original dataset"""
        self.df = pd.read_csv(self.data_path)
        print("Original data shape:", self.df.shape)
        print("Columns:", self.df.columns.tolist())
        
    def parse_stats(self, stats_str):
        """Parse the stats string into a dictionary"""
        try:
            # Remove any extra quotes and whitespace
            stats_str = stats_str.strip('"').strip()
            
            # Handle empty or invalid stats
            if not stats_str or stats_str == 'nan':
                return {}
                
            # Convert the string to a proper JSON format
            stats_str = stats_str.replace("'", '"')
            
            # Parse the JSON string
            stats_dict = json.loads(stats_str)
            
            # Extract base stats
            result = {
                'hp': stats_dict.get('hp_base', 0),
                'mp': stats_dict.get('mp_base', 0),
                'ms': stats_dict.get('ms', 0),
                'armor': stats_dict.get('arm_base', 0),
                'spellblock': stats_dict.get('mr_base', 0),
                'attackrange': stats_dict.get('range', 0),
                'hpregen': stats_dict.get('hp5_base', 0),
                'mpregen': stats_dict.get('mp5_base', 0),
                'crit': 0,  # Not in base stats
                'attackdamage': stats_dict.get('dam_base', 0),
                'attackspeed': stats_dict.get('as_base', 0)
            }
            
            return result
        except Exception as e:
            print(f"Error parsing stats: {stats_str}")
            print(f"Error: {str(e)}")
            return {}
            
    def preprocess_data(self):
        """Preprocess the dataset"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Parse stats into separate columns
        print("Parsing stats...")
        stats_data = self.df['stats'].apply(self.parse_stats)
        
        # Extract individual stats
        for stat in ['hp', 'mp', 'ms', 'armor', 'spellblock', 'attackrange', 'hpregen', 'mpregen', 'crit', 'attackdamage', 'attackspeed']:
            self.df[stat] = stats_data.apply(lambda x: x.get(stat, 0))
            
        # Rename stats for better readability
        self.df = self.df.rename(columns={
            'ms': 'move_speed',
            'mp': 'mana',
            'spellblock': 'magic_resist',
            'attackrange': 'attack_range',
            'hpregen': 'hp_regen',
            'mpregen': 'mana_regen',
            'crit': 'critical_strike',
            'attackdamage': 'attack_damage',
            'attackspeed': 'attack_speed'
        })
        
        # Create roles and positions columns based on herotype
        self.df['roles'] = self.df['herotype'].fillna('')
        self.df['positions'] = self.df['herotype'].fillna('')
        
        # Print sample of processed stats
        print("\nSample of processed stats:")
        print(self.df[['champion', 'move_speed', 'hp', 'attack_damage', 'armor', 'magic_resist']].head())
        
        # Print move speed statistics
        print("\nMove speed statistics:")
        print(f"Min move speed: {self.df['move_speed'].min()}")
        print(f"Max move speed: {self.df['move_speed'].max()}")
        print(f"Average move speed: {self.df['move_speed'].mean()}")
        
        # Print champions with highest move speed
        print("\nTop 5 champions by move speed:")
        top_speed = self.df.nlargest(5, 'move_speed')[['champion', 'move_speed']]
        print(top_speed)
        
        # Create derived features
        self.create_derived_features()
        
        # Normalize features
        self.normalize_features()
        
    def create_derived_features(self):
        """Create derived features from base stats"""
        # Tankiness score (combination of HP, armor, and magic resist)
        self.df['tankiness'] = (
            self.df['hp'] * 0.4 +
            self.df['armor'] * 0.3 +
            self.df['magic_resist'] * 0.3
        )
        
        # Damage score (combination of attack damage and attack speed)
        self.df['damage_score'] = (
            self.df['attack_damage'] * 0.7 +
            self.df['attack_speed'] * 0.3
        )
        
        # Mobility score (based on move speed)
        self.df['mobility_score'] = self.df['move_speed']
        
        # Beginner friendly score (inverse of difficulty)
        self.df['beginner_friendly'] = 10 - self.df['difficulty']
        
        # Combat control score (based on champion type and abilities)
        # Initialize with base value
        self.df['combat_control'] = 0.5
        
        # Increase score for tanks and supports
        tank_mask = self.df['herotype'].str.contains('Tank', case=False, na=False)
        support_mask = self.df['herotype'].str.contains('Support', case=False, na=False)
        self.df.loc[tank_mask, 'combat_control'] += 0.3
        self.df.loc[support_mask, 'combat_control'] += 0.2
        
        # Utility score (based on champion type)
        self.df['utility_score'] = 0.5
        self.df.loc[support_mask, 'utility_score'] += 0.5
        
        # Game phase strengths
        self.df['early_game_strength'] = (
            self.df['damage_score'] * 0.4 +
            self.df['mobility_score'] * 0.3 +
            self.df['tankiness'] * 0.3
        )
        
        self.df['late_game_strength'] = (
            self.df['damage_score'] * 0.5 +
            self.df['tankiness'] * 0.3 +
            self.df['utility_score'] * 0.2
        )
        
        # Team composition scores
        self.df['teamfight_potential'] = (
            self.df['damage_score'] * 0.3 +
            self.df['tankiness'] * 0.3 +
            self.df['combat_control'] * 0.2 +
            self.df['utility_score'] * 0.2
        )
        
        self.df['peel_potential'] = (
            self.df['combat_control'] * 0.5 +
            self.df['utility_score'] * 0.5
        )
        
        self.df['engage_potential'] = (
            self.df['mobility_score'] * 0.4 +
            self.df['combat_control'] * 0.4 +
            self.df['tankiness'] * 0.2
        )
        
    def normalize_features(self):
        """Normalize numerical features using min-max scaling"""
        # List of features to normalize
        features_to_normalize = [
            'hp', 'mana', 'move_speed', 'armor', 'magic_resist',
            'attack_range', 'hp_regen', 'mana_regen', 'critical_strike',
            'attack_damage', 'attack_speed', 'tankiness', 'damage_score',
            'mobility_score', 'beginner_friendly', 'combat_control',
            'utility_score', 'early_game_strength', 'late_game_strength',
            'teamfight_potential', 'peel_potential', 'engage_potential'
        ]
        
        # Normalize each feature using min-max scaling
        for feature in features_to_normalize:
            if feature in self.df.columns:
                min_val = self.df[feature].min()
                max_val = self.df[feature].max()
                if max_val > min_val:  # Avoid division by zero
                    self.df[feature] = (self.df[feature] - min_val) / (max_val - min_val)
                else:
                    self.df[feature] = 0.5  # Set to middle value if all values are the same
                
    def create_recommender_dataset(self):
        """Create the final dataset for the recommender"""
        # Select and order columns
        columns = [
            'champion', 'title', 'herotype', 'alttype', 'roles', 'positions',
            'difficulty', 'hp', 'mana', 'move_speed', 'armor', 'magic_resist',
            'attack_range', 'hp_regen', 'mana_regen', 'critical_strike',
            'attack_damage', 'attack_speed', 'tankiness', 'damage_score',
            'mobility_score', 'beginner_friendly', 'combat_control',
            'utility_score', 'early_game_strength', 'late_game_strength',
            'teamfight_potential', 'peel_potential', 'engage_potential'
        ]
        
        # Create the final dataset
        recommender_df = self.df[columns].copy()
        
        # Save the processed dataset
        output_path = 'data/processed_champion_data.csv'
        recommender_df.to_csv(output_path, index=False)
        print(f"\nProcessed dataset saved to {output_path}")
        print(f"Final dataset shape: {recommender_df.shape}")
        
        return recommender_df
        
    def process(self):
        """Process the data and create the recommender dataset"""
        self.load_data()
        self.preprocess_data()
        return self.create_recommender_dataset()

if __name__ == "__main__":
    processor = DataProcessor('data/140325_LoL_champion_data_original.csv')
    processor.process()