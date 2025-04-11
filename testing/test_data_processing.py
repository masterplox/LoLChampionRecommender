# test_data_processing.py
from data_processor import LoLDataProcessor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Process the data
processor = LoLDataProcessor("data/140325_LoL_champion_data_original.csv")
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