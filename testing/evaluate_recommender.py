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