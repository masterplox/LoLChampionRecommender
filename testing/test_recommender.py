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