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
                print(
                    f"   Position: {', '.join(champ['position']) if isinstance(champ['position'], list) else champ['position']}")
                print(f"   Difficulty: {champ['difficulty']}/3")
                print(f"   Why: {', '.join(champ['reasons'])}")
                print("")

        print("-" * 55)


if __name__ == "__main__":
    main()