import pandas as pd
from feature_engineering import create_feature_vectors
from model import build_recommendation_model

def main():
    users_df = pd.read_csv("users.csv")
    places_df = pd.read_csv("places.csv")
    interactions_df = pd.read_csv("user_interactions.csv")

    feature_vectors = create_feature_vectors(users_df, places_df, interactions_df)

    user_features = feature_vectors["user_item_matrix"] 
    place_features = feature_vectors["place_similarity_matrix"]
    user_item_matrix = feature_vectors["user_item_matrix"]

    model = build_recommendation_model(user_features, place_features, user_item_matrix)

    print("Sample Recommendations:")
    print(model.recommend(user_id=1, top_n=5))

if __name__ == "__main__":
    main()
