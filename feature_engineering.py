import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def compute_user_similarity(interactions_df):
    interaction_mapping = {"like": 1, "visit": 2, "add_to_list": 3}
    interactions_df["interaction_type"] = interactions_df["interaction_type"].map(interaction_mapping)

    user_item_matrix = interactions_df.pivot_table(
        index="user_id", columns="place_id", values="interaction_type", aggfunc="sum", fill_value=0
    )

    print("User-Item Matrix Shape:", user_item_matrix.shape)
    print(user_item_matrix.head())

    similarity_matrix = cosine_similarity(user_item_matrix)

    return similarity_matrix, user_item_matrix

def compute_place_similarity(places_df):
    places_df["combined_features"] = (
        places_df["category"] + " " + places_df["tags"].fillna("")
    )

    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(stop_words="english")
    feature_matrix = tfidf.fit_transform(places_df["combined_features"])
    similarity_matrix = cosine_similarity(feature_matrix)

    return similarity_matrix

def create_feature_vectors(users_df, places_df, interactions_df):

    user_similarity_matrix, user_item_matrix = compute_user_similarity(interactions_df)

    place_similarity_matrix = compute_place_similarity(places_df)

    return {
        "user_similarity_matrix": user_similarity_matrix,
        "user_item_matrix": user_item_matrix,
        "place_similarity_matrix": place_similarity_matrix,
    }
