from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class RecommendationModel:
    def __init__(self, user_features, place_features, user_item_matrix):
        self.user_features = user_features
        self.place_features = place_features
        self.user_item_matrix = user_item_matrix

    def generate_recommendations(self, user_id, top_n):
        if user_id not in self.user_item_matrix.index:
            return f"User {user_id} not found."

        user_vector = self.user_features.loc[user_id].values.reshape(1, -1)
        similarity_scores = cosine_similarity(user_vector, self.user_features).flatten()

        weighted_scores = np.dot(similarity_scores, self.user_item_matrix.values)
        ranked_indices = np.argsort(weighted_scores)[::-1]

        ranked_place_ids = self.user_item_matrix.columns[ranked_indices]

        interacted_places = self.user_item_matrix.loc[user_id]
        unvisited_places = [place for place in ranked_place_ids if interacted_places[place] == 0]

        recommended_place_ids = unvisited_places[:top_n]
        return recommended_place_ids

    def recommend(self, user_id, top_n=5):
        return self.generate_recommendations(user_id, top_n)


def build_recommendation_model(user_features, place_features, user_item_matrix):
    return RecommendationModel(user_features, place_features, user_item_matrix)
