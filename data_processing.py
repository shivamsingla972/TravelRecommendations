import pandas as pd
import numpy as np
import ast

def process_interactions(interactions_df):
    """
    Process interaction data to create binary columns for interaction types.
    """
    interaction_types = ['like', 'add_to_list', 'visit']
    
    for interaction in interaction_types:
        interactions_df[interaction] = interactions_df['interaction_type'].apply(
            lambda x: 1 if interaction in x else 0
        )
    
    # Aggregating by user_id and place_id
    aggregated_df = interactions_df.groupby(['user_id', 'place_id'], as_index=False)[interaction_types].sum()
    
    # Convert interactions to binary (1 if > 0, else 0)
    aggregated_df[interaction_types] = aggregated_df[interaction_types].apply(lambda x: x > 0).astype(int)
    
    return aggregated_df


def process_users(users_df):
    users_df['list_of_places'] = users_df['list_of_places'].apply(lambda x: ast.literal_eval(x))
    return users_df

def process_places(places_df):
    places_df = pd.get_dummies(places_df, columns=['category', 'tags', 'place_name'], drop_first=True)
    places_df.fillna(0, inplace=True)
    
    return places_df



def merge_data(interactions_df, users_df, places_df):
    merged_df = pd.merge(interactions_df, places_df, on='place_id', how='left')
    merged_df = pd.merge(merged_df, users_df, on='user_id', how='left')
    return merged_df
