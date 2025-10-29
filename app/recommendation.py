# ==========================================
# 🍽️ ZOMATO RECOMMENDATION SYSTEM (Optimized)
# Author: Vishwa
# ==========================================

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

# ----------------------------------------------------------
# 1️⃣ Load and Prepare Data
# ----------------------------------------------------------

def load_data(path="data/clean_zomato.csv"):
    """
    Loads the cleaned Zomato dataset and prepares combined text features.
    """
    df = pd.read_csv(path)
    df.dropna(subset=['name'], inplace=True)

    # Convert all features to string type before combining
    df['Primary Cuisine'] = df['Primary Cuisine'].astype(str)
    df['location'] = df['location'].astype(str)
    df['online_order'] = df['online_order'].astype(str)
    df['book_table'] = df['book_table'].astype(str)

    # Combine text-based features
    df['combined_features'] = (
        df['Primary Cuisine'] + ' ' +
        df['location'] + ' ' +
        df['online_order'] + ' ' +
        df['book_table']
    )

    return df


# ----------------------------------------------------------
# 2️⃣ Build Nearest Neighbor Model (Memory Efficient)
# ----------------------------------------------------------

def build_nearest_neighbors(df, n_neighbors=10):
    """
    Builds a NearestNeighbors model instead of computing full NxN similarity matrix.
    Uses cosine distance for finding similar restaurants efficiently.
    """
    cv = CountVectorizer(stop_words='english', max_features=5000)
    vectors = cv.fit_transform(df['combined_features'])

    # Create Nearest Neighbors model
    nn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbors)
    nn.fit(vectors)

    return nn, vectors


# ----------------------------------------------------------
# 3️⃣ Recommend Restaurants Function
# ----------------------------------------------------------

def recommend_restaurants(restaurant_name, df, nn, vectors, n=5):
    """
    Finds top 'n' similar restaurants to the given restaurant_name.
    """
    if restaurant_name not in df['name'].values:
        print(f"❌ Restaurant '{restaurant_name}' not found in dataset.")
        return pd.DataFrame()

    idx = df[df['name'] == restaurant_name].index[0]

    # Find nearest neighbors
    distances, indices = nn.kneighbors(vectors[idx], n_neighbors=n+1)

    recommendations = []
    for i, dist in zip(indices[0][1:], distances[0][1:]):  # skip the first (same restaurant)
        restaurant_info = {
            "Restaurant Name": df.iloc[i].name,
            "Location": df.iloc[i].location,
            "Cuisine": df.iloc[i]['Primary Cuisine'],
            "Cost (for two)": df.iloc[i]['approx_cost(for two people)'],
            "Rating": df.iloc[i].rate,
            "Votes": df.iloc[i].votes,
            "Similarity Score": round(1 - dist, 3)
        }
        recommendations.append(restaurant_info)

    return pd.DataFrame(recommendations)


# ----------------------------------------------------------
# 4️⃣ Test Section (Run directly for local testing)
# ----------------------------------------------------------

if __name__ == "__main__":
    print("🔄 Loading data and building recommendation model...")
    df = load_data()
    nn, vectors = build_nearest_neighbors(df)
    print("✅ Model built successfully!")

    # Example test
    restaurant_name = "Empire Restaurant"  # Change as per your dataset
    print(f"\n🍴 Recommendations for: {restaurant_name}")
    result_df = recommend_restaurants(restaurant_name, df, nn, vectors)
    print(result_df)
