import streamlit as st
import pandas as pd
from PIL import Image
import os
import gzip
import pickle
import dask.array as da
from fuzzywuzzy import fuzz

# Define a function to load images
def load_image(image_path):
    return Image.open(image_path)

# Load images
images_path = os.path.join(os.getcwd(), 'Images')
anime2 = load_image(os.path.join(images_path, 'anime2.png'))
img2 = load_image(os.path.join(images_path, 'img2.jpeg'))
img3 = load_image(os.path.join(images_path, 'img3.jpeg'))
img4 = load_image(os.path.join(images_path, 'img4.jpeg'))
img5 = load_image(os.path.join(images_path, 'img5.jpeg'))

# Define the path to your data folder
data_path = os.path.join(os.getcwd(), 'Data')

@st.cache_data
def load_data(file, dtype=None, sample_size=None, random_state=42):
    df = pd.read_csv(file, dtype=dtype)
    if sample_size:
        return df.sample(n=sample_size, random_state=random_state)
    return df

# Define dtypes for columns
dtype_anime = {
    'episodes': 'object',
}

# Load sampled anime and train datasets
anime = load_data(os.path.join(data_path, 'anime.csv'), dtype=dtype_anime, sample_size=1000)
train = load_data(os.path.join(data_path, 'train.csv'), sample_size=50000)
model_folder = os.path.join(os.getcwd(), 'Models')

# Lazy loading models
@st.cache_resource
def load_tfv_vectorizer():
    with gzip.open(os.path.join(model_folder, 'tfv_vectorizer.pkl.gz'), 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_sigmoid_kernel_matrix():
    with gzip.open(os.path.join(model_folder, 'sigmoid_kernel_matrix.pkl.gz'), 'rb') as f:
        matrix = pickle.load(f)
    return da.from_array(matrix, chunks=(250, 250))

@st.cache_resource
def load_baseline_model():
    with gzip.open(os.path.join(model_folder, 'baseline_model.pkl.gz'), 'rb') as f:
        return pickle.load(f)

# Load models only when needed
tfv = load_tfv_vectorizer()
sig = load_sigmoid_kernel_matrix()
best_baseline_model = load_baseline_model()

# Drop duplicates and reset index
rec_data = anime.drop_duplicates(subset="name", keep="first").reset_index(drop=True)
rec_indices = pd.Series(rec_data.index, index=rec_data["name"]).drop_duplicates()

def get_anime_index(name, rec_indices):
    rec_indices = rec_indices.fillna('').astype(str)
    name_lower = name.lower()
    rec_indices_lower = rec_indices.str.lower()
    
    # Use fuzzy matching with a high score threshold (e.g., 80) for partial matches
    matches = rec_indices_lower[rec_indices_lower.apply(lambda x: fuzz.ratio(x, name_lower) >= 80)]
    
    if not matches.empty:
        return matches.index[0]
    else:
        return None

def top_10_anime_by_average_rating(anime_data):
    return anime_data.sort_values(by="rating", ascending=False).head(10)[['anime_id', 'name', 'rating']]

def top_10_anime_by_user_ratings(train_data, anime_data):
    top_rated_ids = train_data.groupby('anime_id')['rating'].mean().sort_values(ascending=False).head(10).index
    valid_anime_data = anime_data.set_index('anime_id')
    valid_top_rated_ids = [anime_id for anime_id in top_rated_ids if anime_id in valid_anime_data.index]
    top_anime = valid_anime_data.loc[valid_top_rated_ids][['name', 'rating']].reset_index()
    return top_anime

def content_based_recommendations(user_input_list, anime_data, sig, top_n=10):
    all_recommendations = []
    for anime in user_input_list:
        idx = get_anime_index(anime, rec_indices)
        if idx is not None:
            if 0 <= idx < sig.shape[0]:
                sim_scores = list(enumerate(sig[idx].compute()))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_indices = [i[0] for i in sim_scores[1:top_n + 1]]
                recommended_animes = anime_data['name'].iloc[sim_indices].tolist()
                all_recommendations.extend(recommended_animes)
            else:
                st.warning(f"Index {idx} for anime '{anime}' is out of bounds. Skipping.")
        else:
            st.warning(f"Anime '{anime}' not found. Skipping.")
    
    if all_recommendations:
        recommendation_counts = pd.Series(all_recommendations).value_counts()
        top_recommendations = recommendation_counts.head(top_n).index.tolist()
        return top_recommendations, None
    else:
        top_popular_animes = anime_data.sort_values(by='rating', ascending=False).head(top_n)['name'].tolist()
        st.warning(f"No anime found for your input. Here are the top {top_n} popular animes:")
        return [], top_popular_animes

def collaborative_recommendations(user_ratings, model, anime_data, train_data, top_n=5):
    assert isinstance(anime_data, pd.DataFrame), "Expected anime_data to be a DataFrame"
    
    user_id = train_data['user_id'].max() + 1
    anime_ids = []
    unrecognized_animes = []

    for anime_name in user_ratings.keys():
        mask = anime_data['name'].str.lower().apply(lambda x: anime_name.lower() in x)
        anime_id_series = anime_data[mask]['anime_id']
        
        if not anime_id_series.empty:
            anime_ids.append(anime_id_series.iloc[0])
        else:
            unrecognized_animes.append(anime_name)

    if not anime_ids:
        st.warning("Your anime has not been rated yet or it is invalid, in the meantime enjoy some of the most popular animes of all time.")
        return top_10_anime_by_user_ratings(train_data, anime_data)

    if unrecognized_animes:
        st.warning(f"The following anime(s) were not recognized or not yet rated: {', '.join(unrecognized_animes)}. Please enter valid anime names.")
        return top_10_anime_by_user_ratings(train_data, anime_data)

    predictions = [(anime_id, model.predict(user_id, anime_id).est) for anime_id in anime_ids]
    predictions_df = pd.DataFrame(predictions, columns=['anime_id', 'predicted_rating'])

    recommendations = pd.merge(predictions_df, anime_data, on='anime_id').sort_values(by='predicted_rating', ascending=False).head(top_n)
    
    return recommendations[['name', 'predicted_rating']]

def display_team_member(name, role, description, image_path, linkedin_url):
    col1, col2 = st.columns([1, 3])

    with col1:
        st.image(image_path, width=150, caption=name)

    with col2:
        st.markdown(f"""
        ### {name}
        **{role}**

        [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat&logo=linkedin)]({linkedin_url})
        """, unsafe_allow_html=True)
        st.write("---")

def main():
    """Anime Recommender App with Streamlit """

    st.markdown(
        """
        <h1 style="color:#FF5733;">Welcome to Funiverse Network</h1>
        <p>Discover Your Next Favorite Anime!</p>
        <hr>
        """, unsafe_allow_html=True
    )
 
    st.image(anime2, use_column_width=True)  

    st.markdown(
        """
        <div style="border-left: 7px solid #0288d1; padding-left: 15px; background-color: #e1f5fe;">
            <p style="color: #000000; font-weight: bold;">Your personalized gateway to discovering the best anime. 
            Whether you're a seasoned otaku or new to the world of anime, our app helps you find shows that match your interests.</p>
        </div>
        """, unsafe_allow_html=True
    )

    options = ["Recommend", "Information", "Our Team", "EDA", "Contact Us", "App Feedback"]
    selection = st.sidebar.selectbox("Choose Option", options)
 
    if selection == "Information":
        st.markdown('<h1>About the Anime Recommender App:</h1>', unsafe_allow_html=True)
        st.markdown('<h2>Project Overview</h2>', unsafe_allow_html=True)
        st.markdown(
            """Introduction: The world of anime is vast and diverse, with thousands of shows spanning multiple genres, themes, and styles. For anime enthusiasts, discovering new series that align with their tastes can be a challenge. The Anime Recommender App is designed to address this challenge by providing personalized recommendations, making it easier for users to find their next favorite anime.""",
            unsafe_allow_html=True
        )
        st.markdown(
            """Objective: The primary objective of this project is to create a user-friendly web application that delivers tailored anime recommendations based on users' preferences and viewing history. The app employs both content-based and collaborative filtering techniques to generate these recommendations, ensuring a broad and accurate selection of anime titles for each user.""",
            unsafe_allow_html=True
        )
        st.markdown(
            """Features:
            - **Personalized Recommendations:** Users can receive anime recommendations based on the shows they have watched and rated.
            - **Exploratory Data Analysis (EDA):** Users can explore data visualizations related to anime ratings, genres, and more.
            - **Top-Rated Anime:** The app highlights the top-rated anime based on average user ratings, helping users discover critically acclaimed shows.
            - **User-Friendly Interface:** With a clean and intuitive design, users can easily interact with the app and find the information they need.
            """,
            unsafe_allow_html=True
        )
    
    elif selection == "Our Team":
        st.markdown('<h1>Meet the Team:</h1>', unsafe_allow_html=True)
        st.markdown('<h2>Our Team Members</h2>', unsafe_allow_html=True)

        display_team_member("Khululiwe Hlongwane", "Project Manager",
                            "Khululiwe is a passionate data scientist with expertise in machine learning and recommender systems. She oversees the project's development of the Anime Recommender App.",
                            img2, "https://www.linkedin.com/in/funeka-jwambi/")
        display_team_member("Judith Kabongo", "Data Scientist",
                            "Judith handles slide decks and presentation communications".",
                            img4, "https://www.linkedin.com/in/judith-kabongo-568b581b7/")
        display_team_member("Ntembeko Mhlungu", "Data Scientist",
                            "Ntembeko is responsible for developing the app's user interface.",
                            img5, "https://www.linkedin.com/in/ntembeko-mhlungu-899ab5201/")
        display_team_member("Tselani Moeti", "Github Manager",
                            "Tselani is tasked with managing GitHub for version control.",
                            img3, "https://www.linkedin.com/in/tselani-moeti-449a451b1/")

    elif selection == "EDA":
        st.markdown('<h1>Anime Recommender App Exploratory Data Analysis:</h1>', unsafe_allow_html=True)
        st.markdown('<h2>Exploratory Data Analysis (EDA)</h2>', unsafe_allow_html=True)
        st.markdown(
            """Exploratory Data Analysis (EDA) is an approach to analyzing data sets to summarize their main characteristics, often with visual methods. In the context of the Anime Recommender App, EDA is used to explore various aspects of the anime dataset, such as ratings distribution, genre popularity, and the relationship between different features. By gaining insights into the data, we can better understand user preferences and improve the recommendation system.""",
            unsafe_allow_html=True
        )

    elif selection == "Contact Us":
        st.markdown('<h1>Contact Us:</h1>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="border-left: 7px solid #0288d1; padding-left: 15px; background-color: #e1f5fe;">
                <p style="color: #000000; font-weight: bold;">Thank you for using the Anime Recommender App! We value your feedback and are here to assist you. If you have any questions, suggestions, or inquiries, please feel free to contact us using the information below:</p>
            </div>
            """, unsafe_allow_html=True
        )

        contact_info = """
            **Email:** support@funiversenetwork.com  
            **Phone:** +27 322 774 789  
            **Address:** 381 Church Street, Pietermaritzburg, South Africa
        """
        st.markdown(contact_info)

    elif selection == "App Feedback":
        st.markdown('<h1>App Feedback:</h1>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="border-left: 7px solid #0288d1; padding-left: 15px; background-color: #e1f5fe;">
                <p style="color: #000000; font-weight: bold;">We hope you had a fantastic experience using the Anime Recommender App. Your feedback is invaluable to us, and we would love to hear your thoughts on how we can improve the app to better serve you. Please take a moment to share your feedback with us:</p>
            </div>
            """, unsafe_allow_html=True
        )
        st.text_area("Please leave your feedback here...")

    elif selection == "Recommend":
        st.markdown('<h2>Collaborative Filtering Recommendations</h2>', unsafe_allow_html=True)

        # Example user input section for user_ratings
        user_ratings = {}

        st.markdown("### Rate some anime:")
        anime_list = anime['name'].sample(10).tolist()  # Random sample of anime names
        for anime_name in anime_list:
            rating = st.slider(f'Rate {anime_name}', 0, 10, 5)
            user_ratings[anime_name] = rating

        # Call the collaborative_recommendations function
        if user_ratings:
            recommendations = collaborative_recommendations(user_ratings, best_baseline_model, anime, train)
            st.write("### Recommendations for you:")
            st.table(recommendations)
        else:
            st.warning("Please rate at least one anime to get recommendations.")
    
    # Content-based recommendations
    st.markdown('<h2>Content-Based Recommendations</h2>', unsafe_allow_html=True)

    user_input = st.text_area("Enter an anime you like (or multiple, separated by commas):")
    user_input_list = [anime.strip() for anime in user_input.split(',') if anime.strip()]

    if user_input_list:
        recommendations, popular_anime = content_based_recommendations(user_input_list, anime, sig)
        if recommendations:
            st.write(f"### Recommended Animes based on your input: {', '.join(user_input_list)}")
            st.table(recommendations)
        else:
            st.write(f"### Top 10 popular animes:")
            st.table(popular_anime)

if __name__ == '__main__':
    main()
