import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import plotly.express as px

# --- Spotify API Authentication (safe way) ---
client_id = None
client_secret = None

# 1) Streamlit Cloud (secrets)
if "spotify" in st.secrets:
    client_id = st.secrets["spotify"]["client_id"]
    client_secret = st.secrets["spotify"]["client_secret"]

# 2) Local dev (environment variables)
elif os.getenv("SPOTIPY_CLIENT_ID") and os.getenv("SPOTIPY_CLIENT_SECRET"):
    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

# 3) Fallback (local quick test ONLY — don’t commit real keys!)
else:
    client_id = "YOUR_CLIENT_ID"
    client_secret = "YOUR_CLIENT_SECRET"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=client_id,
    client_secret=client_secret
))

# --- Load dataset ---
df = pd.read_csv("data/SpotifyFeatures.csv")

# --- Convert 'key' column (musical notes -> numbers) ---
key_mapping = {
    "C": 0, "C#": 1, "D": 2, "D#": 3,
    "E": 4, "F": 5, "F#": 6, "G": 7,
    "G#": 8, "A": 9, "A#": 10, "B": 11
}
df["key"] = df["key"].map(key_mapping)

# --- Convert 'mode' column (Major/Minor -> numbers) ---
mode_mapping = {"Minor": 0, "Major": 1}
df["mode"] = df["mode"].map(mode_mapping)

# --- Feature matrix for recommendations ---
features = ["danceability", "energy", "key", "loudness", "mode",
            "speechiness", "acousticness", "instrumentalness",
            "liveness", "valence", "tempo"]

scaler = StandardScaler()
feature_matrix = scaler.fit_transform(df[features])

# --- Song Recommendation Function ---
def recommend_songs(song_name, n=5):
    if song_name.lower() not in df["track_name"].str.lower().values:
        return None

    # Find index of the song (case-insensitive)
    idx = df[df["track_name"].str.lower() == song_name.lower()].index[0]

    # Cosine similarity
    sim_scores = cosine_similarity([feature_matrix[idx]], feature_matrix).flatten()

    # Top N similar songs (excluding itself)
    similar_indices = sim_scores.argsort()[-(n + 1):-1][::-1]
    return df.iloc[similar_indices][["track_name", "artist_name", "genre", "popularity"]]

# --- Streamlit Page Config ---
st.set_page_config(page_title="TuNiFy - Your Smart Music Recommender", layout="wide")

# Title
st.markdown("<h1 style='text-align: center;'>🎵 TuNiFy: Smart Music Recommender, Playlist Generator & Dashboard 🎶</h1>", unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🎧 Song Recommendation", "📂 Playlist Generator", "📊 Dashboard Insights"])

# --- Spotify Metadata Function ---
def get_spotify_info(song_name, artist_name):
    try:
        query = f"track:{song_name} artist:{artist_name}"
        result = sp.search(q=query, type="track", limit=1)
        if result["tracks"]["items"]:
            track = result["tracks"]["items"][0]
            return track["album"]["images"][0]["url"], track["preview_url"], track["external_urls"]["spotify"]
    except:
        return None, None, None
    return None, None, None

# ---- SONG RECOMMENDATION TAB ----
with tab1:
    st.subheader("Find similar songs")
    song_name = st.text_input("Enter a song name (e.g. 'Shape of You'):")

    if st.button("Recommend"):
        result = recommend_songs(song_name, n=5)

        if result is not None:
            st.success(f"✅ Showing songs similar to **{song_name}**:")

            for _, row in result.iterrows():
                album_art, preview_url, spotify_url = get_spotify_info(
                    row["track_name"], row["artist_name"]
                )

                col1, col2 = st.columns([1, 4])
                with col1:
                    if album_art:
                        st.image(album_art, width=80)
                with col2:
                    st.markdown(
                        f"**{row['track_name']}** by *{row['artist_name']}* "
                        f"({row['genre']}) 🎶"
                    )
                    st.write(f"🔥 Popularity: {row['popularity']}")

                    if preview_url:
                        st.audio(preview_url, format="audio/mp3")
                    if spotify_url:
                        st.markdown(f"[▶️ Listen on Spotify]({spotify_url})", unsafe_allow_html=True)

                st.markdown("---")

        else:
            st.error("❌ Song not found in dataset. Try another.")

    else:
        st.warning("⚠️ Please enter a song name.")

# ---- PLAYLIST GENERATOR TAB ----
with tab2:
    st.subheader("Generate a Playlist")

    genre_choice = st.selectbox("🎶 Select Genre:", sorted(df["genre"].unique()))
    mood_choice = st.radio("😀 Select Mood:", ["Happy / Energetic", "Calm / Chill"])
    num_songs = st.slider("🎚 Number of Songs:", 5, 20, 10)

    if st.button("Generate Playlist"):
        playlist = df[df["genre"] == genre_choice].copy()

        if mood_choice == "Happy / Energetic":
            playlist = playlist.sort_values(by=["valence", "energy"], ascending=False)
        else:
            playlist = playlist.sort_values(by=["valence", "energy"], ascending=True)

        playlist = playlist.head(num_songs)[["track_name", "artist_name", "genre", "popularity"]]

        st.success(f"🎉 Your {mood_choice} playlist in **{genre_choice}** genre:")

        # Show with Spotify metadata
        for _, row in playlist.iterrows():
            album_art, preview_url, spotify_url = get_spotify_info(row["track_name"], row["artist_name"])
            col1, col2 = st.columns([1, 4])
            with col1:
                if album_art:
                    st.image(album_art, width=80)
            with col2:
                st.markdown(f"**{row['track_name']}** by *{row['artist_name']}* 🎶")
                st.write(f"Genre: {row['genre']} | Popularity: {row['popularity']}")
                if preview_url:
                    st.audio(preview_url, format="audio/mp3")
                if spotify_url:
                    st.markdown(f"[▶️ Listen on Spotify]({spotify_url})", unsafe_allow_html=True)
            st.markdown("---")

        # Allow Download as CSV
        csv = playlist.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Playlist as CSV",
            data=csv,
            file_name="playlist.csv",
            mime="text/csv",
        )

# ---- DASHBOARD TAB ----
with tab3:
    st.subheader("📊 Music Dataset Insights Dashboard")

    # Genre distribution
    st.markdown("### 🎶 Top Genres in Dataset")
    genre_count = df["genre"].value_counts().reset_index()
    genre_count.columns = ["Genre", "Count"]
    fig1 = px.bar(genre_count.head(10), x="Genre", y="Count", color="Genre",
                  title="Top 10 Genres", height=400)
    st.plotly_chart(fig1, use_container_width=True)

    # Popularity vs Danceability scatter
    st.markdown("### 🔥 Popularity vs Danceability")
    fig2 = px.scatter(df, x="danceability", y="popularity", color="genre",
                      hover_data=["track_name", "artist_name"],
                      title="Danceability vs Popularity", height=400)
    st.plotly_chart(fig2, use_container_width=True)

    # Mood distribution pie chart
    st.markdown("### 😀 Mood Distribution (Happy vs Chill)")
    mood_labels = ["Happy / Energetic", "Calm / Chill"]
    mood_counts = [
        len(df[(df["valence"] > 0.5) & (df["energy"] > 0.5)]),
        len(df[(df["valence"] <= 0.5) & (df["energy"] <= 0.5)])
    ]
    fig3 = px.pie(values=mood_counts, names=mood_labels, title="Mood Breakdown")
    st.plotly_chart(fig3, use_container_width=True)
