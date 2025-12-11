import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import base64
import time

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Book Recommender",
    layout="wide",
    page_icon="📚"
)

# ---------------------------------------------------------
# DARK MODE UI
# ---------------------------------------------------------
dark_theme = """
<style>

body {
    background-color: #0e1117 !important;
    color: #e1e1e1 !important;
}

.sidebar .sidebar-content {
    background-color: #111418 !important;
}

.big-title {
    font-size: 50px;
    font-weight: 900;
    color: #ffffff;
    text-align: center;
}

.book-card {
    background: #1a1d23;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    transition: all 0.3s ease-in-out;
    border: 1px solid #2b2f36;
}

.book-card:hover {
    transform: scale(1.05);
    background: #242830;
}

.book-title {
    color: #e5e5e5;
    font-size: 18px;
    margin-top: 10px;
}

img {
    border-radius: 10px;
}

.search-bar input {
    border-radius: 10px !important;
    height: 50px;
    font-size: 18px;
}

</style>
"""

st.markdown(dark_theme, unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    books = pd.read_csv("books.csv")
    books.fillna("", inplace=True)

    # -------------------------------
    # CONTENT FEATURE FOR NLP
    # -------------------------------
    books["combined"] = (
        books["title"].astype(str) + " " +
        books["authors"].astype(str) + " " +
        books["publisher"].astype(str) + " " +
        books["categories"].astype(str)
    )

    # Cosine similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(books["combined"])
    similarity = cosine_similarity(matrix)

    return books, similarity


books, similarity = load_data()

# ---------------------------------------------------------
# ANIMATED HEADER
# ---------------------------------------------------------
st.markdown("<h1 class='big-title'>📚 Book Recommender System</h1>", unsafe_allow_html=True)
st.write("")
st.write("")

# ---------------------------------------------------------
# SEARCH BAR
# ---------------------------------------------------------
search = st.text_input("🔍 Search Book", placeholder="Type a book name...")

if search.strip() != "":
    results = books[books['title'].str.contains(search, case=False, na=False)]
    if results.empty:
        st.warning("No results found!")
    else:
        st.subheader("Search Results:")
        for index, row in results.head(5).iterrows():
            st.write(f"**{row['title']}** — {row['authors']}")

st.write("---")

# ---------------------------------------------------------
# RECOMMENDATION SYSTEM
# ---------------------------------------------------------
st.subheader("📖 Select a Book for Recommendation")

book_list = books["title"].tolist()
selected_book = st.selectbox("Choose a book", book_list)

if selected_book:
    idx = books[books["title"] == selected_book].index[0]

    scores = list(enumerate(similarity[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    st.write("")
    st.write("### 🎯 Top Recommendations")

    cols = st.columns(5)

    for num, (i, score) in enumerate(sorted_scores):
        with cols[num]:
            try:
                img = books.iloc[i]["thumbnail"]
                st.image(img, width=130)
            except:
                st.image("https://via.placeholder.com/150", width=130)

            st.markdown(f"<p class='book-title'>{books.iloc[i]['title']}</p>", unsafe_allow_html=True)
            st.caption(books.iloc[i]["authors"])


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.write("---")
st.markdown(
    "<p style='text-align:center;color:gray;'>Made with ❤️ using Streamlit</p>",
    unsafe_allow_html=True
)
