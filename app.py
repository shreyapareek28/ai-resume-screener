import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Resume Screener")

resume = st.text_area("Paste Resume")
job = st.text_area("Paste Job Description")

if st.button("Check Match"):
    if resume and job:
        data = [resume, job]

        tfidf = TfidfVectorizer()
        vectors = tfidf.fit_transform(data)

        score = cosine_similarity(vectors[0:1], vectors[1:2])

        st.write("Match Score:", round(score[0][0]*100, 2), "%")
    else:
        st.write("Enter both fields")