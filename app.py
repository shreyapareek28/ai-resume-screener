import streamlit as st
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI Resume Analytics Platform",
    page_icon="📊",
    layout="wide"
)

# ---------- CLEAN TEXT ----------
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

# ---------- ANALYSIS ENGINE ----------
def analyze_match(resume, job_desc):

    clean_resume = preprocess_text(resume)
    clean_job = preprocess_text(job_desc)

    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform([clean_resume, clean_job])

    similarity = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]

    feature_names = tfidf.get_feature_names_out()
    job_vec = matrix.toarray()[1]

    job_keywords = [feature_names[i] for i, v in enumerate(job_vec) if v > 0]

    matched = []
    missing = []

    for word in job_keywords:
        if word in clean_resume:
            matched.append(word)
        else:
            missing.append(word)

    return round(similarity * 100, 2), matched, missing

# ---------- ATS SCORE ----------
def ats_score(score, matched, total):
    skill_score = (len(matched) / total * 100) if total else 0
    keyword_score = score
    density_score = min(len(matched) * 5, 100)

    overall = (skill_score + keyword_score + density_score) / 3

    return round(skill_score, 2), round(keyword_score, 2), round(density_score, 2), round(overall, 2)

# ---------- CHART ----------
def plot_chart(matched, missing):
    fig, ax = plt.subplots()
    ax.bar(["Matched Skills", "Missing Skills"], [matched, missing])
    ax.set_title("Skill Match Breakdown")
    return fig

# ---------- MULTI CANDIDATE RANKING ----------
def rank_candidates(resumes, job_desc):

    results = []

    for i, r in enumerate(resumes):
        score, matched, missing = analyze_match(r, job_desc)
        results.append((i + 1, score, matched, missing))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

# ---------- UI ----------
st.title("📊 AI Resume Analytics Platform")
st.write("Advanced Data-Driven Candidate Ranking & ATS System")
st.markdown("---")

# ---------- INPUT ----------
st.subheader("💼 Job Description")
job_desc = st.text_area("Paste job description", height=200)

st.subheader("📝 Multiple Resumes (separate each resume with blank line)")
resumes_input = st.text_area("Paste resumes", height=250)

# ---------- BUTTON ----------
if st.button("🚀 Analyze & Rank Candidates"):

    if resumes_input.strip() == "" or job_desc.strip() == "":
        st.warning("Please enter both resumes and job description")

    else:

        resumes = [r.strip() for r in resumes_input.split("\n\n") if r.strip()]

        ranked = rank_candidates(resumes, job_desc)

        st.markdown("## 🏆 Candidate Ranking Results")

        for rank, score, matched, missing in ranked:

            skill_s, keyword_s, density_s, overall = ats_score(
                score, matched, len(matched) + len(missing)
            )

            st.markdown(f"### Candidate {rank}")
            st.write(f"Similarity Score: {score}%")
            st.write(f"ATS Overall Score: {overall}%")

            if score >= 70:
                st.success("Strong Match 🔥")
            elif score >= 40:
                st.warning("Moderate Match ⚠️")
            else:
                st.error("Weak Match ❌")

            col1, col2 = st.columns(2)

            with col1:
                st.success("Matched Skills")
                st.write(matched if matched else "None")

            with col2:
                st.error("Missing Skills")
                st.write(missing if missing else "None")

            st.pyplot(plot_chart(len(matched), len(missing)))

        best = ranked[0]

        st.markdown("---")
        st.markdown("## 🥇 Final Recommendation")

        st.success(f"Candidate {best[0]} is the BEST MATCH with {best[1]}% similarity score")

        # ---------- JSON REPORT ----------
        report = {
            "best_candidate": best[0],
            "best_score": best[1],
            "total_candidates": len(resumes),
            "ranking": [
                {
                    "candidate": r[0],
                    "score": r[1],
                    "matched_skills": r[2],
                    "missing_skills": r[3]
                }
                for r in ranked
            ]
        }

        json_report = json.dumps(report, indent=4)

        st.download_button(
            label="📥 Download Full Ranking Report (JSON)",
            data=json_report,
            file_name="candidate_ranking_report.json",
            mime="application/json"
        )