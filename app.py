
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Career Match Analyzer",
    page_icon="🎯",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #f7f9fc; }
.hero {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(135deg, #172554, #2563eb);
    color: white;
    margin-bottom: 25px;
}
.card {
    padding: 18px;
    border-radius: 12px;
    background-color: white;
    border: 1px solid #e5e7eb;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🎯 AI Career Match Analyzer</h1>
<p>Discover career paths matching your skills, education and interests.</p>
</div>
""", unsafe_allow_html=True)

career_data = {
    "Data Analyst": {
        "skills": "python sql excel power bi statistics data visualization pandas numpy",
        "education": "computer science data science artificial intelligence statistics",
        "interests": "data analytics business intelligence dashboards insights",
        "experience": "data cleaning reporting visualization analysis"
    },
    "Machine Learning Engineer": {
        "skills": "python machine learning tensorflow keras scikit-learn pandas numpy statistics",
        "education": "artificial intelligence data science computer science",
        "interests": "machine learning artificial intelligence prediction models",
        "experience": "model training feature engineering evaluation"
    },
    "AI/GenAI Specialist": {
        "skills": "python generative ai prompt engineering llm api machine learning",
        "education": "artificial intelligence data science computer science",
        "interests": "generative ai chatbots automation large language models",
        "experience": "prompt engineering ai applications model integration"
    },
    "Business Analyst": {
        "skills": "excel sql power bi communication statistics business analysis",
        "education": "business management computer science data science",
        "interests": "business strategy requirements problem solving analytics",
        "experience": "reporting requirements analysis stakeholder management"
    },
    "Data Scientist": {
        "skills": "python machine learning statistics sql pandas numpy scikit-learn",
        "education": "data science artificial intelligence statistics computer science",
        "interests": "predictive analytics machine learning research",
        "experience": "data analysis model building experimentation"
    }
}

st.subheader("👤 Tell us about yourself")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Your Name", placeholder="e.g. Rishika")
    education = st.text_input(
        "Education / Branch",
        placeholder="e.g. B.Tech AI and Data Science"
    )
    skills = st.text_area(
        "Your Skills",
        placeholder="e.g. Python, SQL, Power BI, Excel"
    )

with col2:
    interests = st.text_area(
        "Career Interests",
        placeholder="e.g. Data Analytics, AI, Business Intelligence"
    )
    experience = st.text_area(
        "Projects / Experience",
        placeholder="e.g. Built a chatbot using Python"
    )

analyze = st.button(
    "🚀 Analyze My Career Match",
    use_container_width=True
)

# Reusable matching function
def analyze_profile(user_profile):
    career_names = list(career_data.keys())
    career_profiles = []

    for career in career_names:
        details = career_data[career]
        profile = " ".join(details.values())
        career_profiles.append(profile)

    vectorizer = TfidfVectorizer(stop_words="english")
    documents = [user_profile] + career_profiles
    vectors = vectorizer.fit_transform(documents)

    scores = cosine_similarity(vectors[0:1], vectors[1:])[0]
    results = sorted(
        zip(career_names, scores),
        key=lambda item: item[1],
        reverse=True
    )
    return results


if analyze:
    if not skills.strip():
        st.warning("Please enter your skills before analyzing.")
    elif not interests.strip():
        st.warning("Please enter your career interests before analyzing.")
    elif len(skills.strip()) < 3 or len(interests.strip()) < 3:
        st.warning("Please enter meaningful skills and interests.")
    else:
        try:
            user_profile = " ".join([
                skills, education, interests, experience
            ]).lower()

            results = analyze_profile(user_profile)
            best_career, best_score = results[0]

            st.divider()
            st.subheader(
                f"✨ Career Analysis for {name.strip() or 'Candidate'}"
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Top Career Match", best_career)
            c2.metric("Match Score", f"{best_score * 100:.1f}%")
            c3.metric("Career Paths Analyzed", len(results))

            st.markdown("### 🏆 Career Match Results")

            for career, score in results:
                st.markdown(
                    f'<div class="card"><h3>{career}</h3>'
                    f'<h2>{score * 100:.1f}%</h2>'
                    f'<p>Text similarity score, not a probability of employment.</p></div>',
                    unsafe_allow_html=True
                )
                st.progress(float(score))

            st.markdown("### 🧠 Skill Gap Analysis")

            required = set(career_data[best_career]["skills"].split())
            user_skills = set(
                skills.lower().replace(",", " ").split()
            )
            missing = sorted(required - user_skills)

            if missing:
                st.write(
                    f"Skills to consider for **{best_career}**:"
                )
                st.write(", ".join(skill.title() for skill in missing[:6]))
            else:
                st.success("Your entered skills cover the listed skills.")

            st.markdown("### 💡 Recommendation")
            st.info(
                f"{best_career} is the closest text match for your profile. "
                "Build practical projects and review the suggested skills."
            )
            st.caption(
                "Method: TF-IDF Vectorization + Cosine Similarity. "
                "Results are suggestions, not guaranteed career outcomes."
            )

        except ValueError:
            st.error(
                "The profile could not be analyzed. Please add more details "
                "to your skills and interests."
            )
        except Exception as error:
            st.error(
                "Something went wrong during analysis. "
                "Please check your inputs and try again."
            )
            st.caption(f"Technical detail: {type(error).__name__}")

# Task 3: test examples
st.divider()
with st.expander("🧪 Task 3: Test Examples and Evaluation"):
    st.write(
        "Use these sample profiles to check whether the matching feature "
        "returns relevant career suggestions."
    )

    test_profiles = {
        "Test 1 — Data Analytics": (
            "python sql excel power bi statistics",
            "dashboards business intelligence data analytics"
        ),
        "Test 2 — Generative AI": (
            "python prompt engineering generative ai llm",
            "chatbots automation large language models"
        ),
        "Test 3 — Machine Learning": (
            "python tensorflow machine learning statistics",
            "prediction models model training experimentation"
        )
    }

    selected_test = st.selectbox(
        "Choose a sample test",
        list(test_profiles.keys())
    )

    if st.button("Run Sample Test"):
        try:
            test_skills, test_interests = test_profiles[selected_test]
            test_results = analyze_profile(
                test_skills + " " + test_interests
            )
            st.success("Test completed successfully.")
            st.write("Sample profile:", selected_test)
            st.write("Top suggested career:", test_results[0][0])
            st.write(
                f"Similarity score: {test_results[0][1] * 100:.1f}%"
            )
            st.write("All ranked results:")
            for career, score in test_results:
                st.write(f"- {career}: {score * 100:.1f}%")
        except Exception as error:
            st.error("Sample test failed. Please try again.")
            st.caption(f"Technical detail: {type(error).__name__}")

st.divider()
st.caption("AI Career Match Analyzer • Python • Streamlit • NLP")
