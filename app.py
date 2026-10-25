import re
import html
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CareerPilot AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- PREMIUM DESIGN ----------------
st.markdown("""
<style>
:root {
    --navy: #172554;
    --indigo: #4F46E5;
    --purple: #7C3AED;
    --muted: #64748B;
}
.stApp {
    background: linear-gradient(135deg, #F8FAFF 0%, #EEF2FF 100%);
    color: #172033;
}
.block-container {
    max-width: 1320px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}
h1, h2, h3, h4 {
    color: #172554 !important;
    letter-spacing: -0.4px;
}
.hero {
    background: linear-gradient(115deg, #172554, #4338CA 58%, #7C3AED);
    padding: 34px;
    border-radius: 22px;
    color: white;
    margin-bottom: 26px;
    box-shadow: 0 12px 30px rgba(67, 56, 202, 0.20);
}
.hero h1 {
    color: white !important;
    font-size: 38px;
    font-weight: 800;
    margin: 0 0 8px 0;
}
.hero p {
    color: #E0E7FF !important;
    font-size: 16px;
    margin: 0;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #172554, #252F68);
}
section[data-testid="stSidebar"] * {
    color: #F8FAFC;
}
[data-testid="stForm"] {
    background: white;
    border: 1px solid #E0E7FF;
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 8px 28px rgba(30, 41, 90, 0.07);
}
[data-testid="stWidgetLabel"] p {
    color: #334155;
    font-weight: 600;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 10px !important;
    border-color: #D9E1F0 !important;
    background: white !important;
}
.stTextArea textarea {
    border: 1px solid #D9E1F0 !important;
    border-radius: 12px !important;
    background: white !important;
}
.stFormSubmitButton > button,
.stButton > button,
.stDownloadButton > button {
    border-radius: 11px;
    font-weight: 650;
    min-height: 44px;
}
.stFormSubmitButton > button,
.stButton > button[kind="primary"] {
    background: linear-gradient(100deg, #4F46E5, #7C3AED);
    color: white;
    border: none;
    box-shadow: 0 5px 14px rgba(79, 70, 229, 0.20);
}
.stDownloadButton > button {
    background: #EEF2FF;
    color: #3730A3;
    border: 1px solid #C7D2FE;
}
.kpi {
    background: linear-gradient(145deg, #FFFFFF, #F9FAFF);
    border: 1px solid #E0E7FF;
    border-radius: 18px;
    padding: 22px 20px;
    min-height: 112px;
    box-shadow: 0 7px 22px rgba(30, 41, 90, 0.06);
}
.kpi-label {
    color: #64748B;
    font-size: 11px;
    font-weight: 750;
    letter-spacing: 1px;
    margin-bottom: 10px;
}
.kpi-value {
    color: #312E81;
    font-size: 24px;
    font-weight: 800;
    line-height: 1.35;
    overflow-wrap: anywhere;
}
.rec-score {
    font-size: 27px;
    font-weight: 800;
    color: #4F46E5;
    text-align: right;
}
.rec-muted {
    color: #64748B;
    font-size: 12px;
    text-align: right;
}
.skill-chip {
    display: inline-block;
    background: #EEF2FF;
    color: #3730A3;
    border: 1px solid #E0E7FF;
    border-radius: 20px;
    padding: 5px 10px;
    margin: 4px 4px 0 0;
    font-size: 12px;
}
[data-testid="stExpander"], [data-testid="stMetric"] {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
}
[data-testid="stMetric"] { padding: 16px; }
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #4F46E5, #8B5CF6);
    border-radius: 99px;
}
[data-testid="stAlert"] { border-radius: 12px; }
hr { border-color: #E2E8F0; margin: 1.5rem 0; }
#MainMenu, footer { visibility: hidden; }
@media (max-width: 768px) {
    .block-container { padding: 1rem; }
    .hero { padding: 23px; }
    .hero h1 { font-size: 29px; }
}
</style>
""", unsafe_allow_html=True)

# ---------------- CAREER DATABASE ----------------
CAREERS = {
    "Data Analyst": {
        "skills": ["python", "sql", "excel", "power bi", "statistics",
                   "data visualization", "tableau", "pandas"],
        "projects": ["Sales Performance Dashboard", "Customer Data Analysis"],
        "learning": ["Practice SQL queries and joins",
                     "Build an interactive Power BI dashboard",
                     "Create a data analysis portfolio"],
    },
    "Data Scientist": {
        "skills": ["python", "sql", "statistics", "pandas", "numpy",
                   "machine learning", "scikit-learn", "data visualization"],
        "projects": ["Customer Churn Prediction", "Exploratory Data Analysis"],
        "learning": ["Strengthen Python and statistics",
                     "Learn Pandas and NumPy",
                     "Build a machine learning project"],
    },
    "Machine Learning Engineer": {
        "skills": ["python", "machine learning", "scikit-learn",
                   "tensorflow", "deep learning", "algorithms", "numpy"],
        "projects": ["House Price Prediction", "ML Model Deployment"],
        "learning": ["Study core ML algorithms",
                     "Learn model evaluation techniques",
                     "Build and deploy an ML application"],
    },
    "Generative AI Specialist": {
        "skills": ["python", "prompt engineering", "generative ai", "llm",
                   "nlp", "api integration", "machine learning"],
        "projects": ["AI Document Assistant", "Question-Answering Chatbot"],
        "learning": ["Practice prompt engineering",
                     "Learn LLM fundamentals",
                     "Build an AI-powered application"],
    },
    "Business Analyst": {
        "skills": ["excel", "sql", "power bi", "communication",
                   "business analysis", "requirements", "data visualization"],
        "projects": ["Business Performance Report", "Business Case Study"],
        "learning": ["Practice Excel and SQL",
                     "Learn requirements analysis",
                     "Present insights using a dashboard"],
    },
    "Python Developer": {
        "skills": ["python", "oop", "git", "api", "debugging",
                   "sql", "testing", "data structures"],
        "projects": ["Task Management Application", "REST API Project"],
        "learning": ["Strengthen Python fundamentals",
                     "Practice Git and software testing",
                     "Build a complete application"],
    },
    "Data Engineer": {
        "skills": ["python", "sql", "etl", "databases", "data pipelines",
                   "cloud", "spark", "data warehousing"],
        "projects": ["ETL Data Pipeline", "Data Warehouse Mini-Project"],
        "learning": ["Learn advanced SQL",
                     "Practice ETL concepts",
                     "Build a data pipeline"],
    },
}

SAMPLES = {
    "Data Analytics": "Python, SQL, Excel, Power BI, statistics",
    "Generative AI": "Python, prompt engineering, LLM, NLP, API integration",
    "Machine Learning": "Python, machine learning, scikit-learn, pandas, numpy",
}

def normalize(text):
    return re.sub(r"\\s+", " ", str(text).lower().strip())

def analyze(skills, interests):
    names = list(CAREERS.keys())
    skill_text = normalize(skills)
    interest_text = " ".join(interests).lower()
    profiles = [" ".join(CAREERS[name]["skills"]) for name in names]

    matrix = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(
        [skill_text + " " + interest_text] + profiles
    )
    similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    results = []

    for name, similarity in zip(names, similarities):
        required = CAREERS[name]["skills"]
        matched = [
            skill for skill in required
            if re.search(
                r"(?<![a-z0-9+#.])" + re.escape(skill.lower()) +
                r"(?![a-z0-9+#.])", skill_text
            )
        ]
        missing = [skill for skill in required if skill not in matched]
        coverage = len(matched) / len(required)
        score = 100 * (0.55 * float(similarity) + 0.45 * coverage)

        results.append({
            "Career": name,
            "Match score": round(score, 1),
            "Skill coverage": round(coverage * 100, 1),
            "Matched skills": matched,
            "Missing skills": missing,
            "Projects": CAREERS[name]["projects"],
            "Learning": CAREERS[name]["learning"],
        })

    return sorted(results, key=lambda item: item["Match score"], reverse=True)

def show_chips(items):
    if not items:
        st.caption("No listed skills matched yet.")
        return
    chips = "".join(
        f'<span class="skill-chip">{html.escape(str(item))}</span>'
        for item in items
    )
    st.markdown(chips, unsafe_allow_html=True)

# ---------------- SIDEBAR AND HERO ----------------
st.sidebar.markdown("# 🎯 CareerPilot AI")
st.sidebar.caption("YOUR CAREER INTELLIGENCE WORKSPACE")
st.sidebar.divider()
page = st.sidebar.radio(
    "NAVIGATION",
    ["Dashboard", "Skill Gap Analysis", "Learning Roadmap", "About"],
)
st.sidebar.divider()
st.sidebar.markdown("**Explore. Learn. Grow.**")
st.sidebar.caption("AI-assisted career exploration prototype")
st.sidebar.caption("SWYNEX Internship Project")

st.markdown("""
<div class="hero">
    <h1>CareerPilot AI</h1>
    <p>Discover career paths that connect your skills, interests and goals.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.subheader("Build your career profile")
    st.write("Share your current skills to explore career options and identify areas where you can grow.")

    with st.form("career_profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            education = st.selectbox(
                "Education level", ["Bachelor's", "Diploma", "Master's", "Other"]
            )
            experience = st.selectbox(
                "Experience",
                ["Student / Fresher", "Less than 1 year", "1–3 years", "3+ years"],
            )
        with col2:
            interests = st.multiselect(
                "Areas of interest",
                ["Data Analytics", "Artificial Intelligence", "Machine Learning",
                 "Generative AI", "Business", "Python Development", "Data Engineering"],
            )
            goal = st.selectbox(
                "Your current goal",
                ["Find an entry-level role", "Build projects",
                 "Improve my skills", "Explore career options"],
            )

        sample = st.selectbox(
            "Try a sample profile", ["Enter my own skills"] + list(SAMPLES.keys())
        )
        skills = st.text_area(
            "Skills you currently have",
            placeholder="For example: Python, SQL, Excel, Power BI",
            height=110,
        )
        submitted = st.form_submit_button(
            "✨ Generate Career Insights", type="primary", use_container_width=True
        )

    if submitted:
        if sample != "Enter my own skills":
            skills = SAMPLES[sample]
        if not skills.strip():
            st.warning("Enter your skills or choose a sample profile.")
        else:
            st.session_state["results"] = analyze(skills, interests)
            st.session_state["profile"] = {
                "skills": skills, "education": education, "experience": experience,
                "interests": interests, "goal": goal,
            }
            st.success("Your career insights are ready!")

    if "results" in st.session_state:
        results = st.session_state["results"]
        profile = st.session_state["profile"]
        top = results[0]
        st.divider()
        st.subheader("Your career overview")
        st.caption(
            f"Profile: {profile['education']} · {profile['experience']} · Goal: {profile['goal']}"
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                '<div class="kpi"><div class="kpi-label">TOP CAREER MATCH</div>'
                f'<div class="kpi-value">{html.escape(top["Career"])}</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                '<div class="kpi"><div class="kpi-label">MATCH SCORE</div>'
                f'<div class="kpi-value">{top["Match score"]}%</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                '<div class="kpi"><div class="kpi-label">CAREER PATHS ANALYZED</div>'
                f'<div class="kpi-value">{len(results)}</div></div>',
                unsafe_allow_html=True,
            )

        st.caption(
            "Prototype scores combine text similarity and listed-skill coverage. "
            "They are not probabilities of getting hired."
        )
        st.divider()
        st.subheader("Career match analytics")
        chart_df = pd.DataFrame([
            {"Career": item["Career"], "Match score": item["Match score"]}
            for item in results[:5]
        ]).set_index("Career")
        st.bar_chart(chart_df)

        st.divider()
        st.subheader("Your top career recommendations")
        for rank, item in enumerate(results[:5], start=1):
            with st.container(border=True):
                left, right = st.columns([3, 1])
                with left:
                    st.markdown(f"#### {rank:02d}  {item['Career']}")
                    st.caption(
                        f"{len(item['Matched skills'])} of "
                        f"{len(CAREERS[item['Career']]['skills'])} listed skills matched"
                    )
                with right:
                    st.markdown(
                        f'<div class="rec-score">{item["Match score"]}%</div>'
                        '<div class="rec-muted">MATCH SCORE</div>',
                        unsafe_allow_html=True,
                    )
                st.progress(item["Skill coverage"] / 100)
                st.caption(f"Skill coverage: {item['Skill coverage']}%")
                tab1, tab2, tab3 = st.tabs(
                    ["Matched skills", "Skills to develop", "Project ideas"]
                )
                with tab1:
                    show_chips(item["Matched skills"])
                with tab2:
                    show_chips(item["Missing skills"])
                with tab3:
                    for project in item["Projects"]:
                        st.write(f"• {project}")

        report = pd.DataFrame([
            {
                "Rank": rank,
                "Career": item["Career"],
                "Match score": item["Match score"],
                "Skill coverage": item["Skill coverage"],
                "Matched skills": ", ".join(item["Matched skills"]),
                "Missing skills": ", ".join(item["Missing skills"]),
            }
            for rank, item in enumerate(results, start=1)
        ])
        st.download_button(
            "⬇️ Download Career Report (CSV)",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="career_insights.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ---------------- SKILL GAP ANALYSIS ----------------
elif page == "Skill Gap Analysis":
    st.subheader("Skill Gap Analysis")
    st.write("Understand which skills you already have and which you can develop next.")
    if "results" not in st.session_state:
        st.info("Generate your career insights from the Dashboard first.")
    else:
        results = st.session_state["results"]
        career = st.selectbox("Choose a career path", [item["Career"] for item in results])
        item = next(r for r in results if r["Career"] == career)
        st.markdown(f"### {career}")
        st.metric("Current skill coverage", f'{item["Skill coverage"]}%')
        st.progress(item["Skill coverage"] / 100)
        left, right = st.columns(2)
        with left:
            st.markdown("#### ✅ Matched skills")
            if item["Matched skills"]:
                for skill in item["Matched skills"]:
                    st.success(skill)
            else:
                st.info("No listed skills matched yet.")
        with right:
            st.markdown("#### 🎯 Skills to develop")
            if item["Missing skills"]:
                for skill in item["Missing skills"]:
                    st.warning(skill)
            else:
                st.success("All listed skills matched!")
        gap_df = pd.DataFrame({
            "Category": ["Matched", "To develop"],
            "Number of skills": [
                len(item["Matched skills"]), len(item["Missing skills"])
            ],
        }).set_index("Category")
        st.markdown("#### Skill gap overview")
        st.bar_chart(gap_df)

# ---------------- LEARNING ROADMAP ----------------
elif page == "Learning Roadmap":
    st.subheader("Your learning roadmap")
    st.write("Use this suggested sequence to plan your next learning steps.")
    if "results" not in st.session_state:
        st.info("Generate your career insights from the Dashboard first.")
    else:
        results = st.session_state["results"]
        career = st.selectbox("Choose your target career", [item["Career"] for item in results])
        item = next(r for r in results if r["Career"] == career)
        st.markdown(f"### Roadmap for {career}")
        st.caption("Suggested learning sequence — adjust it to your own pace.")
        for i, task in enumerate(item["Learning"], start=1):
            with st.container(border=True):
                st.markdown(f"**STEP {i}**")
                st.checkbox(task, key=f"{career}_roadmap_step_{i}")
        st.markdown("#### Portfolio project ideas")
        for project in item["Projects"]:
            st.markdown(f"- {project}")
        st.checkbox("Review my progress and update my portfolio", key=f"{career}_roadmap_review")

# ---------------- ABOUT ----------------
else:
    st.subheader("About CareerPilot AI")
    st.write(
        "CareerPilot AI is an educational career exploration prototype built with "
        "Python, Streamlit, Scikit-learn and Pandas. It uses TF-IDF text features, "
        "cosine similarity and explicit skill coverage to compare entered skills "
        "with predefined career profiles."
    )
    st.markdown("### What you can do")
    st.markdown(
        "- Explore seven predefined career paths.\\n"
        "- Compare your entered skills with career requirements.\\n"
        "- Review skill gaps and suggested learning steps.\\n"
        "- Download your career comparison as a CSV report."
    )
    st.markdown("### Responsible use")
    st.info(
        "Recommendations are exploratory suggestions, not employment predictions. "
        "Career profiles are limited, and wording can affect matching. Review results "
        "critically and consider your interests and human career guidance."
    )

st.divider()
st.caption("CareerPilot AI • SWYNEX Internship • AI-assisted career exploration")
