# SWYNEX Final AI Application – AI Career Match Analyzer

## Project Overview
The AI Career Match Analyzer is a Python-based application that helps users explore suitable career options based on their skills and interests. It compares user-entered skills with predefined career profiles and suggests relevant career paths.

## Objectives
- Help users explore suitable AI and data-related careers.
- Identify skill gaps for selected career options.
- Provide career suggestions using text similarity techniques.
- Demonstrate a simple AI-powered application with an interactive interface.

## Features
- Career recommendations based on user skills.
- Ranked career matches with similarity scores.
- Skill gap analysis and suggested skills to learn.
- Input validation and error handling.
- Sample test profiles for trying the application.

## Technologies Used
- Python
- Streamlit
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity

## How It Works
1. The user enters their skills.
2. The application converts the text into numerical features using TF-IDF.
3. Cosine similarity compares the user's skills with predefined career profiles.
4. The application displays ranked career matches and suggested skills to improve.

## Installation and Setup

Clone or download this repository, then open a terminal in the project folder.

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## Testing
The application was tested using sample profiles for different career interests, including Data Analytics, Machine Learning, and Generative AI. Empty skill input was also tested to verify input validation.

## Limitations
- Career suggestions are based on predefined career profiles.
- Similarity scores are not probabilities of getting a job.
- Recommendations may be incomplete because the career dataset is limited.
- Results should be used as guidance, not as a final career decision.

## Ethical Considerations
Career recommendations may reflect limitations in the predefined profiles. Users should consider their interests, qualifications, and professional guidance before making career decisions. Avoid entering sensitive personal information.

## Conclusion
This project demonstrates how AI-related text-processing techniques can support career exploration through an interactive application.

**Developed as part of the SWYNEX Technologies Internship.**
