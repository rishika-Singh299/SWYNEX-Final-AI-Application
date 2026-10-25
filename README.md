# CareerPilot AI – SWYNEX Internship Project

## Project Overview

CareerPilot AI is an interactive, Python-based career guidance application that helps users explore suitable career paths based on their skills, education, interests, and experience. It provides ranked career recommendations, skill gap insights, and learning guidance through a professional dashboard.

## Objectives

- Help users explore suitable AI, data, and technology careers.
- Identify skills that users may need to develop for their target careers.
- Provide career recommendations using text similarity techniques.
- Offer an interactive dashboard for career exploration and learning guidance.

## Features

- Career recommendations based on user-provided information.
- Ranked career matches with similarity scores.
- Skill gap analysis with suggestions for improvement.
- Learning roadmap and career guidance.
- Interactive dashboard with visualizations.
- Downloadable career matching results in CSV format.
- Input validation for a better user experience.

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity

## How It Works

1. The user enters relevant information, such as skills, education, interests, and experience.
2. The application processes the input using TF-IDF vectorization.
3. Cosine similarity compares the user's information with predefined career profiles.
4. Career matches are ranked based on the calculated similarity scores.
5. The application displays career recommendations, skill gaps, and learning guidance.

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version
- pip package manager

### Steps

1. Clone or download this repository.
2. Open a terminal in the project directory.
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
python -m streamlit run app.py
```

5. Open the local URL displayed in the terminal to access the application.

## Testing

The application was tested with sample career profiles and different user inputs to check career recommendations and skill gap analysis. Empty or incomplete inputs should also be tested to verify input validation.

## Limitations

- Career suggestions depend on predefined career profiles.
- Similarity scores are not probabilities of getting a job.
- Recommendations may be limited by the available career data.
- The application provides general guidance and does not guarantee employment.

## Ethical Considerations

Career recommendations may be limited by the predefined career profiles and the information entered by users. Users should consider their interests, qualifications, and professional guidance before making career decisions. Avoid entering sensitive personal information.

## Future Enhancements

- Expand the career profile dataset.
- Integrate updated job-market information.
- Add personalized course and certification recommendations.
- Explore advanced machine learning models for career matching.

## Conclusion

CareerPilot AI demonstrates how text-processing techniques can support career exploration through an interactive dashboard, ranked recommendations, skill gap analysis, and learning guidance.

**Developed as part of the SWYNEX Technologies Internship.**
