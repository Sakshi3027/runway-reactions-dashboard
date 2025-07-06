# ✨ Runway Reactions - by Saksh

**Live Demo:** [Link to be added after deployment]

## 📖 Overview

Runway Reactions is a comprehensive, interactive web application designed to analyze customer sentiment and discover insights from fashion product reviews. This project transforms over 20,000 raw text reviews into a user-friendly dashboard that reveals not only *if* customers are happy, but *why*. It showcases an end-to-end data science workflow, from data cleaning and analysis to machine learning and final deployment.

## 🚀 Key Features

* **Interactive Dashboard:** A multi-page Streamlit application with a custom, aesthetic user interface for a professional feel.
* **Sentiment Analysis:** Utilizes a pre-trained `transformers` model (DistilBERT) from Hugging Face to perform sentiment analysis on every customer review, classifying them as POSITIVE or NEGATIVE.
* **Exploratory Data Analysis (EDA):** Interactive charts and metrics that allow users to filter data by department and instantly see breakdowns of ratings and sentiment.
* **Topic Modeling:** Employs Latent Dirichlet Allocation (LDA) with `scikit-learn` to automatically discover the key themes and reasons for complaints within negative reviews.
* **Content-Based Recommendation Engine:** A feature that suggests similar products to the user based on the textual content of reviews, calculated using TF-IDF and Cosine Similarity.

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:**
    * **Web App:** Streamlit
    * **Data Manipulation & Analysis:** Pandas
    * **Machine Learning:** Scikit-learn (TF-IDF, LDA), Transformers (Hugging Face for Sentiment Analysis)
    * **Text Processing:** NLTK
    * **Plotting:** Matplotlib, Seaborn

## ⚙️ Setup & How to Run Locally

To run this project on your own machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Sakshi3027/runway-reactions-dashboard.git](https://github.com/Sakshi3027/runway-reactions-dashboard.git)
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run Home.py
    ```