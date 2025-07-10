import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    layout="wide", 
    page_title="RUNWAY REACTIONS", 
    page_icon="✨"
)

# --- Custom CSS ---
# This CSS now uses a direct URL for the background image.
page_bg_img = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&display=swap');

    /* Main app background */
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("./background2.jpg");
        background-image: url("https://www.toptal.com/designers/subtlepatterns/uploads/dark-denim-3.png");
        background-size: cover;
        background-position: center;
        background-repeat: repeat;
        background-attachment: local;
    }}
    
    /* Make headers and text readable */
    h1, h2, h3, p, label {{
        color: #FFFFFF !important; 
    }}
    
    /* Custom Font for the Main Title */
    h1 {{
        font-family: 'Playfair Display', serif !important;
        font-size: 3rem !important;
    }}

    /* Final Button Styles */
    .stButton > button {{
        border: 2px solid #FFFFFF; border-radius:10px; background-color: #FFFFFF;
        color: #000000 !important; transition: all 0.2s ease-in-out; font-weight: bold;
    }}
    .stButton > button * {{ color: #000000 !important; }}
    .stButton > button:hover {{ background-color: transparent; color: #FFFFFF !important; border-color: #FFFFFF; }}
    .stButton > button:hover * {{ color: #FFFFFF !important; }}
    
    /* Style for the dropdown select box */
    [data-testid="stSelectbox"] > div[data-baseweb="select"] > div {{
        background-color: #FFFFFF; border: 2px solid #000000; border-radius: 10px; color: #000000;
    }}
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div > div {{ color: #000000 !important; }}
    [data-testid="stSelectbox"] svg {{ fill: #000000 !important; }}
    [data-baseweb="popover"] ul {{ background-color: #FFFFFF; border-radius: 10px; }}
    [data-baseweb="popover"] ul li:hover {{ background-color: #e0e0e0 !important; }}
    [data-baseweb="popover"] ul li > div {{ color: #000000 !important; }}

    /* Recommendation card style */
    .recommendation-card {{
        background-color: rgba(0, 0, 0, 0.5); 
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 15px;
    }}
    /* Hide the default Streamlit sidebar */
    .st-emotion-cache-16txtl3 {{
        display: none;
    }}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


# --- Data Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv("fashion_data_with_sentiment.csv", index_col=0)
    return df

df = load_data()


# --- App Title and Custom Navigation ---
st.markdown("<h1 style='text-align: center;'>RUNWAY REACTIONS - by SAKSH</h1>", unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state.page = 'Home'

spacer1, nav_col1, nav_col2, spacer2 = st.columns([1.5, 1.2, 1.2, 1.5]) 
with nav_col1:
    if st.button('🏠 Home', use_container_width=True):
        st.session_state.page = 'Home'
with nav_col2:
    if st.button('🚀 Explore Insights', use_container_width=True):
        st.session_state.page = 'Explore Insights'
st.markdown("---")


# --- Page Content ---
if st.session_state.page == 'Home':
    st.markdown("""
    <div style="text-align: center;">
        <h2 style="color:white;">"Style speaks louder when data listens"</h2>
        <p style="color:white;">From delight to disappointment, this dashboard unravels what customers feel and why. Let the numbers speak the language of fashion.</p>
        <br>
        <p style="color:white;"><i>Note: This live demo showcases the interactive dashboard and visualization features. The full version with all AI features is available in the <a href="https://github.com/Sakshi3027/runway-reactions-dashboard" style="color: #A7C7E7;">GitHub repository</a>.</i></p>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.page == 'Explore Insights':
    st.header("📊 Department-Level Insights")
    department_options = ['All'] + sorted(df['Department Name'].dropna().unique().tolist())
    selected_department = st.selectbox("Select a Department to Analyze", department_options)

    if selected_department == 'All':
        filtered_df = df
    else:
        filtered_df = df[df['Department Name'] == selected_department]

    col1, col2 = st.columns(2)
    col1.metric("Total Reviews Analyzed", f"{filtered_df.shape[0]:,}")
    avg_rating = filtered_df['Rating'].mean()
    col2.metric("Average Rating", f"{avg_rating:.2f} ★")

    st.header("📈 Visual Insights")
    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        st.subheader("Distribution of Ratings")
        rating_counts = filtered_df['Rating'].value_counts().sort_index()
        fig1, ax1 = plt.subplots()
        sns.barplot(data=rating_counts.reset_index(), x='Rating', y='count', ax=ax1, palette="viridis", hue='Rating', legend=False)
        st.pyplot(fig1)
    with plot_col2:
        st.subheader("Sentiment Breakdown")
        sentiment_counts = filtered_df['sentiment_label'].value_counts()
        fig2, ax2 = plt.subplots()
        sns.barplot(data=sentiment_counts.reset_index(), x='sentiment_label', y='count', ax=ax2, palette="coolwarm_r", hue='sentiment_label', legend=False)
        st.pyplot(fig2)