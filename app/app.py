import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from fpdf import FPDF
import base64
from sklearn.preprocessing import LabelEncoder
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from recommendation import load_data as load_rec_data, build_nearest_neighbors, recommend_restaurants  # <-- updated import

# ------------------- REMOVE WARNINGS -------------------
warnings.filterwarnings("ignore")
st.set_option('client.showErrorDetails', False)

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="🍽️ Zomato Rating Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------- LOAD MODEL & DATA -------------------
@st.cache_resource
def load_model():
    return joblib.load('outputs/models/best_model_Random_Forest.joblib')

@st.cache_data
def load_data():
    return pd.read_csv('data/clean_zomato.csv')

model = load_model()
df = load_data()

# ------------------- STYLING -------------------

st.markdown("""
    <style>
    /* 🔴 Red color for radio button labels */
    div[role="radiogroup"] label p {
        color: #e53935 !important;   /* Zomato red */
        font-weight: 600 !important;
    }

    /* Optional: make the selected option bolder */
    div[role="radiogroup"] input:checked + div p {
        color: #b71c1c !important;
        font-weight: 700 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #fff5f5 0%, #ffeaea 50%, #ffd6d6 100%);
        font-family: "Poppins", sans-serif;
        color: #b71c1c;
    }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 3px solid #ff4d4d;
        box-shadow: 2px 0px 10px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #c62828 !important;
        font-weight: 700;
    }
    label {
        color: #b71c1c !important;
        font-weight: 600 !important;
    }
    .stSelectbox, .stNumberInput, .stSlider {
        background-color: #fff !important;
        border: 1px solid #ffcdd2 !important;
        border-radius: 10px !important;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.05) !important;
    }
    div.stButton > button {
        background-color: #e53935 !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.2rem !important;
        border: none !important;
        box-shadow: 0 4px 6px rgba(229,57,53,0.4);
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #d32f2f !important;
        box-shadow: 0 6px 10px rgba(229,57,53,0.5);
        transform: scale(1.02);
    }
    header, [data-testid="stHeader"], [data-testid="stToolbar"], div[data-testid="stDecoration"], footer {
        display: none !important;
    }
    [data-testid="stSidebar"]::after {
        content: "";
        position: absolute;
        top: 0;
        right: 0;
        height: 100%;
        width: 3px;
        background-color: #ff4d4d;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------- SIDEBAR -------------------
st.sidebar.markdown("## 📊 Zomato Insights")
st.sidebar.image("outputs/eda/top_cuisines.png", caption="Top 10 Cuisines", use_container_width=True)
st.sidebar.image("outputs/eda/top_locations_by_rating.png", caption="Top Locations", use_container_width=True)
st.sidebar.image("outputs/eda/rating_distribution.png", caption="Rating Distribution", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### ❤️ Designed for Zomato Dataset Analysis")

# ------------------- MAIN TITLE -------------------
st.markdown("<h1 style='text-align:center;'>🍴 Zomato Restaurant Rating Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; color:#e53935;'>Predict customer ratings using Machine Learning</h3>", unsafe_allow_html=True)
st.markdown("---")

# ------------------- USER INPUT SECTION -------------------
st.subheader("🔧 Enter Restaurant Details")

col1, col2 = st.columns(2)

with col1:
    online_order = st.selectbox("Online Order Available?", ["Yes", "No"])
    book_table = st.selectbox("Book Table Available?", ["Yes", "No"])
    cost = st.number_input("Approx Cost for Two (₹)", min_value=100, max_value=5000, value=500, step=1)
    votes = st.number_input("Number of Votes", min_value=0, max_value=10000, value=100, step=1)

with col2:
    cuisines = sorted(df['Primary Cuisine'].dropna().unique().tolist())
    primary_cuisine = st.selectbox("Primary Cuisine", cuisines)
    locations = sorted(df['location'].dropna().unique().tolist())
    location = st.selectbox("Restaurant Location", locations)

# ------------------- DATA PREPARATION -------------------
online_order_val = 1 if online_order == "Yes" else 0
book_table_val = 1 if book_table == "Yes" else 0

le = LabelEncoder()
le.fit(df['location'].astype(str))
location_encoded = le.transform([location])[0]

input_data = pd.DataFrame({
    'online_order': [online_order_val],
    'book_table': [book_table_val],
    'approx_cost(for two people)': [cost],
    'votes': [votes],
    'Primary Cuisine': [primary_cuisine],
    'location encoded': [location_encoded]
})

# ------------------- PREDICTION -------------------
if st.button("🔮 Predict Rating"):
    prediction = model.predict(input_data)[0]
    rating = round(prediction, 2)

    st.markdown(f"""
        <div style="
            background-color:#e53935;
            color:white;
            padding:1.2rem;
            border-radius:12px;
            text-align:center;
            font-size:28px;
            font-weight:700;
            box-shadow:0px 4px 10px rgba(229,57,53,0.4);
            margin-top:1rem;
        ">
        ⭐ Predicted Zomato Rating: {rating} / 5.0
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🧾 Prediction Summary")
    st.write(input_data)

    # ------------------- PDF REPORT -------------------
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("DejaVuSans", "", "fonts/ttf/DejaVuSans.ttf", uni=True)

    pdf.set_font("DejaVuSans", "", 18)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(220, 20, 60)
    pdf.cell(0, 15, txt="🍴 Zomato Rating Prediction Report", ln=True, align="C", fill=True)
    pdf.ln(10)

    pdf.set_font("DejaVuSans", "", 16)
    pdf.set_text_color(220, 20, 60)
    pdf.cell(0, 10, txt=f"⭐ Predicted Rating: {rating} / 5.0", ln=True)
    pdf.ln(8)

    pdf.set_font("DejaVuSans", "", 14)
    pdf.set_text_color(220, 20, 60)
    pdf.cell(0, 10, txt="🍽 Restaurant Details", ln=True)
    pdf.ln(2)

    pdf.set_font("DejaVuSans", "", 12)
    pdf.set_text_color(0, 0, 0)

    row_height = 8
    col_width_label = 60
    col_width_value = 120

    details = [
        ("Online Order", "Yes" if online_order_val else "No"),
        ("Book Table", "Yes" if book_table_val else "No"),
        ("Approx Cost for Two", f"₹{cost}"),
        ("Votes", f"{votes}"),
        ("Primary Cuisine", primary_cuisine),
        ("Location", location)
    ]

    for label, value in details:
        pdf.set_fill_color(245, 245, 245)
        pdf.cell(col_width_label, row_height, txt=label, border=1, fill=True)
        pdf.cell(col_width_value, row_height, txt=value, border=1, ln=True, fill=False)

    pdf.ln(5)
    pdf.set_draw_color(220, 20, 60)
    pdf.set_line_width(0.8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    pdf_bytes = pdf.output(dest='S')
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="Zomato_Rating_Report.pdf">📥 Download Prediction Report (PDF)</a>'
    st.markdown(href, unsafe_allow_html=True)

# ------------------- EDA SECTION -------------------
st.markdown("---")
st.subheader("📊 Exploratory Data Analysis (EDA)")

eda_section = st.radio(
    "Select EDA Chart:",
    ["Top 10 Cuisines", "Cost vs Rating", "Top Locations", "Online Order vs Rating", "Rating Distribution"],
    index=0,
    horizontal=True
)

if eda_section == "Top 10 Cuisines":
    st.markdown("**Top 10 Popular Cuisines**")
    top_cuisines = df['Primary Cuisine'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x=top_cuisines.values, y=top_cuisines.index, palette='viridis', ax=ax)
    ax.set_xlabel("Number of Restaurants")
    ax.set_ylabel("Cuisine")
    st.pyplot(fig)

elif eda_section == "Cost vs Rating":
    st.markdown("**Cost vs Rating Scatter Plot**")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(x='approx_cost(for two people)', y='rate', hue='Primary Cuisine', data=df, palette='tab10', ax=ax)
    ax.set_xlabel("Approx Cost (₹) for Two")
    ax.set_ylabel("Rating")
    st.pyplot(fig)

elif eda_section == "Top Locations":
    st.markdown("**Top Locations by Number of Restaurants**")
    top_locations = df['location'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x=top_locations.index, y=top_locations.values, palette='magma', ax=ax)
    ax.set_xlabel("Location")
    ax.set_ylabel("Number of Restaurants")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

elif eda_section == "Online Order vs Rating":
    st.markdown("**Online Order Availability vs Ratings**")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(x='online_order', y='rate', data=df, palette='coolwarm', ax=ax)
    ax.set_xlabel("Online Order (0=No, 1=Yes)")
    ax.set_ylabel("Rating")
    st.pyplot(fig)

elif eda_section == "Rating Distribution":
    st.markdown("**Restaurant Rating Distribution**")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(df['rate'], bins=20, kde=True, color='coral', ax=ax)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Restaurants")
    st.pyplot(fig)

# ------------------- 🍴 RECOMMENDATION SYSTEM TAB -------------------
st.markdown("---")
st.subheader("🍴 Restaurant Recommendation System")

st.write("Find restaurants similar to your favorite one based on cuisine, location, and services.")

@st.cache_resource
def load_recommendation_model():
    df_rec = load_rec_data("data/clean_zomato.csv")
    nn, vectors = build_nearest_neighbors(df_rec)
    return df_rec, nn, vectors

df_rec, nn, vectors = load_recommendation_model()

restaurant_names = sorted(df_rec['name'].unique())
selected_restaurant = st.selectbox("Select a restaurant to get recommendations:", restaurant_names)

if st.button("🔍 Show Recommendations"):
    with st.spinner("Finding similar restaurants..."):
        results = recommend_restaurants(selected_restaurant, df_rec, nn, vectors)
    if results.empty:
        st.error("No similar restaurants found.")
    else:
        st.markdown(f"""
        <div style="
            background-color:#e53935;
            color:white;
            padding:1.2rem;
            border-radius:12px;
            text-align:center;
            font-size:28px;
            font-weight:700;
            box-shadow:0px 4px 10px rgba(229,57,53,0.4);
            margin-top:1rem;
        ">
        ⭐ ✅ Top {len(results)} restaurants similar to **{selected_restaurant}**)
        </div>
        """, unsafe_allow_html=True)
    
        st.dataframe(results, use_container_width=True)
