# ZOMATO-RESTAURANT-RATING-PREDICTION-AND-RECOMMENDATION-SYSTEM
zomato data science project

zomato restaurant dataset link - https://www.kaggle.com/datasets/rajeshrampure/zomato-dataset

# 🍽️ Zomato Restaurant Rating Prediction & Recommendation System  

## 🧠 Overview  
This project is a **Machine Learning and Recommendation System** that predicts restaurant ratings and recommends similar restaurants based on cuisine, location, and other features.  
It is built using **Streamlit** for the web interface and **scikit-learn** for the machine learning model.  

The goal is to help users and restaurant owners understand rating patterns and discover the best restaurants nearby.  

---

## ⚙️ Technologies Used  
- **Python** 🐍  
- **Streamlit** – for the interactive web app  
- **scikit-learn** – for model training and evaluation  
- **Pandas, NumPy** – for data cleaning and manipulation  
- **Matplotlib, Seaborn** – for EDA and data visualization  
- **Joblib** – for model serialization  

---

## 🧩 Project Modules  
1. **Restaurant Rating Prediction** – Predicts restaurant ratings based on features like cost, votes, online order availability, etc.  
2. **Restaurant Recommendation System** – Suggests similar restaurants using content-based filtering.  
3. **EDA Dashboard** – Visualizes important insights about cuisines, cost, ratings, and location.  

---

## 💡 Features  
- 🔹 Interactive Streamlit interface  
- 🔹 Machine learning–based rating prediction  
- 🔹 Smart restaurant recommendations  
- 🔹 Beautiful UI with color-coded themes  
- 🔹 Exploratory Data Analysis (EDA) charts  
- 🔹 Downloadable PDF prediction report  

---

## 📊 Exploratory Data Analysis  
The app provides a separate **EDA section** showing:  
- Top 10 cuisines  
- Cost vs Rating scatterplot  
- Top locations  
- Online order vs Rating comparison  
- Rating distribution  

---

## 🚀 How to Run  

### 1️⃣ Clone the repository

git clone git@github.com:vvishwarajput/ZOMATO-RESTAURANT-RATING-PREDICTION-AND-RECOMMENDATION-SYSTEM.git

2️⃣ Create a virtual environment

python -m venv venv
source venv/bin/activate    # For Linux/Mac
venv\Scripts\activate       # For Windows

3️⃣ Install dependencies

pip install -r requirements.txt


4️⃣ Run the app

streamlit run app.py


📁 Folder Structure

ZOMATO-RESTAURANT-RATING-PREDICTION-AND-RECOMMENDATION-SYSTEM/
│
├── app.py                        # Main Streamlit app  
├── recommendation.py              # Recommendation system logic  
├── data/                          # Dataset folder (ignored in Git)  
├── outputs/                       # EDA charts, saved models, etc.  
├── fonts/                         # Fonts for PDF reports  
├── requirements.txt               # Dependencies  
├── .gitignore  
└── README.md  





