import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score,mean_absolute_error
import joblib

#load data

df = pd.read_csv('data/clean_zomato.csv')
print("shape : ",df.shape)

print("\n\ndf.isna().sum():- \n")
print(df.isna().sum())

print("\n\n dropping rows with NaN values : \n")
df.dropna(inplace=True)

print("\n\nCleaned data : \n")
print(df.head(20)[['online_order','book_table','approx_cost(for two people)','votes','Primary Cuisine','location encoded','location']])

#select features(X) and target(y)

X = df[['online_order','book_table','approx_cost(for two people)','votes','Primary Cuisine','location encoded']]

Y = df['rate']

print("Missing values per column :\n")
print(X.isna().sum())

# handle categorical features using one-hot encoding

categorical_cols = ['Primary Cuisine']

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(handle_unknown='ignore'),categorical_cols)],remainder='passthrough')

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# function to train and evaluate a model

def evaluate_model(model,name):
    pipe = Pipeline(steps=[('transform',ct),('model',model)])
    pipe.fit(x_train,y_train)
    y_pred = pipe.predict(x_test)
    r2 = r2_score(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    print(f"{name} → R²: {r2:.3f}, MAE: {mae:.3f}")
    return pipe,r2

#train multiple models and pick the best model

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree" : DecisionTreeRegressor(random_state=42),
    "Random Forest" : RandomForestRegressor(n_estimators=200,random_state=42),
    "XGBoost" : XGBRegressor(n_estimators=300,learning_rate=0.05,random_state=42)
}

best_model,best_score =  None,-999

for name,model in models.items():
    pipe,score = evaluate_model(model,name)

    if score > best_score:
        best_model,best_score = pipe,score
        best_name = name


#save the best model (pipeline)
#saving the trained model with joblib.dump()

joblib.dump(best_model, f'outputs/models/best_model_{best_name.replace(" ", "_")}.joblib')

print(f"\n✅ Best model saved: {best_name} (R² = {best_score:.3f})")
     