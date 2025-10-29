import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_clean(path='data/zomato.csv'):

    # step 1:- load dataset
    df = pd.read_csv(path,encoding='utf-8')
    print(df.shape)
 
    # step 2:- clean the rate column

    df['rate'] = df['rate'].astype(str)

    df['rate'] = df['rate'].replace(['NEW','-','nan'],np.nan)

    df['rate'] = df['rate'].astype('str')

    df['rate'] = df['rate'].apply(lambda x : x.split('/')[0] if '/' in x else x)

    df['rate'] = pd.to_numeric(df['rate'],errors='coerce')

    # print(df[df['rate'].isna()]['rate'])

    # print(type(df['rate'][72]))
    # print(type(df['rate'][75]))


    #step 3:- clean the approx cost(for two) column

    print(df['approx_cost(for two people)'])

    if 'approx_cost(for two people)' in df.columns:
        df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(str).str.replace(',','').astype(float)

    print("after removing , from approx cost :\n",df['approx_cost(for two people)'])

    #step 4:- extract primary cuisine
    
    print('Cuisines :\n')
    print(df['cuisines'])
    print(df[df['cuisines'].isna()]['cuisines'])

    #cuisines column might have multiple values in single record like (North Indian, Mughlai, Chinese)
    #but we will only take first one (primary cuisine)
    df['Primary Cuisine'] = df['cuisines'].fillna('Unknown').apply(lambda x:x.split(',')[0])
    print(df['Primary Cuisine'])
    print(df['Primary Cuisine'][31400])
    print(df['Primary Cuisine'][438])
    print(df['Primary Cuisine'][1662])

   # step 5:- convert "yes" or "no" column to binary (0 or 1)
    
    print("\n\n online order column : ")
    print(df['online_order'])
    print("\n\n Book Table Column: ")
    print(df['book_table'])
    df['online_order'] = df['online_order'].map({"Yes":1,"No":0})
    df['book_table'] = df['book_table'].map({"Yes":1,"No":0})

    #we can't train ML models on text: so we convert:

    #yes - 1
    #No - 0

    print("\n\n after coverting text to binary : ")
    print("\n\online order : \n",df['online_order'])
    print("\nbook table : \n",df['book_table'])

    #step 6:- keep only important columns

    keep_cols = ['name','location','Primary Cuisine','approx_cost(for two people)','rate','votes','online_order','book_table']

    df = df[keep_cols].dropna(subset=['rate'])

    # We only keep the columns relevant to our analysis and drop all others.

    # dropna(subset=['Rate']) removes rows where the rating is missing —
    # because we can’t train a model on missing targets.

    #step 7:- Label encode the location column

    print("\n\n location column : \n",df['location'])

    le = LabelEncoder()
    df['location encoded'] = le.fit_transform(df['location'])

    print("\n\nencoded location: \n\n",df['location encoded'])

    #This turns city names into numeric values for the ML model.

    #step 8: - save cleaned data

    df.to_csv('data/clean_zomato.csv',index=False)
    print("Cleaned data saved to clean_zomato.csv file")
    print("Final shape : ",df.shape)

    print("\n\n First 5 rows of cleaned data (clean_zomato.csv)")
    print(df.head())

    #now we can use this cleaned dataset clean_zomato.csv for EDA and ML models.



    


    
    



if __name__ == '__main__':
    load_and_clean()