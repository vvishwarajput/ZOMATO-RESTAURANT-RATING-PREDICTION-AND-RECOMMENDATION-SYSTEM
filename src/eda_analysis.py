import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#read the cleaned data

df = pd.read_csv('data/clean_zomato.csv')

print("shape of the dataset : ",df.shape)
print("\n\n Columns : ",df.columns.to_list())
print("\n\nmissing values : \n",df.isna().sum())

#distribution of ratings

plt.figure(figsize=(6,4))
sns.histplot(df['rate'],bins=20,kde=True)
plt.title("Distribution of Restaurant ratings")
plt.xlabel("Ratings")
plt.ylabel("Count")
plt.savefig("outputs/eda/rating_distribution.png")
plt.close()

#top 10 cuisines

plt.figure(figsize=(8,4))
print("\n\nPrimary Cuisine Count : \n\n")
print(df['Primary Cuisine'].value_counts().head(10))
df['Primary Cuisine'].value_counts().head(10).plot(kind='bar',color='skyblue')
plt.title("Top 10 most popular cuisines")
plt.ylabel("Number of restaurants")
plt.xlabel("Cuisine")
plt.tight_layout()
plt.savefig('outputs/eda/top_10_cuisines.png')
plt.show()
plt.close()

# average rating by location


print("\n\n Average rating by location :\n")

print(df[df['location']=='Yeshwantpur']['rate'].mean())

avg_rating = df.groupby('location')['rate'].mean().sort_values(ascending=False).head(10)
print("\n\n",avg_rating)

print("\n avg rating index : \n",avg_rating.index)
print("\n avg rating values : \n",avg_rating.values)
plt.figure(figsize=(8,4))
sns.barplot(x=avg_rating.values,y=avg_rating.index,palette='viridis')
plt.title("Top 10 locations by average rating")
plt.xlabel("Average rating")
plt.ylabel("location")
plt.tight_layout()
plt.savefig('outputs/eda/top_locations_by_rating.png')
plt.close()

# cost vs rating

plt.figure(figsize=(6,4))
sns.scatterplot(data=df,x='approx_cost(for two people)',y='rate',alpha=0.5)
plt.title("Cost vs Rating")
plt.xlabel("Cost for two people")
plt.ylabel("Rating")
plt.tight_layout()
plt.savefig("outputs/eda/cost_vs_rating.png")
plt.close()