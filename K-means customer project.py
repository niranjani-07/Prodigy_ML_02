import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 🔹 Step 1: Load or Create Sample Data
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'AnnualIncome': [15, 16, 17, 18, 19, 70, 71, 72, 73, 74],
    'SpendingScore': [39, 81, 6, 77, 40, 10, 99, 95, 12, 88]
}
df = pd.DataFrame(data)

# 🔹 Step 2: Preprocessing
X = df[['AnnualIncome', 'SpendingScore']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Step 3: Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=0)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("Grouped Customer Data:\n")
print(df)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='AnnualIncome',
    y='SpendingScore',
    hue='Cluster',
    data=df,
    palette='Set2',
    s=100
)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='AnnualIncome', y='SpendingScore', hue='Cluster', palette='Set2', s=100)
plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income (in thousands)')
plt.ylabel('Spending Score (1–100)')
plt.grid(True)
plt.show()