import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans

df = pd.read_csv("D:\OPERATION GROWTH\PERSONALITY SYSTEM\IPIP-FFM-data-8Nov2018\data-final.csv",
                 delimiter="\t")

print(df)
print('\n')

columns = df.columns
for column in columns:
    print(column)

print('\n')

X = df[df.columns[0:50]]
pd.set_option("display.max_columns", None)

print(X)
print('\n')

X = X.fillna(0)
kmeans = MiniBatchKMeans(n_clusters=10, random_state=10, batch_size=100, max_iter=100).fit(X)
print(len(kmeans.cluster_centers_))
print('\n')

one = kmeans.cluster_centers_[0]
two = kmeans.cluster_centers_[1]
three = kmeans.cluster_centers_[2]
four =kmeans.cluster_centers_[3]
five = kmeans.cluster_centers_[4]
six = kmeans.cluster_centers_[5]
seven = kmeans.cluster_centers_[6]
eight = kmeans.cluster_centers_[7]
nine = kmeans.cluster_centers_[8]
ten = kmeans.cluster_centers_[9]

print(one)