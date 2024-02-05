import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt

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
four = kmeans.cluster_centers_[3]
five = kmeans.cluster_centers_[4]
six = kmeans.cluster_centers_[5]
seven = kmeans.cluster_centers_[6]
eight = kmeans.cluster_centers_[7]
nine = kmeans.cluster_centers_[8]
ten = kmeans.cluster_centers_[9]

print(one)
print('\n')

all_types = {'one': one, 'two': two, 'three': three, 'four': four, 'five': five, 'six': six, 'seven': seven,
             'eight': eight,
             'nine': nine, 'ten': ten}

all_types_scores = {}

for name, personality_type in all_types.items():
    personality_trait = {}

    personality_trait['extroversion_score'] = personality_type[0] - personality_type[1] + personality_type[2] - \
                                              personality_type[3] + personality_type[4] - personality_type[5] + \
                                              personality_type[6] - personality_type[7] + personality_type[8] - \
                                              personality_type[9]
    personality_trait['neuroticism_score'] = personality_type[0] - personality_type[1] + personality_type[2] - \
                                             personality_type[3] + personality_type[4] + personality_type[5] + \
                                             personality_type[6] + personality_type[7] + personality_type[8] + \
                                             personality_type[9]
    personality_trait['agreeableness_score'] = -personality_type[0] + personality_type[1] - personality_type[2] + \
                                               personality_type[3] - personality_type[4] - personality_type[5] + \
                                               personality_type[6] - personality_type[7] + personality_type[8] + \
                                               personality_type[9]
    personality_trait['conscientiousness_score'] = personality_type[0] - personality_type[1] + personality_type[2] - \
                                                   personality_type[3] + personality_type[4] - personality_type[5] + \
                                                   personality_type[6] - personality_type[7] + personality_type[8] + \
                                                   personality_type[9]
    personality_trait['openness_score'] = personality_type[0] - personality_type[1] + personality_type[2] - \
                                          personality_type[3] + personality_type[4] - personality_type[5] + \
                                          personality_type[6] + personality_type[7] + personality_type[8] + \
                                          personality_type[9]

    all_types_scores[name] = personality_trait

print(all_types_scores)
print('\n')

all_extroversion = []
all_neuroticism = []
all_agreeableness = []
all_conscientiousness = []
all_openness = []

for personality_type, personality_trait in all_types_scores.items():
    all_extroversion.append(personality_trait['extroversion_score'])
    all_neuroticism.append(personality_trait['neuroticism_score'])
    all_agreeableness.append(personality_trait['agreeableness_score'])
    all_conscientiousness.append(personality_trait['conscientiousness_score'])
    all_openness.append(personality_trait['openness_score'])

all_extroversion_normalized = (all_extroversion-min(all_extroversion))/(max(all_extroversion)-min(all_extroversion))
all_neuroticism_normalized = (all_neuroticism-min(all_neuroticism))/(max(all_neuroticism)-min(all_neuroticism))
all_agreeableness_normalized = (all_agreeableness-min(all_agreeableness))/(max(all_agreeableness)-min(all_agreeableness))
all_conscientiousness_normalized = (all_conscientiousness-min(all_conscientiousness))/(max(all_conscientiousness)-min(all_conscientiousness))
all_openness_normalized = (all_openness-min(all_openness))/(max(all_openness)-min(all_openness))

print(all_extroversion_normalized)
print('\n')

counter = 0

normalized_all_types_scores = {}

for personality_type, personality_trait in all_types_scores.items():
    normalized_personality_trait = {}
    normalized_personality_trait['extroversion_score'] = all_extroversion_normalized[counter]
    normalized_personality_trait['neuroticism_score'] = all_neuroticism_normalized[counter]
    normalized_personality_trait['agreeableness_score'] = all_agreeableness_normalized[counter]
    normalized_personality_trait['conscientiousness_score'] = all_conscientiousness_normalized[counter]
    normalized_personality_trait['openness_score'] = all_openness_normalized[counter]

    normalized_all_types_scores[personality_type] = normalized_personality_trait

    counter += 1

print(normalized_all_types_scores)
print('\n')

plt.figure(figsize=(15,5))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['one'].keys()), normalized_all_types_scores['one'].values(), color='g')
plt.show()

plt.figure(figsize=(15,5))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['two'].keys()), normalized_all_types_scores['two'].values(), color='g')
plt.show()

plt.figure(figsize=(15,5))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['three'].keys()), normalized_all_types_scores['three'].values(), color='g')
plt.show()

plt.figure(figsize=(15,5))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['four'].keys()), normalized_all_types_scores['four'].values(), color='g')
plt.show()

plt.figure(figsize=(15,5))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['five'].keys()), normalized_all_types_scores['five'].values(), color='g')
plt.show()

plt.figure(figsize=(15,5))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['six'].keys()), normalized_all_types_scores['six'].values(), color='g')
plt.show()

plt.figure(figsize=(15,5))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['seven'].keys()), normalized_all_types_scores['seven'].values(), color='g')
plt.show()

plt.figure(figsize=(15,5))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['eight'].keys()), normalized_all_types_scores['eight'].values(), color='g')
plt.show()

plt.figure(figsize=(15,5))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['nine'].keys()), normalized_all_types_scores['nine'].values(), color='g')
plt.show()

plt.figure(figsize=(15,5))
plt.ylim(0, 1)
plt.bar(list(normalized_all_types_scores['ten'].keys()), normalized_all_types_scores['ten'].values(), color='g')
plt.show()