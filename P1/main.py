import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time

dataset_name = "rand"


def is_infeasible(pair):
    if restrictions[pair[0]][pair[1]] == -1:
        return True
    else:
        return False


def assignation(df, centroids):
    for i in range(len(centroids)):
        # sqrt((x1 - c1)^2 - (x2 - c2)^2)
        df['distance_from_{}'.format(i)] = df[col_names].apply(lambda row : np.linalg.norm(row-centroids[i]), axis=1)

    centroid_distance_cols = ['distance_from_{}'.format(i) for i in range(len(centroids))]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))

    df['color'] = df['closest'].apply(lambda x: colmap[x])
    return df


if dataset_name == "iris":
    k = 3
    data_path = "./iris_set.dat"
    restrictions_path = "./iris_set_const_10.const"
elif dataset_name == "ecoli":
    k = 8
    data_path = "./ecoli_set.dat"
    restrictions_path = "./ecoli_set_const_10.const"
else:
    k = 3
    data_path = "./rand_set.dat"
    restrictions_path = "./rand_set_const_10.const"

colmap = ['r', 'g','b']
data = pd.read_csv(data_path, header=None)
restrictions = pd.read_csv(restrictions_path, header=None)
# print(restrictions.head())

col_names = []
for i in range(len(data.columns)):
    col_names.append("c"+str(i))
data.columns = col_names
print(data.head())
#####################################################################################################

minim = data.min()
# print(minim)
maxim = data.max()
# print(maxim)
dec_precision = 5

seed = int(round(time.time()))
random.seed(100)

n_carac = len(minim)
# centroid = np.zeros((k, n_carac))
centroids = []
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.gca().set_aspect('equal')
for i in range(k):
    centroids.append(random.uniform(minim, maxim))
    plt.scatter(*centroids[i], color=colmap[i],marker=(5, 1),edgecolors='b')


# print(random.uniform(minim, maxim))

centroids = np.array(centroids)

data = assignation(data, centroids)
print(data['distance_from_2'])

print(data.head())
plt.scatter(data['c0'], data['c1'], color=data["color"])

act = data[col_names+['closest']]
print(act.groupby('closest').mean())
# plt.show()