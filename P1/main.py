import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time

dataset_name = "rand"

def same_cluster(df, p1, p2):
    return get_own_cluster(df, p1) == get_own_cluster(df, p2)

def get_own_cluster(df, index):
    return df['closest'].iloc[index,]

def row_infeasibility(df, row_index):
    infeasibility_points = 0
    r = restrictions.iloc[row_index]
    own_group = get_own_cluster(df, row_index)
    for c in range(r.size):
        if r[c] == 1:
            if not same_cluster(df, row_index, c):
                infeasibility_points = infeasibility_points + 1
        elif r[c] == -1:
            if same_cluster(df, row_index, c):
               infeasibility_points = infeasibility_points +1

    return infeasibility_points

def infeasibility(df):
    infeasibility_count = 0
    for i in range(df.shape[0]):
        infeasibility_count += row_infeasibility(df, i)

    return infeasibility_count


# Return a numpy matrix with new centroids given the clusters
def update_centroids(df):
    act = df[col_names + ['closest']]
    return act.groupby('closest').mean().to_numpy()


# Return True if a pair of points CANNOT be linked
def is_infeasible(pair):
    if restrictions[pair[0]][pair[1]] == -1:
        return True
    else:
        return False


#  Para seguir restricciones, calcular closest fila por fila
#  Mirar en la tabla de restricciones todos los anteriores a su columna
#  Si hay 1, asignar al mismo grupo. Si no calcular normal
#  Luego de calcular, revisar si hay algun -1 y ver si lo incumple (tal vez ir acumulando en lista cuando veo 1)

# Given df of points and matrix of centroids, calculate the distance of each point in the df to every centroid.
# Then, it selects the closest one.
def assignation(df, centr):
    global k
    for i in range(len(centr)):
        df['distance_from_{}'.format(i)] = df[col_names].apply(lambda row : np.linalg.norm(row-centr[i]), axis=1)

    centroid_distance_cols = ['distance_from_{}'.format(i) for i in range(len(centr))]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))

    df['color'] = df['closest'].apply(lambda x: colmap[x])

    # Return empty if not enough clusters
    if(df[col_names + ['closest']].groupby('closest').count().shape[0]) < k:
        return

    return df


########################################################################################################################
# Dataset selection. It establish number of clusters and read data and restrictions.
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
# The name of the df's columns will be 'cX' being X the column's number
for i in range(len(data.columns)):
    col_names.append("c"+str(i))
data.columns = col_names
########################################################################################################################

# Get minimum and maximum values for each column
minim = data.min()
maxim = data.max()

# Start random generator
seed = int(round(time.time()))
random.seed(100)


f1 = plt.figure(1)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.gca().set_aspect('equal')
# Generate random centroids
centroids = []
for i in range(k):
    centroids.append(random.uniform(minim, maxim))
    plt.scatter(*centroids[i], color=colmap[i],marker=(5, 1),edgecolors='b')

centroids = np.array(centroids)

data = assignation(data, centroids)
plt.scatter(data['c0'], data['c1'], color=data["color"])

print(restrictions.head(5))
print(infeasibility(data))

# print(data.shape[0])
# for j in range(10):
#     plt.figure(j+2)
#     plt.xlim(0, 10)
#     plt.ylim(0, 10)
#     plt.gca().set_aspect('equal')
#     centroids = update_centroids(data)
#
#
#     data = assignation(data, centroids)
#     plt.scatter(data['c0'], data['c1'], color=data["color"])
#     for i in range(k):
#         plt.scatter(*centroids[i], color=colmap[i],marker=(5, 1),edgecolors='b')
#     print(centroids)
# plt.show()
# print(data[col_names + ['closest']].groupby('closest').count().shape[0])



# crear el vecindario de forma parejas (i, k), que pasaria si a punto i le asigno k
# si i ya esta en cluster k, voy al siguiente