import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from sklearn.utils import shuffle

dataset_name = "rand"

def same_cluster(df, p1, p2):
    return get_own_cluster(df, p1) == get_own_cluster(df, p2)

def get_own_cluster(df, index):
    return df['closest'].iloc[index,]

def calculate_distance_closest(df, centr):
    for i in range(len(df.index)):
        df.at[i,'distance_closest'] = np.linalg.norm(df.loc[i,col_names] - centr[int(df.at[i, 'closest'])])
    # s =
    # df['distance_closest'] = df.apply(lambda row: np.linalg.norm(row[col_names] - centr[row['closest']]), axis=1)
    # print (df.loc[i,col_names])
    return df

def row_infeasibility(df, row_index):
    infeasibility_points = 0
    # print(restrictions.shape)
    r = restrictions.iloc[row_index]
    # print(len(r), "yyyy")

    own_group = get_own_cluster(df, row_index)
    for c in range(r.size):
        if r[c] == 1:
            if not same_cluster(df, row_index, c):
                infeasibility_points = infeasibility_points + 1
        elif r[c] == -1:
            if same_cluster(df, row_index, c):
                infeasibility_points = infeasibility_points +1

    return infeasibility_points

def row_infeasibility_first(df, row_index):
    infeasibility_points = 0
    r = restrictions.iloc[row_index]
    own_group = get_own_cluster(df, row_index)
    for c in range(row_index):
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
        # print("In row ", i, " inf is ", infeasibility_count)

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

def first_assignation(df, centr):
    global k
    for i in range(len(df.index)):
        for cluster_id in range(len(centr)):
            df.at[i,'closest'] = cluster_id
            df.at[i,'infeasility_cluster_{}'.format(cluster_id)] = row_infeasibility_first(df, i)

        feas_points = ['infeasility_cluster_{}'.format(i) for i in range(len(centr))]
        df.at[i, 'closest'] = df.loc[i, feas_points].idxmin().lstrip('infeasility_cluster_')
        own = int(df.at[i, 'closest'])

        for cluster_id in range(len(centr)):
            if cluster_id != own and df.at[i,'infeasility_cluster_{}'.format(own)] == df.at[i, 'infeasility_cluster_{}'.format(cluster_id)]:
                d1 = np.linalg.norm(df.loc[i, col_names]-centr[own])
                d2 = np.linalg.norm(df.loc[i, col_names]-centr[cluster_id])
                if(d1 > d2):
                    df.at[i, 'closest'] = cluster_id
                    own = int(cluster_id)

        # df['closest'] = df['closest'].map(lambda x: int(x.lstrip('infeasility_cluster_')))
        # print(df.loc[i, feas_points].min())


    return df

def regular_assignation(df, centr):
    global k
    for i in range(len(df.index)):
        for cluster_id in range(len(centr)):
            df.at[i,'closest'] = cluster_id
            df.at[i,'infeasility_cluster_{}'.format(cluster_id)] = row_infeasibility(df, i)

        feas_points = ['infeasility_cluster_{}'.format(i) for i in range(len(centr))]
        df.at[i, 'closest'] = df.loc[i, feas_points].idxmin().lstrip('infeasility_cluster_')
        own = int(df.at[i, 'closest'])

        for cluster_id in range(len(centr)):
            if cluster_id != own and df.at[i,'infeasility_cluster_{}'.format(own)] == df.at[i, 'infeasility_cluster_{}'.format(cluster_id)]:
                d1 = np.linalg.norm(df.loc[i, col_names]-centr[own])
                d2 = np.linalg.norm(df.loc[i, col_names]-centr[cluster_id])
                if(d1 > d2):
                    df.at[i, 'closest'] = cluster_id
                    own = int(cluster_id)

        # df['closest'] = df['closest'].map(lambda x: int(x.lstrip('infeasility_cluster_')))
        # print(df.loc[i, feas_points].min())


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
    restrictions_path = "./rand_set_const_20.const"

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
pd.set_option('display.max_columns', None)

# Get minimum and maximum values for each column
minim = data.min()
maxim = data.max()

# Start random generator
seed = int(round(time.time()))
random.seed(seed)
idx = np.random.permutation(data.index)
data = data.reindex(idx)
restrictions = restrictions.reindex(idx)
# data = shuffle(data)


f1 = plt.figure(1)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.gca().set_aspect('equal')
# Generate random centroids
centroids = []
for i in range(k):
    centroids.append(random.uniform(minim, maxim))

centroids = np.array(centroids)

data['closest']=np.nan
data['distance_closest']=np.nan

data = first_assignation(data, centroids)
data = calculate_distance_closest(data, centroids)
# data2 = data


# print(data[['closest']+['distance_closest']].groupby('closest').mean().mean())
# print(infeasibility(data))
#

# print(data.head(5))
# print(data.dtypes)

# print(restrictions.iloc[6].shape)

x = 0
old_inf = 0
new_inf = infeasibility(data)
data_old = data

while (old_inf != new_inf):
    print(data[['closest']+['distance_closest']].groupby('closest').mean().mean())
    print("Inf:  ", new_inf, "           It: ", x)
    data_old = data
    centroids = update_centroids(data)
    data = regular_assignation(data, centroids)
    data = calculate_distance_closest(data, centroids)
    x = x+1
    old_inf = new_inf
    new_inf = infeasibility(data)

# print(data[['closest']+['distance_closest']].groupby('closest').mean().mean())
# print(infeasibility(data))

if(data_old.equals(data)):
    print("They are equal")
else:
    print("They are not equal")
#
# plt.figure(1)
# plt.xlim(0, 10)
# plt.ylim(0, 10)
# plt.gca().set_aspect('equal')
# data2['color'] = data2['closest'].apply(lambda x: colmap[int(x)])
#
# plt.scatter(data2['c0'], data2['c1'], color=data2["color"])
# for i in range(k):
#     plt.scatter(*centroids[i], color=colmap[i],marker=(5, 1),edgecolors='k')
#
#
#
# plt.figure(2)
# data['color'] = data['closest'].apply(lambda x: colmap[int(x)])
#
# plt.xlim(0, 10)
# plt.ylim(0, 10)
# plt.gca().set_aspect('equal')
# plt.scatter(data['c0'], data['c1'], color=data["color"])
#
# centroids = update_centroids(data)
# for i in range(k):
#     plt.scatter(*centroids[i], color=colmap[i],marker=(5, 1),edgecolors='k')
#
#
# # plt.scatter(data['c0'], data['c1'], color=data["color"])
#
# # print(data[['closest']+['infeasility_cluster_0']+['infeasility_cluster_1']+['infeasility_cluster_2']])
#
# # print(data[['closest']+['infeasility_cluster_0']+['infeasility_cluster_1']+['infeasility_cluster_2']])
# # print(restrictions.head(5))
# # print(infeasibility(data))
# # print(get_own_cluster(data, 0))
#
# # print(data.shape[0])
# # for j in range(10):
# #     plt.figure(j+2)
# #     plt.xlim(0, 10)
# #     plt.ylim(0, 10)
# #     plt.gca().set_aspect('equal')
# #     centroids = update_centroids(data)
# #
# #
# #     data = assignation(data, centroids)
# #     plt.scatter(data['c0'], data['c1'], color=data["color"])
# #     for i in range(k):
# #         plt.scatter(*centroids[i], color=colmap[i],marker=(5, 1),edgecolors='b')
# #     print(centroids)
# plt.show()
# # print(data[col_names + ['closest']].groupby('closest').count().shape[0])
#
#

# crear el vecindario de forma parejas (i, k), que pasaria si a punto i le asigno k
# si i ya esta en cluster k, voy al siguiente