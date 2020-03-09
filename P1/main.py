import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from scipy.spatial.distance import pdist, squareform, cdist
import math
dataset_name = "rand"

def count_each_cluster(df):
    result = np.zeros(k)
    for row in df:
        result[int(row[closest_index])]+=1

    return result


def cost_function(df, centroids):
    centroids = update_centroids(df)
    df = calculate_distance_closest(df, centroids)

def calculate_av_distance_numpy(df):
    return np.sum(df[:,distance_closest_index])/df.shape[0]


def get_neightbour(neightbour_list):
    neighbour = neightbour_list[0]
    neightbour_list = neightbour_list[1:] + [neighbour]
    return neightbour_list, neighbour

def same_cluster(df, p1, p2):
    return get_own_cluster(df, p1) == get_own_cluster(df, p2)

def get_own_cluster(df, index):
    return df['closest'].iloc[index,]

def calculate_distance_closest(df, centr):
    for i in range(len(df.index)):
        # df.at[i,'distance_closest'] = np.linalg.norm(df.loc[i,col_names] - centr[int(df.at[i, 'closest'])])
        # df.at[i,'distance_closest'] = np.sqrt(((df.loc[i,col_names]-centr[int(df.at[i, 'closest'])])**2).sum())
        df.at[i, 'distance_closest'] = math.sqrt(sum([(a - b) ** 2 for a, b in zip(df.loc[i,col_names], centr[int(df.at[i, 'closest'])])]))
        # print(np.linalg.norm(df.loc[i,col_names] - centr[int(df.at[i, 'closest'])]))
        # print(np.sqrt(((df.loc[i,col_names]-centr[int(df.at[i, 'closest'])])**2).sum()))
        # print(math.sqrt(sum([(a - b) ** 2 for a, b in zip(df.loc[i,col_names], centr[int(df.at[i, 'closest'])])])))

    # s =
    # df['distance_closest'] = df.apply(lambda row: np.linalg.norm(row[col_names] - centr[row['closest']]), axis=1)
    return df

def calculate_distance_closest_numpy(df, centr):
    for row in df:
        row[distance_closest_index] = math.sqrt(sum([(a - b) ** 2 for a, b in zip(row[0:closest_index], centr[int(row[closest_index])])]))

    return df

def row_infeasibility(df, row_index):
    infeasibility_points = 0
    # print(restrictions.shape)
    r = restrictions.iloc[row_index]
    # print(len(r), "yyyy")

    for c in range(r.size):
        if c != row_index:
            if r[c] == 1:
                if not same_cluster(df, row_index, c):
                    infeasibility_points = infeasibility_points + 1
            elif r[c] == -1:
                if same_cluster(df, row_index, c):
                    infeasibility_points = infeasibility_points +1

    return infeasibility_points

def row_infeasibility_numpy(df, row_index):
    infeasibility_points = 0
    r = restrictions.iloc[row_index]
    for c in range(r.size):
        if c != row_index:
            if r[c] == 1:
                if not (df[c][closest_index] == df[row_index][closest_index]):
                    infeasibility_points = infeasibility_points + 1
            elif r[c] == -1:
                if df[c][closest_index] == df[row_index][closest_index]:
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

    return infeasibility_count

def infeasibility_numpy(df):
    inf_c = 0
    for i in range(df.shape[0]):
        inf_c = inf_c + row_infeasibility_numpy(df, i)

    return inf_c


# Return a numpy matrix with new centroids given the clusters
def update_centroids(df):
    act = df[col_names + ['closest']]
    return act.groupby('closest').mean().to_numpy()

# Return a numpy matrix with new centroids given the clusters
def update_centroids_numpy(df):
    sum = np.zeros((k, closest_index))
    count = np.zeros(k)
    for row in df:
        sum[int(row[closest_index])] += row[0:closest_index]
        count[int(row[closest_index])]+=1

    for i in range(k):
        sum[i] = sum[i]/count[i]
    return sum

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
    #data[['closest'] + ['c0']].groupby('closest').count()

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
    clusters_assigned = (data[['closest']+['c0']].groupby('closest').count().index)
    # print(clusters_assigned)
    for i in range(k):
        if not(clusters_assigned.any == i):
            df.at[i, 'closest'] = i

    return df

def regular_assignation(df, centr):
    global k

    count_clusters = (np.asarray(data[['closest'] + ['c0']].groupby('closest').count()))
    for i in range(len(df.index)):
        own = int(df.at[i,'closest'])
        if(count_clusters[own] == 1):
            continue
        count_clusters[own]-=1

        for cluster_id in range(len(centr)):
            df.at[i,'closest'] = cluster_id
            df.at[i,'infeasility_cluster_{}'.format(cluster_id)] = row_infeasibility(df, i)

        feas_points = ['infeasility_cluster_{}'.format(i) for i in range(len(centr))]
        df.at[i, 'closest'] = df.loc[i, feas_points].idxmin().lstrip('infeasility_cluster_')
        own = int(df.at[i, 'closest'])
        count_clusters[own] += 1
        for cluster_id in range(len(centr)):
            if cluster_id != own and df.at[i,'infeasility_cluster_{}'.format(own)] == df.at[i, 'infeasility_cluster_{}'.format(cluster_id)]:
                d1 = np.linalg.norm(df.loc[i, col_names]-centr[own])
                d2 = np.linalg.norm(df.loc[i, col_names]-centr[cluster_id])
                if(d1 > d2):
                    count_clusters[int(df.at[i, 'closest'])]-=1
                    df.at[i, 'closest'] = cluster_id
                    own = int(cluster_id)
                    count_clusters[own]+=1

        # df['closest'] = df['closest'].map(lambda x: int(x.lstrip('infeasility_cluster_')))
        # print(df.loc[i, feas_points].min())

    # print(count_clusters)
    # print(count_clusters)

    return df


########################################################################################################################
# Dataset selection. It establish number of clusters and read data and restrictions.
if dataset_name == "iris":
    k = 3
    data_path = "./iris_set.dat"
    restrictions_path = "./iris_set_const_20.const"
elif dataset_name == "ecoli":
    k = 8
    data_path = "./ecoli_set.dat"
    restrictions_path = "./ecoli_set_const_20.const"
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
seed = 200
random.seed(seed)
np.random.seed(seed)
idx = np.random.permutation(data.index)
data = data.reindex(idx)
restrictions = restrictions.reindex(idx)
# data = shuffle(data)


D = squareform(pdist(data))
max_distance, [I_row, I_col] = np.nanmax(D), np.unravel_index( np.argmax(D), D.shape )
n_restrictions = ((len(restrictions.index)**2)-(restrictions.isin([0]).sum().sum()))/2
lambda_value = max_distance/n_restrictions
# print(lambda_value)

# Local
####################################################################
# Generate neighbourhood
possible_changes = []
for i in range(len(data.index)):
    for w in range(k):
        possible_changes.append((i, w))
np.random.shuffle(possible_changes)

# Generate initial solution
data['closest'] = np.random.randint(0,k, data.shape[0])

# Count how many points in each cluster
cluster_count = np.array(data[['c0']+['closest']].groupby('closest').count())

# Exit if initial solution doesn't have at least a point in each cluster
if len(cluster_count) != k:
    exit(1)

n_iterations = 0

# Calculate initial infeasibility
total_infeasibility = infeasibility(data)

# Create necessary columns
data['distance_closest']=np.nan
centroids = update_centroids(data)
calculate_distance_closest(data, centroids)

# Calculate initial variables
av_dist = data[['closest']+['distance_closest']].groupby('closest').mean().mean().iloc[0]
objective_value = av_dist + lambda_value * total_infeasibility
possible_changes, neigh = get_neightbour(possible_changes)

# Calculate indexes for numpy array
closest_index = int(len(col_names))
distance_closest_index = closest_index+1

# Transform it into a numpy array
pandas_df = data
data = np.asarray(data)

# print(infeasibility_numpy(data)/2)
# This variable will help us print values
limit = 0
while n_iterations < 100000:
    # n_iterations += 1

    # We get the first neightbour
    possible_changes, first_neigh = get_neightbour(possible_changes)
    first_neigh = neigh

    # Save old values
    old_objective_value = objective_value
    old_infeasibility = total_infeasibility
    old_av_dist = av_dist
    old_cluster = data[neigh[0]][closest_index]
    # print("NEW: ", total_infeasibility, "      real: ", infeasibility_numpy(data))
    print("NEW")
    while old_objective_value <= objective_value and n_iterations < 100000:
        n_iterations += 1

        # Print values in intervals to check if everything is going all right
        if n_iterations > limit:
            print("Iteration:    ", n_iterations, "    Av. Dist: ", av_dist, "    Inf: ", total_infeasibility, "    Obj:", objective_value)
            limit += 1000

        # Save the original cluster from the point we are going to change
        old_cluster = int(data[neigh[0]][closest_index])
        p_index = neigh[0]
        new_cluster = neigh[1]

        # Check if we don't empty a cluster nor make an useless change
        if cluster_count[old_cluster] == 1 or old_cluster == new_cluster:
            possible_changes, neigh = get_neightbour(possible_changes)
            continue

        # print("In: ", total_infeasibility)
        # Subtract the infeasibility and add the new one to get new infeasibility
        print("ANT: ", total_infeasibility, "      real: ", infeasibility_numpy(data), "    calc: ",row_infeasibility_numpy(data, p_index))

        total_infeasibility = total_infeasibility - row_infeasibility_numpy(data, p_index)*2
        data[p_index][closest_index] = new_cluster
        total_infeasibility = total_infeasibility + row_infeasibility_numpy(data, p_index)*2
        print("NEW: ", total_infeasibility, "      real: ", infeasibility_numpy(data), "    calc: ",row_infeasibility_numpy(data, p_index))

        # print("Out: ", total_infeasibility)

        # Calculate new average
        centroids = update_centroids_numpy(data)
        calculate_distance_closest_numpy(data, centroids)

        av_dist = calculate_av_distance_numpy(data)

        # Calculate new objective value
        objective_value = av_dist + lambda_value * total_infeasibility

        # if (n_iterations > 50):
        #     print("Iteration:    ", n_iterations)
        #     print("The inf is: ", total_infeasibility, "    and av.dist:   ", av_dist)
        #     print("Old, ", old_objective_value, "       new   ", objective_value)

        # Restore values
        if old_objective_value <= objective_value:
            data[p_index][closest_index] = old_cluster
            total_infeasibility = old_infeasibility

        # print((data[['closest'] + ['c0']].groupby('closest').count()))

        possible_changes, neigh = get_neightbour(possible_changes)

        if neigh == first_neigh:
            print("No more neighbours")
            break

print("Total inf: ", total_infeasibility)
print(count_each_cluster(data))
print("Aver ", calculate_av_distance_numpy(data))
print("Obj ",objective_value)

pandas_df['c0'] = data[:,0]
pandas_df['c1'] = data[:,1]
pandas_df['closest'] = data[:,2]
pandas_df['distance_closest'] = data[:,3]
print("Pandas inf: ", infeasibility(pandas_df))
print((pandas_df[['closest']+['c0']].groupby('closest').count()))
print("Pandas av dist: ", pandas_df[['closest'] + ['distance_closest']].groupby('closest').mean().mean().iloc[0])


# while n_iterations < 100000:
#     n_iterations += 1
#     possible_changes, first_neigh = get_neightbour(possible_changes)
#     first_neigh = neigh
#     old_objective_value = objective_value
#
#     # Save old values in case we need to restore them
#     old_infeasibility = total_infeasibility
#     old_av_dist = av_dist
#     old_cluster = get_own_cluster(data, neigh[0])
#     while old_objective_value <= objective_value:
#         # print((data[['closest'] + ['c0']].groupby('closest').count()))
#         ## old_cluster = get_own_cluster(data, neigh[0])
#         old_cluster = data[neigh[0]][closest_index]
#
#         p_index = neigh[0]
#         new_cluster = neigh[1]
#
#         n_iterations += 1
#
#
#         if cluster_count[old_cluster] == 1 or old_cluster == new_cluster:
#             possible_changes, neigh = get_neightbour(possible_changes)
#             continue
#
#
#         # Subtract the infeasibility and add the new one to get new infeasibility
#         total_infeasibility = total_infeasibility - row_infeasibility(data, p_index)*2
#         data.at[p_index, 'closest'] = new_cluster
#         total_infeasibility = total_infeasibility + row_infeasibility(data, p_index)*2
#
#         # Calculate new average
#         centroids = update_centroids(data)
#         calculate_distance_closest(data, centroids)
#         av_dist = data[['closest'] + ['distance_closest']].groupby('closest').mean().mean().iloc[0]
#
#         # Calculate new objective value
#         objective_value = av_dist + lambda_value * (total_infeasibility/2.0)
#
#         # if (n_iterations > 50):
#         #     print("Iteration:    ", n_iterations)
#         #     print("The inf is: ", total_infeasibility, "    and av.dist:   ", av_dist)
#         #     print((data[['closest'] + ['c0']].groupby('closest').count()))
#         #     print("Old, ", old_objective_value, "       new   ", objective_value)
#
#         # Restore values
#         if old_objective_value <= objective_value:
#             data.at[p_index, 'closest'] = old_cluster
#             total_infeasibility = old_infeasibility
#
#         possible_changes, neigh = get_neightbour(possible_changes)
#
#         if neigh == first_neigh:
#             break

#     if neigh == first_neigh:
#         break


###################################################################
# GREEDY
###############################################################3
# f1 = plt.figure(1)
# plt.xlim(0, 10)
# plt.ylim(0, 10)
# plt.gca().set_aspect('equal')
# # Generate random centroids
# centroids = []
# for i in range(k):
#     centroids.append(random.uniform(minim, maxim)/1.5)
#
# centroids = np.array(centroids)
#
# data['closest']=np.nan
# data['distance_closest']=np.nan
# print(data.head(5))
# data = first_assignation(data, centroids)
# print("First infeasibility  :", infeasibility(data))
# print((data[['closest']+['c0']].groupby('closest').count()))
#
# data = calculate_distance_closest(data, centroids)
# # data2 = data
#
# print((data[['closest']+['c0']].groupby('closest').count()))
#
# # print(data[['closest']+['distance_closest']].groupby('closest').mean().mean())
# # print(infeasibility(data))
# #
#
# # print(data.head(5))
# # print(data.dtypes)
#
# # print(restrictions.iloc[6].shape)
#
# x = 0
# old_inf = -1
# new_inf = infeasibility(data)
# data_old = data
#
# while (old_inf != new_inf):
#     print(data[['closest']+['distance_closest']].groupby('closest').mean().mean())
#     print("Inf:  ", new_inf, "           It: ", x)
#     data_old = data
#     centroids = update_centroids(data)
#     data = regular_assignation(data, centroids)
#     data = calculate_distance_closest(data, centroids)
#     x = x+1
#     old_inf = new_inf
#     new_inf = infeasibility(data)
#
# # print(data[['closest']+['distance_closest']].groupby('closest').mean().mean())
# # print(infeasibility(data))
#
# if(data_old.equals(data)):
#     print("They are equal")
# else:
#     print("They are not equal")
#
# print((data[['closest']+['c0']].groupby('closest').count()))

#############################################3

# crear el vecindario de forma parejas (i, k), que pasaria si a punto i le asigno k
# si i ya esta en cluster k, voy al siguiente