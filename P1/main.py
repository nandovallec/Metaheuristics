import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from scipy.spatial.distance import pdist, squareform, cdist
import math
import sys
# dataset_name = str(sys.argv[1])
dataset_name = "iris"

def count_each_cluster(df):
    result = np.zeros(k)
    for row in df:
        result[int(row[closest_index])]+=1
    return result


def calculate_av_distance_numpy(df):
    return np.sum(df[:,distance_closest_index])/df.shape[0]


def get_neightbour(neightbour_list):
    neighbour = neightbour_list[0]
    neightbour_list = neightbour_list[1:] + [neighbour]
    return neightbour_list, neighbour

def calculate_distance_closest_numpy(df, centr):
    for row in df:
        row[distance_closest_index] = math.sqrt(sum([(a - b) ** 2 for a, b in zip(row[0:closest_index], centr[int(row[closest_index])])]))

    return df

def row_infeasibility_numpy(df, row_index):
    infeasibility_points = 0
    r = restrictions_numpy[row_index]
    for c in range(r.size):
        if c != row_index:
            if r[c] == 1:
                if not (df[c][closest_index] == df[row_index][closest_index]):
                    infeasibility_points += 1
            elif r[c] == -1:
                if df[c][closest_index] == df[row_index][closest_index]:
                    infeasibility_points += 1

    return infeasibility_points

def infeasibility_numpy(df):
    inf_c = 0
    for i in range(df.shape[0]):
        inf_c = inf_c + row_infeasibility_numpy(df, i)

    return inf_c


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
pd.set_option('display.max_columns', None)

# Get minimum and maximum values for each column
minim = data.min()
maxim = data.max()

# Start random generator
seed = int(round(time.time()))
# seed = 52
# seed = 103
# seed = 123456789
#seed = 502     # SEMILLA MUY BUENA
random.seed(seed)
np.random.seed(seed)
idx = np.random.permutation(data.index)
# data = data.reindex(idx)
# restrictions = restrictions.reindex(idx)
# data = shuffle(data)


D = squareform(pdist(data))
max_distance, [I_row, I_col] = np.nanmax(D), np.unravel_index( np.argmax(D), D.shape )
n_restrictions = ((len(restrictions.index)**2)-(restrictions.isin([0]).sum().sum()))/2
lambda_value = max_distance/n_restrictions
print(n_restrictions,"    ",lambda_value)

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

# Calculate indexes for numpy array
closest_index = int(len(col_names))
distance_closest_index = closest_index+1

# Transform it into a numpy array
pandas_df = data
restrictions_numpy = np.asarray(restrictions)

# Create necessary columns
data['distance_closest']=np.nan
data = np.asarray(data)
centroids = update_centroids_numpy(data)
calculate_distance_closest_numpy(data, centroids)





# This variable will help us print values
limit = 0
n_iterations = 0

# Calculate initial infeasibility
total_infeasibility = infeasibility_numpy(data)/2.0

av_dist = calculate_av_distance_numpy(data)
objective_value = av_dist + lambda_value * total_infeasibility
print("Dist: ", av_dist, "  obj: ", objective_value)
while n_iterations < 100000:
    # n_iterations += 1

    # We get the first neightbour
    possible_changes, neigh = get_neightbour(possible_changes)
    first_neigh = neigh

    # Save old values
    old_objective_value = objective_value
    old_infeasibility = total_infeasibility
    old_av_dist = av_dist
    old_cluster = data[neigh[0]][closest_index]
    # print("NEW: ", total_infeasibility, "      real: ", infeasibility_numpy(data))

    first_iteration = False
    while old_objective_value <= objective_value and n_iterations < 100000:
        n_iterations += 1
        possible_changes, neigh = get_neightbour(possible_changes)
        if(possible_changes.index(first_neigh) < 5):
            print(possible_changes[:5], "     ", possible_changes.index(first_neigh))
        # print(neigh, "      ", first_neigh, "    ", possible_changes.index(first_neigh), "     ", first_iteration, "    ", possible_changes[0])
        if neigh == first_neigh and first_iteration:
            # Break if we already got every possible neighbour
            break

        first_iteration = True
        # Print values in intervals to check if everything is going all right
        if n_iterations > limit:
            print("Iteration:    ", n_iterations, "    Av. Dist: ", av_dist, "    Inf: ", total_infeasibility, "    Obj:", old_objective_value)
            limit += 1000

        # Save the original cluster from the point we are going to change
        old_cluster = int(data[neigh[0]][closest_index])
        p_index = neigh[0]
        new_cluster = neigh[1]


        # print("In: ", total_infeasibility)
        # Subtract the infeasibility and add the new one to get new infeasibility
        # print("ANT: ", total_infeasibility, "      real: ", infeasibility_numpy(data)/2, "    calc: ",row_infeasibility_numpy(data, p_index))

        total_infeasibility = total_infeasibility - row_infeasibility_numpy(data, p_index)
        data[p_index][closest_index] = new_cluster
        total_infeasibility = total_infeasibility + row_infeasibility_numpy(data, p_index)
        # print("NEW: ", total_infeasibility, "      real: ", infeasibility_numpy(data)/2, "    calc: ",row_infeasibility_numpy(data, p_index))

        # print("Out: ", total_infeasibility)

        # Calculate new average
        centroids = update_centroids_numpy(data)
        calculate_distance_closest_numpy(data, centroids)
        av_dist = calculate_av_distance_numpy(data)

        # Calculate new objective value
        objective_value = av_dist + lambda_value * total_infeasibility

        # Restore values
        if old_objective_value <= objective_value:
            data[p_index][closest_index] = old_cluster
            total_infeasibility = old_infeasibility

    if neigh == first_neigh:
        print("No more neighbours")
        break

print("Total inf: ", total_infeasibility)
print("Inf. Calculated: " , infeasibility_numpy(data)/2.0)
print(count_each_cluster(data))
print("Aver ", calculate_av_distance_numpy(data))
print("Obj ",objective_value)

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