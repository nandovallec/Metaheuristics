import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import pdist, squareform, cdist
import math
import sys
##########################
if len(sys.argv) == 7:
    dataset_name = sys.argv[1]
    restr_level = int(sys.argv[2])
    seed_asigned = int(sys.argv[3])
    lambda_var = float(sys.argv[4])
    cauchy = sys.argv[5] == "si"
    alpha = float(sys.argv[6])
elif len(sys.argv) == 1:
    dataset_name = "rand"
    restr_level = 10
    seed_asigned = 123
    lambda_var = 1
    cauchy = True
    alpha = 0.98
else:
    print("Wrong number of arguments.")
    exit(1)
#############################
it = []
ob = []

def printIndexCluster(df):
    d = [[] for i in range(k)]
    for i in range(df.shape[0]):
        d[int(df[i][cluster_index])].append(i)

    for r in range(k):
        print(d[r])


def count_each_cluster(df):
    result = np.zeros(k)
    for row in df:
        result[int(row[cluster_index])] += 1
    return result


def get_neightbour(neightbour_list):
    neighbour = neightbour_list[0]
    neightbour_list = neightbour_list[1:] + [neighbour]
    return neightbour_list, neighbour


def calculate_distance_cluster_numpy(df, centr):
    av = (np.zeros(k, dtype=np.float128))
    av_count = (np.zeros(k))
    for i in range(df.shape[0]):
        cluster = int(df[i][cluster_index])
        distance = np.float128(math.sqrt(sum([(a - b) ** 2 for a, b in zip(df[i][0:cluster_index], centr[cluster])])))
        df[i][distance_cluster_index] = (distance)
        av[cluster] += (distance)
        av_count[cluster] += 1
        # print("Iter: ",i,"        Sum: ", av[0])

    return df, av, av_count

def calculate_distance_cluster_numpy2(df, centr, old, new):
    old_d = 0
    new_d = 0
    for i in range(df.shape[0]):
        cluster = int(df[i][cluster_index])
        if cluster == old:
            distance = np.float128(math.sqrt(sum([(a - b) ** 2 for a, b in zip(df[i][0:cluster_index], centr[cluster])])))
            df[i][distance_cluster_index] = (distance)
            old_d += (distance)
        elif cluster == new:
            distance = np.float128(math.sqrt(sum([(a - b) ** 2 for a, b in zip(df[i][0:cluster_index], centr[cluster])])))
            df[i][distance_cluster_index] = (distance)
            new_d += (distance)
        # print("Iter: ",i,"        Sum: ", av[0])

    return df, old_d, new_d

def update_distance(df, centr, sum_dis, ind, old_clu, new_clu):

    old_dist = df[ind][distance_cluster_index]
    new_dist = np.float128(math.sqrt(sum([(a - b) ** 2 for a, b in zip(df[ind][0:cluster_index], centr[new_clu])])))
    df[ind][distance_cluster_index] = new_dist

    sum_dis[old_clu] -= old_dist
    sum_dis[new_clu] += new_dist

    return df, sum_dis, old_dist

def undo_distance(df, sum_dist, ind, old_cluster, new_cluster, old_dist):
    new_dist = df[ind][distance_cluster_index]

    df[ind][distance_cluster_index] = old_dist
    sum_dist[old_cluster] += old_dist
    sum_dist[new_cluster] -= new_dist

    return df, sum_dist


def row_infeasibility_numpy(df, row_index):
    infeasibility_points = 0
    own = df[row_index][cluster_index]

    for c in pos_ones[row_index]:
        if not (df[c][cluster_index] == own):
            infeasibility_points += 1


    for c in pos_neg_ones[row_index]:
        if df[c][cluster_index] == own:
            infeasibility_points += 1


    return infeasibility_points



def row_infeasibility_first_numpy(df, row_index):
    infeasibility_points = 0
    r = restrictions_numpy[row_index]
    for c in np.nonzero(r == 1)[0]:
        if (not np.isnan(df[c][cluster_index]))and(not (df[c][cluster_index] == df[row_index][cluster_index])):
            infeasibility_points += 1

    for c in np.nonzero(r == -1)[0]:
        if (not np.isnan(df[c][cluster_index]))and(df[c][cluster_index] == df[row_index][cluster_index]):
            infeasibility_points += 1
    return infeasibility_points


def row_infeasibility_partial(df, row_index):
    infeasibility_points = 0
    r = restrictions_numpy[row_index]
    for c in range(row_index + 1, r.size):
        if r[c] == 1:
            if not (df[c][cluster_index] == df[row_index][cluster_index]):
                infeasibility_points += 1
        elif r[c] == -1:
            if df[c][cluster_index] == df[row_index][cluster_index]:
                infeasibility_points += 1

    return infeasibility_points


def infeasibility_numpy(df):
    inf_c = 0
    for i in range(df.shape[0]):
        inf_c = inf_c + row_infeasibility_partial(df, i)

    return inf_c


# Return a numpy matrix with new centroids given the clusters
def update_centroids_numpy(df):
    sum = np.zeros((k, cluster_index))
    count = np.zeros(k)
    for row in df:
        sum[int(row[cluster_index])] += row[0:cluster_index]
        count[int(row[cluster_index])] += 1
    # print(sum)
    for i in range(k):
        sum[i] = sum[i] / count[i]
    return sum

# Return a numpy matrix with new centroids given the clusters
def update_centroids_optimized(df, centr, sum_values, ind, old_cluster, new_cluster, count):
    sum_values[old_cluster] -= df[ind][0:cluster_index]
    sum_values[new_cluster] += df[ind][0:cluster_index]

    centr[old_cluster] = sum_values[old_cluster]/count[old_cluster]
    centr[new_cluster] = sum_values[new_cluster]/count[new_cluster]

    return centr, sum_values


def sum_instances(df):
    sum = np.zeros((k, cluster_index))
    for row in df:
        sum[int(row[cluster_index])] += row[0:cluster_index]

    return sum

########################################################################################################################
# Dataset selection. It establish number of clusters and read data and restrictions.
if dataset_name == "iris":
    k = 3
    data_path = "./iris_set.dat"
    restrictions_path = "./iris_set_const_" + str(restr_level) + ".const"
elif dataset_name == "ecoli":
    k = 8
    data_path = "./ecoli_set.dat"
    restrictions_path = "./ecoli_set_const_" + str(restr_level) + ".const"
elif dataset_name == "rand":
    k = 3
    data_path = "./rand_set.dat"
    restrictions_path = "./rand_set_const_" + str(restr_level) + ".const"

elif dataset_name == "newthyroid":
    k = 3
    data_path = "./newthyroid_set.dat"
    restrictions_path = "./newthyroid_set_const_" + str(restr_level) + ".const"
else:
    print("Wrong dataset name")
    exit(1)

colmap = ['r', 'g', 'b']
data = pd.read_csv(data_path, header=None)
restrictions = pd.read_csv(restrictions_path, header=None)
col_names = []
# The name of the df's columns will be 'cX' being X the column's number
for i in range(len(data.columns)):
    col_names.append("c" + str(i))
data.columns = col_names
n_characteristics = len(col_names)
n_instances = data.shape[0]

########################################################################################################################
pd.set_option('display.max_columns', None)

# Get minimum and maximum values for each column
minim = data.min()
maxim = data.max()

# Start random generator
# seed = int(round(time.time()))
seed = seed_asigned
random.seed(seed)
np.random.seed(seed)
idx = np.random.permutation(data.index)

# Calculate indexes for numpy array
cluster_index = int(len(col_names))
distance_cluster_index = cluster_index + 1

# Start timing
start_time = time.perf_counter()

D = squareform(pdist(data))
max_distance, [I_row, I_col] = np.nanmax(D), np.unravel_index(np.argmax(D), D.shape)
n_restrictions = (((len(restrictions.index) ** 2) - (restrictions.isin([0]).sum().sum())) / 2)-data.shape[0]
# print(max_distance)
lambda_value = (max_distance / n_restrictions) * lambda_var
mu_phi = 0.3
final_temp = 0.01
max_generated = 10*n_instances
max_accepted = 0.1*max_generated

# Generate neighbourhood
possible_changes = []
for i in range(len(data.index)):
    for w in range(k):
        possible_changes.append((i, w))
np.random.shuffle(possible_changes)

# Generate initial solution
data['cluster'] = np.random.randint(0, k, data.shape[0])
# data['cluster'] = pd.Series([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
# ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
# ,0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
# ,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
# ,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
# ,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
# ,1,1,1,1,1,1])
# Count how many points in each cluster
cluster_count = np.array(data[['c0'] + ['cluster']].groupby('cluster').count())

# Exit if initial solution doesn't have at least a point in each cluster
if len(cluster_count) != k:
    exit(1)

# Transform it into a numpy array
restrictions_numpy = np.asarray(restrictions)

# Save index of restrictions for each row
pos_ones = []
pos_neg_ones = []

for i in range(n_instances):
    r = restrictions_numpy[i][:]
    pos_ones.append(np.nonzero(r == 1)[0])
    pos_neg_ones.append(np.nonzero(r == -1)[0])

pos_ones = np.asarray(pos_ones)
pos_neg_ones = np.asarray(pos_neg_ones)


# Create necessary columns
data['distance_cluster'] = np.nan
data = np.asarray(data)

centroids = update_centroids_numpy(data)
data, sum_dist, av_count = calculate_distance_cluster_numpy(data, centroids)

n_evaluations = 0

# Calculate initial infeasibility
total_infeasibility = infeasibility_numpy(data)

# Initialize variables
objective_value = np.mean(sum_dist/av_count) + lambda_value * total_infeasibility
repeated = False
sum_values_clusters = sum_instances(data)

temperature = (mu_phi*objective_value)/(-np.log(mu_phi))
while temperature < final_temp:
    final_temp = final_temp / 10.0

beta = (temperature - final_temp) / ((100000.0/max_generated) * temperature * final_temp)
n_iterations = 1

best_objective = np.copy(objective_value)
best_deviation = np.copy(np.mean(sum_dist/av_count))
best_inf = np.copy(total_infeasibility)
# print(beta)
# print((100000.0/max_generated) * temperature * final_temp)
# print(temperature, "ss", beta)
# exit(1)
kk = 0
best_ss = np.zeros(n_instances)
while temperature > final_temp and n_evaluations < 100000:
    # Get first neighbour to be able to compare it later on
    first_neigh = possible_changes[0]

    # number_cluster = count_each_cluster(data)
    accepted = 0
    generated = 0
    # print("BBB",temperature,"  ", final_temp, "  ", total_infeasibility)
    kk = 0
    # np.random.shuffle(possible_changes)

    while accepted < max_accepted and generated < max_generated:
        possible_changes, neigh = get_neightbour(possible_changes)
        # Save the original cluster from the point we are going to change
        old_cluster = int(data[neigh[0]][cluster_index])
        p_index = neigh[0]
        new_cluster = neigh[1]

        # Save old values
        old_objective_value = objective_value
        old_infeasibility = total_infeasibility
        old_sum = np.copy(sum_dist)

        # Skip if the cluster only have 1 element
        if av_count[old_cluster] == 1 or old_cluster == new_cluster:
            # print(av_count, "   ", p_index, "  ", old_cluster, "  ", new_cluster, "   ", len(possible_changes))
            continue
        n_evaluations += 1
        generated+=1

        total_infeasibility = total_infeasibility - row_infeasibility_numpy(data, p_index)
        av_count[old_cluster] -= 1

        data[p_index][cluster_index] = new_cluster

        total_infeasibility = total_infeasibility + row_infeasibility_numpy(data, p_index)
        av_count[new_cluster] += 1

        centroids, sum_values_clusters = update_centroids_optimized(data, centroids, sum_values_clusters, p_index, old_cluster, new_cluster, av_count)
        # Calculate new average
        # centroids = update_centroids_numpy(data)

        data, old_d, new_d = calculate_distance_cluster_numpy2(data, centroids, old_cluster, new_cluster)
        sum_dist[old_cluster] = old_d
        sum_dist[new_cluster] = new_d
        # data, sum_dist, av_count = calculate_distance_cluster_numpy(data, centroids)

        # Calculate new objective value
        objective_value = np.mean(sum_dist/av_count) + lambda_value * total_infeasibility
        # print(generated, ":  ",objective_value, "   ", old_objective_value)
        delta = objective_value - old_objective_value
        ll = np.random.rand()
        ra = ll < np.exp((-delta)/temperature)

        # Restore values
        it.append(n_evaluations)
        ob.append(old_objective_value)
        if delta <= 0 or ra:
            if delta > 0:
                kk+=1
            # print(objective_value, "   ",delta<0,"  ",best_objective,"     ", ll)

            accepted += 1
            # if total_infeasibility != infeasibility_numpy(data):
            #     print(total_infeasibility, " +++ ", infeasibility_numpy(data))
            if objective_value < best_objective:
                best_objective = np.copy(objective_value)
                best_deviation = np.copy(np.mean(sum_dist / av_count))
                best_inf = np.copy(total_infeasibility)
                best_ss = np.copy(data[:, cluster_index])

        else:
            # if total_infeasibility != infeasibility_numpy(data):
            #     print(total_infeasibility, " --- ", infeasibility_numpy(data))
            objective_value = old_objective_value
            data[p_index][cluster_index] = old_cluster
            total_infeasibility = old_infeasibility
            av_count[old_cluster] += 1
            av_count[new_cluster] -= 1
            sum_dist = np.copy(old_sum)
            centroids, sum_values_clusters = update_centroids_optimized(data, centroids, sum_values_clusters,
                                                                        p_index, new_cluster, old_cluster, av_count)

    if cauchy:
        temperature = temperature / (1.0 + (beta * temperature))
        # print(temperature)
    else:
        temperature = temperature * alpha
    n_iterations += 1
    if accepted == 0:
        # print("NO ENCUNETRO NA")
        break
    # print("CCC",temperature,"  ", final_temp,"   ", generated, "  ac:", kk, "   ", best_objective)


#Finish timing
elapsed_time = time.perf_counter() - start_time
# print("Ev: ", n_evaluations)

# print("For lambda var:", lambda_var)
print("Tasa C:", best_deviation)
# print("Iter:", n_evaluations)
print("Tasa Inf:", best_inf)
print("Agr:", best_objective)
print("Time:", elapsed_time)
# print(best_ss)
# print("KK",kk)
# plt.plot(it,ob)
# plt.show()
# print(elapsed_time)

# print(count_each_cluster(data))
