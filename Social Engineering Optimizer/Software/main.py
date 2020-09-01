import pandas as pd
import numpy as np
import time
# import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.special import comb
import math
import sys
##########################
if len(sys.argv) == 8:
    dataset_name = sys.argv[1]
    restr_level = int(sys.argv[2])
    seed_asigned = int(sys.argv[3])
    lambda_var = float(sys.argv[4])
    alpha = float(sys.argv[5])
    beta = float(sys.argv[6])
    n_attacks = int(sys.argv[7])

elif len(sys.argv) == 1:
    dataset_name = "rand"
    restr_level = 10
    seed_asigned = 456
    lambda_var = 1
    alpha = 0.2
    beta = 0.5
    n_attacks = 50
    technique = 2
    optimization = True
    optimization2 = True
else:
    print("Wrong number of arguments.")
    exit(1)
#############################
inf_gr = []
inf_lo = []
it_gr = []
it_lo = []

def printIndexCluster(df):
    d = [[] for i in range(k)]
    for i in range(df.shape[0]):
        d[int(df[i][cluster_index])].append(i)

    for r in range(k):
        print(d[r])

def first_assignation_numpy(df, centr):
    clusters_assigned = np.zeros(k)
    for w in range(df.shape[0]):
        i = idx[w]
        for cluster_id in range(k):
            df[i][cluster_index] = cluster_id
            df[i][infeasibility_cluster_index + cluster_id] = row_infeasibility_first_numpy(df, i)



        df[i][cluster_index] = np.argmin(df[i, infeasibility_cluster_index:])

        own = int(df[i][cluster_index])
        clusters_assigned[own]+=1
        for cluster_id in range(k):
            if cluster_id != own and df[i][infeasibility_cluster_index + own] == df[i][
                infeasibility_cluster_index + cluster_id]:
                d1 = np.float128(math.sqrt(sum([(a - b) ** 2 for a, b in zip(df[i][0:cluster_index], centr[own])])))
                d2 = np.float128(math.sqrt(sum([(a - b) ** 2 for a, b in zip(df[i][0:cluster_index], centr[cluster_id])])))
                if d1 > d2:
                    clusters_assigned[own]-=1
                    df[i][cluster_index] = cluster_id
                    own = cluster_id
                    clusters_assigned[own] += 1


    for i in range(k):
        if clusters_assigned[i] == 0:
            w = i
            while (clusters_assigned[int(df[w][cluster_index])] <= k and w < df.shape[0] - 1):
                w += 1
            df[w][cluster_index] = i
    return df


def regular_assignation_numpy(df, centr):
    clusters_assigned = count_each_cluster(df)
    for w in range(df.shape[0]):
        i = idx[w]
        own = int(df[i][cluster_index])
        if (clusters_assigned[own] == 1):
            continue
        clusters_assigned[own] -= 1
        for cluster_id in range(k):
            df[i][cluster_index] = cluster_id
            df[i][infeasibility_cluster_index + cluster_id] = row_infeasibility_numpy(df, i)

        df[i][cluster_index] = np.argmin(df[i, infeasibility_cluster_index:])
        own = int(df[i][cluster_index])
        clusters_assigned[own] += 1
        for cluster_id in range(k):
            if cluster_id != own and df[i][infeasibility_cluster_index + own] == df[i][
                infeasibility_cluster_index + cluster_id]:
                d1 = np.float128(math.sqrt(sum([(a - b) ** 2 for a, b in zip(df[i][0:cluster_index], centr[own])])))
                d2 = np.float128(math.sqrt(sum([(a - b) ** 2 for a, b in zip(df[i][0:cluster_index], centr[cluster_id])])))
                if d1 > d2:
                    clusters_assigned[own] -= 1
                    df[i][cluster_index] = cluster_id
                    own = int(cluster_id)
                    clusters_assigned[own] += 1

    return df


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
    av = (np.zeros(k))
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
    new_dist = math.sqrt(sum([(a - b) ** 2 for a, b in zip(df[ind][0:cluster_index], centr[new_clu])]))
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
    r = restrictions_numpy[row_index][row_index+1:]

    own = df[row_index][cluster_index]

    for c in pos_ones[row_index][pos_ones[row_index]>row_index]:
        if not (df[c][cluster_index] == own):
            infeasibility_points += 1

    # infeasibility_points += len([1 for i in np.where(r == -1)[0] if (cluster[i + row_index + 1] == own)])
    for c in pos_neg_ones[row_index][pos_neg_ones[row_index]>row_index]:
        if df[c][cluster_index] == own:
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

# Spot attack technique 1 (ambos random son el mismo segun el codigo, cambiar?)
def obtaining_technique(defender, attacker, beta):
    uniform_numbers = np.random.rand(n_instances)
    defender = (defender * (1-np.sin(beta)*uniform_numbers) + ((defender + attacker)*np.sin(beta) * uniform_numbers)/2.0)
    return defender

# Spot attack technique 2
def phishing_technique(defender, attacker, beta):
    uniform_numbers1 = np.random.rand(n_instances)
    uniform_numbers2 = np.random.rand(n_instances)
    def1 = (attacker*(1-np.sin(beta)*uniform_numbers1)+(((defender+attacker)*np.sin(beta)*uniform_numbers2)/2.0))

    uniform_numbers1 = np.random.rand(n_instances)
    uniform_numbers2 = np.random.rand(n_instances)
    def2 = (defender*(1-np.sin(np.pi/2.0 - beta)*uniform_numbers1)+((defender+attacker)*np.sin(np.pi/2.0 - beta)*uniform_numbers2))

    return def1, def2

# Spot attack technique 3
def diversion_theft_technique(defender, attacker, beta):
    uniform_numbers1 = np.random.rand(n_instances)
    uniform_numbers2 = np.random.rand(n_instances)
    uniform_numbers3 = np.random.rand(n_instances)
    defender = (defender * (1-np.sin(beta)*uniform_numbers1)+(((defender+attacker*uniform_numbers2*np.sin(np.pi/2.0-beta))*np.sin(beta)*uniform_numbers3)/2.0))
    return defender

# Spot attack technique 4
def pretext_technique(defender, attacker, beta):
    uniform_numbers1 = np.random.rand(n_instances)
    uniform_numbers2 = np.random.rand(n_instances)
    uniform_numbers3 = np.random.rand(n_instances)
    uniform_numbers4 = np.random.rand(n_instances)
    defender = (defender*uniform_numbers1*np.sin(np.pi/2.0-beta))*(1-np.sin(beta)*np.sin(uniform_numbers2))+(((defender*uniform_numbers3*np.sin(np.pi/2.0 - beta)+attacker)*np.sin(beta)*uniform_numbers4)/2.0)
    return defender

def training(gamma, defender_data, defender_centroids, defender_sum_dist, defender_av_count, defender_total_infeasibility, defender_sum_values_clusters, attacker):
    global n_evaluations
    # Save old values
    old_infeasibility = defender_total_infeasibility
    old_sum = np.copy(defender_sum_dist)

    best_obj_value = np.mean(defender_sum_dist / defender_av_count) + lambda_value * defender_total_infeasibility
    best_p = -1
    result = np.copy(defender_data)
    result_centroids = np.copy(defender_centroids)
    result_sum_dist = np.copy(defender_sum_dist)
    result_av_count = np.copy(defender_av_count)
    result_total_inf = np.copy(defender_total_infeasibility)
    result_sum_values_clusters = np.copy(defender_sum_values_clusters)
    qq = 0
    it = 0
    orr = best_obj_value
    while it < gamma:
        p_index = np.random.randint(0, n_instances)
        # Save the original cluster from the point we are going to change
        old_cluster = int(defender_data[p_index][cluster_index])
        new_cluster = int(attacker[p_index][cluster_index])

        # Skip if the cluster only have 1 element
        if defender_av_count[old_cluster] == 1 or old_cluster == new_cluster:
            continue
        qq+=1
        it+=1
        n_evaluations +=1
        defender_total_infeasibility = defender_total_infeasibility - row_infeasibility_numpy(defender_data, p_index)
        defender_av_count[old_cluster] -= 1

        defender_data[p_index][cluster_index] = new_cluster

        defender_total_infeasibility = defender_total_infeasibility + row_infeasibility_numpy(defender_data, p_index)
        defender_av_count[new_cluster] += 1

        defender_centroids, defender_sum_values_clusters = update_centroids_optimized(defender_data, defender_centroids, defender_sum_values_clusters, p_index,
                                                                    old_cluster, new_cluster, defender_av_count)
        defender_data, old_d, new_d = calculate_distance_cluster_numpy2(defender_data, defender_centroids, old_cluster, new_cluster)
        defender_sum_dist[old_cluster] = old_d
        defender_sum_dist[new_cluster] = new_d

        # Calculate new objective value
        objective_value = np.mean(defender_sum_dist / defender_av_count) + lambda_value * defender_total_infeasibility

        if objective_value < best_obj_value:
            best_obj_value = objective_value
            best_p = p_index
            result = np.copy(defender_data)
            result_centroids  = np.copy(defender_centroids)
            result_sum_dist = np.copy(defender_sum_dist)
            result_av_count = np.copy(defender_av_count)
            result_total_inf = np.copy(defender_total_infeasibility)
            result_sum_values_clusters = np.copy(defender_sum_values_clusters)

        first_it = False
        defender_data[p_index][cluster_index] = old_cluster
        defender_total_infeasibility = old_infeasibility
        defender_av_count[old_cluster] += 1
        defender_av_count[new_cluster] -= 1
        defender_sum_dist = np.copy(old_sum)
        defender_centroids, defender_sum_values_clusters = update_centroids_optimized(defender_data, defender_centroids, defender_sum_values_clusters,
                                                                    p_index, new_cluster, old_cluster, defender_av_count)

    print("Añado ", qq, "  de ", gamma)
    print("B : ", orr-best_obj_value)
    return result, result_centroids, result_sum_dist, result_av_count, result_total_inf, result_sum_values_clusters, best_obj_value, best_p
    # We need to update with best_p after returning

def training2(gamma, defender_data, defender_centroids, defender_sum_dist, defender_av_count, defender_total_infeasibility, defender_sum_values_clusters, attacker):
    global n_evaluations, possible_changes

    # Save old values
    old_infeasibility = defender_total_infeasibility
    old_sum = np.copy(defender_sum_dist)

    best_obj_value = np.mean(defender_sum_dist / defender_av_count) + lambda_value * defender_total_infeasibility
    best_p = -1
    result = np.copy(defender_data)
    result_centroids = np.copy(defender_centroids)
    result_sum_dist = np.copy(defender_sum_dist)
    result_av_count = np.copy(defender_av_count)
    result_total_inf = np.copy(defender_total_infeasibility)
    result_sum_values_clusters = np.copy(defender_sum_values_clusters)
    qq = 0
    it = 0
    orr = best_obj_value
    rep = False
    rep2 = False
    while it < gamma:
        possible_changes, neigh = get_neightbour(possible_changes)
        p_index = neigh[0]
        old_cluster = int(defender_data[p_index][cluster_index])
        new_cluster = int(attacker[p_index][cluster_index])
        # Skip if the cluster only have 1 element
        if defender_av_count[old_cluster] == 1 or old_cluster == new_cluster:
            if rep:
                if rep2:
                    it +=1
                rep2 = True
            rep = True
            continue
        rep = False
        rep2 = False
        qq+=1
        it+=1
        n_evaluations +=1
        defender_total_infeasibility = defender_total_infeasibility - row_infeasibility_numpy(defender_data, p_index)
        defender_av_count[old_cluster] -= 1

        defender_data[p_index][cluster_index] = new_cluster

        defender_total_infeasibility = defender_total_infeasibility + row_infeasibility_numpy(defender_data, p_index)
        defender_av_count[new_cluster] += 1

        defender_centroids, defender_sum_values_clusters = update_centroids_optimized(defender_data, defender_centroids, defender_sum_values_clusters, p_index,
                                                                    old_cluster, new_cluster, defender_av_count)
        defender_data, old_d, new_d = calculate_distance_cluster_numpy2(defender_data, defender_centroids, old_cluster, new_cluster)
        defender_sum_dist[old_cluster] = old_d
        defender_sum_dist[new_cluster] = new_d

        # Calculate new objective value
        objective_value = np.mean(defender_sum_dist / defender_av_count) + lambda_value * defender_total_infeasibility
        # print("Encontrado: ", objective_value)

        # print("Best: ", best_obj_value, "  act: ", objective_value)
        # Restore values
        if best_obj_value <= objective_value:
            defender_data[p_index][cluster_index] = old_cluster
            defender_total_infeasibility = old_infeasibility
            defender_av_count[old_cluster] += 1
            defender_av_count[new_cluster] -= 1
            defender_sum_dist = np.copy(old_sum)
            defender_centroids, defender_sum_values_clusters = update_centroids_optimized(defender_data, defender_centroids, defender_sum_values_clusters,
                                                                        p_index, new_cluster, old_cluster, defender_av_count)
        if objective_value < best_obj_value:
            best_obj_value = objective_value
            best_p = p_index
            result = np.copy(defender_data)
            result_centroids  = np.copy(defender_centroids)
            result_sum_dist = np.copy(defender_sum_dist)
            result_av_count = np.copy(defender_av_count)
            result_total_inf = np.copy(defender_total_infeasibility)
            result_sum_values_clusters = np.copy(defender_sum_values_clusters)


    return result, result_centroids, result_sum_dist, result_av_count, result_total_inf, result_sum_values_clusters, best_obj_value, best_p
    # We need to update with best_p after returning

def discretize_solution(df, continous):
    temp = np.copy(np.floor(1+(float(k)*continous))-1)
    temp[temp>=k] = k-1

    count_new = np.zeros(k)
    full_c, count = np.unique(temp, return_counts=True)
    for x in range(len(full_c)):
        count_new[int(full_c[x])] = count[x]
    if len(full_c) != k:
        w = 0
        for h in range(k):
            if h not in full_c:
                while count_new[int(temp[w])] < k:
                    w += 1
                temp[w] = h
                w += 1


    return temp





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
gamma = int(np.round(n_instances*alpha))


# Start timing
start_time = time.perf_counter()

D = squareform(pdist(data))
max_distance, [I_row, I_col] = np.nanmax(D), np.unravel_index(np.argmax(D), D.shape)
n_restrictions = (((len(restrictions.index) ** 2) - (restrictions.isin([0]).sum().sum())) / 2)-data.shape[0]
lambda_value = (max_distance / n_restrictions) * lambda_var
# Generate neighbourhood
possible_changes = []
for i in range(len(data.index)):
    possible_changes.append((i, 0))
np.random.shuffle(possible_changes)

data['cluster'] = np.zeros(data.shape[0])

# Transform it into a numpy array
restrictions_numpy = np.asarray(restrictions)


# Save index of restrictions for each row
pos_ones = []
pos_neg_ones = []

for i in range(n_instances):
    r = restrictions_numpy[i][:]
    pos_ones.append(np.nonzero(r == 1)[0])
    pos_neg_ones.append(np.nonzero(r == -1)[0])

pos_ones = np.asarray(pos_ones, dtype=object)
pos_neg_ones = np.asarray(pos_neg_ones, dtype=object)


# Create necessary columns
data['distance_cluster'] = np.nan
data = np.asarray(data)

n_iterations = 0

repeated = False


defender = np.random.rand(n_instances)
data_defender = np.copy(data)
data_defender[:,cluster_index] = discretize_solution(data, defender)
def_centroids = update_centroids_numpy(data_defender)
data_defender, def_sum_dist, def_av_count = calculate_distance_cluster_numpy(data_defender, def_centroids)
def_total_infeasibility = infeasibility_numpy(data_defender)
def_objective_value = np.mean(def_sum_dist / def_av_count) + lambda_value * def_total_infeasibility
def_sum_values_clusters = sum_instances(data_defender)

new_data_defender = np.copy(data)
new_data_defender2 = np.copy(data)


attacker = np.random.rand(n_instances)
data_attacker = np.copy(data)
data_attacker[:,cluster_index] = discretize_solution(data, attacker)
att_centroids = update_centroids_numpy(data_attacker)
data_attacker, att_sum_dist, att_av_count = calculate_distance_cluster_numpy(data_attacker, att_centroids)
att_total_infeasibility = infeasibility_numpy(data_attacker)
att_objective_value = np.mean(att_sum_dist / att_av_count) + lambda_value * att_total_infeasibility
att_sum_values_clusters = sum_instances(data_attacker)

if def_objective_value < att_objective_value:
    attacker, defender = defender, attacker
    data_attacker, data_defender = data_defender, data_attacker
    att_centroids, def_centroids = def_centroids, att_centroids
    att_sum_dist, def_sum_dist = def_sum_dist, att_sum_dist
    att_av_count, def_av_count = def_av_count, att_av_count
    att_total_infeasibility, def_total_infeasibility = def_total_infeasibility, att_total_infeasibility
    att_objective_value, def_objective_value = def_objective_value, att_objective_value
    att_sum_values_clusters, def_sum_values_clusters = def_sum_values_clusters, att_sum_values_clusters

n_evaluations = 0
###################
while n_evaluations < 100000:

    if optimization2:
        data_defender, def_centroids, def_sum_dist, def_av_count, def_total_infeasibility, def_sum_values_clusters, def_objective_value, index_p =\
            training2(gamma, data_defender, def_centroids, def_sum_dist, def_av_count, def_total_infeasibility, def_sum_values_clusters, data_attacker)
    else:
        data_defender, def_centroids, def_sum_dist, def_av_count, def_total_infeasibility, def_sum_values_clusters, def_objective_value, index_p =\
            training(gamma, data_defender, def_centroids, def_sum_dist, def_av_count, def_total_infeasibility, def_sum_values_clusters, data_attacker)


    if index_p != -1:
        defender[index_p] = attacker[index_p]

    for w in range(n_attacks):
        if technique == 1:
            new_defender = obtaining_technique(defender, attacker, beta)
        elif technique == 2:
            new_defender, new_defender2 = phishing_technique(defender, attacker, beta)
        elif technique == 3:
            new_defender = diversion_theft_technique(defender, attacker, beta)
        elif technique == 4:
            new_defender = pretext_technique(defender, attacker, beta)


        new_data_defender[:,cluster_index] = discretize_solution(data, new_defender)
        new_def_centroids = update_centroids_numpy(new_data_defender)
        new_data_defender, new_def_sum_dist, new_def_av_count = calculate_distance_cluster_numpy(new_data_defender, new_def_centroids)
        new_def_total_infeasibility = infeasibility_numpy(new_data_defender)
        new_def_objective_value = np.mean(new_def_sum_dist / new_def_av_count) + lambda_value * new_def_total_infeasibility
        new_def_sum_values_clusters = sum_instances(new_data_defender)
        n_evaluations +=1

        if technique == 2:
            new_data_defender2[:,cluster_index] = discretize_solution(data, new_defender2)
            new_def2_centroids = update_centroids_numpy(new_data_defender2)
            new_data_defender2, new_def2_sum_dist, new_def2_av_count = calculate_distance_cluster_numpy(new_data_defender2, new_def2_centroids)
            new_def2_total_infeasibility = infeasibility_numpy(new_data_defender2)
            new_def2_objective_value = np.mean(new_def2_sum_dist / new_def2_av_count) + lambda_value * new_def2_total_infeasibility
            new_def2_sum_values_clusters = sum_instances(new_data_defender2)
            n_evaluations += 1

            if new_def2_objective_value < new_def_objective_value:
                new_defender2, new_defender = new_defender, new_defender2
                new_data_defender2, new_data_defender = new_data_defender, new_data_defender2
                new_def2_centroids, new_def_centroids = new_def_centroids, new_def2_centroids
                new_def2_sum_dist, new_def_sum_dist = new_def_sum_dist, new_def2_sum_dist
                new_def2_av_count, new_def_av_count = new_def_av_count, new_def2_av_count
                new_def2_total_infeasibility, new_def_total_infeasibility = new_def_total_infeasibility, new_def2_total_infeasibility
                new_def2_objective_value, new_def_objective_value = new_def_objective_value, new_def2_objective_value
                new_def2_sum_values_clusters, new_def_sum_values_clusters = new_def_sum_values_clusters, new_def2_sum_values_clusters

        if new_def_objective_value < def_objective_value:
            defender = np.copy(new_defender)
            data_defender[:,cluster_index] = np.copy(new_data_defender[:, cluster_index])
            def_centroids = np.copy(new_def_centroids)
            def_sum_dist = np.copy(new_def_sum_dist)
            def_av_count = np.copy(new_def_av_count)
            def_total_infeasibility = np.copy(new_def_total_infeasibility)
            def_objective_value = np.copy(new_def_objective_value)
            def_sum_values_clusters = np.copy(new_def_sum_values_clusters)

        if def_objective_value < att_objective_value:
            attacker, defender = defender, attacker
            data_attacker, data_defender = data_defender, data_attacker
            att_centroids, def_centroids = def_centroids, att_centroids
            att_sum_dist, def_sum_dist = def_sum_dist, att_sum_dist
            att_av_count, def_av_count = def_av_count, att_av_count
            att_total_infeasibility, def_total_infeasibility = def_total_infeasibility, att_total_infeasibility
            att_objective_value, def_objective_value = def_objective_value, att_objective_value
            att_sum_values_clusters, def_sum_values_clusters = def_sum_values_clusters, att_sum_values_clusters

    defender = np.random.rand(n_instances)
    if optimization:
        for i in range(n_instances):
            if np.random.rand() < 0.7:
                defender[i] = attacker[i]
    data_defender = np.copy(data)
    data_defender[:, cluster_index] = discretize_solution(data, defender)
    def_centroids = update_centroids_numpy(data_defender)
    data_defender, def_sum_dist, def_av_count = calculate_distance_cluster_numpy(data_defender, def_centroids)
    def_total_infeasibility = infeasibility_numpy(data_defender)
    def_objective_value = np.mean(def_sum_dist / def_av_count) + lambda_value * def_total_infeasibility
    def_sum_values_clusters = sum_instances(data_defender)




#Finish timing
elapsed_time = time.perf_counter() - start_time

# print("For lambda var:", lambda_var)
print("Tasa C:", np.mean(att_sum_dist/att_av_count))
# print("Iter:", n_iterations)
print("Tasa Inf:", att_total_infeasibility)
print("Agr:", att_objective_value)
print("Time:", elapsed_time)


