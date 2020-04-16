import pandas as pd
import numpy as np
import time
# import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import pdist, squareform, cdist
import math
import sys
##########################
if len(sys.argv) == 6:
    mode = sys.argv[1]
    dataset_name = sys.argv[2]
    restr_level = int(sys.argv[3])
    seed_asigned = int(sys.argv[4])
    lambda_var = float(sys.argv[5])
elif len(sys.argv) == 1:
    mode = "generational"          # generational or steady (state)
    dataset_name = "ecoli"
    restr_level = 10
    seed_asigned = 456
    lambda_var = 1
    n_population = 50
    uni_cross = False
    two_point_best_first = True
else:
    print("Wrong number of arguments.")
    exit(1)
#############################


def calculate_distance_cluster_numpy(data, centr, clusters, distance_clusters):
    av = (np.zeros(k))
    for i in range(n_instances):
        cluster_des = clusters[i]
        distance_clusters[i] = np.float128(math.sqrt(sum([(a - b) ** 2 for a, b in zip(data[i], centr[cluster_des])])))
        av[cluster_des] += distance_clusters[i]

    return data, av, clusters, distance_clusters


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
    r = restrictions_numpy[row_index]
    for c in np.nonzero(r == 1)[0]:
        if not (df[c][cluster_index] == df[row_index][cluster_index]):
            infeasibility_points += 1


    for c in np.nonzero(r == -1)[0]:
        if df[c][cluster_index] == df[row_index][cluster_index]:
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

def row_infeasibility_partial(cluster, row_index):
    infeasibility_points = 0
    r = restrictions_numpy[row_index][row_index+1:]

    own = cluster[row_index]

    # infeasibility_points += len([1 for i in np.where(r == 1)[0] if not(cluster[i + row_index + 1] == own)])
    for c in np.where(r == 1)[0]:
        if not (cluster[c + row_index + 1] == own):
            infeasibility_points += 1

    # infeasibility_points += len([1 for i in np.where(r == -1)[0] if (cluster[i + row_index + 1] == own)])
    for c in np.where(r == -1)[0]:
        if cluster[c + row_index + 1] == own:
            infeasibility_points += 1
    return infeasibility_points


def infeasibility_numpy(cluster):
    inf_c = 0
    for i in range(n_instances):
        inf_c = inf_c + row_infeasibility_partial(cluster, i)

    return inf_c


# Return a numpy matrix with new centroids given the clusters
def update_centroids_numpy(df, cluster):
    sum = np.zeros((k, n_characteristics))
    count = np.zeros(k)
    for i in range(n_instances):
        sum[cluster[i]] += df[i]
        count[cluster[i]] += 1
    # print(sum)
    for i in range(k):
        sum[i] = sum[i] / count[i]
    return sum, count

# Return a numpy matrix with new centroids given the clusters
def update_centroids_optimized(df, centr, sum_values, ind, old_cluster, new_cluster, count):
    sum_values[old_cluster] -= df[ind][0:cluster_index]
    sum_values[new_cluster] += df[ind][0:cluster_index]

    centr[old_cluster] = sum_values[old_cluster]/count[old_cluster]
    centr[new_cluster] = sum_values[new_cluster]/count[new_cluster]

    return centr, sum_values


def sum_instances(data, cluster):
    sum = np.zeros((k, cluster_index))
    for i in range(n_instances):
        sum[cluster[i]] += data[i]

    return sum


def count_each_cluster(clusters):
    result = np.zeros(k)
    # print(clusters)
    for row in clusters:
        result[row] += 1
    return result


def uniform_crossover(p1, p2):
    result = np.copy(p2)
    ind_copy = np.random.choice(range(n_instances), round(n_instances/2), replace=False)
    # print("A", len(ind_copy),"bb", len(np.unique(ind_copy)))
    for i in ind_copy:
        result[i] = p1[i]

    return result


def two_point_uniform_crossover(p1, p2):
    result = np.copy(p1)
    start, leng = np.random.randint(0, n_instances, 2)
    end = start + leng - (0 if (start + leng < n_instances) else n_instances)
    left = int(n_instances - leng)
    ind_copy = np.random.choice(range(left), round(left/2), replace=False)
    for i in ind_copy:
        ind = i + end - (0 if (i + end < n_instances) else n_instances)
        result[ind] = p2[ind]

    return result



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


data = pd.read_csv(data_path, header=None)
restrictions = pd.read_csv(restrictions_path, header=None)
col_names = []
# The name of the df's columns will be 'cX' being X the column's number
for i in range(len(data.columns)):
    col_names.append("c" + str(i))
data.columns = col_names
n_characteristics = len(col_names)
n_instances = data.shape[0]
mutation_prob = 0.001

if mode == "steady":
    n_selected = 2
    n_new_children = 2
elif mode == "generational":
    n_selected = n_population
    n_new_children = int(n_population * .7)
    if n_new_children % 2 == 1:
        n_new_children += 1
else:
    print("Wrong mode inserted")
    exit(1)
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


#####################################################################


D = squareform(pdist(data))
max_distance, [I_row, I_col] = np.nanmax(D), np.unravel_index(np.argmax(D), D.shape)
n_restrictions = (((len(restrictions.index) ** 2) - (restrictions.isin([0]).sum().sum())) / 2) - data.shape[0]
# print(max_distance)
lambda_value = (max_distance / n_restrictions) * lambda_var

start_time = time.perf_counter()

population_cluster = np.empty([n_population, n_instances], dtype=int)
population_distance_cluster = np.empty([n_population, n_instances], dtype=float)
population_cluster_count = np.empty([n_population, k], dtype=int)
population_centroids =  np.empty([n_population,k, n_characteristics], dtype=float)
population_sum_dist = np.empty([n_population, k], dtype=float)
population_av_count = np.empty([n_population, k], dtype=float)
population_infeasibility = np.empty(n_population, dtype=float)
population_objetive_value = np.empty(n_population, dtype=float)
# population_sum_values_clusters = np.empty([n_population, k, n_characteristics], dtype=float)

# Transform it into a numpy array
restrictions_numpy = np.asarray(restrictions)
data_arr = np.asarray(data)

for i in range(n_population):
    # Generate initial solution
    population_cluster[i] = np.random.randint(0, k, data.shape[0])

    # Exit if initial solution doesn't have at least a point in each cluster
    if len(population_cluster_count[i]) != k:
        exit(1)



    # Create necessary columns
    distance_cluster = np.empty(n_instances, dtype=float)


    population_centroids[i], population_av_count[i] = update_centroids_numpy(data_arr, population_cluster[i])

    data_arr, population_sum_dist[i], population_cluster[i], population_distance_cluster[i] = calculate_distance_cluster_numpy(data_arr, population_centroids[i], population_cluster[i], distance_cluster)

    population_infeasibility[i] = infeasibility_numpy(population_cluster[i])

    population_objetive_value[i] = np.mean(population_sum_dist[i]/population_av_count[i]) + lambda_value * population_infeasibility[i]

    # population_sum_values_clusters[i] = sum_instances(data_arr, population_cluster[i])



finish_ini = time.perf_counter()
elapsed_time = finish_ini - start_time
print("Initializing: ", elapsed_time)

evaluations = 0
ev_l = 100
# Modelo Estacionario

children_distance_cluster = np.empty([n_selected, n_instances], dtype=float)
children_cluster_count = np.empty([n_selected, k], dtype=int)
children_centroids = np.empty([n_selected, k, n_characteristics], dtype=float)
children_sum_dist = np.empty([n_selected, k], dtype=float)
children_av_count = np.empty([n_selected, k], dtype=float)
children_infeasibility = np.empty(n_selected, dtype=float)
children_objetive_value = np.empty(n_selected, dtype=float)
# children_sum_values_clusters = np.empty([n_selected, k, n_characteristics], dtype=float)

selected = np.empty(n_selected, dtype=int)

while evaluations < 100000:
    best_selected = False
    ######################## Selection #######################################
    for i in range(n_selected):
        a, b = np.random.randint(0, n_population, 2)
        if population_objetive_value[a]> population_objetive_value[b]:
            selected[i] = b
        else:
            selected[i] = a

    if(mode=="generational"):
        index_best_popu = np.argmin(population_objetive_value)
        best_selected = index_best_popu in selected
        if best_selected:
            best_selected = np.where(selected == index_best_popu)[0][0] >= n_new_children
        if not best_selected:
            elite_obj = np.copy(population_objetive_value[index_best_popu])
            elite_cluster = np.copy(population_cluster[index_best_popu])
    #########################################################################


    ######################## Offspring ######################################

    children = np.empty([n_selected, n_instances], dtype=int)

    for i in range(0,n_new_children,2):
        if uni_cross:
            children[i] = uniform_crossover(population_cluster[selected[i]], population_cluster[selected[i+1]])
            children[i+1] = uniform_crossover(population_cluster[selected[i]], population_cluster[selected[i+1]])
        else:
            a = i
            b = i + 1
            if population_objetive_value[selected[i]] < population_objetive_value[selected[i+1]]:
                if not two_point_best_first:
                    a = i + 1
                    b = i
            else:
                if two_point_best_first:
                    a = i + 1
                    b = i

            children[i] = two_point_uniform_crossover(population_cluster[selected[a]], population_cluster[selected[b]])
            children[i+1] = two_point_uniform_crossover(population_cluster[selected[a]], population_cluster[selected[b]])

    if mode == "generational":
        for i in range(n_new_children, n_selected):
            children[i] = np.copy(population_cluster[selected[i]])

    #########################################################################

    if np.any(population_objetive_value == np.nan) or np.any(children == np.nan):
        print("aaaaaaaaaaaaaaa")

    ######################## Mutation ######################################
    # Since is for the stationary approach we do the mutation at the level of the gen

    if mode == "steady":
        mutation_limit_steady = n_instances * mutation_prob

        if np.random.rand() < mutation_limit_steady:
            # print("MUT")
            mutation_index = np.random.randint(0, n_instances)
            n_c = np.random.randint(0, k)
            if(children[0][mutation_index] == n_c):
                n_c= (n_c+1)%k

            children[1][mutation_index] = n_c

        if np.random.rand() < mutation_limit_steady:
            # print("MUT")
            mutation_index = np.random.randint(0, n_instances)
            n_c = np.random.randint(0, k)
            if children[1][mutation_index] == n_c:
                n_c= (n_c+1)%k

            children[1][mutation_index] = n_c

    else:
        n_mutation = round(n_instances * mutation_prob * n_selected)
        mutation_index = np.random.randint(0, n_instances * n_selected, n_mutation)
        for i in range(n_mutation):
            ind_crom, ind_gen = divmod(mutation_index[i], n_instances)
            n_c = np.random.randint(0, k)
            if children[ind_crom][ind_gen] == n_c:
                n_c= (n_c+1)%k

            children[ind_crom][ind_gen] = n_c


    #########################################################################

    # print(children[0])
    # for i in range(n_instances):
    #     children[0][i] = 0
    # print(children[0])


    ######################## Fixing ########################################
    # print(children[5][n_instances-2])
    # for i in range(n_instances):
    #     # if children[5][i] == 3 or children[5][i] == 4 or children[5][i] == 5or children[5][i] == 6:
    #         children[5][i] = 3
    # Fix if there are empty clusters
    i = 0
    while i < n_selected:
        full_c, count= np.unique(children[i], return_counts=True)
        if len(full_c) != k:
            w = 0
            c = np.zeros(k)
            for x in range(len(full_c)):
                c[full_c[x]] = count[x]
            # print(full_c)
            # print(c)
            for h in range(k):
                if h not in full_c:
                    while c[children[i][w]] < k:
                        w += 1
                    children[i][w] = h
                    w+=1

        i += 1

    #########################################################################

    for i in range(n_selected):
        if len(np.unique(children[i])) != k:
            print("En iteracion ",i, "solo hay ", len(np.unique(children[i])) ," hijos: ", np.unique(children[i]))
    # print(np.unique(children))
    # children_cluster = np.empty(n_selected, dtype=int)

    # exit(0)
    if evaluations == 1155:
        print("AAAAAAAAA", best_selected, "   BB", np.min(population_objetive_value))

    ######################## Calculate Fitness  #############################

    children_cluster = children

    evaluations += n_new_children
    for i in range(n_new_children):
        # print(children_cluster.shape)
        children_centroids[i], children_av_count[i] = update_centroids_numpy(data_arr, children_cluster[i])
        data_arr, children_sum_dist[i], children_cluster[i], children_distance_cluster[i] \
            = calculate_distance_cluster_numpy(data_arr, children_centroids[i], children_cluster[i], children_distance_cluster[i])

        children_infeasibility[i] = infeasibility_numpy(children_cluster[i])
        children_objetive_value[i] = np.mean(children_sum_dist[i]/children_av_count[i]) + lambda_value * children_infeasibility[i]

        # children_sum_values_clusters[i] = sum_instances(data_arr, children_cluster[i])

    if mode == "generational":
        for i in range(n_new_children, n_selected):
            children_objetive_value[i] = population_objetive_value[selected[i]]
            if evaluations == 1190:
                print("DD", children_objetive_value[i])

    #########################################################################



    ######################## Reinsert #######################################

    if mode == "steady":
        if children_objetive_value[0] < children_objetive_value[1]:
            ind_to_ins = 0
        else:
            ind_to_ins = 1

        worst_ind = np.argmax(population_objetive_value)

        if(children_objetive_value[ind_to_ins] < population_objetive_value[worst_ind]):
            # if z>7000:
            #     print(children_objetive_value[ind_to_ins], " < " , population_objetive_value[worst_ind])
            population_cluster[worst_ind] = np.copy(children_cluster[ind_to_ins])
            population_objetive_value[worst_ind] = np.copy(children_objetive_value[ind_to_ins])
            # population_sum_values_clusters[worst_ind] = children_sum_values_clusters[ind_to_ins]

            worst_ind = np.argmax(population_objetive_value)
            ind_to_ins = (ind_to_ins+1) % 2

            if (children_objetive_value[ind_to_ins] < population_objetive_value[worst_ind]):
                population_cluster[worst_ind] = np.copy(children_cluster[ind_to_ins])
                population_objetive_value[worst_ind] = np.copy(children_objetive_value[ind_to_ins])
                # population_sum_values_clusters[worst_ind] = children_sum_values_clusters[ind_to_ins]
    else:
        if not best_selected:
            index_worse_child = np.argmax(children_objetive_value)
            children_cluster[index_worse_child] = np.copy(elite_cluster)
            children_objetive_value[index_worse_child] = np.copy(elite_obj)

        population_cluster = np.copy(children_cluster)
        population_objetive_value = np.copy(children_objetive_value)
    #########################################################################


    if evaluations>ev_l:
        ev_l += 100
        d = sorted(population_objetive_value)
        print("It: ",evaluations, "   ",d, "   ", n_new_children)




elapsed_time2 = time.perf_counter() - finish_ini

print("Initializing: ", elapsed_time)
print("Calculating for 1000: ", elapsed_time2)