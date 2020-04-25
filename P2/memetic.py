import pandas as pd
import numpy as np
import time
# import matplotlib.pyplot as plt
# import random
from scipy.spatial.distance import pdist, squareform, cdist
import math
import sys
##########################
if len(sys.argv) == 12:
    mode = sys.argv[1]
    dataset_name = sys.argv[2]
    restr_level = int(sys.argv[3])
    seed_asigned = int(sys.argv[4])
    lambda_var = float(sys.argv[5])
    n_population = int(sys.argv[6])
    mutation_prob = float(sys.argv[7])
    uni_cross = sys.argv[8] == "si"
    two_point_best_first = sys.argv[9] == "si"
    interval_local_search = int(sys.argv[10])
    memetic_subset = float(sys.argv[11])
elif len(sys.argv) == 1:
    dataset_name = "rand"
    mode = "all"             # best, random, all
    restr_level = 10
    seed_asigned = 123
    lambda_var = 1
    n_population = 10
    interval_local_search = 10
    memetic_subset = 0.1
    mutation_prob = 0.001
    uni_cross = False
    two_point_best_first = True
else:
    print("Wrong number of arguments.")
    exit(1)

print("BBBB")
#############################
if uni_cross:
    name_file = ("memetic_"+mode+"_"+str(restr_level)+"_seed_"+str(seed_asigned)+"_uni")
elif two_point_best_first:
    name_file = ("memetic_"+mode+"_"+str(restr_level)+"_seed_"+str(seed_asigned)+"_two_BestFirst")
else:
    name_file = ("memetic_"+mode+"_"+str(restr_level)+"_seed_"+str(seed_asigned)+"_two_BestLast")

f = open("./data_collected/" + name_file + ".csv", "w")


def calculate_distance_cluster_numpy(data, centr, clusters, distance_clusters):
    av = (np.zeros(k))
    for i in range(n_instances):
        cluster_des = clusters[i]
        distance_clusters[i] = np.float128(math.sqrt(sum([(a - b) ** 2 for a, b in zip(data[i], centr[cluster_des])])))
        av[cluster_des] += distance_clusters[i]

    return data, av, clusters, distance_clusters


def update_distance(data, centr, sum_dis, ind, old_clu, new_clu, distance_cluster):

    old_dist = distance_cluster[ind]
    new_dist = math.sqrt(sum([(a - b) ** 2 for a, b in zip(data[ind], centr[new_clu])]))
    distance_cluster[ind] = new_dist

    sum_dis[old_clu] -= old_dist
    sum_dis[new_clu] += new_dist

    return data, sum_dis, old_dist, distance_cluster

def undo_distance(df, sum_dist, ind, old_cluster, new_cluster, old_dist):
    new_dist = df[ind][distance_cluster_index]

    df[ind][distance_cluster_index] = old_dist
    sum_dist[old_cluster] += old_dist
    sum_dist[new_cluster] -= new_dist

    return df, sum_dist


def row_infeasibility_numpy(cluster, row_index):
    infeasibility_points = 0
    r = restrictions_numpy[row_index]

    own = cluster[row_index]


    for c in np.where(r == 1)[0]:
        if not (cluster[c] == own):
            infeasibility_points += 1

    for c in np.where(r == -1)[0]:
        # print(c)
        if cluster[c] == own:
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

    for c in pos_ones[row_index]:
        if not (cluster[c + row_index + 1] == own):
            infeasibility_points += 1

    # infeasibility_points += len([1 for i in np.where(r == -1)[0] if (cluster[i + row_index + 1] == own)])
    for c in pos_neg_ones[row_index]:
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
def update_centroids_optimized(data, centr, sum_values, ind, old_cluster, new_cluster, count):
    sum_values[old_cluster] -= data[ind]
    sum_values[new_cluster] += data[ind]

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

def soft_local_seach(p_index, arr_data, cluster_assignated, distance_cluster, sum_values_clusters, centroids, sum_dist,
                     av_count, total_infeasibility, objetive_value):

    best_obj = np.copy(objetive_value)
    original_cluster = np.copy(cluster_assignated[p_index])
    best_cluster = np.copy(original_cluster)
    all_clusters = range(0, k)
    # print("Starting with c: ", cluster_assignated[p_index],"  obj:", objetive_value)

    best_distance_cluster = np.copy(distance_cluster)
    best_sum_values_clusters = np.copy(sum_values_clusters)
    best_centroids = np.copy(centroids)
    best_sum_dist = np.copy(sum_dist)
    best_av_count = np.copy(av_count)
    best_total_infeasibility = np.copy(total_infeasibility)

    for new_cluster in all_clusters:
        # Save the original cluster from the point we are going to change
        old_cluster = cluster_assignated[p_index]

        # Skip if the cluster only have 1 element
        if av_count[old_cluster] == 1 or original_cluster == new_cluster:
            continue

        total_infeasibility = total_infeasibility - row_infeasibility_numpy(cluster_assignated, p_index)
        av_count[old_cluster] -= 1

        cluster_assignated[p_index] = new_cluster

        total_infeasibility = total_infeasibility + row_infeasibility_numpy(cluster_assignated, p_index)
        av_count[new_cluster] += 1


        centroids, sum_values_clusters = update_centroids_optimized(arr_data, centroids, sum_values_clusters, p_index, old_cluster, new_cluster, av_count)


        data, sum_dist, old_distance, distance_cluster = update_distance(arr_data, centroids, sum_dist, p_index, old_cluster, new_cluster, distance_cluster)


        objetive_value = np.mean(sum_dist / av_count) + lambda_value * total_infeasibility

        if objetive_value < best_obj:
            best_obj = np.copy(objetive_value)
            best_cluster = np.copy(new_cluster)
            best_distance_cluster = np.copy(distance_cluster)
            best_sum_values_clusters = np.copy(sum_values_clusters)
            best_centroids = np.copy(centroids)
            best_sum_dist = np.copy(sum_dist)
            best_av_count = np.copy(av_count)
            best_total_infeasibility = np.copy(total_infeasibility)


    #     print("If we change ", old_cluster, " for ", new_cluster, " we get obj: ", objetive_value)
    # print("Best cluster is ", best_cluster, " with ", best_obj)

    changed = best_cluster != original_cluster
    cluster_assignated[p_index] = best_cluster
    return cluster_assignated, best_distance_cluster, best_sum_values_clusters, best_centroids, best_sum_dist, \
                best_av_count, best_total_infeasibility, best_obj, changed

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
limit_errors = 0.1 * n_instances



n_selected = n_population
n_new_children = int(n_population * .7)
if n_new_children % 2 == 1:
    n_new_children += 1

########################################################################################################################
pd.set_option('display.max_columns', None)

# Get minimum and maximum values for each column
minim = data.min()
maxim = data.max()

# Start random generator
# seed = int(round(time.time()))
seed = seed_asigned
# random.seed(seed)
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
population_sum_values_clusters = np.empty([n_population, k, n_characteristics], dtype=float)

# Transform it into a numpy array
restrictions_numpy = np.asarray(restrictions)
data_arr = np.asarray(data)
mean_deviation = np.zeros(n_population)

pos_ones = []
pos_neg_ones = []

for i in range(n_instances):
    r = restrictions_numpy[i][i+1:]
    pos_ones.append(np.nonzero(r == 1)[0])
    pos_neg_ones.append(np.nonzero(r == -1)[0])
    # print(pos_ones[0])

# print(pos_ones[150])
pos_ones = np.asarray(pos_ones)
pos_neg_ones = np.asarray(pos_neg_ones)

for i in range(n_population):
    # Generate initial solution
    population_cluster[i] = np.random.randint(0, k, data.shape[0])

    # Exit if initial solution doesn't have at least a point in each cluster
    if len(np.unique(population_cluster[i])) != k:
        exit(1)



    # Create necessary columns
    distance_cluster = np.empty(n_instances, dtype=float)


    population_centroids[i], population_av_count[i] = update_centroids_numpy(data_arr, population_cluster[i])

    data_arr, population_sum_dist[i], population_cluster[i], population_distance_cluster[i] = calculate_distance_cluster_numpy(data_arr, population_centroids[i], population_cluster[i], distance_cluster)

    population_infeasibility[i] = infeasibility_numpy(population_cluster[i])

    mean_deviation[i] = np.mean(population_sum_dist[i]/population_av_count[i])

    population_objetive_value[i] = mean_deviation[i] + lambda_value * population_infeasibility[i]

    population_sum_values_clusters[i] = sum_instances(data_arr, population_cluster[i])



finish_ini = time.perf_counter()
elapsed_time = finish_ini - start_time
print("Initializing: ", elapsed_time)

evaluations = n_population
generations = 0
ev_l = 100
# Modelo Estacionario

children_distance_cluster = np.empty([n_selected, n_instances], dtype=float)
children_cluster_count = np.empty([n_selected, k], dtype=int)
children_centroids = np.empty([n_selected, k, n_characteristics], dtype=float)
children_sum_dist = np.empty([n_selected, k], dtype=float)
children_av_count = np.empty([n_selected, k], dtype=float)
children_infeasibility = np.empty(n_selected, dtype=float)
children_objetive_value = np.empty(n_selected, dtype=float)
children_sum_values_clusters = np.empty([n_selected, k, n_characteristics], dtype=float)

selected = np.empty(n_selected, dtype=int)

f.write("Evaluations,BestObjetivo,MediaDesviacionHijos,MediaInfeasibilityHijos,MediaObjetivoHijos,MediaObjetivo")
f.write('\n')

f.write(str(evaluations) + "," + str(np.min(population_objetive_value)) + "," + str(np.mean(mean_deviation)) + "," +
        str(np.mean(children_infeasibility)) + "," + str(np.mean(children_objetive_value)) + "," + str(
    np.mean(population_objetive_value)))
f.write('\n')

while evaluations < 100000:
    best_selected = False
    ######################## Selection #######################################
    for i in range(n_selected):
        a, b = np.random.randint(0, n_population, 2)
        if population_objetive_value[a]> population_objetive_value[b]:
            selected[i] = b
        else:
            selected[i] = a


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

    for i in range(n_new_children, n_selected):
        children[i] = np.copy(population_cluster[selected[i]])

    #########################################################################





    ######################## Fixing ########################################
    # print(children[5][n_instances-2])
    # for i in range(n_instances):
    #     # if children[5][i] == 3 or children[5][i] == 4 or children[5][i] == 5or children[5][i] == 6:
    #         children[5][i] = 3
    # Fix if there are empty clusters
    count_new = np.empty((n_selected,k))
    i = 0
    while i < n_new_children:
        full_c, count= np.unique(children[i], return_counts=True)
        for x in range(len(full_c)):
            count_new[i][full_c[x]] = count[x]
        if len(full_c) != k:
            w = 0
            for h in range(k):
                if h not in full_c:
                    while count_new[i][children[i][w]] < k:
                        w += 1
                    children[i][w] = h
                    w+=1

        i += 1

    #########################################################################


    ######################## Mutation ######################################
    # Since is for the stationary approach we do the mutation at the level of the gen

    n_mutation = round(n_instances * mutation_prob * n_selected)
    mutation_index = np.random.randint(0, n_instances * n_selected, n_mutation)
    for i in range(n_mutation):
        ind_crom, ind_gen = divmod(mutation_index[i], n_instances)
        n_c = np.random.randint(0, k)

        recalc = False
        if ind_crom >= n_new_children:
            recalc = True
            full_c, count = np.unique(children[ind_crom], return_counts=True)
            for x in range(len(full_c)):
                count_new[ind_crom][full_c[x]] = count[x]

        while count_new[ind_crom][children[ind_crom][ind_gen]] <= 1:
            ind_gen += 1
            if ind_gen >= n_instances:
                ind_gen -= n_instances

        if children[ind_crom][ind_gen] == n_c:
            n_c = (n_c + 1) % k

        children[ind_crom][ind_gen] = n_c

        if(recalc):
            w = ind_crom
            popu_index = selected[w]

            children_centroids[w], children_av_count[w] = update_centroids_numpy(data_arr, children[w])

            data_arr, children_sum_dist[w], children[w], children_distance_cluster[w] \
                = calculate_distance_cluster_numpy(data_arr, children_centroids[w], children[w],
                                                   children_distance_cluster[w])

            children_infeasibility[w] = infeasibility_numpy(children[w])
            mean_deviation[w] = np.mean(children_sum_dist[w] / children_av_count[w])
            children_objetive_value[w] = mean_deviation[w] + lambda_value * children_infeasibility[w]

            population_cluster[popu_index] = np.copy(children[w])
            population_objetive_value[popu_index] = np.copy(children_objetive_value[w])
            population_distance_cluster[popu_index] = np.copy(children_distance_cluster[w])
            population_cluster_count[popu_index] = np.copy(children_cluster_count[w])
            population_centroids[popu_index] = np.copy(children_centroids[w])
            population_sum_dist[popu_index] = np.copy(children_sum_dist[w])
            population_av_count[popu_index] = np.copy(children_av_count[w])
            population_infeasibility[popu_index] = np.copy(children_infeasibility[w])



    #########################################################################



    ######################## Calculate Fitness  #############################

    children_cluster = children

    evaluations += n_new_children
    for i in range(n_new_children):
        # print(children_cluster.shape)
        children_centroids[i], children_av_count[i] = update_centroids_numpy(data_arr, children_cluster[i])

        data_arr, children_sum_dist[i], children_cluster[i], children_distance_cluster[i] \
            = calculate_distance_cluster_numpy(data_arr, children_centroids[i], children_cluster[i], children_distance_cluster[i])

        children_infeasibility[i] = infeasibility_numpy(children_cluster[i])
        mean_deviation[i] =  np.mean(children_sum_dist[i]/children_av_count[i])
        children_objetive_value[i] = mean_deviation[i] + lambda_value * children_infeasibility[i]

        # children_sum_values_clusters[i] = sum_instances(data_arr, children_cluster[i])

    for i in range(n_new_children, n_selected):
        children_objetive_value[i] = population_objetive_value[selected[i]]

        children_distance_cluster[i] = np.copy(population_distance_cluster[selected[i]])
        children_cluster_count[i] = np.copy(population_cluster_count[selected[i]])
        children_centroids[i] = np.copy(population_centroids[selected[i]])
        children_sum_dist[i] = np.copy(population_sum_dist[selected[i]])
        children_av_count[i] = np.copy(population_av_count[selected[i]])
        children_infeasibility[i] = np.copy(population_infeasibility[selected[i]])

    #########################################################################


    ######################## Reinsert #######################################

    if not best_selected:
        index_worse_child = np.argmax(children_objetive_value)
        children_cluster[index_worse_child] = np.copy(elite_cluster)
        children_objetive_value[index_worse_child] = np.copy(elite_obj)
        children_distance_cluster[index_worse_child] = np.copy(population_distance_cluster[index_best_popu])
        children_cluster_count[index_worse_child] = np.copy(population_cluster_count[index_best_popu])
        children_centroids[index_worse_child] = np.copy(population_centroids[index_best_popu])
        children_sum_dist[index_worse_child] = np.copy(population_sum_dist[index_best_popu])
        children_av_count[index_worse_child] = np.copy(population_av_count[index_best_popu])
        children_infeasibility[index_worse_child] = np.copy(population_infeasibility[index_best_popu])

    population_cluster = np.copy(children_cluster)
    population_objetive_value = np.copy(children_objetive_value)

    population_distance_cluster = np.copy(children_distance_cluster)
    population_cluster_count = np.copy(children_cluster_count)
    population_centroids = np.copy(children_centroids)
    population_sum_dist = np.copy(children_sum_dist)
    population_av_count = np.copy(children_av_count)
    population_infeasibility = np.copy(children_infeasibility)

    #########################################################################

    # print("GEn+1")
    # Problema av count
    generations += 1
    if evaluations > ev_l:
        ev_l = evaluations+ 100
        d = sorted(population_objetive_value)
        print("It: ",evaluations, "   ",d, "   ", ev_l)

    # print("Evaluacion: ", evaluations, "Best Objetivo: ", np.min(population_objetive_value), "Media Objetivo: ", np.mean(population_objetive_value))
    f.write(str(evaluations) + "," + str(np.min(population_objetive_value))+ ","+ str(np.mean(mean_deviation))+ ","+
            str(np.mean(children_infeasibility))+","+str(np.mean(children_objetive_value))+","+str(np.mean(population_objetive_value)))
    f.write('\n')
    if generations == interval_local_search:
        generations = 0
        # print("BBBBBBB",evaluations)

        if mode == "best":
            n = round(n_population * memetic_subset)
            population_selected = np.argpartition(children_objetive_value, n)[:n]
        elif mode == "random":
            n = round(n_population * memetic_subset)
            population_selected = np.random.choice(range(n_population), n, replace=False)
        else:
            population_selected = range(n_population)
        for x in population_selected:
            gen_selected = np.random.choice(range(n_instances), n_instances, replace=False)
            w = 0
            n_errors = 0
            while w < len(gen_selected) and n_errors < limit_errors:
                changed = False
                if population_av_count[x][population_cluster[x][w]] != 1:
                    evaluations += (k - 1)
                    population_cluster[x], population_distance_cluster[x], population_sum_values_clusters[x], \
                        population_centroids[x],population_sum_dist[x], population_av_count[x], population_infeasibility[x], \
                        population_objetive_value[x], changed = \
                                        soft_local_seach(gen_selected[w], data_arr, population_cluster[x], population_distance_cluster[x],
                                         population_sum_values_clusters[x], population_centroids[x], population_sum_dist[x],
                                         population_av_count[x], population_infeasibility[x], population_objetive_value[x])
                    f.write(str(evaluations) + "," + str(np.min(population_objetive_value)) + "," + str(
                        np.mean(mean_deviation)) + "," +
                            str(np.mean(children_infeasibility)) + "," + str(
                        np.mean(children_objetive_value)) + "," + str(np.mean(population_objetive_value)))
                    f.write('\n')

                w += 1
                if not changed:
                    n_errors += 1

        # exit()

elapsed_time2 = time.perf_counter() - finish_ini
f.write("Tiempo Consumido: "+ str(elapsed_time2))
print("Initializing: ", elapsed_time)
t = np.argmin(population_objetive_value)
f.write('\n')
f.write("Mejor: obj"+ str(population_objetive_value[t])+ "  inf:"+str(population_infeasibility[t])+ "  dev:"+str(np.mean(population_sum_dist[t]/population_av_count[t])))
print("Initializing: ", elapsed_time)
print("Calculating for 1000: ", elapsed_time2)













