import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
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
    mode = "local"
    dataset_name = "ecoli"
    restr_level = 10
    seed_asigned = 456
    lambda_var = 1
else:
    print("Wrong number of arguments.")
    exit(1)
#############################
inf_list1 = []
it_list1 = []
inf_list2 = []
it_list2 = []

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
                d1 = np.linalg.norm(df[i][:n_characteristics] - centr[own])
                d2 = np.linalg.norm(df[i][:n_characteristics] - centr[cluster_id])
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
                d1 = np.linalg.norm(df[i][:n_characteristics] - centr[own])
                d2 = np.linalg.norm(df[i][:n_characteristics] - centr[cluster_id])
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
###################################################################
# LOCAL
# #################################################################
def prueba():
    global data, infeasibility_cluster_index, restrictions_numpy, k, data_path, dataset_name, restrictions_path, restrictions, col_names, n_characteristics, distance_cluster_index, cluster_index, minim, maxim, seed, idx
    if dataset_name == "iris":
        k = 3
        data_path = "./iris_set.dat"
        restrictions_path = "./iris_set_const_" + str(restr_level) + ".const"
    elif dataset_name == "ecoli":
        k = 8
        data_path = "./ecoli_set.dat"
        restrictions_path = "./ecoli_set_const_" + str(restr_level) + ".const"
    else:
        k = 3
        data_path = "./rand_set.dat"
        restrictions_path = "./rand_set_const_" + str(restr_level) + ".const"

    colmap = ['r', 'g', 'b']
    data = pd.read_csv(data_path, header=None)
    restrictions = pd.read_csv(restrictions_path, header=None)
    col_names = []
    # The name of the df's columns will be 'cX' being X the column's number
    for i in range(len(data.columns)):
        col_names.append("c" + str(i))
    data.columns = col_names
    n_characteristics = len(col_names)
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

    inf_list1 = []
    it_list1 = []
    if mode == "local":
        # Start timing
        start_time = time.perf_counter()

        D = squareform(pdist(data))
        max_distance, [I_row, I_col] = np.nanmax(D), np.unravel_index(np.argmax(D), D.shape)
        n_restrictions = (((len(restrictions.index) ** 2) - (restrictions.isin([0]).sum().sum())) / 2) - data.shape[0]
        lambda_value = (max_distance / n_restrictions) * lambda_var

        # Generate neighbourhood
        possible_changes = []
        for i in range(len(data.index)):
            for w in range(k):
                possible_changes.append((i, w))
        np.random.shuffle(possible_changes)

        # Generate initial solution
        data['cluster'] = np.random.randint(0, k, data.shape[0])

        # Count how many points in each cluster
        cluster_count = np.array(data[['c0'] + ['cluster']].groupby('cluster').count())

        # Exit if initial solution doesn't have at least a point in each cluster
        if len(cluster_count) != k:
            exit(1)

        # Transform it into a numpy array
        restrictions_numpy = np.asarray(restrictions)

        # Create necessary columns
        data['distance_cluster'] = np.nan
        data = np.asarray(data)

        centroids = update_centroids_numpy(data)
        data, sum_dist, av_count = calculate_distance_cluster_numpy(data, centroids)

        n_iterations = 0

        # Calculate initial infeasibility
        total_infeasibility = infeasibility_numpy(data)

        # Initialize variables
        objective_value = np.mean(sum_dist/av_count) + lambda_value * total_infeasibility
        repeated = False
        sum_values_clusters = sum_instances(data)

        while n_iterations < 100000 and not repeated:
            # Get first neighbour to be able to compare it later on
            first_neigh = possible_changes[0]

            # Save old values
            old_objective_value = objective_value
            old_infeasibility = total_infeasibility
            # number_cluster = count_each_cluster(data)
            first_iteration = True
            while old_objective_value <= objective_value and n_iterations < 100000:
                n_iterations += 1

                it_list1.append(n_iterations)
                inf_list1.append(total_infeasibility)

                possible_changes, neigh = get_neightbour(possible_changes)
                if neigh == first_neigh and not first_iteration:
                    repeated = True
                    # Break if we already got every possible neighbour
                    break

                first_iteration = False

                # Save the original cluster from the point we are going to change
                old_cluster = int(data[neigh[0]][cluster_index])
                p_index = neigh[0]
                new_cluster = neigh[1]

                # Skip if the cluster only have 1 element
                if av_count[old_cluster] == 1 or old_cluster == new_cluster:
                    continue

                total_infeasibility = total_infeasibility - row_infeasibility_numpy(data, p_index)
                av_count[old_cluster] -= 1

                data[p_index][cluster_index] = new_cluster

                total_infeasibility = total_infeasibility + row_infeasibility_numpy(data, p_index)
                av_count[new_cluster] += 1



                centroids, sum_values_clusters = update_centroids_optimized(data, centroids, sum_values_clusters, p_index, old_cluster, new_cluster, av_count)
                # Calculate new average
                # centroids = update_centroids_numpy(data)




                data, sum_dist, old_distance = update_distance(data, centroids, sum_dist, p_index, old_cluster, new_cluster)
                # data, sum_dist, av_count = calculate_distance_cluster_numpy(data, centroids)

                # Calculate new objective value
                objective_value = np.mean(sum_dist/av_count) + lambda_value * total_infeasibility

                # Restore values
                if old_objective_value <= objective_value:
                    data[p_index][cluster_index] = old_cluster
                    total_infeasibility = old_infeasibility
                    av_count[old_cluster] += 1
                    av_count[new_cluster] -= 1
                    data, sum_dist = undo_distance(data, sum_dist, p_index, old_cluster, new_cluster, old_distance)
                    centroids, sum_values_clusters = update_centroids_optimized(data, centroids, sum_values_clusters,
                                                                                p_index, new_cluster, old_cluster, av_count)


        #Finish timing
        elapsed_time = time.perf_counter() - start_time

        centroids = update_centroids_numpy(data)
        data, sum_dist, av_count = calculate_distance_cluster_numpy(data, centroids)
        total_infeasibility = infeasibility_numpy(data)
        objective_value = np.mean(sum_dist / av_count) + lambda_value * total_infeasibility

        # print("For lambda var:", lambda_var)
        print("Tasa C:", np.mean(sum_dist/av_count))
        # print("Iter:", n_iterations)
        print("Tasa Inf:", total_infeasibility)
        print("Agr:", objective_value)
        print("Time:", elapsed_time)
        print("NumIter:", n_iterations)
        # print("aaaaaaaa", total_infeasibility, "sdasd",inf_list1[-1])
        return it_list1, inf_list1, objective_value, np.mean(sum_dist/av_count)

        # print(elapsed_time)

        # print(count_each_cluster(data))

    ###################################################################
    # GREEDY
    # #################################################################
    elif mode == "greedy":
        # Start timing
        start_time = time.perf_counter()

        # Generate random centroids
        centroids = []
        for i in range(k):
            centroids.append(random.uniform(minim, maxim) / 1.5)

        centroids = np.array(centroids)
        data['cluster'] = np.nan
        data['distance_cluster'] = np.nan
        for i in range(k):
            data['infeasility_cluster_{}'.format(i)] = np.nan

        infeasibility_cluster_index = distance_cluster_index + 1
        restrictions_numpy = np.asarray(restrictions)
        data = np.asarray(data)

        data = first_assignation_numpy(data, centroids)
        data, sum_dist, av_count = calculate_distance_cluster_numpy(data, centroids)

        data_old = data
        old_clusters = np.zeros(0)
        n_iterations = 0
        while (not np.array_equal(old_clusters, data[:, cluster_index])):
            it_list1.append(n_iterations)
            inf_list1.append(infeasibility_numpy(data))
            n_iterations += 1
            old_clusters = np.copy(data[:, cluster_index])
            centroids = update_centroids_numpy(data)
            data = regular_assignation_numpy(data, centroids)

        #Finish timing
        elapsed_time = time.perf_counter() - start_time

        centroids = update_centroids_numpy(data)
        data, sum_dist, av_count = calculate_distance_cluster_numpy(data, centroids)

        print("Tasa C:", np.mean(sum_dist/av_count))
        print("Tasa Inf:", infeasibility_numpy(data))
        print("Time:", elapsed_time)
        print("NumIter:", n_iterations)
        return it_list1, inf_list1


################RTRYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
l_values = [.5,2,10]

obj_values = []
lam_values =[]
dist_values=[]
inf_values=[]
# for l in l_values:
for l in range(10, 101, 5):
    lambda_var = l/10.0
    lam_values.append(lambda_var)
    it_list1, inf_list1, obj, desv= prueba()
    obj_values.append(obj)
    dist_values.append(desv)
    inf_values.append(inf_list1[-1])
    # p = plt.plot(it_list1, inf_list1, label=str("L: "+str(l)) )
    # plt.plot(it_list1[-1], inf_list1[-1], 'ro', color=p[0].get_color())
fig, ax1 = plt.subplots()

ax1.plot(lam_values, obj_values, label="Objective")
ax1.plot(lam_values, dist_values, label="Deviation")
ax1.set_xlabel("Lambda")
ax1.set_ylabel("Objective/Deviation")
ax1.set_xlim([.5, 11])
plt.title(mode.capitalize()+" "+dataset_name.capitalize()+" "+str(restr_level)+"%")
plt.legend()

ax2 = ax1.twinx()
ax2.set_ylabel("Infeasibility")
ax2.plot(lam_values, inf_values, label="Infeasibility", color="red")



plt.legend(loc=0)
plt.show()


