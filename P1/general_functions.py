def same_cluster(df, p1, p2):
    return get_own_cluster(df, p1) == get_own_cluster(df, p2)

def get_own_cluster(df, index):
    return df['closest'].iloc[index,]

def calculate_distance_closest(df, centr):
    for i in range(len(df.index)):
        df.at[i, 'distance_closest'] = math.sqrt(sum([(a - b) ** 2 for a, b in zip(df.loc[i,col_names], centr[int(df.at[i, 'closest'])])]))

    return df



def row_infeasibility(df, row_index):
    infeasibility_points = 0
    r = restrictions.iloc[row_index]

    for c in range(r.size):
        if c != row_index:
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
# Return a numpy matrix with new centroids given the clusters
def update_centroids(df):
    act = df[col_names + ['closest']]
    return act.groupby('closest').mean().to_numpy()

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
