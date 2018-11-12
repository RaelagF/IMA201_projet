# contributor: FANG Guyu, GU Yuanzhe
# %% Segmentation


def gathering(graph, threshold):
    flag = 1
    factor = 0
    while flag:
        flag = 0
        l = sorted(graph.dic_content, reverse=True)
        for i in l:
            k = min(
                graph.dic_neigh[i], key=lambda x: graph.index_mixed_distance(i, x, factor))
            print(graph.index_mixed_distance(i, k, factor))
            if graph.index_mixed_distance(i, k, factor) < threshold:
                graph.combine_index(i, k)
                flag = 1
