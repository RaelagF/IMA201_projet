# contributor: FANG Guyu, GU Yuanzhe
# %% post-processing


def simple_processing(graph, threshold=30):
    l = sorted(graph.dic_content, reverse=True)
    for i in l:
        if len(graph.dic_content[i]) < threshold:
            graph.combine_index(i, graph.dic_neigh[i][0])


def distance_based_processing(graph, threshold=30):
    l = sorted(graph.dic_content, reverse=True)
    for i in l:
        if len(graph.dic_content[i]) < threshold:
            k = min(graph.dic_neigh[i],
                    key=lambda x: graph.index_distance(i, x))
            graph.combine_index(i, k)
