# contributor: FANG Guyu, GU Yuanzhe

import numpy as np
import cv2
import math
import pickle
import os
os.chdir("C:\\Users\\limit\\Desktop\\IMA_PROJET\\IMA201_PROJET")

# %% class graph


class graph:
    def __init__(self, filename=None):

        # dict of graph index
        self.dic_content = {}
        # dict of graph neighbour
        self.dic_neigh = {}

        self.filename = filename
        self.im = cv2.imread(filename)

    # a fonction add content to graph
    def add_content(self, position, index):
        # position: the position of the point
        # index: the index of the point
        if index in self.dic_content:
            self.dic_content[index].append(position)
        else:
            self.dic_content[index] = [position]
            self.dic_neigh[index] = []

    # a fonction built connnection between two index
    def add_neigh(self, index1, index2):
        # default
        if index1 == -1 or index2 == -1 or index1 == index2:
            return
        if index1 not in self.dic_content or index2 not in self.dic_content:
            print("Error: Index doesn't exists when adding neighbour")
            return
        # two-side
        if index1 not in self.dic_neigh[index2]:
            self.dic_neigh[index2].append(index1)
        if index2 not in self.dic_neigh[index1]:
            self.dic_neigh[index1].append(index2)

    # a fonction combine connexe composants of two indexs
    def combine_index(self, index1, index2):

        if index1 not in self.dic_content or index2 not in self.dic_content:
            print("Error: Index doesn't exists when combining")
            return

        index_min, index_max = min(index1, index2), max(index1, index2)

        self.dic_content[index_min].extend(self.dic_content.pop(index_max))

        res = self.dic_neigh.pop(index_max)
        for i in res:
            self.add_neigh(i, index_min)
            self.dic_neigh[i].remove(index_max)

    def generate_graph(self, slic):

        height, weight = slic.shape

        # label matrix
        label = (-1) * np.ones((height, weight)).astype(np.int32)  # label

        # a global variable which determiners the graph index
        index = 1

        for i in range(height):
            for j in range(weight):
                # check the point above
                if label[i][j-1] != -1 and slic[i][j-1] == slic[i][j]:
                    res = label[i][j-1]
                # check the point left
                elif label[i-1][j] != -1 and slic[i-1][j] == slic[i][j]:
                    res = label[i-1][j]
                # new index
                else:
                    res = index
                    index += 1

                # add the point to the graph
                self.add_content([i, j], res)
                label[i][j] = res
                self.add_neigh(res, label[i][j-1])
                self.add_neigh(res, label[i-1][j])

                if slic[i][j-1] == slic[i][j] and slic[i-1][j] == slic[i][j] and label[i][j-1] != label[i-1][j]:
                    # if two composants can combine
                    index_min, index_max = min(
                        label[i][j-1], label[i-1][j]), max(label[i][j-1], label[i-1][j])
                    if index_min != -1 and index_min != index_max:
                        for point in self.dic_content[index_max]:
                            label[point[0]][point[1]] = index_min
                        self.combine_index(index_max, index_min)
        return label

    def translate_2_label_matrix(self):
        height, weight, _ = self.im.shape
        mat = np.ones((height, weight)).astype(np.int32)
        for i in self.dic_content:
            for j in self.dic_content[i]:
                mat[j[0], j[1]] = i
        return mat

    def count_of_element(self):
        l = {}
        for j in self.dic_content:
            l[j] = len(self.dic_content[j])
        return l

    def index_distance(self, index1, index2):
        center1 = np.average(self.dic_content[index1], axis=0)
        center2 = np.average(self.dic_content[index2], axis=0)
        return np.linalg.norm(center1-center2)

    def index_mixed_distance(self, index1, index2, factor):
        im_Lab = cv2.cvtColor(self.im, cv2.COLOR_BGR2Lab)
        X1f = np.average(
            list(map(lambda x: im_Lab[x[0], x[1]], self.dic_content[index1])), axis=0)
        X1r = np.average(self.dic_content[index1], axis=0)
        X1 = np.concatenate((X1f, X1r))
        X2f = np.average(
            list(map(lambda x: im_Lab[x[0], x[1]], self.dic_content[index2])), axis=0)
        X2r = np.average(self.dic_content[index2], axis=0)
        X2 = np.concatenate((X2f, X2r))
        return mixed_distance(X1, X2, 1, factor)

    def graph_save(self, savename):
        f = open(savename, 'wb')
        pickle.dump([self.dic_content, self.dic_neigh,
                     self.filename, self.im], f)
        f.close()

    def graph_load(self, filename):
        f = open(filename, 'rb')
        [self.dic_content, self.dic_neigh,
            self.filename, self.im] = pickle.load(f)
        f.close()
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


# %% test slic
# np.set_printoptions(threshold=np.inf)
# print(SLIC("luangai.jpg", 100, 10))
k = 100
m = 30
threshold = 0.1
filename = "test.png"
slic = SLIC(filename, k=k, m=m, threshold=threshold)
np.save(filename[:-3]+'npy', slic)
savename = filename[:-4]+'_'+'slic_' + \
    str(k)+'_'+str(m)+'_'+str(threshold)+filename[-4:]
im = show_segmentation(filename, savename, slic, show_im=True)
# %% graph
'''
test graph
#a = np.array([[1,0,1,0,2],[1,0,1,0,2],[1,1,1,2,2]])
filename = "girl.jpg"
slic = np.load(filename[:-3]+'npy')
slic_graph = graph()
im_graph = slic_graph.generate_graph(slic)
show_segmentation(filename, filename[:-4]+'_graph'+filename[-4:], im_graph, show_im=True)
#im_graph = slic_graph.generate_graph(a)
'''
filename = "test.png"
slic = np.load(filename[:-3]+'npy')
slic_graph = graph(filename)
im_graph = slic_graph.generate_graph(slic)
slic_graph.graph_save(filename[:-3]+'pkl')
# %% test post-processing
# simple_processing(slic_graph)
new_graph = graph()
new_graph.graph_load(filename[:-3]+'pkl')
distance_based_processing(new_graph)
res = new_graph.translate_2_label_matrix()
im_seg = show_segmentation(
    filename, filename[:-4]+'_post_processing'+filename[-4:], res, show_im=True)
im_seg = show_segmentation(
    filename, filename[:-4]+'_post_processing_white'+filename[-4:], res, show_im=True, white=True)
# %% test gathering
for i in range(10, 55, 5):
    gathering(new_graph, i)
    res = new_graph.translate_2_label_matrix()
    im_seg = show_segmentation(filename, filename[:-4]+'_gathering_' + str(
        i) + filename[-4:], res, show_im=False, color=[255, 255, 255])
    im_seg = show_segmentation(filename, filename[:-4]+'_gathering_white_' + str(
        i) + filename[-4:], res, show_im=False, white=True)
