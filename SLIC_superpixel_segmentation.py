# contributor: guyu guyuan

import numpy as np
import cv2
import math


def mixed_distance(X1, X2, S, m):
    # mixed distance for both color similarity and spatial proximity
    # X1, X2: two vectors of the form [lk, ak, bk, yk, xk]
    # S: length in the SLIC algorithm
    # m: allows to weigh the relative importance between color
    li, ai, bi, yi, xi = X1
    lj, aj, bj, yj, xj = X2

    dc_square = (lj-li)**2 + (aj-ai)**2 + (bj-bi)**2
    ds_square = (xj-xi)**2 + (yj-yi)**2

    distance = math.sqrt(dc_square + ds_square * (m**2) / (S**2))
    return distance


def SLIC(filename, k, m, threshold=0.1):
    # k: number of clusters
    # m: allows to weigh the relative importance between color
    # similarity and spatial proximity
    # threshold: stop the iterations when the error_improvement <= threshold
    # returns a matrix for clustering labels and the number of clusters

    im = cv2.imread(filename)
    height, weight, num_channels = im.shape
    im_Lab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    N = height * weight
    S = int(math.sqrt(N/k))

    # initialize cluster centers Ck by sampling pixels at regular grid steps S
    h_ = round(height/S)  # number of blocks along the height
    w_ = round(weight/S)  # number of blocks along the weight
    k_ = h_ * w_  # number of clusters in the real case

    Ck = []  # Ck = k_ * [lk, ak, bk, yk, xk] * the positions of x and y have been inversed compared to the paper
    y_aux = (np.arange(0.5, h_, 1) * S).astype(np.int32)
    x_aux = (np.arange(0.5, w_, 1) * S).astype(np.int32)
    for j in y_aux:
        for i in x_aux:
            C = list(im_Lab[j, i])  # each cluster center
            C.extend([j, i])
            Ck.append(C)
    Ck = np.array(Ck)

    # move cluster centers to the lowest gradient position in a 3 * 3 neighborhood
    im_GRAY = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(im_GRAY, cv2.CV_64F)
    for C in Ck:
        y, x = C[3:5]
        min_gradient = laplacian[y, x]
        for j in range(3):
            for i in range(3):
                if (laplacian[y-1+j, x-1+i] < min_gradient):
                    min_gradient = laplacian[y-1+j, x-1+i]
                    C[3:5] = y-1+i, x-1+j
        C[:3] = im_Lab[C[3], C[4]]

    # initialize the label and the distance matrix
    l = (-1) * np.ones((height, weight)).astype(np.int32)  # label
    INF = 1e9
    d = INF * np.ones((height, weight))  # distance

    # begin the iterations
    global_error = 1e20
    error_improvement = 1
    iter_cmt = 0
    while (error_improvement > threshold):
        print(iter_cmt)
        for (cluster_label, C) in enumerate(Ck):
            y, x = C[3:5]
            for j in range(max(0, y-S), min(height, y+S)):
                for i in range(max(0, x-S), min(weight, x+S)):
                    tmp_vector = np.concatenate((im_Lab[j, i], [j, i]))
                    D = mixed_distance(C, tmp_vector, S, m)
                    if (D < d[j, i]):
                        d[j, i] = D
                        l[j, i] = cluster_label

        # updadate
        # compute new cluster centers and residual error E
        E = 0
        for (cluster_label, C) in enumerate(Ck):
            cluster = []
            for j in range(height):
                for i in range(weight):
                    if (l[j, i] == cluster_label):
                        tmp_vector = np.concatenate((im_Lab[j, i], [j, i]))
                        cluster.append(tmp_vector)
            new_C = np.average(cluster, axis=0)
            E = E + np.linalg.norm(C - new_C, ord=2)
            #print (E)
            C[:] = new_C

        # update error_imrovement to decide whether to stop
        error_improvement = abs(global_error - E) / global_error
        global_error = E

        iter_cmt += 1
    # cv2.namedWindow("Image")
    # cv2.imshow("Image", laplacian)
    # cv2.waitKey (0)
    return l


def show_segmentation(filename, k, m, threshold=0.05, show_im=False):
    # label matrix
    l = SLIC(filename, k, m, threshold)

    # turn pixels on the bords to black
    height, weight = l.shape
    im = cv2.imread(filename)

    for j in range(1, height):
        for i in range(0, weight):
            if (l[j, i] != l[j-1, i]):
                im[j, i, :] = [0, 0, 0]

    for j in range(0, height):
        for i in range(1, weight):
            if (l[j, i] != l[j, i-1]):
                im[j, i, :] = [0, 0, 0]

    cv2.imwrite('slic_'+str(k)+'_'+str(m)+'_'+str(threshold)+'_'+filename,im)
    if show_im:
        cv2.namedWindow("Segmentation")
        cv2.imshow("Segmentation", im)
    cv2.waitKey(0)
    
    return l,im

#%% class graph
class graph:
    def __init__(self):
        
        #dict of graph index
        self.dic_content = {}
        #dict of graph neighbour
        self.dic_neigh = {}
        
    #a fonction add index to index dict
    def add_content(self, position, index):
        if index in self.dic_content:
            self.dic_content[index].append(position)
        else:
            self.dic_content[index]=[position]
            self.dic_neigh[index]=[]
            
    #a fonction add neighbour to neigh dict
    def add_neigh(self, index1, index2):
        
        if index1 == -1 or index2 == -1 or index1 == index2:
            return
        
        if index1 not in self.dic_neigh[index2]:
            self.dic_neigh[index2].append(index1)
        if index2 not in self.dic_neigh[index1]:
            self.dic_neigh[index1].append(index2)
            
    #a fonction combine connexe composants with different index
    def combine_index(self, index1, index2):
        
        index_min, index_max = min(index1, index2), max(index1, index2)
        
        self.dic_content[index_min].extend(self.dic_content.pop(index_max))
        
        res = self.dic_neigh.pop(index_max)      
        for i in res:
            self.add_neigh(i, index_min)
            self.dic_neigh[i].remove(index_max)

    def generate_graph(self,slic):
        
        height, weight = slic.shape
        
        #label matrix
        label = (-1) * np.ones((height, weight)).astype(np.int32)  # label
        
        #a global variable which determiners the graph index
        index=1
        
        for i in range(height):
            for j in range(weight):
                #check the point above
                if label[i][j-1] != -1 and slic[i][j-1] == slic[i][j]:
                    res = label[i][j-1]
                #check the point left
                elif label[i-1][j] != -1 and slic[i-1][j] == slic[i][j]:
                    res = label[i-1][j]
                #new index
                else:
                    res = index
                    index += 1
                
                self.add_content([i,j],res)
                label[i][j]=res
                self.add_neigh(res,label[i][j-1])
                self.add_neigh(res,label[i-1][j])
                
                if slic[i][j-1] == slic[i][j] and slic[i-1][j] == slic[i][j] and label[i][j-1] != label[i-1][j]:
                    index_min, index_max = min(label[i][j-1], label[i-1][j]), max(label[i][j-1], label[i-1][j])
                    if index_min != -1 and index_min != index_max:
                        for point in self.dic_content[index_max]:
                            label[point[0]][point[1]] = index_min
                        self.combine_index(index_max, index_min)
        return label
    
    def sum_of_element(self):
        l = {}
        for j in self.dic_content:
            l[j] = len(self.dic_content[j])
        return l
    
#%% test part
# np.set_printoptions(threshold=np.inf)
# print(SLIC("luangai.jpg", 100, 10))
slic,im = show_segmentation("lena_petit.tif", k=100, m=30)
#a = np.array([[1,0,1,0,2],[1,0,1,0,2],[1,1,1,2,2]])
slic_graph = graph()
im_graph = slic_graph.generate_graph(slic)
#im_graph = slic_graph.generate_graph(a)
