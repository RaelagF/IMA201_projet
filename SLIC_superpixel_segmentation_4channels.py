# contributor: FANG Guyu, GU Yuanzhe

import numpy as np
import cv2
import math


def mixed_distance(X1, X2, S, m):
    # mixed distance for both color similarity and spatial proximity
    # X1, X2: two vectors of the form [lk, ak, bk, yk, xk]
    # S: length in the SLIC algorithm
    # m: allows to weigh the relative importance between color
    ri, gi, bi, rii, yi, xi = X1
    rj, gj, bj, rij, yj, xj = X2

    dc_square = (rj-ri)**2 + (gj-gi)**2 + (bj-bi)**2 + (rij-rii)**2
    ds_square = (xj-xi)**2 + (yj-yi)**2

    distance = math.sqrt(dc_square + ds_square * (m**2) / (S**2))
    return distance


def SLIC_4channels(filenameR, filenameG, filenameB, filenameIR, k, m, threshold=0.1):
    # k: number of clusters
    # m: allows to weigh the relative importance between color
    # similarity and spatial proximity
    # threshold: stop the iterations when the error_improvement <= threshold
    # returns a matrix for clustering labels and the number of clusters

    imR = cv2.imread(filenameR)[:,:,0]
    imG = cv2.imread(filenameG)[:,:,0]
    imB = cv2.imread(filenameB)[:,:,0]
    imIR = cv2.imread(filenameIR)[:,:,0]
    im = np.stack((imR, imG, imB, imIR), axis = 2)

    height, weight = imR.shape
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
            C = list(im[j, i])  # each cluster center
            C.extend([j, i])
            Ck.append(C)
    Ck = np.array(Ck)

    # move cluster centers to the lowest gradient position in a 3 * 3 neighborhood
    im_GRAY = np.average(im, axis = 2)
    # laplacian = cv2.Laplacian(im_GRAY, cv2.CV_64F)
    gradient = cv2.Sobel(im_GRAY, ddepth=-1, dx=1, dy=1)
    print (gradient.shape)
    for C in Ck:
        y, x = C[4:6]
        min_gradient = gradient[y, x]
        for j in range(3):
            for i in range(3):
                if (gradient[y-1+j, x-1+i] < min_gradient):
                    min_gradient = gradient[y-1+j, x-1+i]
                    C[4:6] = y-1+i, x-1+j
        C[:4] = im[C[4], C[5]]

    # initialize the label and the distance matrix
    l = (-1) * np.ones((height, weight)).astype(np.int32)  # label
    INF = 1e9
    d = INF * np.ones((height, weight))  # distance

    # begin the iterations
    global_error = 1e20
    error_improvement = 1
    iter_cnt = 0
    while (error_improvement > threshold):
        print(iter_cnt)
        for (cluster_label, C) in enumerate(Ck):
            y, x = C[4:6]
            for j in range(max(0, y-S), min(height, y+S)):
                for i in range(max(0, x-S), min(weight, x+S)):
                    tmp_vector = np.concatenate((im[j, i], [j, i]))
                    D = mixed_distance(C, tmp_vector, S, m)
                    if (D < d[j, i]):
                        d[j, i] = D
                        l[j, i] = cluster_label

        # update
        # compute new cluster centers and residual error E
        E = 0
        for (cluster_label, C) in enumerate(Ck):
            cluster = []
            for j in range(height):
                for i in range(weight):
                    if (l[j, i] == cluster_label):
                        tmp_vector = np.concatenate((im[j, i], [j, i]))
                        cluster.append(tmp_vector)
            new_C = np.average(cluster, axis=0)
            E = E + np.linalg.norm(C - new_C, ord=2)
            #print (E)
            C[:] = new_C

        # update error_imrovement to decide whether to stop
        error_improvement = abs(global_error - E) / global_error
        global_error = E

        iter_cnt += 1
    # cv2.namedWindow("Image")
    # cv2.imshow("Image", laplacian)
    # cv2.waitKey (0)
    return l


def show_segmentation(filename, savename, l, show_im=False, white=False, color=[0, 0, 0]):

    # turn pixels on the bords to black
    height, weight = l.shape
    im = cv2.imread(filename)

    if white:
        im[:] = 255

    for j in range(1, height):
        for i in range(0, weight):
            if (l[j, i] != l[j-1, i]):
                im[j, i, :] = color

    for j in range(0, height):
        for i in range(1, weight):
            if (l[j, i] != l[j, i-1]):
                im[j, i, :] = color

    cv2.imwrite(savename, im)
    if show_im:
        cv2.namedWindow("Segmentation")
        cv2.imshow("Segmentation", im)
    cv2.waitKey(0)

    return im


l = SLIC_4channels("extraitR.tif", "extraitG.tif", "extraitB.tif", "extraitIR.tif", 50, 20)
show_segmentation("extraitBGR.tif", "tmp_4channels.tif", l)