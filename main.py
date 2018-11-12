import numpy as np

import SLIC_superpixel_segmentation as SLIC
from graph import Graph
import post_processing
import fusion

def main():
    k = 100
    m = 30
    threshold = 0.1
    filename = "lena_petit.tif"
    slic = SLIC.SLIC(filename, k=k, m=m, threshold=threshold)
    np.save(filename[:-3]+'npy', slic)
    savename = filename[:-4]+'_'+'slic_' + \
        str(k)+'_'+str(m)+'_'+str(threshold)+filename[-4:]
    im = SLIC.show_segmentation(filename, savename, slic, show_im=True)

    slic = np.load(filename[:-3]+'npy')
    slic_graph = Graph(filename)
    im_graph = slic_graph.generate_graph(slic)
    slic_graph.graph_save(filename[:-3]+'pkl')

    # %% test post-processing
    new_graph = Graph()
    new_graph.graph_load(filename[:-3]+'pkl')
    post_processing.distance_based_processing(new_graph)
    res = new_graph.translate_2_label_matrix()
    im_seg = SLIC.show_segmentation(
        filename, filename[:-4]+'_post_processing'+filename[-4:], res, show_im=True)
    im_seg = SLIC.show_segmentation(
        filename, filename[:-4]+'_post_processing_white'+filename[-4:], res, show_im=True, white=True)

    # # %% test fusion
    for i in range(54, 55, 5):
        fusion.fusion(new_graph, i)
        res = new_graph.translate_2_label_matrix()
        im_seg = SLIC.show_segmentation(filename, filename[:-4]+'_fusion_' + str(
            i) + filename[-4:], res, show_im=False, color=[255, 255, 255])
        im_seg = SLIC.show_segmentation(filename, filename[:-4]+'_fusion_white_' + str(
            i) + filename[-4:], res, show_im=False, white=True)    



if __name__ == '__main__':
    main()