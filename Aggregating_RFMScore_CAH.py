#################################################################################
####################### Import python packages ##################################
#################################################################################
import pandas as pd
import RFMScore
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

def RFM_segmentation_using_CAH(rfm):
    #################################################################################
    # We compute the centroid of each RFMScore segment
    #################################################################################
    rfm_centroid = rfm.groupby('RFMScore')[['recency','frequency','monetary_value']].mean()

    #################################################################################
    #Generate Links matrix
    #################################################################################
    Z = linkage(rfm_centroid,method='ward',metric='euclidean')

    #################################################################################
    #Print dendrogram
    #################################################################################
    plt.title("CAH : Aggregated segments")
    dendrogram(Z,labels=rfm_centroid.index,orientation='left',color_threshold=2300)
    plt.show()

    #################################################################################
    # Get new segments aggregated
    #################################################################################
    groupes_cah = fcluster(Z,t=2300,criterion='distance')
    idg = np.argsort(groupes_cah)
    new_segments = pd.DataFrame({'RFMScore':rfm_centroid.index[idg],'segment_id':groupes_cah[idg]})

    rfm = pd.merge(rfm, new_segments, how='inner', on='RFMScore')
    rfm['segment_id'] = rfm.segment_id.map({1:'Big Spenders', 2:'Almost Lost', 3:'Promising', 4:'Best Customers'})
    return(rfm)