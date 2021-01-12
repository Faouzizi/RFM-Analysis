#################################################################################
####################### Import python packages ##################################
#################################################################################
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import RFMScore
import Aggregating_RFMScore_CAH

#################################################################################
#######################       Import data      ##################################
#################################################################################
df = pd.read_excel('./data/HotelCustomersDataset.xlsx')

#################################################################################
#######################       Univariate analysis      ##########################
#################################################################################
profile = ProfileReport(df, title="Pandas Profiling Report")
profile.to_file("./results/univariate_report.html")


rfm = RFMScore.create_RFMScore(df)

# Get the number of customers for each segment
rfm.groupby(['RFMScore'])['DocIDHash'].nunique().reset_index().rename(columns={'DocIDHash':'Nb Customers'})

################################################################
# Create segments by aggregating RFM SCores
################################################################
# First we download the segments names by RFM Score
segments_names = pd.read_csv('./results/segments.csv')
rfm = pd.merge(rfm, segments_names, how='inner', left_on='RFMScore', right_on='segments').drop('segments', axis=1)

# Get the number of customers for each segment after aggregation
rfm.groupby(['segment_name'])['DocIDHash'].nunique().reset_index().rename(columns={'DocIDHash':'Nb Customers'})


################################################################
# Create sub-classes from RFM SCores using CAH
################################################################
rfm_by_CAH = Aggregating_RFMScore_CAH.RFM_segmentation_using_CAH(rfm)
