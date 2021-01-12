#################################################################################
####################### Import python packages ##################################
#################################################################################
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt

def create_RFMScore(df):
    ############################################################
    # We keep only some variable for the segmentation
    ############################################################
    colonnes_RFM = ['DocIDHash','ID', 'LodgingRevenue','OtherRevenue','DaysSinceLastStay']
    df_rfm = df[colonnes_RFM]

    ############################################################
    # Create Frequency variable
    ############################################################
    rfm = df_rfm.groupby(['DocIDHash'])[['ID']].nunique().reset_index().rename(columns={'ID':'frequency'})

    ############################################################
    # Create Recency variable
    ############################################################
    df_rfm.rename(columns={'DaysSinceLastStay':'recency'}, inplace=True)
    rfm = pd.merge(rfm, df_rfm.groupby(['DocIDHash'])[['recency']].min().reset_index(), how='inner', on='DocIDHash')

    ############################################################
    # Create Monetary Value variable
    ############################################################
    df_rfm['monetary_value'] = df_rfm['LodgingRevenue'] + df_rfm['OtherRevenue']
    rfm = pd.merge(rfm, df_rfm.groupby(['DocIDHash'])[['monetary_value']].sum().reset_index(), how='inner',on='DocIDHash')

    ############################################################
    # Then we compute quantiles
    ############################################################
    quantiles = rfm.quantile(q=[0.20,0.40, 0.60,0.80])
    quantiles = quantiles.to_dict()

    ############################################################
    # This two function to cast the continues variables Recency,
    # Frequency and Monetary Value to discontinues variables
    ############################################################
    # Convert recency variable
    def RClass(x,p,d):
        if x <= d[p][0.2]:
            return 1
        elif x <= d[p][0.40]:
            return 2
        elif x <= d[p][0.60]:
            return 3
        elif x <= d[p][0.8]:
            return 4
        else:
            return 5

    # Convert Frequency and Monetary Value variables
    def FMClass(x,p,d):
        if x <= d[p][0.20]:
            return 5
        elif x <= d[p][0.40]:
            return 4
        elif x <= d[p][0.60]:
            return 3
        elif x <= d[p][0.80]:
            return 2
        else:
            return 1


    # Create New discontinue variables from continues ones
    rfm['R_Quartile'] = rfm['recency'].apply(RClass, args=('recency',quantiles,))
    rfm['F_Quartile'] = rfm['frequency'].apply(FMClass, args=('frequency',quantiles,))
    rfm['M_Quartile'] = rfm['monetary_value'].apply(FMClass, args=('monetary_value',quantiles,))
    ############################################################
    # Get The RFM Score
    ############################################################
    rfm['RFMScore'] = rfm['R_Quartile'].astype('str') + rfm['F_Quartile'].astype('str') + rfm['M_Quartile'].astype('str')
    return(rfm)