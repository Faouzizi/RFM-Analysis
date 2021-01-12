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



































################
# compute frequency
rfm = df_rfm.groupby(['DocIDHash'])[['ID']].nunique().reset_index().rename(columns={'ID':'frequency'})
rfm = pd.merge(rfm, df_rfm.groupby(['DocIDHash'])[['recency']].min().reset_index(), how='inner', on='DocIDHash')
df_rfm['monetary_value'] = df_rfm['LodgingRevenue'] + df_rfm['OtherRevenue']
rfm = pd.merge(rfm, df_rfm.groupby(['DocIDHash'])[['monetary_value']].sum().reset_index(), how='inner', on='DocIDHash')


################################################################
# compute Recency, Frequency, Monetary Value quantile
################################################################
quantiles = rfm.quantile(q=[0.20,0.40, 0.60,0.80])
quantiles = quantiles.to_dict()
# Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)
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

# Arguments (x = value, p = recency, monetary_value, frequency, k = quartiles dict)
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

rfm['R_Quartile'] = rfm['recency'].apply(RClass, args=('recency',quantiles,))
rfm['F_Quartile'] = rfm['frequency'].apply(FMClass, args=('frequency',quantiles,))
rfm['M_Quartile'] = rfm['monetary_value'].apply(FMClass, args=('monetary_value',quantiles,))

################################################################
# RFM Score
################################################################
rfm['RFMScore'] = rfm['R_Quartile'].astype('str') + rfm['F_Quartile'].astype('str')  + rfm['M_Quartile'].astype('str')


rfm.groupby(['RFMScore'])['DocIDHash'].nunique().reset_index().rename(columns={'DocIDHash':'Nb Customers'})

################################################################
# Create sub-classes from RFM SCores
################################################################
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
rfm_centroid = rfm.groupby('RFMScore')[['recency','frequency','monetary_value']].mean()
from scipy.spatial import Delaunay
tri = Delaunay(rfm[['recency','frequency','monetary_value']])
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
##générer la matrice des liens
Z = linkage(rfm_centroid,method='ward',metric='euclidean')
##affichage du dendrogramme
plt.title("CAH : Aggregate classes")
dendrogram(Z,labels=rfm_centroid.index,orientation='left',color_threshold=700)
plt.show()

plt.title("CAH : Aggregate classes")
dendrogram(Z,labels=rfm_centroid.index,orientation='left',color_threshold=900)
plt.show()

groupes_cah = fcluster(Z,t=700,criterion='distance')
idg = np.argsort(groupes_cah)
print(pd.DataFrame(rfm_centroid.index[idg],groupes_cah[idg]))
#matérialisation des 4 classes (hauteur t = 7)

plt.title('CAH : Aggregate classes')


dendrogram(Z,labels=fromage.index,orientation='left',color_threshold=7)
plt.show()

scores = pd.read_clipboard()
scores.to_csv('./results/scores_segments.csv', index=False)
listes = scores.Scores.map(lambda x: x.split(','))
res = pd.DataFrame()
for x in scores.iterrows():
    segment_name = x[1][0]
    segments = [x.strip() for x in x[1][1].split(',')]
    res = res.append(pd.DataFrame({'segment_name':np.repeat(segment_name, len(segments)), 'segments': segments}))



rfm = pd.merge(rfm, res, how='inner', left_on='RFMScore', right_on='segments')
rfm.to_excel('./results/rfm_results.xlsx')


df.iloc[0]
# Age
profiling = pd.merge(rfm, df, how='inner', on='DocIDHash')
q1 = profiling['Age'].quantile([0.25]).values[0]
q3 = profiling['Age'].quantile([0.75]).values[0]
IC_valeur_non_aberantes = [q1 - 2*(q3-q1), q3 + 2*(q3-q1)]
profiling.loc[((profiling['Age']<IC_valeur_non_aberantes[0]) & (profiling.index<profiling.index[-1])) | ((profiling['Age']>IC_valeur_non_aberantes[1]) & (profiling.index<df.index[-1])), 'Age'] = profiling['Age'].mean()
return(df)
traiter_valeurs_extremes_continues(profiling, 'Age')


def imputation_statique(df, statique):
    ###############################################################
    # Cette fonction vous permettra d'imputer les données manquantes
    # Si statique=True alors l'imputation se fera par la median ou le mode
    # selon le type des données en entrée
    ###############################################################
    missing_data = df.apply(lambda x: np.round(x.isnull().value_counts()*100.0/len(x),2)).iloc[0]
    columns_MissingData = missing_data[missing_data<100].index
    if imputation_statique:
        for col in columns_MissingData:
            if df[col].dtype=='O':
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            else:
                df[col] = df[col].fillna(df[col].median())
    else:
        imputer = KNNImputer(n_neighbors=3)
        ids = df.CustomerID
        X = pd.concat([pd.get_dummies(df.drop('CustomerID', axis=1).select_dtypes('O')), df.drop('CustomerID', axis=1).select_dtypes(exclude='O')], axis=1)
        X_filled_knn = pd.DataFrame(imputer.fit_transform(X))
        X_filled_knn.columns = X.columns
        for col in columns_MissingData:
            print(col)
            if df[col].dtypes=='O':
                df_temp =X_filled_knn.filter(regex='^'+col+'*')
                df_temp.columns = [x.replace(col+'_', '') for x in df_temp.columns]
                df[col] = df_temp.idxmax(1)
            else:
                df[col] = np.round(X_filled_knn[col],2)
    return(df)
profiling = imputation_statique(profiling, False)
profiling.groupby('segment_name')['Age'].mean()
profiling.groupby('segment_name')['Age'].mean()
# Nationality
profiling.groupby(['segment_name', 'Nationality'])['DocIDHash'].nunique()
# LodgingRevenue
profiling.groupby(['segment_name'])['LodgingRevenue'].mean()
# OtherRevenue
# AverageLeadTime
# BookingsCanceled
# BookingsNoShowed
# BookingsCheckedIn
# PersonsNights
# RoomNights
# DistributionChannel

