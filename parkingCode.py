import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import seaborn as sns
import numpy as np 
import folium as fl

#Filtering Dataset to only include DC MPD/Department of Transportation/Department of Public Works

parkingCSV = pd.read_csv('DCViolationsMarch24.csv')
dcParkingAgencyNumbers = [1, 2, 3, 4, 5, 6, 7, 10, 13, 14, 15, 21, 25, 32]

parkingCSV = parkingCSV[parkingCSV['ISSUING_AGENCY_CODE'].isin(dcParkingAgencyNumbers)]
parkingCSV = parkingCSV[["TICKET_NUMBER", "ISSUE_DATE", "ISSUE_TIME", "ISSUING_AGENCY_NAME", "ISSUING_AGENCY_CODE", "VIOLATION_PROC_DESC"
                         ,"LOCATION", "LATITUDE", "LONGITUDE"]]

parkingCSV['ISSUE_DATE'] = parkingCSV['ISSUE_DATE'].str.split().str[0]
parkingCSV['ISSUE_TIME'] = parkingCSV['ISSUE_TIME'].astype(str).str.zfill(4)
parkingCSV['ISSUE_TIME'] = pd.to_datetime(parkingCSV['ISSUE_TIME'], format='%H%M').dt.strftime('%H:%M:%S')

#Calculate Unix Time for clustering by date and time
parkingCSV['Time in Unix'] = pd.to_datetime(parkingCSV['ISSUE_DATE'].astype(str) + " " + parkingCSV['ISSUE_TIME'].astype(str))
parkingCSV['Time in Unix'] = parkingCSV['Time in Unix'].values.astype(np.int64) // 10 ** 9


#Clustering by Agency Code and Time issued
x = parkingCSV[['ISSUING_AGENCY_CODE']]
x['Hour of Day'] = pd.to_datetime(parkingCSV['ISSUE_TIME']).dt.hour
kmeans = KMeans(n_clusters=4)
yPred = kmeans.fit_predict(x)
x['Cluster'] = yPred

print(kmeans.cluster_centers_)
df1 = x[x['Cluster'] == 0]
df2 = x[x['Cluster'] == 1]
df3 = x[x['Cluster'] == 2]
df4 = x[x['Cluster'] == 3]

plt.scatter(df1['Hour of Day'],df1[['ISSUING_AGENCY_CODE']], color = 'green')
plt.scatter(df2['Hour of Day'],df2[['ISSUING_AGENCY_CODE']], color = 'red')
plt.scatter(df3['Hour of Day'],df3[['ISSUING_AGENCY_CODE']], color = 'black')
plt.scatter(df4['Hour of Day'],df4[['ISSUING_AGENCY_CODE']], color = 'blue')

plt.xlabel("Hour of Day")
plt.ylabel('Issuing Agency Code')
plt.legend()
plt.show()

print(x['Cluster'].value_counts())

#Clustering based off of Time and Coordinates
y = parkingCSV[['LATITUDE', 'LONGITUDE']].dropna()
kmean = KMeans(init ="random", n_clusters=8, n_init=10)
kmean.fit(y)
label = kmean.labels_

y['Cluster'] = label

_clusters = y.groupby('Cluster').size()

#Visualizing on Map
colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', \
     'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', \
     'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray', \
     'black', 'lightgray', 'red', 'blue', 'green', 'purple', \
     'orange', 'darkred', 'lightred', 'beige', 'darkblue', \
     'darkgreen', 'cadetblue', 'darkpurple','pink', 'lightblue', \
     'lightgreen', 'gray', 'black', 'lightgray' ]
lat = y.iloc[215]['LATITUDE']
lng = y.iloc[215]['LONGITUDE']
map = fl.Map(location=[lng, lat], zoom_start=12)

for _, row in y.iterrows():
    fl.CircleMarker(
        location=[row["LATITUDE"], row["LONGITUDE"]],
        radius=12, 
        weight=2, 
        fill=True, 
        fill_color=colors[int(row["Cluster"])],
        color=colors[int(row["Cluster"])]
    ).add_to(map)

map.save("DCParking.html")
