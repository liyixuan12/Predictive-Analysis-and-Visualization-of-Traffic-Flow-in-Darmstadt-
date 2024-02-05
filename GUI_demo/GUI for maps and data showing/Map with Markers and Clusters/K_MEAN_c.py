import folium
import pandas as pd
from sklearn.cluster import KMeans

# Read a CSV file containing latitude and longitude information
csv_file_path = '../../data/Darmstadt_Roads_Detid_Data.csv'
data = pd.read_csv(csv_file_path)

# Create map object
map_object = folium.Map(location=[data['lat'].mean(), data['long'].mean()], zoom_start=12)

# Add title
title_html = '''
             <h3 align="center" style="font-size:16px"><b>Darmstadt Map with Sensor Markers</b></h3>
             '''
map_object.get_root().html.add_child(folium.Element(title_html))

# Mark each road segment with a number
for index, row in data.iterrows():
    popup_text = f"cross ID: {row['detid']}"
    # Determine whether it is K1D43, if so, mark it in red
    if row['detid'] == 'K1D43':
        folium.Marker([row['lat'], row['long']], popup=folium.Popup(popup_text), icon=folium.Icon(color='red')).add_to(map_object)
    else:
        folium.Marker([row['lat'], row['long']], popup=folium.Popup(popup_text)).add_to(map_object)

# Clustering using K-Means
coordinates = data[['lat', 'long']]
kmeans = KMeans(n_clusters=12, n_init=10)  # Adjust the number of clusters
data['cluster'] = kmeans.fit_predict(coordinates)

# Add circles of clustering results to the map
for i, cluster_center in enumerate(kmeans.cluster_centers_):
    color = 'blue' if i % 3 == 0 else 'green' if i % 3 == 1 else 'purple'  # The color of each cluster circle alternates
    folium.CircleMarker(location=cluster_center, radius=100, color=color, fill=True, fill_opacity=0.2, opacity=0.5).add_to(map_object)

# Add Click for Lat/Lon plugin
map_object.add_child(folium.LatLngPopup())

# Add JavaScript code to display the last clicked coordinates in the lower right corner of the map
map_object.add_child(folium.Html('<div id="coords" style="position:fixed;bottom:10px;left:10px;background-color:white;padding:5px;"></div>',
                                script="""
                                    function show_coords(e) {
                                        document.getElementById('coords').innerHTML = 'Latitude: ' + e.latlng.lat.toFixed(5) + '<br>Longitude: ' + e.latlng.lng.toFixed(5);
                                    }
                                    map.on('click', show_coords);
                                    document.getElementById('coords').style.display = 'none';  // 初始时隐藏
                                """))

# Modify the JavaScript code and add it to the popup
popup_script = """
    <script>
        function openNewPage() {
            window.open("traffic_flow_analysis.html", "_blank");
        }

        document.querySelector(".leaflet-popup-content-wrapper").addEventListener("click", openNewPage);
    </script>
"""

map_object.get_root().html.add_child(folium.Element(popup_script))

# Save map as HTML file
map_object.save('map_with_markers_and_clusters_darmstadt.html')
