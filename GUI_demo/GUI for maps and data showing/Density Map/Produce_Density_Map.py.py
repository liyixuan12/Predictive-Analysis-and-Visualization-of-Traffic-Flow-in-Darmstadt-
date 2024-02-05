import pandas as pd
import folium
from folium.plugins import HeatMap

# Read traffic data
file_path_traffic = '../../data/Gui/final_darmstadt_data.csv'
data_traffic = pd.read_csv(file_path_traffic)

# Read location data
file_path_locations = '../../data/Gui/processed_location_data.csv'
data_locations = pd.read_csv(file_path_locations)

# Merge traffic and location data, based on 'detid'
merged_data = pd.merge(data_traffic, data_locations, on='detid', how='inner')

# Select data at a specific time point, here select six points
selected_data = merged_data[merged_data['interval'] == 6 * 3600]

# Calculate traffic density
selected_data['density'] = selected_data['flow'] / selected_data['length']

# Create a map
m = folium.Map(location=[selected_data['lat'].mean(), selected_data['long'].mean()], zoom_start=14)  # 使用均值作为起始位置

# Add heat map
heat_data = [[point[0], point[1], weight] for point, weight in zip(selected_data[['lat', 'long']].values, selected_data['density'])]
HeatMap(heat_data).add_to(m)

# Add title
title_html = '''
             <h3 align="center" style="font-size:16px"><b>Traffic Density Map of Darmstadt</b></h3>
             '''
m.get_root().html.add_child(folium.Element(title_html))

# Save map as HTML file
m.save('density_map.html')
