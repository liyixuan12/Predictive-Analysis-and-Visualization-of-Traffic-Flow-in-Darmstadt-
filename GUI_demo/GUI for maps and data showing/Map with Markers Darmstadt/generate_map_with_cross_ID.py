import folium
import pandas as pd

# Read a CSV file containing latitude and longitude information
#csv_file_path = 'from_detectors_public.csv_extract_only_darmstadt_data.csv'
csv_file_path = r'C:\Users\rensh\Desktop\CE Semester 3\Data Science 2\Summarized and organized documents files datas\GUI for maps and data showing\data\from_detectors_public.csv_extract_only_darmstadt_data.csv'



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

    # If it's K1D43, it's marked red
    if row['detid'] == 'K1D43':
        marker_color = 'red'
    else:
        marker_color = 'blue'

    # Create a marker and set the Popup to pop up when clicked
    marker = folium.Marker([row['lat'], row['long']], popup=folium.Popup(popup_text),
                           icon=folium.Icon(color=marker_color))

    # Add link in Popup
    popup_html = f'''
                 <a href="traffic_flow_analysis.html" target="_blank">Click for Traffic Flow Analysis</a>
                 '''
    marker.add_child(folium.Element(popup_html))

    marker.add_to(map_object)

# Add Click for Lat/Lon plugin
map_object.add_child(folium.LatLngPopup())

# Add JavaScript code to display the last clicked coordinates in the lower right corner of the map
map_object.add_child(folium.Html(
    '<div id="coords" style="position:fixed;bottom:10px;left:10px;background-color:white;padding:5px;"></div>',
    script="""
                                    function show_coords(e) {
                                        document.getElementById('coords').innerHTML = 'Latitude: ' + e.latlng.lat.toFixed(5) + '<br>Longitude: ' + e.latlng.lng.toFixed(5);
                                    }
                                    map.on('click', show_coords);
                                    document.getElementById('coords').style.display = 'none';  // 初始时隐藏
                                """))

# Save map as HTML file
map_object.save('map_with_markers_darmstadt_1.html')
