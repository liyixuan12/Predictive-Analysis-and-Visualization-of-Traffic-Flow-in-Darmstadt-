import folium

#Create map object
map_object = folium.Map(location=[49.8728, 8.6512], zoom_start=12)
# Use the default latitude and longitude coordinates, you can modify them as needed

# Add a title to the map
title_html = """
             <h3 align="center" style="font-size:16px"><b>Darmstadt City Map</b></h3>
             """
map_object.get_root().html.add_child(folium.Element(title_html))

# Save map as HTML file
map_object.save('Darmstadt_City_Map.html')
