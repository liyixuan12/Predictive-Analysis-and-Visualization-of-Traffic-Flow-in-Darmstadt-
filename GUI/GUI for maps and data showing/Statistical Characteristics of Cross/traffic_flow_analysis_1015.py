import pandas as pd
import matplotlib.pyplot as plt

# Read data file
data_file_path = '../../data/Preprocessed/K1D43/K1D43_data.csv'
data = pd.read_csv(data_file_path)

# Convert date column to datetime type
data['day'] = pd.to_datetime(data['day'])

# Filter data for specific intersections and dates
selected_data = data[(data['detid'] == 'K1D43') & (data['day'] == '2015-10-15')]

# Calculate mean and variance
mean_flow = selected_data['flow'].mean()
var_flow = selected_data['flow'].var()

# Draw a traffic flow line chart
plt.figure(figsize=(10, 6))
plt.plot(selected_data['interval'], selected_data['flow'], label='Traffic Flow')
plt.axhline(mean_flow, color='red', linestyle='--', label=f'Mean Flow: {mean_flow:.2f}')
plt.title('Traffic Flow on 2015-10-15 at K1D43')
plt.xlabel('Interval (seconds)')
plt.ylabel('Traffic Flow')
plt.legend()
plt.grid(True)

# Save graphics as image files
plt.savefig('traffic_flow_plot.png')

# Generate HTML content
html_content = f'''
    <h2>Traffic Flow Analysis for K1D43 on 2015-10-15</h2>
    <p>Mean Traffic Flow: {mean_flow:.2f}</p>
    <p>Variance of Traffic Flow: {var_flow:.2f}</p>
    <img src="traffic_flow_plot.png" alt="Traffic Flow Plot">
'''

# Save HTML content to file
with open('traffic_flow_analysis.html', 'w') as html_file:
    html_file.write(html_content)

# display graphics
plt.show()
