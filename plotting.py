import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def convert_lat_lon_to_cartesian(latitude, longitude, altitude):
    # Radius of the Earth (assuming a spherical Earth)
    earth_radius = 6371  # in kilometers

    # Convert latitude and longitude to radians
    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude)

    # Convert spherical coordinates to Cartesian coordinates
    x = (earth_radius + altitude) * math.cos(lat_rad) * math.cos(lon_rad)
    y = (earth_radius + altitude) * math.cos(lat_rad) * math.sin(lon_rad)
    z = (earth_radius + altitude) * math.sin(lat_rad)

    return x, y, z
#%%
def plot_satellites_3d_interactive(satellite_data, earth_radius=6371, export_filename=None, row_index=None):
    # Create a sphere for the Earth surface
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_earth = earth_radius * np.outer(np.cos(u), np.sin(v))
    y_earth = earth_radius * np.outer(np.sin(u), np.sin(v))
    z_earth = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

    earth_trace = go.Surface(x=x_earth, y=y_earth, z=z_earth, colorscale='blues')

    satellite_traces = []
    if row_index is not None:
        nearby_satellites = satellite_data.iloc[row_index]['Nearby_Satellites']
        distances = satellite_data.iloc[row_index]['Distances']

        for satellite, distance in zip(nearby_satellites, distances):
            x, y, z = satellite['X-position'], satellite['Y-position'], satellite['Z-position']
            satellite_trace = go.Scatter3d(
                x=[x],
                y=[y],
                z=[z],
                mode='markers',
                marker=dict(size=8, symbol='circle', line=dict(width=2, color='black')),
                name=f'{satellite["Satellite ID"]} - Distance: {distance:.2f}'
            )
            satellite_traces.append(satellite_trace)
    else:
        for index, satellite in satellite_data.iterrows():
            x, y, z = satellite['X-position'], satellite['Y-position'], satellite['Z-position']
            satellite_trace = go.Scatter3d(
                x=[x],
                y=[y],
                z=[z],
                mode='markers',
                marker=dict(size=8, symbol='circle', line=dict(width=2, color='black')),
                name=f'{satellite["Satellite ID"]}'
            )
            satellite_traces.append(satellite_trace)

    layout = go.Layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )

    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'surface'}]])
    fig.add_trace(earth_trace)
    for satellite_trace in satellite_traces:
        fig.add_trace(satellite_trace)

    fig.update_layout(layout)
    if export_filename:
        fig.write_html(export_filename)
        print(f"Plot exported to {export_filename}")
    else:
        fig.show()
#%%
#x, y, z = convert_lat_lon_to_cartesian(latitude, longitude, altitude)


satellite_data = pd.read_csv(r'C:\Users\illes\Downloads\ConstellationEphemerides.csv')
satellite_data = satellite_data[['Satellite ID','Time','X-position','Y-position','Z-position']]
#satellite_data = satellite_data[satellite_data['Time']==10]
satellite_data[['X-position', 'Y-position', 'Z-position']] = satellite_data[['X-position', 'Y-position', 'Z-position']].astype(float)

def calculate_distance(satellite1, satellite2):
    # Calculate the Euclidean distance between two satellites using their X, Y, and Z positions
    dx = satellite1['X-position'] - satellite2['X-position']
    dy = satellite1['Y-position'] - satellite2['Y-position']
    dz = satellite1['Z-position'] - satellite2['Z-position']
    
    distance = math.sqrt(dx**2 + dy**2 + dz**2)
    return distance

def is_satellite_in_front_of_earth(target_satellite, nearby_satellite):
    # Check if the nearby satellite is in front of the Earth relative to the target satellite
    target_to_earth = target_satellite[['X-position', 'Y-position', 'Z-position']].values
    target_to_nearby = nearby_satellite[['X-position', 'Y-position', 'Z-position']].values

    # Calculate the dot product
    dot_product = np.dot(target_to_nearby, target_to_earth)

    return dot_product > 0

def find_satellites_in_radius(target_satellite, all_satellites, radius):
    nearby_satellites = []
    distances = []
    
    for index, satellite in all_satellites.iterrows():
        if satellite['Satellite ID'] != target_satellite['Satellite ID']:
            distance = calculate_distance(target_satellite, satellite)
            if distance <= radius and is_satellite_in_front_of_earth(target_satellite, satellite):
                nearby_satellites.append(satellite)
                distances.append(distance)
    
    return {'Nearby Satellites': nearby_satellites, 'Distances': distances}


# Create new columns 'Nearby_Satellites' and 'Distances' to store the lists
'''
result = satellite_data.apply(lambda row: find_satellites_in_radius(row, satellite_data, radius=6000), axis=1)
satellite_data['Nearby_Satellites'] = result.apply(lambda x: x['Nearby Satellites'])
satellite_data['Distances'] = result.apply(lambda x: x['Distances'])
'''
#%%exapmle
first_instance_nearby_satellites = satellite_data.iloc[1578]['Distances']

print(f"The first instance has {len(first_instance_nearby_satellites)} nearby satellites.")
#%%example
row_index = 1578
plot_satellites_3d_interactive(satellite_data, export_filename='single_row_plot.html', row_index=row_index)
#%%whole
plot_satellites_3d_interactive(satellite_data, export_filename='all_satellites_plot.html')
#%%Kapcsolatok gráffá alakítása
import networkx as nx
def create_network_graph(satellite_data):
    G = nx.Graph()

    for index, satellite in satellite_data.iterrows():
        G.add_node(satellite['Satellite ID'],pos=(satellite['X-position'], satellite['Y-position'], satellite['Z-position']))
        nearby_satellites = satellite['Nearby Satellites']
        for nearby_satellite in nearby_satellites:
            if nearby_satellite['Satellite ID'] != satellite['Satellite ID']:
                distance = calculate_distance(satellite, nearby_satellite)
                weight = distance #latency
                capacity=15 #bitrate
                G.add_edge(satellite['Satellite ID'], nearby_satellite['Satellite ID'], weight=weight, capacity=capacity)

    # Use a different layout algorithm (e.g., shell_layout or kamada_kawai_layout)
    pos = nx.shell_layout(G)

    # Draw nodes with a lighter color
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=50)

    # Draw edges with a contrasting color
    nx.draw_networkx_edges(G, pos, edge_color='darkred', width=0.5)

    # Draw labels with a white font color
    nx.draw_networkx_labels(G, pos, font_color='white')

    return G

network_graph = create_network_graph(satellite_data)

#%%routing probléma megoldás
def find_shortest_path(graph, source, destination):
    try:
        shortest_path = nx.shortest_path(graph, source=source, target=destination, weight='weight')
        shortest_distance = nx.shortest_path_length(graph, source=source, target=destination, weight='weight')
        return shortest_path, shortest_distance
    except nx.NetworkXNoPath:
        print(f"No path found from {source} to {destination}.")
        return None, float('inf')

# Example usage:
# Assuming you have the network graph (G) and source-destination pairs
# (Modify the parameters according to your specific use case)
source_nodes = ['s01001', 's01001', 's01001']
destination_nodes = ['s65008', 's65008', 's72010']

for source, destination in zip(source_nodes, destination_nodes):
    shortest_path, shortest_distance = find_shortest_path(network_graph, source, destination)
    
    if shortest_path:
        total_capacity = sum(network_graph[u][v]['capacity'] for u, v in zip(shortest_path[:-1], shortest_path[1:]))
        if total_capacity >= 15:
            print(f"Optimal route from {source} to {destination}: {shortest_path}")
            print(f"Total distance: {shortest_distance}")
            print(f"Total capacity: {total_capacity}")
        else:
            print(f"Capacity insufficient from {source} to {destination}.")
    print()
#%% optimális satelithez csatlakozás
# Observer's latitude and longitude (example coordinates for New York)
observer_latitude = 40.7128
observer_longitude = -74.0060

# Calculate elevation angles for each satellite
elevation_angles = []
for index, row in satellite_data.iterrows():
    # Calculate satellite's position vector relative to the observer
    observer_to_satellite = np.array([row['X-position'], row['Y-position'], row['Z-position']])
    
    # Calculate observer's position vector pointing towards the zenith
    observer_to_zenith = np.array([
        np.cos(np.radians(observer_latitude)) * np.cos(np.radians(observer_longitude)),
        np.cos(np.radians(observer_latitude)) * np.sin(np.radians(observer_longitude)),
        np.sin(np.radians(observer_latitude))
    ])
    
    # Calculate elevation angle using the dot product
    elevation_angle = np.degrees(np.arcsin(np.dot(observer_to_satellite, observer_to_zenith) / np.linalg.norm(observer_to_satellite)))
    
    elevation_angles.append({'Satellite ID': row['Satellite ID'], 'Elevation Angle': elevation_angle})

# Convert to DataFrame
elevation_df = pd.DataFrame(elevation_angles)

# Filter visible satellites (elevation angle above a threshold)
threshold_elevation = 5  # Example threshold angle in degrees
visible_satellites = elevation_df[elevation_df['Elevation Angle'] > threshold_elevation]

# Identify satellite spending the most time above the horizon
most_visible_satellite = visible_satellites.loc[visible_satellites['Elevation Angle'].idxmin()] #max would be the one that is closest to zenith

# Make connections or perform other actions with the most visible satellite
print(f"The most visible satellite is {most_visible_satellite['Satellite ID']}")

#%%
def calculate_distance_between_nodes(graph, node1, node2):
    pos = nx.get_node_attributes(graph, 'pos')
    x1, y1, z1 = pos[node1]
    x2, y2, z2 = pos[node2]
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance
#%%Idősoros megoldás
satellite_data = satellite_data[satellite_data['Time']<50]
observation_points = {
    "Villenave d’Ornon": (44.7754, -0.5565),
    "Aerzen": (52.0331, 9.2406),
    "Frankfurt": (50.1109, 8.6821),
    "Ballinspittle": (51.6664, -8.7539),
    "Elfordstown": (51.8290, -8.2383),
    "Foggia": (41.4626, 15.5430),
    "Marsala": (37.8046, 12.4384),
    "Milano": (45.4642, 9.1900),
    "Kaunas": (54.8985, 23.9036),
    "Tromsø": (69.6496, 18.9560),
    "Wola Krobowska": (52.1844, 20.8539),
    "Alfouvar de Cima": (40.1951, -8.4430),
    "Coviha": (40.0741, -8.6796),
    "Ibi": (38.6176, -0.5729),
    "Lepe": (37.2575, -7.1970),
    "Villarejo de Salvanes": (40.8674, -3.2673),
    "Chalfont Grove": (51.6392, -0.5622),
    "Fawley": (50.828, 1.352),  # Unknown location
    "Goonhilly": (50.0486, -5.2232),
    "Hoo": (51.423, 0.558),  # Construction pending
    "Isle of Man": (54.2361, -4.5481),
    "Morn Hill": (51.0620, -1.258),  # Unknown location
    "Wherstead": (52.022, 1.1428),  # Construction ongoing
    "Woodwalton": (52.400, -0.217),  # Unknown location
}
#convert ground station locations to cartesian
ground_station_coordinates = {location: convert_lat_lon_to_cartesian(lat, lon, 0) for location, (lat, lon) in observation_points.items()}

traffic_data = pd.read_csv("sampled_traffic.csv")
traffic_data = traffic_data[traffic_data['timestamp']=='2022-09-01 08:50:00+00:00']
user_coordinates = {
    user: convert_lat_lon_to_cartesian(lat, lon, 0)
    for user, lat, lon in zip(
        traffic_data['icao_address'],
        traffic_data['latitude'],
        traffic_data['longitude']
    )
}

previous_elevation_angles = pd.DataFrame()
previous_connections = {}
results_df = pd.DataFrame(columns=['Time', 'Location', 'Satellite_ID'])
user_results=pd.DataFrame()
# Iterate over unique time values
for current_time in satellite_data['Time'].unique():
    # Filter satellites for the current time
    current_satellites = satellite_data[satellite_data['Time'] == current_time]
    current_satellites = current_satellites[current_satellites['Y-position'] > 0]
    # Iterate over observation points
    result = current_satellites.apply(lambda row: find_satellites_in_radius(row, satellite_data, radius=6000), axis=1)
    current_satellites['Nearby Satellites'] = result.apply(lambda x: x['Nearby Satellites'])
    current_satellites['Distances'] = result.apply(lambda x: x['Distances'])

    for location, coordinates in observation_points.items():
        observer_latitude, observer_longitude = coordinates

        # Select the most visible satellite in the first iteration or if the previous one is not visible anymore
        elevation_angles = []
        for index, row in current_satellites.iterrows():
            observer_to_satellite = np.array([row['X-position'], row['Y-position'], row['Z-position']])
            observer_to_zenith = np.array([
                np.cos(np.radians(observer_latitude)) * np.cos(np.radians(observer_longitude)),
                np.cos(np.radians(observer_latitude)) * np.sin(np.radians(observer_longitude)),
                np.sin(np.radians(observer_latitude))
            ])
            elevation_angle = np.degrees(np.arcsin(np.dot(observer_to_satellite, observer_to_zenith) / np.linalg.norm(observer_to_satellite)))

            satellite_id = row['Satellite ID']
            elevation_angles.append({'Location':location,'Satellite ID': satellite_id, 'Elevation Angle': elevation_angle})

        elevation_df = pd.DataFrame(elevation_angles)
        if current_time>0:
            previous_state = previous_elevation_angles[(previous_elevation_angles['Location']==location) &(previous_elevation_angles['Time']==current_time-10)]
            previous_state=previous_state.rename(columns={"Elevation Angle": "Prev Elevation Angle"})
            merged_elevation_df = pd.merge(elevation_df, previous_state, on=['Satellite ID', 'Location'], how='left')
            merged_elevation_df = merged_elevation_df[(merged_elevation_df['Elevation Angle']>merged_elevation_df['Prev Elevation Angle'])| (merged_elevation_df['Satellite ID'] == previous_connections[location])]
            merged_elevation_df=merged_elevation_df[['Satellite ID','Elevation Angle']]
        else:
            merged_elevation_df=elevation_df
        threshold_elevation = 5
        visible_satellites = merged_elevation_df[merged_elevation_df['Elevation Angle'] > threshold_elevation]
        merged_elevation_df['Location']=location
        merged_elevation_df['Time']=current_time
        previous_elevation_angles=previous_elevation_angles.append(merged_elevation_df)
        if not visible_satellites.empty:
            most_visible_satellite = visible_satellites.loc[visible_satellites['Elevation Angle'].idxmin()]

            if location not in previous_connections or location not in visible_satellites['Satellite ID'].values:
                # The selected satellite has changed, update connections
                results_df = results_df.append({'Time': current_time, 'Location': location, 'Satellite_ID': most_visible_satellite['Satellite ID']}, ignore_index=True)
                previous_connections[location] = most_visible_satellite['Satellite ID']
                print(f"At {current_time}, for {location}, the most visible satellite is {most_visible_satellite['Satellite ID']}. Updating connections.")
            else:
                # The selected satellite is still visible, no need to change
                results_df = results_df.append({'Time': current_time, 'Location': location, 'Satellite_ID': most_visible_satellite['Satellite ID']}, ignore_index=True)
                print(f"At {current_time}, for {location}, the most visible satellite is still {previous_connections[location]}")
        else:
            print(f"At {current_time}, for {location}, no visible satellites in this iteration.")


    #create network of satelites
    network_graph = create_network_graph(current_satellites)
    
    #Add ground stations as nodes to the existing network
    for location, coordinates in ground_station_coordinates.items():
        network_graph.add_node(location, pos=coordinates, node_color='green', node_size=10)
        
    # Step 3: Create connections between satellites and ground stations based on the 'results_df' dataframe
    for index, row in results_df.iterrows():
        satellite_id = row['Satellite_ID']
        location = row['Location']
    
        # Check if the satellite and ground station nodes exist in the network
        if network_graph.has_node(satellite_id) and network_graph.has_node(location):
            network_graph.add_edge(satellite_id, location, weight=calculate_distance_between_nodes(network_graph, satellite_id, location), capacity=15)  # Adjust capacity as needed
    
    for user, coordinates in user_coordinates.items():
        network_graph.add_node(user, pos=coordinates)
    
    #Traffic addition
    # Iterate over users
    for user, coordinates in user_coordinates.items():
        # Find the nearest ground station for the user
        nearest_ground_station = min(ground_station_coordinates.keys(), key=lambda station: calculate_distance_between_nodes(network_graph, user, station))

        # Use maximum_flow_min_cost to find the optimal flow in terms of both latency and throughput
        flow_value, flow_dict = nx.maximum_flow_min_cost(
            network_graph, user, nearest_ground_station, capacity='capacity', weight='cost'
        )

        # Get the path taken by the user
        user_path = nx.shortest_path(network_graph, user, nearest_ground_station, weight='cost')

        # Update the network_graph based on the optimal flow
        for source, targets in flow_dict.items():
            for target, flow in targets.items():
                if flow > 0:
                    network_graph[source][target]['flow'] = flow
                    network_graph[source][target]['used_capacity'] = flow
                    network_graph[source][target]['remaining_capacity'] -= flow

        # Add the routing information to user_results
        user_results = user_results.append({
            'Time': current_time,
            'Location': user,  # Assuming user is considered a location in your user_results
            'Satellite_ID': nearest_ground_station,  # The selected ground station
            'Flow': flow_value,
            'Path': user_path,
        }, ignore_index=True)
    # Draw nodes and edges
    pos = nx.shell_layout(network_graph)
    nx.draw_networkx_nodes(network_graph, pos, node_color='lightblue', node_size=10)
    nx.draw_networkx_edges(network_graph, pos, edge_color='darkred', width=0.5)
    nx.draw_networkx_labels(network_graph, pos, font_color='white')
    
    # Show the plot
    plt.show()
