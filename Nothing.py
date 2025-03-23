from geopy.distance import geodesic

# Define aircraft positions
waypoints = {
    "HAWK": (34.30776639159571, 21.246895925692375),
    "EAGLE": (34.70176325859763, 16.940876508875004),
    "FALCON": (35.070410492339434, 20.12760095399451),
    "SCORPION": (36.170842067744005, 20.364581388623865),
    "VIPER": (39.753678370708556, 20.45356807094306),
    "RAVEN": (36.70064882343415, 19.78422450593784),
    "PHOENIX": (33.8998441355122, 12.244464367446312),
    "SPARROW": (39.45553659043508, 17.07935815876537),
    "HORNET": (34.94096693461544, 19.445448594167964),
    "PEGASUS": (35.27522942481725, 20.41070957837147),
    "TALON": (33.716030529929455, 20.135941532594202),
    "GRYPHON": (38.74123496867829, 15.47847072657443),
    "WRAITH": (35.52591523425711, 8.655649026135887),
    "LIGHTNING": (37.33790294999774, 12.95380569069244),
    "DRAGON": (34.72672514317918, 18.060263703813867),
    "THUNDERBIRD": (36.93170226079151, 11.425038648290911),
    "STORM": (39.86464063464124, 8.422176933149194),
    "BLADE": (36.59683420566159, 13.714132422333778),
}

# Define destination airports
destinations = {
    "HAWK": (35.8575, 14.4775),  # LMML (Malta)
    "SCORPION": (35.8575, 14.4775),  # LMML
    "PHOENIX": (35.8575, 14.4775),  # LMML
    "PEGASUS": (35.8575, 14.4775),  # LMML
    "WRAITH": (35.8575, 14.4775),  # LMML
    "THUNDERBIRD": (35.8575, 14.4775),  # LMML
    "EAGLE": (38.1864, 13.0914),  # LICJ (Palermo)
    "VIPER": (38.1864, 13.0914),  # LICJ
    "SPARROW": (38.1864, 13.0914),  # LICJ
    "TALON": (38.1864, 13.0914),  # LICJ
    "LIGHTNING": (38.1864, 13.0914),  # LICJ
    "STORM": (38.1864, 13.0914),  # LICJ
    "FALCON": (37.4667, 15.0664),  # LICC (Catania)
    "RAVEN": (37.4667, 15.0664),  # LICC
    "HORNET": (37.4667, 15.0664),  # LICC
    "GRYPHON": (37.4667, 15.0664),  # LICC
    "DRAGON": (37.4667, 15.0664),  # LICC
    "BLADE": (37.4667, 15.0664),  # LICC
}

# Calculate distances
for aircraft, waypoint in waypoints.items():
    if aircraft in destinations:
        destination = destinations[aircraft]
        print(type(waypoint[0]), type(destination[0]))
        distance = geodesic(waypoint, destination).kilometers
        print(f"Distance from {aircraft} to destination: {distance:.2f} km")
        
        
        

