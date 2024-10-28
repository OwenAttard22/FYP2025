""" BlueSky Test Plugin. This plugin prints a message every 5 seconds. """

from Remove import calculate_triangle_points, nm_to_km
from bluesky import core, stack
import random

LMML_LAT = 35.857012
LMML_LONG = 14.477101
LMML_WPT_LAT = 35.727825
LMML_WPT_LONG = 14.628304
# CIRCLE_RADIUS = nm_to_km(20)
CIRCLE_RADIUS = 20

SAFETY = 0.98
AREA_MAX_LAT = 40.06*SAFETY
AREA_MIN_LAT = 33.62*SAFETY
AREA_MAX_LONG = 21.87*SAFETY
AREA_MIN_LONG = 7.97*SAFETY

COOL_PLANE_NAMES = ['HAWK','EAGLE', 'FALCON', 'SCORPION', 'VIPER', 'RAVEN', 'PHOENIX', 'SPARROW', 'HORNET', 'PEGASUS']

def generate_random_lat_long():
    """ Generate a random latitude and longitude within the defined area. """
    lat = random.uniform(AREA_MIN_LAT, AREA_MAX_LAT)
    long = random.uniform(AREA_MIN_LONG, AREA_MAX_LONG)
    return lat, long

def init_plugin():
    """ Initialization function for the test plugin. """
    config = {
        'plugin_name': 'TestPlugin',  # Name of the plugin
        'plugin_type': 'sim'  # Type of plugin ('sim' for simulation-side)
    }

    # Instantiate the plugin class so it runs during simulation
    TestPlugin()

    return config

class TestPlugin(core.Entity):
    ''' A simple test plugin that prints a message periodically. '''

    def __init__(self):
        super().__init__()
        # LMML_top, LMML_bottom = calculate_triangle_points(LMML_LAT, LMML_LONG)
        # stack.stack(f'POLY RUNWAY, {LMML_LAT},{LMML_LONG},{LMML_top[0]},{LMML_top[1]},{LMML_bottom[0]},{LMML_bottom[1]}')
        stack.stack(f'LINE LMML_LINE {LMML_WPT_LAT},{LMML_WPT_LONG} LMML')
        stack.stack('AREA 40.06,7.97 33.62,21.87')
        stack.stack(f'CIRCLE LMML_CIRCLE, {LMML_LAT},{LMML_LONG}, {CIRCLE_RADIUS}')
        
        stack.stack('PAN 37, 14.24')
        stack.stack('ZOOM 0.3')
        
        stack.stack('CRECMD COLOUR 255,255,255')
        
        '''PLANE SPECIFIC COMMANDS'''
        for i in range(10):
            plane_name = COOL_PLANE_NAMES[i]
            lat, long = generate_random_lat_long()
            stack.stack(f'CRE {plane_name}, B747, {lat},{long}, 50, 10000, 300')
            
            # stack.stack(f'ADDWPTMODE {plane_name} FLYOVER')
            # stack.stack(f'ADDWPT {plane_name}, {LMML_WPT_LAT},{LMML_WPT_LONG}')
            stack.stack(f'ADDWPT {plane_name}, {LMML_WPT_LAT},{LMML_WPT_LONG}')
            # stack.stack(f'VNAV {plane_name} ON')
            
            stack.stack(f'{plane_name} AT {plane_name}001 DO DELRTE {plane_name}')
            stack.stack(f'{plane_name} AT {plane_name}001 DO ECHO {plane_name} ON APPROACH')
            stack.stack(f'{plane_name} AT {plane_name}001 DO DEST {plane_name}, LMML')
            stack.stack(f'{plane_name} AT {plane_name}001 DO SPD {plane_name} 50')
            stack.stack(f'{plane_name} AT {plane_name}001 DO COLOUR {plane_name} 255,68,51')
            stack.stack(f'{plane_name} AT {plane_name}001 DO ALT {plane_name} 1000')
            
            # stack.stack(f'DEST {plane_name}, LMML')
            
            stack.stack(f'{plane_name} ATDIST LMML 20 SPD {plane_name} 100')
            stack.stack(f'{plane_name} ATDIST LMML 20 COLOUR {plane_name} 249,166,2')
            stack.stack(f'{plane_name} ATDIST LMML 20 ALT {plane_name} 5000')
            
            stack.stack(f'{plane_name} AT {plane_name}001 DO {plane_name} AT LMML DO SPD {plane_name} 0')
            stack.stack(f'{plane_name} AT {plane_name}001 DO {plane_name} AT LMML DO ALT {plane_name} 0')
            stack.stack(f'{plane_name} AT {plane_name}001 DO {plane_name} AT LMML DO COLOUR {plane_name} 56,0,0')
            stack.stack(f'{plane_name} AT {plane_name}001 DO {plane_name} AT LMML DO ECHO {plane_name} LANDED')
            # SCHEDULE COMMANDS NOT WORKING AS INTENDED  
            stack.stack(f'{plane_name} AT {plane_name}001 DO {plane_name} AT LMML DO SCHEDULE 00:20:00 DEL {plane_name}')
            stack.stack(f'{plane_name} AT {plane_name}001 DO {plane_name} AT LMML DO SCHEDULE 00:20:00 ECHO {plane_name} SAFE')
            
            # stack.stack(f'{plane_name} AT LMML DO SPD {plane_name} 0')
            # stack.stack(f'{plane_name} AT LMML DO COLOUR {plane_name} 255,68,51')
            # stack.stack(f'{plane_name} AT LMML DO ECHO {plane_name} LANDED')
        
        stack.stack('OP')
        stack.stack('FF')
        
        # stack.stack('SCHEDULE DEL HAWK +600')

    @core.timed_function(name='testplugin_update', dt=600)
    def update(self):
        # stack.stack('HAWK')
        # stack.stack('DIST HAWK, LMML')
        # print(info)
        pass