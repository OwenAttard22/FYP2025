from bluesky import core, stack, traf


def init_plugin():
    config = {
        'plugin_name': 'TRAFFIC_INFO',
        'plugin_type': 'sim'
    }
    TrafficInfo()
    return config


class TrafficInfo(core.Entity):
    def __init__(self):
        super().__init__()

    @stack.command(name='GETTRAFFIC')
    def get_traffic(self):
        if traf.ntraf == 0:
            return True, "---"

        traffic_data = []
        for i in range(traf.ntraf):
            name, lat, lon, alt, hdg, spd = traf.id[i], traf.lat[i], traf.lon[i], traf.alt[i], traf.hdg[i], traf.tas[i]
            traffic_data.append(f"{name},{lat},{lon},{alt},{hdg},{spd}")

        return True, "\n".join(traffic_data)
    
    