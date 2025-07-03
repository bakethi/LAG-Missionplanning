import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
import re
import matplotlib.pyplot as plt
import seaborn as sns

class FlightHeatmap:
    def __init__(self, acmi_path):
        self.acmi_path = acmi_path
        self.agent_lons = []
        self.agent_lats = []
        self.wp_lons = []
        self.wp_lats = []
        self._parse_acmi()

    def _parse_acmi(self):
        with open(self.acmi_path, 'r') as file:
            acmi_data = file.read()

        # Extract aircraft telemetry (A0100)
        aircraft_pattern = r'A0100,T=([\d\.\-e\|]+)'
        aircraft_matches = re.findall(aircraft_pattern, acmi_data)

        for match in aircraft_matches:
            fields = match.split('|')
            if len(fields) >= 2:
                lon = float(fields[0])
                lat = float(fields[1])
                self.agent_lons.append(lon)
                self.agent_lats.append(lat)

        # Extract UNIQUE waypoints (100000XXX)
        waypoint_pattern = r'(100000\d\d\d),T=([\d\.\-e\|]+)'
        waypoint_matches = re.findall(waypoint_pattern, acmi_data)

        waypoints = {}
        for wp_id, telemetry in waypoint_matches:
            if wp_id not in waypoints:  # only take the first occurrence
                fields = telemetry.split('|')
                if len(fields) >= 2:
                    lon = float(fields[0])
                    lat = float(fields[1])
                    waypoints[wp_id] = (lon, lat)

        self.wp_lons = [lon for lon, lat in waypoints.values()]
        self.wp_lats = [lat for lon, lat in waypoints.values()]

    def plot(self, save_path=None, show=True):
        plt.figure(figsize=(10, 8))

        # Heatmap
        sns.kdeplot(
            x=self.agent_lons, y=self.agent_lats,
            cmap="Reds", fill=True, bw_adjust=0.1, alpha=0.8, levels=100
        )

        # Waypoints (unique)
        plt.scatter(self.wp_lons, self.wp_lats, c='blue', label='Waypoints', edgecolors='k', s=80, marker='o', zorder=5)
        for i, (lon, lat) in enumerate(zip(self.wp_lons, self.wp_lats)):
            plt.annotate(f'WP{i+1}', xy=(lon, lat), xytext=(-5, 5),
                         textcoords='offset points', fontsize=9, ha='right',
                         va='center', color='blue')


        # Starting position
        if self.agent_lons and self.agent_lats:
            plt.scatter(self.agent_lons[0], self.agent_lats[0], c='green', label='Start Position', edgecolors='k', s=100, marker='*', zorder=6)
            plt.annotate('Start', xy=(self.agent_lons[0], self.agent_lats[0]), xytext=(5, 5),
                         textcoords='offset points', fontsize=10, ha='left', va='center', color='green')

        # Decorations
        plt.title("F-16 Flight Path Heatmap with Unique Waypoints")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()

# Example usage:
# from heatmap import FlightHeatmap
# heatmap = FlightHeatmap('./renders/multiwaypoint.txt.acmi')
# heatmap.plot(save_path='flight_heatmap.png')