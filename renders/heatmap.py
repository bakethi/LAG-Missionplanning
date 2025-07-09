import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)  # Suppress font manager warnings
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class FlightHeatmap:
    def __init__(self, acmi_path):
        self.acmi_path = acmi_path
        self.agent_lons = []
        self.agent_lats = []
        self.agent_alts = []
        self.wp_lons = []
        self.wp_lats = []
        self.wp_alts = []
        self._parse_acmi()

    def _parse_acmi(self):
        aircraft_pattern = re.compile(r'^A0100,T=([\d\.\-e\|]+)')
        waypoint_pattern = re.compile(r'^(100000\d{3}),T=([\d\.\-e\|]+)')
        waypoints = {}

        try:
            with open(self.acmi_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            logging.error(f"Could not read ACMI file {self.acmi_path}: {e}")
            return

        for line in tqdm(lines, desc="Parsing ACMI file", unit="lines", ncols=100):
            match = aircraft_pattern.match(line)
            if match:
                telemetry = match.group(1)
                fields = telemetry.split('|')
                if len(fields) >= 3:
                    try:
                        lon = float(fields[0])
                        lat = float(fields[1])
                        alt = float(fields[2])
                        self.agent_lons.append(lon)
                        self.agent_lats.append(lat)
                        self.agent_alts.append(alt)
                    except (ValueError, IndexError):
                        logging.warning(f"Could not parse aircraft telemetry: {line.strip()}")
                continue

            match = waypoint_pattern.match(line)
            if match:
                wp_id, telemetry = match.groups()
                if wp_id not in waypoints:
                    fields = telemetry.split('|')
                    if len(fields) >= 3:
                        try:
                            lon = float(fields[0])
                            lat = float(fields[1])
                            alt = float(fields[2])
                            waypoints[wp_id] = (lon, lat, alt)
                        except (ValueError, IndexError):
                            logging.warning(f"Could not parse waypoint telemetry: {line.strip()}")

        sorted_waypoints = sorted(waypoints.items())
        self.wp_lons = [lon for wp_id, (lon, lat, alt) in sorted_waypoints]
        self.wp_lats = [lat for wp_id, (lon, lat, alt) in sorted_waypoints]
        self.wp_alts = [alt for wp_id, (lon, lat, alt) in sorted_waypoints]

    def plot_3d(self, save_path=None, show=True):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        # --- SCALE ALTITUDE ---
        ALT_SCALE = 0.0001
        agent_alts_scaled = [alt * ALT_SCALE for alt in self.agent_alts]
        wp_alts_scaled = [alt * ALT_SCALE for alt in self.wp_alts]

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot aircraft path with altitude gradient
        if len(self.agent_lons) > 1:
            points = np.array([self.agent_lons, self.agent_lats, agent_alts_scaled]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(min(agent_alts_scaled), max(agent_alts_scaled))
            lc = Line3DCollection(segments, cmap='viridis', norm=norm)
            lc.set_array(np.array(agent_alts_scaled))
            lc.set_linewidth(2)
            ax.add_collection3d(lc)
        else:
            ax.plot(self.agent_lons, self.agent_lats, agent_alts_scaled, color='red', label='Aircraft Path', linewidth=2)

        # Plot waypoints as vertical columns
        for i, (lon, lat, alt) in enumerate(zip(self.wp_lons, self.wp_lats, wp_alts_scaled)):
            min_alt = min(agent_alts_scaled) if agent_alts_scaled else 0
            max_alt = max(agent_alts_scaled) if agent_alts_scaled else alt + 1
            ax.plot([lon, lon], [lat, lat], [min_alt, max_alt], color='blue', linewidth=2, alpha=0.5)
            ax.scatter(lon, lat, alt, c='blue', s=80, marker='o', edgecolors='k', zorder=5)
            ax.text(lon, lat, max_alt, f'WP{i+1}', color='blue', fontsize=9)

        if self.agent_lons and self.agent_lats and agent_alts_scaled:
            ax.scatter(self.agent_lons[0], self.agent_lats[0], agent_alts_scaled[0], c='green', label='Start', s=100, marker='*', edgecolors='k', zorder=6)
            ax.text(self.agent_lons[0], self.agent_lats[0], agent_alts_scaled[0], 'Start', color='green', fontsize=10)

        ax.set_title("3D Flight Path with Waypoint Columns (Altitude Gradient, Alt in km)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_zlabel("Altitude (km)")
        ax.legend()
        ax.set_xlim(119.1015, 120.8985)
        ax.set_ylim(59.5505, 60.4495)
        ax.set_box_aspect([1, 1, 0.3])  # Make z-axis 30% of x/y axes
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()

    def plotly_3d(self, save_path="flight_heatmap_3d.html", show=False):
        import plotly.graph_objects as go

        ALT_SCALE = 0.00005
        agent_alts_scaled = [alt * ALT_SCALE for alt in self.agent_alts]
        wp_alts_scaled = [alt * ALT_SCALE for alt in self.wp_alts]

        fig = go.Figure()

        # Aircraft path with altitude gradient
        fig.add_trace(go.Scatter3d(
            x=self.agent_lons, y=self.agent_lats, z=agent_alts_scaled,
            mode='lines',
            line=dict(color=agent_alts_scaled, colorscale='Viridis', width=5),
            name='Aircraft Path'
        ))

        # Waypoints as vertical columns
        for i, (lon, lat, alt) in enumerate(zip(self.wp_lons, self.wp_lats, wp_alts_scaled)):
            fig.add_trace(go.Scatter3d(
                x=[lon, lon], y=[lat, lat], z=[min(agent_alts_scaled), max(agent_alts_scaled)],
                mode='lines',
                line=dict(color='blue', width=3),
                showlegend=False
            ))
            fig.add_trace(go.Scatter3d(
                x=[lon], y=[lat], z=[alt],
                mode='markers',
                marker=dict(size=6, color='blue'),
                showlegend=False
            ))

        # Start position
        if self.agent_lons and self.agent_lats and agent_alts_scaled:
            fig.add_trace(go.Scatter3d(
                x=[self.agent_lons[0]], y=[self.agent_lats[0]], z=[agent_alts_scaled[0]],
                mode='markers',
                marker=dict(size=8, color='green', symbol='diamond'),
                name='Start'
            ))

        fig.update_layout(
            title="3D Flight Path with Waypoints",
            scene=dict(
                xaxis_title="Longitude",
                yaxis_title="Latitude",
                zaxis_title="Altitude (km)",
                xaxis=dict(nticks=10, range=[119.1015, 120.8985]),
                yaxis=dict(nticks=10, range=[59.5505, 60.4495]),
                zaxis=dict(nticks=10),
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=800,
            width=800,
        )

        fig.write_html(save_path)
        print(f"Interactive 3D plot saved to {save_path}")
        if show:
            import webbrowser
            webbrowser.open(save_path)


# Example usage:
# from heatmap import FlightHeatmap
# heatmap = FlightHeatmap('./renders/multiwaypoint.txt.acmi')
# heatmap.plot_3d(save_path='flight_heatmap_3d.png')
# heatmap.plotly_3d(save_path='flight_heatmap_3d.html')