import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import random
from collections import defaultdict


class SDWANComponents:
    """
    Implementation of SD-WAN components for visualization and simulation.
    Provides building blocks for a virtual SD-WAN environment.
    """

    def __init__(self, senaste=None):
        """Initialize SD-WAN components."""
        self.transport_types = {
            "MPLS": {
                "color": "blue",
                "style": "solid",
                "width": 2.5,
                "reliability": 0.99,
                "typical_latency": (10, 30),
                "typical_jitter": (1, 5),
                "typical_loss": (0.001, 0.01),
                "typical_capacity": (50, 200),
                "cost": "High"
            },
            "Internet": {
                "color": "green",
                "style": "dashed",
                "width": 2.0,
                "reliability": 0.95,
                "typical_latency": (20, 100),
                "typical_jitter": (5, 20),
                "typical_loss": (0.01, 0.05),
                "typical_capacity": (100, 500),
                "cost": "Medium"
            },
            "LTE": {
                "color": "orange",
                "style": "dotted",
                "width": 1.5,
                "reliability": 0.90,
                "typical_latency": (30, 150),
                "typical_jitter": (10, 30),
                "typical_loss": (0.02, 0.08),
                "typical_capacity": (20, 100),
                "cost": "High"
            }
        }

        self.application_types = {
            "Voice": {
                "color": "red",
                "icon": "VOICE",
                "latency_sensitive": True,
                "jitter_sensitive": True,
                "bandwidth_requirement": "Low",
                "max_latency": 100,
                "max_jitter": 20,
                "max_loss": 0.02,
                "priority": 1,
                "preferred_links": ["MPLS", "Internet", "LTE"]
            },
            "Video": {
                "color": "purple",
                "icon": "VIDEO",
                "latency_sensitive": True,
                "jitter_sensitive": True,
                "bandwidth_requirement": "High",
                "max_latency": 150,
                "max_jitter": 30,
                "max_loss": 0.03,
                "priority": 2,
                "preferred_links": ["MPLS", "Internet", "LTE"]
            },
            "Web": {
                "color": "cyan",
                "icon": "WEB",
                "latency_sensitive": False,
                "jitter_sensitive": False,
                "bandwidth_requirement": "Medium",
                "max_latency": 300,
                "max_jitter": 50,
                "max_loss": 0.05,
                "priority": 3,
                "preferred_links": ["Internet", "MPLS", "LTE"]
            },
            "Database": {
                "color": "blue",
                "icon": "DB",
                "latency_sensitive": True,
                "jitter_sensitive": False,
                "bandwidth_requirement": "Medium",
                senaste: 200,
                "max_jitter": 40,
                "max_loss": 0.01,
                "priority": 2,
                "preferred_links": ["MPLS", "Internet", "LTE"]
            },
            "File Transfer": {
                "color": "green",
                "icon": "FILE",
                "latency_sensitive": False,
                "jitter_sensitive": False,
                "bandwidth_requirement": "High",
                "max_latency": 500,
                "max_jitter": 100,
                "max_loss": 0.08,
                "priority": 4,
                "preferred_links": ["Internet", "MPLS", "LTE"]
            }
        }

        self.site_types = {
            "Headquarters": {
                "color": "red",
                "icon": "HQ",
                "typical_capacity": (500, 1000),
                "typical_devices": (50, 200),
                "priority": 1
            },
            "Data Center": {
                "color": "blue",
                "icon": "DC",
                "typical_capacity": (800, 2000),
                "typical_devices": (20, 100),
                "priority": 1
            },
            "Branch": {
                "color": "green",
                "icon": "BR",
                "typical_capacity": (50, 300),
                "typical_devices": (10, 50),
                "priority": 2
            }
        }

        self.event_types = {
            "LinkFailure": {
                "color": "red",
                "icon": "X",
                "description": "Link failure",
                "typical_duration": (1, 10),
                "probability": 0.005
            },
            "TrafficBurst": {
                "color": "orange",
                "icon": "BURST",
                "description": "Traffic burst",
                "typical_duration": (1, 5),
                "typical_factor": (1.5, 3.0),
                "probability": 0.1
            },
            "SiteFailure": {
                "color": "black",
                "icon": "DOWN",
                "description": "Site outage",
                "typical_duration": (5, 20),
                "probability": 0.001
            }
        }

    def get_component_properties(self):
        """
        Get all component properties.

        Returns:
            dict: Transport, application, site, and event types
        """
        return {
            "transport_types": self.transport_types,
            "application_types": self.application_types,
            "site_types": self.site_types,
            "event_types": self.event_types
        }

    def create_application_flows(self, num_flows=10, num_sites=5, seed=42):
        """
        Create application flows for the SD-WAN.

        Args:
            num_flows (int): Number of flows
            num_sites (int): Number of sites
            seed (int): Random seed

        Returns:
            list: List of (source, destination, flow_rate, app_type) tuples
        """
        np.random.seed(seed)
        random.seed(seed)
        flows = []
        for _ in range(num_flows):
            source = random.randint(0, num_sites - 1)
            destination = random.randint(0, num_sites - 1)
            while source == destination:
                destination = random.randint(0, num_sites - 1)
            app_type = random.choice(list(self.application_types.keys()))
            app = self.application_types[app_type]
            flow_rate = (
                random.randint(1, 10) if app["bandwidth_requirement"] == "Low" else
                random.randint(10, 30) if app["bandwidth_requirement"] == "Medium" else
                random.randint(30, 100)
            )
            flows.append((source, destination, flow_rate, app_type))
        print(f"Created {len(flows)} flows")
        return flows

    def visualize_sdwan_topology(self, G, transport_links, site_properties, link_loads=None, save_path=None):
        """
        Visualize the SD-WAN topology.

        Args:
            G: NetworkX graph
            transport_links: Dict of transport links
            site_properties: Dict of site properties
            link_loads: Dict of link loads (optional)
            save_path: Path to save visualization (optional)
        """
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)

        node_colors = []
        node_sizes = []
        for node in G.nodes():
            site_type = site_properties[node]["type"]
            node_colors.append(self.site_types[site_type]["color"])
            node_sizes.append(800 if site_type == "Headquarters" else 700 if site_type == "Data Center" else 600)
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)

        labels = {node: f"Site-{node}\n({site_properties[node]['type']})" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

        for link_type, props in self.transport_types.items():
            edges = [(u, v) for u, v in G.edges() if
                     link_type in transport_links.get((u, v), {}) and transport_links[(u, v)][link_type]]
            if edges:
                nx.draw_networkx_edges(G, pos, edgelist=edges, width=props["width"],
                                       edge_color=props["color"], style=props["style"],
                                       alpha=0.7, connectionstyle="arc3,rad=0.1")

        edge_labels = {}
        for u, v in G.edges():
            label_parts = []
            for link_type in self.transport_types:
                if link_type in transport_links.get((u, v), {}) and transport_links[(u, v)][link_type]:
                    load = link_loads.get((u, v, link_type), 0) if link_loads else 0
                    label_parts.append(f"{link_type}: {load:.1f}")
            if label_parts:
                edge_labels[(u, v)] = "\n".join(label_parts)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        plt.title("SD-WAN Topology")
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def visualize_application_flows(self, G, flows, site_properties, paths=None, save_path=None):
        """
        Visualize application flows.

        Args:
            G: NetworkX graph
            flows: List of (source, destination, flow_rate, app_type)
            site_properties: Dict of site properties
            paths: Dict of flow_id to path (optional)
            save_path: Path to save visualization (optional)
        """
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)

        node_colors = []
        node_sizes = []
        for node in G.nodes():
            site_type = site_properties[node]["type"]
            node_colors.append(self.site_types[site_type]["color"])
            node_sizes.append(800 if site_type == "Headquarters" else 700 if site_type == "Data Center" else 600)
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)

        labels = {node: f"Site-{node}\n({site_properties[node]['type']})" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

        nx.draw_networkx_edges(G, pos, width=1, alpha=0.3, connectionstyle="arc3,rad=0.1")

        for idx, (source, destination, flow_rate, app_type) in enumerate(flows):
            if source not in G.nodes() or destination not in G.nodes():
                continue
            app = self.application_types.get(app_type, {"color": "gray"})
            width = 1 + 3 * (flow_rate / 100)
            flow_id = idx  # Simple flow ID
            path = paths.get(flow_id, [(source, None), (destination, None)]) if paths else [(source, None),
                                                                                            (destination, None)]

            # Draw path segments
            for i in range(len(path) - 1):
                u, u_link = path[i]
                v, _ = path[i + 1]
                if u not in pos or v not in pos:
                    continue
                edge = FancyArrowPatch(
                    posA=pos[u],
                    posB=pos[v],
                    arrowstyle='-|>',
                    mutation_scale=20,
                    lw=width,
                    color=app["color"],
                    alpha=0.7,
                    connectionstyle="arc3,rad=0.2"
                )
                plt.gca().add_patch(edge)

            # Label at midpoint
            u, _ = path[0]
            v, _ = path[-1]
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            midx = (x1 + x2) / 2
            midy = (y1 + y2) / 2
            dx = -(y2 - y1) * 0.15
            dy = (x2 - x1) * 0.15
            plt.text(
                midx + dx, midy + dy,
                f"{app_type}\n{flow_rate} Mbps",
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                horizontalalignment='center',
                verticalalignment='center'
            )

        plt.title("SD-WAN Application Flows")
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    components = SDWANComponents()
    from sdwan_environment import SDWANEnvironment

    env = SDWANEnvironment(num_sites=6, connectivity_prob=0.6)
    G = env.G
    transport_links = env.transport_links
    site_properties = env.site_properties
    components.visualize_sdwan_topology(G, transport_links, site_properties)
    flows = components.create_application_flows(num_flows=8, num_sites=6)
    components.visualize_application_flows(G, flows, site_properties)