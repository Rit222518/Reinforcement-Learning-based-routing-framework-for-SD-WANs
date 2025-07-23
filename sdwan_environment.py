import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import time


class SDWANEnvironment:
    """
    SD-WAN environment for reinforcement learning-based routing optimization.
    This class simulates a Software-Defined Wide Area Network with multiple sites,
    transport links (MPLS, Internet, LTE), and application-aware routing.
    """

    def __init__(self, num_sites=5, connectivity_prob=0.7,
                 link_types=["MPLS", "Internet", "LTE"],
                 app_types=["Voice", "Video", "Web", "Database", "File Transfer"],
                 seed=42):
        """
        Initialize the SD-WAN environment.

        Args:
            num_sites (int): Number of sites/branches in the SD-WAN
            connectivity_prob (float): Probability of connection between sites
            link_types (list): Types of transport links available
            app_types (list): Types of applications with different QoS requirements
            seed (int): Random seed for reproducibility
        """
        self.num_sites = num_sites
        self.connectivity_prob = connectivity_prob
        self.link_types = link_types
        self.app_types = app_types
        self.seed = seed

        # Set random seed
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Create topology
        self._create_sdwan_topology()

        # Initialize flow tracking
        self.active_flows = {}
        self.flow_paths = {}
        self.flow_rates = {}
        self.flow_apps = {}
        self.link_loads = defaultdict(float)
        self.rejected_flows = 0
        self.total_flows = 0
        self.flow_counter = 0

        # Performance metrics
        self.throughput = 0
        self.avg_delay = 0
        self.packet_loss = 0
        self.app_satisfaction = defaultdict(float)

        # SD-WAN metrics
        self.link_utilization = defaultdict(lambda: defaultdict(float))
        self.link_health = defaultdict(lambda: defaultdict(float))
        self.site_status = defaultdict(lambda: "Operational")

        # Policy engine
        self.policies = self._create_default_policies()

        # Time tracking
        self.current_time = 0
        self.events = []

    def _create_sdwan_topology(self):
        """Create a SD-WAN topology with multiple sites and transport links."""
        # Create base graph
        self.G = nx.erdos_renyi_graph(n=self.num_sites, p=self.connectivity_prob,
                                      directed=True, seed=self.seed)

        # Ensure strong connectivity
        while not nx.is_strongly_connected(self.G):
            components = list(nx.strongly_connected_components(self.G))
            if len(components) > 1:
                source_comp = random.choice(range(len(components)))
                target_comp = random.choice([i for i in range(len(components)) if i != source_comp])
                source_node = random.choice(list(components[source_comp]))
                target_node = random.choice(list(components[target_comp]))
                self.G.add_edge(source_node, target_node)

        # Initialize transport links
        self.transport_links = {}
        self.link_capacities = {}
        self.link_latencies = {}
        self.link_jitters = {}
        self.link_loss_rates = {}

        for u, v in self.G.edges():
            self.transport_links[(u, v)] = {}
            for link_type in self.link_types:
                if link_type == "MPLS" and random.random() < 0.8:
                    capacity = random.randint(50, 200)
                    latency = random.randint(10, 30)
                    jitter = random.randint(1, 5)
                    loss_rate = random.uniform(0.001, 0.01)
                elif link_type == "Internet" and random.random() < 0.95:
                    capacity = random.randint(100, 500)
                    latency = random.randint(20, 100)
                    jitter = random.randint(5, 20)
                    loss_rate = random.uniform(0.01, 0.05)
                elif link_type == "LTE" and random.random() < 0.6:
                    capacity = random.randint(20, 100)
                    latency = random.randint(30, 150)
                    jitter = random.randint(10, 30)
                    loss_rate = random.uniform(0.02, 0.08)
                else:
                    continue
                self.transport_links[(u, v)][link_type] = True
                self.link_capacities[(u, v, link_type)] = capacity
                self.link_latencies[(u, v, link_type)] = latency
                self.link_jitters[(u, v, link_type)] = jitter
                self.link_loss_rates[(u, v, link_type)] = loss_rate

        # Compute max capacity for visualization
        capacities = [c for c in self.link_capacities.values() if c > 0]
        self.max_capacity = max(capacities) if capacities else 100

        # Calculate all paths
        self.all_paths = {}
        for source in range(self.num_sites):
            for destination in range(self.num_sites):
                if source != destination:
                    paths = list(nx.all_simple_paths(self.G, source, destination, cutoff=self.num_sites))
                    if paths:
                        self.all_paths[(source, destination)] = paths

        # Create site properties
        self.site_properties = {}
        for site in range(self.num_sites):
            site_type = "Headquarters" if site == 0 else random.choice(["Branch", "Data Center"])
            self.site_properties[site] = {
                "name": f"Site-{site}",
                "type": site_type,
                "capacity": random.randint(200, 1000),
                "devices": random.randint(10, 100),
                "priority": random.randint(1, 5)
            }

    def _create_default_policies(self):
        """Create default SD-WAN policies for application routing."""
        return {
            "Voice": {"preferred_links": ["MPLS", "Internet", "LTE"], "max_latency": 100, "max_jitter": 20, "max_loss": 0.02, "priority": 1},
            "Video": {"preferred_links": ["MPLS", "Internet", "LTE"], "max_latency": 150, "max_jitter": 30, "max_loss": 0.03, "priority": 2},
            "Web": {"preferred_links": ["Internet", "MPLS", "LTE"], "max_latency": 300, "max_jitter": 50, "max_loss": 0.05, "priority": 3},
            "Database": {"preferred_links": ["MPLS", "Internet", "LTE"], "max_latency": 200, "max_jitter": 40, "max_loss": 0.01, "priority": 2},
            "File Transfer": {"preferred_links": ["Internet", "MPLS", "LTE"], "max_latency": 500, "max_jitter": 100, "max_loss": 0.08, "priority": 4}
        }

    def get_possible_paths(self, source, destination, app_type=None):
        """
        Get possible paths from source to destination, considering app requirements.

        Args:
            source (int): Source site
            destination (int): Destination site
            app_type (str, optional): Application type for policy-based routing

        Returns:
            list: List of possible paths (each path is a list of (node, link_type) tuples)
        """
        if (source, destination) not in self.all_paths:
            print(f"No paths from Site-{source} to Site-{destination}")
            return []

        base_paths = self.all_paths[(source, destination)]
        enhanced_paths = []

        for path in base_paths:
            # Generate variations with link types
            valid_path = True
            enhanced_path = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                available_links = self.transport_links.get((u, v), {})

                # Filter by app policy
                if app_type and app_type in self.policies:
                    preferred_links = self.policies[app_type]["preferred_links"]
                    link_options = [lt for lt in preferred_links if lt in available_links and available_links[lt]]
                else:
                    link_options = [lt for lt in self.link_types if lt in available_links and available_links[lt]]

                if not link_options:
                    valid_path = False
                    break

                # Choose first available link
                chosen_link = link_options[0]
                enhanced_path.append((u, chosen_link))

            if valid_path:
                enhanced_path.append((path[-1], None))
                enhanced_paths.append(enhanced_path)

        # Limit paths for efficiency
        if enhanced_paths:
            print(f"Found {len(enhanced_paths)} paths from Site-{source} to Site-{destination}")
        return enhanced_paths[:10]

    def calculate_path_metrics(self, path, app_type=None):
        """
        Calculate metrics for a given path.

        Args:
            path (list): List of (node, link_type) tuples
            app_type (str, optional): Application type for QoS evaluation

        Returns:
            tuple: (delay, available_bandwidth, loss_probability, jitter, app_satisfaction)
        """
        if not path or len(path) < 2:
            print(f"Invalid path: {path}")
            return 1000, 0, 0.1, 50, 0.0

        delay = 0
        jitter = 0
        available_bandwidth = float('inf')
        loss_probability = 0

        for i in range(len(path) - 1):
            u, u_link = path[i]
            v, _ = path[i + 1]
            if u_link is None:
                continue

            link_key = (u, v, u_link)
            link_load = self.link_loads.get(link_key, 0)
            link_capacity = self.link_capacities.get(link_key, 100)
            utilization = link_load / link_capacity if link_capacity > 0 else 1

            base_latency = self.link_latencies.get(link_key, 20)
            link_delay = base_latency * (1 + utilization) if utilization < 0.8 else base_latency * (1 + 5 * (utilization - 0.8) ** 2)
            delay += link_delay

            base_jitter = self.link_jitters.get(link_key, 5)
            link_jitter = base_jitter * (1 + utilization)
            jitter += link_jitter

            available_bandwidth = min(available_bandwidth, max(0, link_capacity - link_load))

            base_loss = self.link_loss_rates.get(link_key, 0.01)
            link_loss = base_loss * (1 + utilization) if utilization < 0.9 else base_loss * (1 + 10 * (utilization - 0.9) ** 2)
            loss_probability = 1 - (1 - loss_probability) * (1 - link_loss)

        app_satisfaction = 1.0
        if app_type and app_type in self.policies:
            policy = self.policies[app_type]
            if delay > policy["max_latency"]:
                app_satisfaction -= 0.4 * min(1.0, (delay - policy["max_latency"]) / policy["max_latency"])
            if jitter > policy["max_jitter"]:
                app_satisfaction -= 0.3 * min(1.0, (jitter - policy["max_jitter"]) / policy["max_jitter"])
            if loss_probability > policy["max_loss"]:
                app_satisfaction -= 0.3 * min(1.0, (loss_probability - policy["max_loss"]) / policy["max_loss"])
            app_satisfaction = max(0.0, app_satisfaction)

        return delay, available_bandwidth, loss_probability, jitter, app_satisfaction

    def add_flow(self, source, destination, flow_rate, app_type, path=None):
        """
        Add a new flow to the SD-WAN.

        Args:
            source (int): Source site
            destination (int): Destination site
            flow_rate (float): Flow rate in Mbps
            app_type (str): Application type
            path (list, optional): Predefined path

        Returns:
            bool: True if flow added, False if rejected
        """
        self.total_flows += 1
        self.flow_counter += 1
        flow_id = self.flow_counter

        if path is None:
            possible_paths = self.get_possible_paths(source, destination, app_type)
            if not possible_paths:
                self.rejected_flows += 1
                print(f"Flow rejected: No path from Site-{source} to Site-{destination}")
                return False
            path = max(possible_paths, key=lambda p: self.calculate_path_metrics(p, app_type)[4], default=None)
            if not path:
                self.rejected_flows += 1
                return False

        for i in range(len(path) - 1):
            u, u_link = path[i]
            v, _ = path[i + 1]
            if u_link is None:
                continue
            link_key = (u, v, u_link)
            if self.link_loads.get(link_key, 0) + flow_rate > self.link_capacities.get(link_key, 100):
                self.rejected_flows += 1
                print(f"Flow rejected: Insufficient bandwidth on {u_link} from Site-{u} to Site-{v}")
                return False

        self.active_flows[flow_id] = (source, destination, flow_rate)
        self.flow_paths[flow_id] = path
        self.flow_rates[flow_id] = flow_rate
        self.flow_apps[flow_id] = app_type

        for i in range(len(path) - 1):
            u, u_link = path[i]
            v, _ = path[i + 1]
            if u_link is None:
                continue
            self.link_loads[(u, v, u_link)] += flow_rate

        self._update_performance_metrics()
        print(f"Flow {flow_id} added: Site-{source} to Site-{destination}, {flow_rate} Mbps, {app_type}")
        return True

    def remove_flow(self, flow_id):
        """
        Remove a flow from the SD-WAN.

        Args:
            flow_id (int): Flow ID

        Returns:
            bool: True if removed, False otherwise
        """
        if flow_id not in self.active_flows:
            return False

        path = self.flow_paths.get(flow_id, [])
        flow_rate = self.flow_rates.get(flow_id, 0)

        for i in range(len(path) - 1):
            u, u_link = path[i]
            v, _ = path[i + 1]
            if u_link is None:
                continue
            self.link_loads[(u, v, u_link)] = max(0, self.link_loads.get((u, v, u_link), 0) - flow_rate)

        del self.active_flows[flow_id]
        self.flow_paths.pop(flow_id, None)
        self.flow_rates.pop(flow_id, None)
        self.flow_apps.pop(flow_id, None)

        self._update_performance_metrics()
        return True

    def _update_performance_metrics(self):
        """Update network performance metrics."""
        self.throughput = sum(self.flow_rates.values())
        total_delay = 0
        total_loss = 0
        total_flows = len(self.active_flows)
        app_satisfaction = defaultdict(list)

        for flow_id, (source, destination, _) in self.active_flows.items():
            path = self.flow_paths.get(flow_id, [])
            app_type = self.flow_apps.get(flow_id, None)
            if path:
                delay, _, loss, _, satisfaction = self.calculate_path_metrics(path, app_type)
                total_delay += delay
                total_loss += loss
                if app_type:
                    app_satisfaction[app_type].append(satisfaction)

        self.avg_delay = total_delay / total_flows if total_flows > 0 else 0
        self.packet_loss = total_loss / total_flows if total_flows > 0 else 0
        for app_type, satisfactions in app_satisfaction.items():
            self.app_satisfaction[app_type] = sum(satisfactions) / len(satisfactions) if satisfactions else 0

    def _generate_network_events(self):
        """Generate random network events."""
        if random.random() < 0.05:
            event_type = random.choice(["LinkFailure", "TrafficBurst", "SiteFailure"])
            if event_type == "LinkFailure":
                edges = list(self.G.edges())
                if edges:
                    u, v = random.choice(edges)
                    link_types = [lt for lt in self.transport_links.get((u, v), {}) if self.transport_links[(u, v)][lt]]
                    if link_types:
                        link_type = random.choice(link_types)
                        duration = random.randint(5, 20)
                        event = {
                            "type": "LinkFailure",
                            "description": f"Link failure: Site-{u} to Site-{v} ({link_type})",
                            "u": u, "v": v, "link_type": link_type,
                            "start_time": self.current_time,
                            "end_time": self.current_time + duration,
                            "resolved": False
                        }
                        self.transport_links[(u, v)][link_type] = False
                        self.events.append(event)
            elif event_type == "TrafficBurst":
                if self.active_flows:
                    flow_id = random.choice(list(self.active_flows.keys()))
                    source, destination, flow_rate = self.active_flows[flow_id]
                    burst_factor = random.uniform(1.5, 3.0)
                    new_flow_rate = flow_rate * burst_factor
                    duration = random.randint(3, 10)
                    event = {
                        "type": "TrafficBurst",
                        "description": f"Traffic burst: Flow {flow_id} ({self.flow_apps.get(flow_id, 'Unknown')})",
                        "flow_id": flow_id,
                        "original_rate": flow_rate,
                        "burst_rate": new_flow_rate,
                        "start_time": self.current_time,
                        "end_time": self.current_time + duration,
                        "resolved": False
                    }
                    self.flow_rates[flow_id] = new_flow_rate
                    path = self.flow_paths.get(flow_id, [])
                    for i in range(len(path) - 1):
                        u, u_link = path[i]
                        v, _ = path[i + 1]
                        if u_link is None:
                            continue
                        self.link_loads[(u, v, u_link)] = max(0, self.link_loads.get((u, v, u_link), 0) - flow_rate)
                        self.link_loads[(u, v, u_link)] += new_flow_rate
                    self.events.append(event)
            elif event_type == "SiteFailure":
                non_hq_sites = [s for s, p in self.site_properties.items() if p["type"] != "Headquarters"]
                if non_hq_sites:
                    site = random.choice(non_hq_sites)
                    duration = random.randint(10, 30)
                    event = {
                        "type": "SiteFailure",
                        "description": f"Site failure: Site-{site} ({self.site_properties[site]['type']})",
                        "site": site,
                        "start_time": self.current_time,
                        "end_time": self.current_time + duration,
                        "resolved": False
                    }
                    self.site_status[site] = "Down"
                    flows_to_remove = [fid for fid, (s, d, _) in self.active_flows.items() if s == site or d == site]
                    for fid in flows_to_remove:
                        self.remove_flow(fid)
                    self.events.append(event)

    def _resolve_network_events(self):
        """Resolve network events."""
        for event in self.events:
            if not event["resolved"] and event["end_time"] <= self.current_time:
                event["resolved"] = True
                if event["type"] == "LinkFailure":
                    self.transport_links[(event["u"], event["v"])][event["link_type"]] = True
                elif event["type"] == "TrafficBurst":
                    flow_id = event["flow_id"]
                    if flow_id in self.active_flows:
                        original_rate = event["original_rate"]
                        burst_rate = event["burst_rate"]
                        self.flow_rates[flow_id] = original_rate
                        path = self.flow_paths.get(flow_id, [])
                        for i in range(len(path) - 1):
                            u, u_link = path[i]
                            v, _ = path[i + 1]
                            if u_link is None:
                                continue
                            self.link_loads[(u, v, u_link)] = max(0, self.link_loads.get((u, v, u_link), 0) - burst_rate)
                            self.link_loads[(u, v, u_link)] += original_rate
                elif event["type"] == "SiteFailure":
                    self.site_status[event["site"]] = "Operational"

    def get_state(self):
        """
        Get the current state of the SD-WAN environment.

        Returns:
            tuple: (link_loads, congestion_level, flow_level, health_level)
        """
        load_levels = {}
        for (u, v, link_type), load in self.link_loads.items():
            capacity = self.link_capacities.get((u, v, link_type), 100)
            utilization = load / capacity if capacity > 0 else 1
            load_levels[(u, v, link_type)] = 0 if utilization < 0.3 else 1 if utilization < 0.7 else 2

        congested_links = sum(1 for k, load in self.link_loads.items() if load / self.link_capacities.get(k, 1) > 0.8)
        congestion_level = 0 if congested_links == 0 else 1 if congested_links < 3 else 2 if congested_links < 6 else 3

        num_flows = len(self.active_flows)
        flow_level = 0 if num_flows < 5 else 1 if num_flows < 15 else 2

        active_events = sum(1 for e in self.events if not e["resolved"] and e["start_time"] <= self.current_time <= e["end_time"])
        health_level = 0 if active_events == 0 else 1 if active_events < 2 else 2

        return (load_levels, congestion_level, flow_level, health_level)

    def step(self, action):
        """
        Take a step in the SD-WAN environment.

        Args:
            action (tuple): (source, destination, flow_rate, app_type, path)

        Returns:
            tuple: (next_state, reward, done, info)
        """
        source, destination, flow_rate, app_type, path = action
        self.current_time += 1
        self._generate_network_events()
        self._resolve_network_events()

        success = self.add_flow(source, destination, flow_rate, app_type, path)
        if success:
            delay, available_bw, loss_prob, jitter, satisfaction = self.calculate_path_metrics(path, app_type)
            reward = (flow_rate / 100) - 0.3 * min(1, delay / 500) - 0.3 * loss_prob + 0.4 * satisfaction
        else:
            reward = -1.0

        next_state = self.get_state()
        done = False
        info = {
            'success': success,
            'throughput': self.throughput,
            'avg_delay': self.avg_delay,
            'packet_loss': self.packet_loss,
            'rejection_rate': self.rejected_flows / max(1, self.total_flows),
            'app_satisfaction': dict(self.app_satisfaction)
        }
        return next_state, reward, done, info

    def reset(self):
        """
        Reset the SD-WAN environment.

        Returns:
            tuple: Initial state
        """
        self.active_flows = {}
        self.flow_paths = {}
        self.flow_rates = {}
        self.flow_apps = {}
        self.link_loads = defaultdict(float)
        self.rejected_flows = 0
        self.total_flows = 0
        self.throughput = 0
        self.avg_delay = 0
        self.packet_loss = 0
        self.app_satisfaction = defaultdict(float)
        self.link_utilization = defaultdict(lambda: defaultdict(float))
        self.link_health = defaultdict(lambda: defaultdict(float))
        self.site_status = defaultdict(lambda: "Operational")
        self.current_time = 0
        self.events = []
        for u, v in self.G.edges():
            for link_type in self.link_types:
                if link_type in self.transport_links.get((u, v), {}):
                    self.transport_links[(u, v)][link_type] = True
        return self.get_state()

    def visualize_network(self, save_path=None):
        """
        Visualize the SD-WAN topology with link loads.

        Args:
            save_path (str, optional): Path to save visualization
        """
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(self.G, seed=self.seed)

        node_colors = []
        for node in self.G.nodes():
            if self.site_status[node] == "Down":
                node_colors.append('red')
            elif self.site_properties[node]["type"] == "Headquarters":
                node_colors.append('blue')
            elif self.site_properties[node]["type"] == "Data Center":
                node_colors.append('green')
            else:
                node_colors.append('orange')
        nx.draw_networkx_nodes(self.G, pos, node_size=700, node_color=node_colors, alpha=0.8)

        labels = {node: f"Site-{node}\n({self.site_properties[node]['type']})" for node in self.G.nodes()}
        nx.draw_networkx_labels(self.G, pos, labels=labels, font_size=10)

        for link_type in self.link_types:
            edges = []
            edge_colors = []
            edge_widths = []
            for u, v in self.G.edges():
                if link_type in self.transport_links.get((u, v), {}) and self.transport_links[(u, v)][link_type]:
                    edges.append((u, v))
                    load = self.link_loads.get((u, v, link_type), 0)
                    capacity = self.link_capacities.get((u, v, link_type), 100)
                    utilization = load / capacity if capacity > 0 else 0
                    edge_widths.append(1 + 3 * (capacity / self.max_capacity))
                    edge_colors.append('red' if utilization > 0.8 else 'orange' if utilization > 0.5 else 'green')
            style = 'solid' if link_type == "MPLS" else 'dashed' if link_type == "Internet" else 'dotted'
            nx.draw_networkx_edges(self.G, pos, edgelist=edges, width=edge_widths,
                                  edge_color=edge_colors, style=style, connectionstyle="arc3,rad=0.1")

        edge_labels = {}
        for u, v in self.G.edges():
            label_parts = []
            for link_type in self.link_types:
                if link_type in self.transport_links.get((u, v), {}) and self.transport_links[(u, v)][link_type]:
                    load = self.link_loads.get((u, v, link_type), 0)
                    capacity = self.link_capacities.get((u, v, link_type), 100)
                    label_parts.append(f"{link_type}: {load:.1f}/{capacity}")
            if label_parts:
                edge_labels[(u, v)] = "\n".join(label_parts)
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_size=8)

        plt.title("SD-WAN Topology with Link Loads")
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    env = SDWANEnvironment(num_sites=6, connectivity_prob=0.6)
    env.visualize_network()
    env.add_flow(0, 3, 20, "Voice")
    env.add_flow(1, 4, 50, "Video")
    env.add_flow(2, 5, 30, "Web")
    env.visualize_network()
    print(f"Throughput: {env.throughput}")
    print(f"Average Delay: {env.avg_delay}")
    print(f"Packet Loss: {env.packet_loss}")
    for app_type, satisfaction in env.app_satisfaction.items():
        print(f"{app_type} Satisfaction: {satisfaction:.2f}")