import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch, Circle
import matplotlib.patheffects as path_effects
import networkx as nx
import os
import random
import time
from collections import defaultdict
from sdwan_environment import SDWANEnvironment
from sdwan_components import SDWANComponents
from sdwan_tensor_rl import SDWANTensorRL


class SDWANPacketVisualizer:
    """
    Enhanced packet visualization for SD-WAN with RL-based routing.
    This class creates detailed visualizations showing packet transfers between PCs
    in an SD-WAN environment with RL-based routing decisions.
    """

    def __init__(self, env, agent=None, figsize=(16, 12)):
        """
        Initialize the SD-WAN packet visualizer.

        Args:
            env: SDWANEnvironment instance
            agent: SDWANTensorRL agent (optional)
            figsize: Figure size for visualization
        """
        self.env = env
        self.agent = agent
        self.figsize = figsize
        self.G = env.G
        self.pos = nx.spring_layout(self.G, seed=env.seed)

        # SD-WAN components for styling
        self.components = SDWANComponents()

        # Visualization elements
        self.fig = None
        self.axes = {}
        self.node_objects = {}
        self.edge_objects = {}
        self.packet_objects = []
        self.packet_trails = {}
        self.active_packets = []
        self.packet_history = []
        self.event_texts = []
        self.stats_text = None
        self.decision_text = None
        self.time_step = 0

        # PC icons and styling
        self.pc_radius = 0.06
        self.pc_colors = plt.cm.tab10(np.linspace(0, 1, env.num_sites))

        # Packet styling
        self.packet_size = 150
        self.packet_colors = {
            "Voice": "red",
            "Video": "purple",
            "Web": "cyan",
            "Database": "blue",
            "File Transfer": "green"
        }

        # Animation properties
        self.animation = None
        self.is_paused = False
        self.action_index = 0  # To track actions_history

    def setup_visualization(self):
        """Set up the visualization dashboard."""
        self.fig = plt.figure(figsize=self.figsize)

        # Create grid spec
        gs = self.fig.add_gridspec(3, 4)

        # Create subplots
        self.axes = {
            "topology": self.fig.add_subplot(gs[0:2, 0:3]),  # Main topology view
            "events": self.fig.add_subplot(gs[0, 3]),  # Network events
            "metrics": self.fig.add_subplot(gs[1, 3]),  # Performance metrics
            "flows": self.fig.add_subplot(gs[2, 0:2]),  # Active flows
            "decisions": self.fig.add_subplot(gs[2, 2:4])  # RL decisions
        }

        # Set titles
        self.axes["topology"].set_title("SD-WAN Topology with Packet Transfers", fontsize=14)
        self.axes["events"].set_title("Network Events", fontsize=12)
        self.axes["metrics"].set_title("Performance Metrics", fontsize=12)
        self.axes["flows"].set_title("Active Flows", fontsize=12)
        self.axes["decisions"].set_title("RL Routing Decisions", fontsize=12)

        # Remove axis from topology view
        self.axes["topology"].axis('off')

        # Initialize PC nodes
        self._init_pc_nodes()

        # Initialize transport links
        self._init_transport_links()

        # Initialize stats text
        self.stats_text = self.axes["metrics"].text(
            0.5, 0.5, "Initializing...",
            horizontalalignment='center', verticalalignment='center',
            fontsize=10, transform=self.axes["metrics"].transAxes,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5')
        )

        # Initialize decision text
        self.decision_text = self.axes["decisions"].text(
            0.5, 0.5, "No routing decisions yet",
            horizontalalignment='center', verticalalignment='center',
            fontsize=10, transform=self.axes["decisions"].transAxes,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5')
        )

        # Remove axis from other views
        self.axes["events"].axis('off')
        self.axes["metrics"].axis('off')
        self.axes["decisions"].axis('off')

        # Adjust layout
        plt.tight_layout()

        # Add pause/resume functionality
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

    def _on_key_press(self, event):
        """Handle key press events for pause/resume."""
        if event.key == ' ':  # Space bar
            self.is_paused = not self.is_paused
            if self.is_paused:
                self.animation.event_source.stop()
            else:
                self.animation.event_source.start()

    def _init_pc_nodes(self):
        """Initialize PC nodes in the visualization."""
        for node in self.G.nodes():
            # Get site properties
            site_props = self.env.site_properties[node]
            site_type = site_props["type"]

            # Get styling from components
            color = self.components.site_types[site_type]["color"]
            icon = self.components.site_types[site_type]["icon"]

            # Create PC node
            pc = Circle(
                self.pos[node],
                radius=self.pc_radius,
                facecolor=color,
                edgecolor='black',
                alpha=0.8,
                zorder=10
            )
            self.axes["topology"].add_patch(pc)

            # Add site label with icon
            label = self.axes["topology"].text(
                self.pos[node][0],
                self.pos[node][1],
                f"{icon}\nSite-{node}",
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=9,
                fontweight='bold',
                color='white',
                zorder=11
            )

            # Add outline to text for better visibility
            label.set_path_effects([
                path_effects.Stroke(linewidth=2, foreground='black'),
                path_effects.Normal()
            ])

            # Store node objects
            self.node_objects[node] = {
                'pc': pc,
                'label': label,
                'flash_time': 0
            }

    def _init_transport_links(self):
        """Initialize transport links in the visualization."""
        self.edge_objects = {}

        for link_type, properties in self.components.transport_types.items():
            style = properties["style"]
            base_width = properties["width"]
            base_color = properties["color"]

            for u, v in self.G.edges():
                if link_type in self.env.transport_links.get((u, v), {}) and self.env.transport_links[(u, v)][link_type]:
                    # Create edge
                    edge = FancyArrowPatch(
                        posA=self.pos[u],
                        posB=self.pos[v],
                        arrowstyle='-|>',
                        mutation_scale=15,
                        lw=base_width,
                        color=base_color,
                        alpha=0.6,
                        linestyle=style,
                        connectionstyle="arc3,rad=0.1",  # Curved edges
                        zorder=5
                    )
                    self.axes["topology"].add_patch(edge)

                    # Add link label
                    x1, y1 = self.pos[u]
                    x2, y2 = self.pos[v]
                    midx = (x1 + x2) / 2
                    midy = (y1 + y2) / 2
                    # Add small offset for label
                    dx = -(y2 - y1) * 0.1
                    dy = (x2 - x1) * 0.1

                    label = self.axes["topology"].text(
                        midx + dx,
                        midy + dy,
                        f"{link_type}",
                        fontsize=7,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                        horizontalalignment='center',
                        verticalalignment='center',
                        zorder=6
                    )

                    # Store edge objects
                    key = (u, v, link_type)
                    self.edge_objects[key] = {
                        'edge': edge,
                        'label': label,
                        'base_width': base_width,
                        'base_color': base_color
                    }

    def _create_packet(self, source, destination, flow_rate, app_type, path):
        """
        Create a packet object for visualization.

        Args:
            source: Source site
            destination: Destination site
            flow_rate: Flow rate
            app_type: Application type
            path: Path for the packet

        Returns:
            dict: Packet object
        """
        # Extract nodes from path
        nodes = [node for node, _ in path]

        # Extract link types from path
        link_types = []
        for i in range(len(path) - 1):
            _, link_type = path[i]
            link_types.append(link_type)

        # Get packet color based on application type
        color = self.packet_colors.get(app_type, "gray")

        # Create packet object
        packet = {
            'id': len(self.packet_history) + 1,
            'source': source,
            'destination': destination,
            'flow_rate': flow_rate,
            'app_type': app_type,
            'path': path,
            'nodes': nodes,
            'link_types': link_types,
            'color': color,
            'position': 0,  # Position along the path (0 to 1)
            'segment': 0,  # Current segment of the path
            'speed': 0.05 + (0.05 * random.random()),  # Random speed variation
            'size': self.packet_size,
            'active': True,
            'delivered': False,
            'creation_time': self.time_step,
            'delivery_time': None,
            'visual': None,  # Matplotlib object
            'trail': []  # Trail of positions for visualization
        }

        # Create visual representation
        if len(nodes) > 1:
            u, v = nodes[0], nodes[1]
            x1, y1 = self.pos[u]
            x2, y2 = self.pos[v]

            # Start at the source node
            packet_x, packet_y = x1, y1

            # Create packet visual (scatter point)
            visual = self.axes["topology"].scatter(
                packet_x, packet_y,
                s=packet['size'],
                color=packet['color'],
                alpha=0.8,
                edgecolor='black',
                marker='o',
                zorder=15
            )

            packet['visual'] = visual

            # Initialize trail
            packet['trail'] = [(packet_x, packet_y)]

            # Add app type icon
            icon = self.components.application_types.get(app_type, {}).get("icon", "📦")
            label = self.axes["topology"].text(
                packet_x, packet_y,
                icon,
                fontsize=8,
                color='white',
                horizontalalignment='center',
                verticalalignment='center',
                zorder=16
            )

            # Add outline to text for better visibility
            label.set_path_effects([
                path_effects.Stroke(linewidth=1, foreground='black'),
                path_effects.Normal()
            ])

            packet['label'] = label

        return packet

    def _update_packet_position(self, packet):
        """
        Update packet position along its path.

        Args:
            packet: Packet object

        Returns:
            bool: True if packet is still active, False if delivered or lost
        """
        if not packet['active']:
            return False

        # Get current segment
        segment = packet['segment']
        nodes = packet['nodes']

        if segment >= len(nodes) - 1:
            # Packet has reached destination
            packet['active'] = False
            packet['delivered'] = True
            packet['delivery_time'] = self.time_step

            # Flash destination node
            dest_node = nodes[-1]
            self.node_objects[dest_node]['flash_time'] = 5  # Flash for 5 frames

            return False

        # Get current segment endpoints
        u, v = nodes[segment], nodes[segment + 1]
        x1, y1 = self.pos[u]
        x2, y2 = self.pos[v]

        # Update position along segment
        packet['position'] += packet['speed']

        if packet['position'] >= 1:
            # Move to next segment
            packet['segment'] += 1
            packet['position'] = 0

            # Check if packet has reached destination
            if packet['segment'] >= len(nodes) - 1:
                # Packet has reached destination
                packet['active'] = False
                packet['delivered'] = True
                packet['delivery_time'] = self.time_step

                # Flash destination node
                dest_node = nodes[-1]
                self.node_objects[dest_node]['flash_time'] = 5  # Flash for 5 frames

                return False

        # Calculate new position
        segment = packet['segment']
        u, v = nodes[segment], nodes[segment + 1]
        x1, y1 = self.pos[u]
        x2, y2 = self.pos[v]

        # Add curve to path for better visualization
        t = packet['position']

        # Get link type for this segment
        link_type = packet['link_types'][segment] if segment < len(packet['link_types']) else None

        # Check if link is active
        if link_type and not self.env.transport_links.get((u, v), {}).get(link_type, True):
            # Link failure, packet is lost
            packet['active'] = False
            packet['delivered'] = False
            return False

        # Calculate curved path position using quadratic Bezier curve
        control_x = (x1 + x2) / 2 - (y2 - y1) * 0.1  # Control point with offset
        control_y = (y1 + y2) / 2 + (x2 - x1) * 0.1
        # Quadratic Bezier: B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
        packet_x = (1 - t)**2 * x1 + 2 * (1 - t) * t * control_x + t**2 * x2
        packet_y = (1 - t)**2 * y1 + 2 * (1 - t) * t * control_y + t**2 * y2

        # Update visual position
        if packet['visual']:
            packet['visual'].set_offsets([(packet_x, packet_y)])

            # Update label position
            if 'label' in packet:
                packet['label'].set_position((packet_x, packet_y))

        # Update trail
        packet['trail'].append((packet_x, packet_y))
        if len(packet['trail']) > 10:  # Limit trail length
            packet['trail'] = packet['trail'][-10:]

        # Draw trail
        if packet['id'] not in self.packet_trails:
            # Create new trail
            trail_line, = self.axes["topology"].plot(
                [p[0] for p in packet['trail']],
                [p[1] for p in packet['trail']],
                color=packet['color'],
                alpha=0.4,
                linewidth=2,
                zorder=14
            )
            self.packet_trails[packet['id']] = trail_line
        else:
            # Update existing trail
            self.packet_trails[packet['id']].set_data(
                [p[0] for p in packet['trail']],
                [p[1] for p in packet['trail']]
            )

        return True

    def _update_link_utilization(self):
        """Update link utilization visualization."""
        for (u, v, link_type), load in self.env.link_loads.items():
            if (u, v, link_type) in self.edge_objects:
                edge_obj = self.edge_objects[(u, v, link_type)]
                edge = edge_obj['edge']
                base_width = edge_obj['base_width']
                base_color = edge_obj['base_color']

                # Get capacity
                capacity = self.env.link_capacities.get((u, v, link_type), 100)

                # Calculate utilization
                utilization = load / capacity if capacity > 0 else 0

                # Update edge width based on utilization
                width = base_width * (1 + utilization)
                edge.set_linewidth(width)

                # Update edge color based on utilization
                if utilization > 0.8:
                    color = 'red'
                elif utilization > 0.5:
                    color = 'orange'
                else:
                    color = base_color

                edge.set_color(color)

                # Update label
                label = edge_obj['label']
                label.set_text(f"{link_type}\n{load:.1f}/{capacity}")

                # Update label color based on utilization
                if utilization > 0.8:
                    label.set_bbox(dict(facecolor='red', alpha=0.7, edgecolor='none', pad=1))
                    label.set_color('white')
                elif utilization > 0.5:
                    label.set_bbox(dict(facecolor='orange', alpha=0.7, edgecolor='none', pad=1))
                    label.set_color('black')
                else:
                    label.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                    label.set_color('black')

    def _update_node_status(self):
        """Update node status visualization."""
        for node, obj in self.node_objects.items():
            pc = obj['pc']

            # Check if node is flashing
            if obj['flash_time'] > 0:
                # Flash node
                pc.set_alpha(0.5 + 0.5 * np.sin(obj['flash_time'] * np.pi))
                obj['flash_time'] -= 1
            else:
                # Reset alpha
                pc.set_alpha(0.8)

            # Check if site is down
            if self.env.site_status.get(node, "Operational") == "Down":
                pc.set_facecolor('gray')
                pc.set_alpha(0.5)
            else:
                # Get site properties
                site_props = self.env.site_properties[node]
                site_type = site_props["type"]

                # Get styling from components
                color = self.components.site_types[site_type]["color"]
                pc.set_facecolor(color)

    def _update_events_panel(self):
        """Update network events panel."""
        # Clear existing event texts
        for text in self.event_texts:
            text.remove()
        self.event_texts = []

        # Get active events
        active_events = [e for e in self.env.events
                         if not e["resolved"] and e["start_time"] <= self.time_step <= e["end_time"]]

        # Display events
        y_pos = 0.9
        for event in active_events[-5:]:  # Show last 5 events
            event_type = event["type"]
            description = event["description"]

            # Get event properties
            event_props = self.components.event_types.get(event_type, {})
            icon = event_props.get("icon", "⚠")
            color = event_props.get("color", "red")

            # Create event text
            text = self.axes["events"].text(
                0.05, y_pos,
                f"{icon} {description}",
                fontsize=9,
                color=color,
                transform=self.axes["events"].transAxes,
                verticalalignment='top'
            )

            self.event_texts.append(text)
            y_pos -= 0.15

    def _update_metrics_panel(self):
        """Update performance metrics panel."""
        # Get current metrics
        throughput = self.env.throughput
        avg_delay = self.env.avg_delay
        packet_loss = self.env.packet_loss
        rejection_rate = self.env.rejected_flows / max(1, self.env.total_flows)

        # Get application satisfaction
        app_satisfaction = self.env.app_satisfaction

        # Create metrics text
        metrics_text = f"Network Performance Metrics:\n\n"
        metrics_text += f"Throughput: {throughput:.2f} Mbps\n"
        metrics_text += f"Average Delay: {avg_delay:.2f} ms\n"
        metrics_text += f"Packet Loss: {packet_loss:.4f}\n"
        metrics_text += f"Flow Rejection Rate: {rejection_rate:.2f}\n\n"

        metrics_text += f"Application Satisfaction:\n"
        for app_type, satisfaction in app_satisfaction.items():
            metrics_text += f"  {app_type}: {satisfaction:.2f}\n"

        # Update stats text
        self.stats_text.set_text(metrics_text)

    def _update_flows_panel(self):
        """Update active flows panel."""
        # Clear existing flows
        self.axes["flows"].clear()
        self.axes["flows"].set_title("Active Flows", fontsize=12)
        self.axes["flows"].axis('off')

        # Get active flows
        active_flows = self.env.active_flows

        # Display flows
        y_pos = 0.95
        for flow_id, (source, destination, flow_rate) in list(active_flows.items())[:10]:  # Show top 10 flows
            app_type = self.env.flow_apps.get(flow_id, "Unknown")
            path = self.env.flow_paths.get(flow_id, [])

            # Get app properties
            app_props = self.components.application_types.get(app_type, {})
            icon = app_props.get("icon", "📦")
            color = app_props.get("color", "gray")

            # Create path description
            path_desc = ""
            if path:
                path_nodes = [f"Site-{node}" for node, _ in path]
                path_desc = " → ".join(path_nodes)

            # Create flow text
            flow_text = f"{icon} Flow {flow_id}: {app_type}\n"
            flow_text += f"  Source: Site-{source} → Destination: Site-{destination}\n"
            flow_text += f"  Rate: {flow_rate} Mbps\n"
            if path_desc:
                flow_text += f"  Path: {path_desc}\n"

            # Add flow text
            self.axes["flows"].text(
                0.05, y_pos,
                flow_text,
                fontsize=8,
                color=color,
                transform=self.axes["flows"].transAxes,
                verticalalignment='top'
            )

            y_pos -= 0.15

    def _update_decisions_panel(self, decision=None):
        """
        Update RL decisions panel.

        Args:
            decision (dict, optional): Latest routing decision
        """
        if decision:
            source = decision.get('source')
            destination = decision.get('destination')
            flow_rate = decision.get('flow_rate')
            app_type = decision.get('app_type')
            path = decision.get('path')
            q_values = decision.get('q_values', {})

            # Create decision text
            decision_text = f"RL Routing Decision:\n\n"
            decision_text += f"Flow: Site-{source} → Site-{destination}\n"
            decision_text += f"Application: {app_type} ({flow_rate} Mbps)\n\n"

            if path:
                path_nodes = [f"Site-{node}" for node, _ in path]
                path_desc = " → ".join(path_nodes)
                decision_text += f"Selected Path: {path_desc}\n\n"

                # Add transport types used
                transport_types = {}
                for i in range(len(path) - 1):
                    _, link_type = path[i]
                    if link_type:
                        transport_types[link_type] = transport_types.get(link_type, 0) + 1

                decision_text += "Transport Links Used:\n"
                for link_type, count in transport_types.items():
                    decision_text += f"  {link_type}: {count} segments\n"

            if q_values:
                decision_text += "\nQ-Values for Top Paths:\n"
                for i, (p, q) in enumerate(sorted(q_values.items(), key=lambda x: x[1], reverse=True)[:3]):
                    decision_text += f"  Path {i + 1}: {q:.2f}\n"

            # Update decision text
            self.decision_text.set_text(decision_text)

    def _init_animation(self):
        """Initialize animation (required by FuncAnimation)."""
        return []

    def _update_animation(self, frame):
        """
        Update animation frame.

        Args:
            frame: Frame number

        Returns:
            list: Updated artists
        """
        if self.is_paused:
            return []

        # Increment time step
        self.time_step += 1

        # Update network events
        self.env._generate_network_events()
        self.env._resolve_network_events()

        # Update link utilization visualization
        self._update_link_utilization()

        # Update node status visualization
        self._update_node_status()

        # Update events panel
        self._update_events_panel()

        # Update metrics panel
        self._update_metrics_panel()

        # Update flows panel
        self._update_flows_panel()

        # Update active packets
        active_packets = []
        for packet in self.active_packets:
            if self._update_packet_position(packet):
                active_packets.append(packet)
        self.active_packets = active_packets

        # Generate new packets using actions_history if available
        if frame % 10 == 0 and len(self.active_packets) < 10:
            if self.agent and self.agent.actions_history and self.action_index < len(self.agent.actions_history):
                # Use action from history
                source, destination, flow_rate, app_type, path = self.agent.actions_history[self.action_index]
                self.action_index = (self.action_index + 1) % len(self.agent.actions_history)  # Cycle through actions
            else:
                # Fallback to random flow
                source = np.random.randint(0, self.env.num_sites)
                destination = np.random.randint(0, self.env.num_sites)
                while source == destination:
                    destination = np.random.randint(0, self.env.num_sites)
                flow_rate = np.random.randint(5, 50)
                app_type = np.random.choice(list(self.env.policies.keys()))

                # Select path using RL agent if available
                if self.agent:
                    state = self.env.get_state()
                    epsilon_backup = self.agent.epsilon
                    self.agent.epsilon = 0.0  # No exploration for visualization
                    action = self.agent.select_action(state, source, destination, flow_rate, app_type)
                    self.agent.epsilon = epsilon_backup
                    _, _, _, _, path = action
                else:
                    possible_paths = self.env.get_possible_paths(source, destination, app_type)
                    path = possible_paths[0] if possible_paths else None

            if path:
                # Get Q-values for decision panel
                q_values = {}
                if self.agent:
                    state = self.env.get_state()
                    state_idx = self.agent._state_to_index(state)
                    possible_paths = self.env.get_possible_paths(source, destination, app_type)
                    for p in possible_paths:
                        action = (source, destination, flow_rate, app_type, p)
                        action_idx = self.agent._action_to_index(action)
                        q_value = self.agent.Q[state_idx + action_idx]
                        q_values[str(p)] = q_value

                # Create packet
                packet = self._create_packet(source, destination, flow_rate, app_type, path)
                self.active_packets.append(packet)
                self.packet_history.append(packet)

                # Add flow to environment
                self.env.add_flow(source, destination, flow_rate, app_type, path)

                # Update decisions panel
                self._update_decisions_panel({
                    'source': source,
                    'destination': destination,
                    'flow_rate': flow_rate,
                    'app_type': app_type,
                    'path': path,
                    'q_values': q_values
                })

        # Remove old flows
        if frame % 30 == 0:
            flow_ids = list(self.env.active_flows.keys())
            if flow_ids:
                oldest_flow = flow_ids[0]
                self.env.remove_flow(oldest_flow)

        return []

    def simulate_rl_routing(self, num_steps=200, interval=100, save_path=None):
        """
        Simulate RL-based routing with packet visualization.

        Args:
            num_steps (int): Number of simulation steps
            interval (int): Interval between frames in milliseconds
            save_path (str, optional): Path to save the animation

        Returns:
            animation.FuncAnimation: Animation object
        """
        # Set up visualization
        self.setup_visualization()

        # Create animation
        self.animation = animation.FuncAnimation(
            self.fig,
            self._update_animation,
            frames=num_steps,
            init_func=self._init_animation,
            interval=interval,
            blit=False
        )

        # Save animation if path is provided
        if save_path:
            try:
                self.animation.save(save_path, writer='ffmpeg', fps=10, dpi=100)
                print(f"Animation saved to {save_path}")
                plt.close()
                return save_path
            except Exception as e:
                print(f"Error saving animation: {e}")
                print("Ensure 'ffmpeg' is installed. On Windows, download from https://ffmpeg.org/download.html and add to PATH.")
                plt.show()
                return None

        # Show animation
        plt.show()
        return self.animation


def run_packet_visualization(num_sites=6, num_episodes=50, save_dir='sdwan_results'):
    """
    Run packet visualization for SD-WAN with RL-based routing.

    Args:
        num_sites (int): Number of sites in the SD-WAN
        num_episodes (int): Number of episodes to train the RL agent
        save_dir (str): Directory to save results

    Returns:
        dict: Results including trained agent, environment, and paths to saved files
    """
    # Create results directory
    os.makedirs(save_dir, exist_ok=True)

    # Create SD-WAN environment
    print("Creating SD-WAN environment...")
    env = SDWANEnvironment(num_sites=num_sites, connectivity_prob=0.4)

    # Define state and action dimensions
    state_dim = (4, 4, 3, 3)  # (load_level, congestion_level, flow_level, health_level)
    action_dim = (num_sites, num_sites, 3, 5, 4)  # (source_site, destination_site, flow_rate_level, app_type, path_index)

    # Create tensor-based RL agent
    print("Initializing tensor-based RL agent...")
    components = SDWANComponents()
    agent = SDWANTensorRL(env, components, state_dim, action_dim)

    # Train the agent
    print(f"Training agent for {num_episodes} episodes...")
    training_history = agent.train(num_episodes=num_episodes, max_steps=30)

    # Create packet transfer visualizer
    print("Creating packet transfer visualization...")
    visualizer = SDWANPacketVisualizer(env, agent)

    # Run RL routing simulation with packet visualization
    print("Running RL routing simulation with packet visualization...")
    animation_path = os.path.join(save_dir, 'packet_transfer_animation.mp4')
    ani = visualizer.simulate_rl_routing(num_steps=200, interval=100, save_path=animation_path)

    print(f"Visualization complete! Animation saved to {animation_path}")

    return {
        'agent': agent,
        'env': env,
        'visualizer': visualizer,
        'animation_path': animation_path
    }


if __name__ == "__main__":
    # Run the packet visualization
    results = run_packet_visualization()