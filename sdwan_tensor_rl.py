import numpy as np
import tensorflow as tf
import tensorly as tl
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import random
from collections import defaultdict
from sdwan_environment import SDWANEnvironment
from sdwan_components import SDWANComponents


class SDWANTensorRL:
    """
    Tensor-Based Reinforcement Learning for SD-WAN routing optimization.
    Integrates Q-learning with Tucker decomposition for efficient routing.
    """

    def __init__(self, env, components, state_dim=(4, 4, 3, 3), action_dim=(6, 6, 3, 5, 4),
                 learning_rate=0.01, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995,
                 tucker_rank=[5, 5, 5, 5, 5, 5, 5, 5], update_frequency=10):
        """
        Initialize the SD-WAN Tensor-Based RL agent.

        Args:
            env: SDWANEnvironment instance
            components: SDWANComponents instance
            state_dim: (load_level, congestion_level, flow_level, health_level)
            action_dim: (source, destination, flow_rate_level, app_type, path_index)
            learning_rate: Learning rate for Q-learning
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            tucker_rank: Rank for Tucker decomposition
            update_frequency: Frequency of Tucker updates
        """
        self.env = env
        self.components = components
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tucker_rank = tucker_rank
        self.update_frequency = update_frequency

        # Initialize Q-tensor
        self._init_q_tensor()

        # Tucker decomposition components
        self.core = None
        self.factors = None
        self.use_tucker = True

        # Training metrics
        self.rewards_history = []
        self.loss_history = []
        self.rejection_rate_history = []
        self.app_satisfaction_history = defaultdict(list)
        self.actions_history = []  # For sdwan_packet_visualization.py
        self.steps_count = 0

        # Set tensorly backend
        tl.set_backend('tensorflow')

    def _init_q_tensor(self):
        """Initialize Q-tensor with zeros."""
        tensor_shape = self.state_dim + self.action_dim
        print(f"Initializing Q-tensor with shape: {tensor_shape}")
        self.Q = np.zeros(tensor_shape, dtype=np.float32)

    def _state_to_index(self, state):
        """
        Convert state to Q-tensor index.

        Args:
            state: (load_levels, congestion_level, flow_level, health_level)

        Returns:
            tuple: (load_idx, congestion_idx, flow_idx, health_idx)
        """
        try:
            load_levels, congestion_level, flow_level, health_level = state
            # Summarize load_levels dict
            if isinstance(load_levels, dict):
                load_values = list(load_levels.values())
                avg_load = sum(load_values) / len(load_values) if load_values else 0
                load_level = min(3, int(avg_load))  # 0: low, 1: med, 2: high, 3: critical
            else:
                load_level = 0
            congestion_idx = min(int(congestion_level), self.state_dim[1] - 1)
            flow_idx = min(int(flow_level), self.state_dim[2] - 1)
            health_idx = min(int(health_level), self.state_dim[3] - 1)
        except (ValueError, TypeError) as e:
            print(f"State indexing error: {e}, state: {state}")
            load_level = congestion_idx = flow_idx = health_idx = 0
        return (load_level, congestion_idx, flow_idx, health_idx)

    def _action_to_index(self, action):
        """
        Convert action to Q-tensor index.

        Args:
            action: (source, destination, flow_rate, app_type, path)

        Returns:
            tuple: (source_idx, dest_idx, flow_idx, app_idx, path_idx)
        """
        try:
            source, destination, flow_rate, app_type, path = action
            # Flow rate levels
            flow_idx = 0 if flow_rate < 10 else 1 if flow_rate < 30 else 2
            # App type index
            app_types = list(self.components.application_types.keys())
            app_idx = app_types.index(app_type) if app_type in app_types else 0
            # Path index
            if not path or len(path) < 2:
                path_idx = 0
            else:
                link_count = sum(1 for _, lt in path if lt)
                path_idx = min(3, link_count)  # 0: none, 1: short, 2: med, 3: long
            source_idx = source % self.action_dim[0]
            dest_idx = destination % self.action_dim[1]
        except (ValueError, TypeError) as e:
            print(f"Action indexing error: {e}, action: {action}")
            source_idx = dest_idx = flow_idx = app_idx = path_idx = 0
        return (source_idx, dest_idx, flow_idx, app_idx, path_idx)

    def _apply_tucker_decomposition(self):
        """Apply Tucker decomposition to Q-tensor."""
        if not self.use_tucker or not self.tucker_rank:
            return
        try:
            # Convert Q-tensor to tensorly format
            Q_tensor = tl.tensor(self.Q, dtype=tl.float32)
            # Perform Tucker decomposition
            core, factors = tl.decomposition.tucker(Q_tensor, rank=self.tucker_rank)
            self.core = core
            self.factors = factors
            # Reconstruct Q-tensor
            self.Q = tl.tucker_to_tensor((core, factors))
            print(f"Applied Tucker decomposition, core shape: {self.core.shape}")
        except Exception as e:
            print(f"Tucker decomposition failed: {e}")
            self.use_tucker = False

    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value using Q-learning.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state

        Returns:
            float: TD error
        """
        state_idx = self._state_to_index(state)
        action_idx = self._action_to_index(action)
        current_q = self.Q[state_idx + action_idx]

        next_state_idx = self._state_to_index(next_state)
        max_next_q = np.max(self.Q[next_state_idx])

        target_q = reward + self.gamma * max_next_q
        td_error = target_q - current_q

        self.Q[state_idx + action_idx] += self.learning_rate * td_error

        self.steps_count += 1
        if self.steps_count % self.update_frequency == 0:
            self._apply_tucker_decomposition()

        return td_error

    def select_action(self, state, source, destination, flow_rate, app_type):
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            source: Source site
            destination: Destination site
            flow_rate: Flow rate
            app_type: Application type

        Returns:
            tuple: (source, destination, flow_rate, app_type, path)
        """
        possible_paths = self.env.get_possible_paths(source, destination, app_type)
        if not possible_paths:
            print(f"No paths from Site-{source} to Site-{destination}")
            return (source, destination, flow_rate, app_type, None)

        if np.random.rand() < self.epsilon:
            path = random.choice(possible_paths)
        else:
            state_idx = self._state_to_index(state)
            best_path = None
            best_q = float('-inf')
            for path in possible_paths:
                action = (source, destination, flow_rate, app_type, path)
                action_idx = self._action_to_index(action)
                q_value = self.Q[state_idx + action_idx]
                if q_value > best_q:
                    best_q = q_value
                    best_path = path
            path = best_path if best_path else random.choice(possible_paths)

        action = (source, destination, flow_rate, app_type, path)
        self.actions_history.append(action)
        return action

    def train(self, num_episodes=100, max_steps=50, flow_rate_range=(5, 50)):
        """
        Train the agent using Q-learning.

        Args:
            num_episodes: Number of episodes
            max_steps: Max steps per episode
            flow_rate_range: Flow rate range

        Returns:
            dict: Training history
        """
        self.rewards_history = []
        self.loss_history = []
        self.rejection_rate_history = []
        self.app_satisfaction_history = defaultdict(list)
        self.actions_history = []

        app_types = list(self.components.application_types.keys())

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            rejection_count = 0
            episode_losses = []

            for step in range(max_steps):
                source = random.randint(0, self.env.num_sites - 1)
                destination = random.randint(0, self.env.num_sites - 1)
                while source == destination:
                    destination = random.randint(0, self.env.num_sites - 1)
                flow_rate = random.randint(flow_rate_range[0], flow_rate_range[1])
                app_type = random.choice(app_types)

                action = self.select_action(state, source, destination, flow_rate, app_type)
                next_state, reward, done, info = self.env.step(action)
                td_error = self.update_q_value(state, action, reward, next_state)

                episode_losses.append(abs(td_error))
                state = next_state
                total_reward += reward
                if not info['success']:
                    rejection_count += 1
                if done:
                    break

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.rewards_history.append(total_reward)
            self.loss_history.append(np.mean(episode_losses) if episode_losses else 0)
            rejection_rate = rejection_count / max_steps if max_steps > 0 else 0
            self.rejection_rate_history.append(rejection_rate)

            for app_type, satisfaction in self.env.app_satisfaction.items():
                self.app_satisfaction_history[app_type].append(satisfaction)

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes}, "
                      f"Reward: {total_reward:.2f}, "
                      f"Rejection Rate: {rejection_rate:.2f}, "
                      f"Epsilon: {self.epsilon:.2f}")

        return {
            'rewards': self.rewards_history,
            'losses': self.loss_history,
            'rejection_rate': self.rejection_rate_history,
            'app_satisfaction': dict(self.app_satisfaction_history),
            'actions_history': self.actions_history
        }
    def plot_training_history(self, save_path=None):
        """
        Plot training history.

        Args:
            save_path: Path to save plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].plot(self.rewards_history)
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')

        axes[0, 1].plot(self.loss_history)
        axes[0, 1].set_title('TD Errors')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Mean Absolute TD Error')

        axes[1, 0].plot(self.rejection_rate_history)
        axes[1, 0].set_title('Flow Rejection Rate')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Rejection Rate')

        for app_type, satisfaction in self.app_satisfaction_history.items():
            axes[1, 1].plot(satisfaction, label=app_type)
        axes[1, 1].set_title('Application Satisfaction')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Satisfaction')
        axes[1, 1].legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def evaluate(self, num_episodes=10, max_steps=50, flow_rate_range=(5, 50)):
        """
        Evaluate the agent.

        Args:
            num_episodes: Number of episodes
            max_steps: Max steps per episode
            flow_rate_range: Flow rate range

        Returns:
            dict: Evaluation metrics
        """
        rewards = []
        rejection_rates = []
        throughputs = []
        avg_delays = []
        packet_losses = []
        app_satisfactions = defaultdict(list)

        app_types = list(self.components.application_types.keys())

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            rejection_count = 0

            for step in range(max_steps):
                source = random.randint(0, self.env.num_sites - 1)
                destination = random.randint(0, self.env.num_sites - 1)
                while source == destination:
                    destination = random.randint(0, self.env.num_sites - 1)
                flow_rate = random.randint(flow_rate_range[0], flow_rate_range[1])
                app_type = random.choice(app_types)

                epsilon_backup = self.epsilon
                self.epsilon = 0
                action = self.select_action(state, source, destination, flow_rate, app_type)
                self.epsilon = epsilon_backup

                next_state, reward, done, info = self.env.step(action)
                state = next_state
                total_reward += reward
                if not info['success']:
                    rejection_count += 1
                if done:
                    break

            rewards.append(total_reward)
            rejection_rates.append(rejection_count / max_steps if max_steps > 0 else 0)
            throughputs.append(self.env.throughput)
            avg_delays.append(self.env.avg_delay)
            packet_losses.append(self.env.packet_loss)
            for app_type, satisfaction in self.env.app_satisfaction.items():
                app_satisfactions[app_type].append(satisfaction)

        avg_metrics = {
            'reward': np.mean(rewards),
            'rejection_rate': np.mean(rejection_rates),
            'throughput': np.mean(throughputs),
            'delay': np.mean(avg_delays),
            'packet_loss': np.mean(packet_losses),
            'app_satisfaction': {app_type: np.mean(satisfactions)
                               for app_type, satisfactions in app_satisfactions.items()}
        }

        print(f"Evaluation Results:")
        for key, value in avg_metrics.items():
            if key != 'app_satisfaction':
                print(f"Average {key.capitalize()}: {value:.2f}")
        for app_type, satisfaction in avg_metrics['app_satisfaction'].items():
            print(f"{app_type} Satisfaction: {satisfaction:.2f}")

        return avg_metrics

    def save_model(self, filepath):
        """
        Save the model.

        Args:
            filepath: Path to save model
        """
        model_data = {
            'Q': self.Q,
            'core': self.core,
            'factors': self.factors,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'tucker_rank': self.tucker_rank,
            'epsilon': self.epsilon,
            'actions_history': self.actions_history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load the model.

        Args:
            filepath: Path to load model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.Q = model_data['Q']
        self.core = model_data.get('core')
        self.factors = model_data.get('factors')
        self.state_dim = model_data['state_dim']
        self.action_dim = model_data['action_dim']
        self.tucker_rank = model_data['tucker_rank']
        self.epsilon = model_data['epsilon']
        self.actions_history = model_data.get('actions_history', [])
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    env = SDWANEnvironment(num_sites=6, connectivity_prob=0.6)
    components = SDWANComponents()
    state_dim = (4, 4, 3, 3)  # load_level, congestion_level, flow_level, health_level
    action_dim = (6, 6, 3, 5, 4)  # source, destination, flow_rate_level, app_type, path_index
    agent = SDWANTensorRL(env, components, state_dim, action_dim)
    training_history = agent.train(num_episodes=50, max_steps=30)
    agent.plot_training_history(save_path="training_history.png")
    evaluation_metrics = agent.evaluate(num_episodes=10)
    agent.save_model('sdwan_tensor_rl_model.pkl')