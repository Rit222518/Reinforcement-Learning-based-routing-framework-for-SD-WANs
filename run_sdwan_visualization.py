import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sdwan_environment import SDWANEnvironment
from sdwan_components import SDWANComponents
from sdwan_tensor_rl import SDWANTensorRL
from sdwan_packet_visulization import SDWANPacketVisualizer, run_packet_visualization


def visualize_sdwan_network_routing(num_sites=6,
                                    training_episodes=50,
                                    visualization_steps=200,
                                    save_dir='sdwan_results'):
    """
    Integrated function to train the RL model and visualize SD-WAN packet transfers.

    Args:
        num_sites (int): Number of sites in the SD-WAN
        training_episodes (int): Number of episodes to train the RL agent
        visualization_steps (int): Number of steps to visualize
        save_dir (str): Directory to save results

    Returns:
        dict: Results including trained agent, environment, and paths to saved files
    """
    # Create results directory
    os.makedirs(save_dir, exist_ok=True)

    print("=== Tensor-Based RL for SD-WAN Routing Visualization ===")
    print(f"SD-WAN size: {num_sites} sites")
    print(f"Training episodes: {training_episodes}")
    print(f"Visualization steps: {visualization_steps}")
    print("=" * 50)

    # Step 1: Create SD-WAN environment
    print("\nStep 1: Creating SD-WAN environment...")
    env = SDWANEnvironment(num_sites=num_sites, connectivity_prob=0.4)

    # Step 2: Define state and action dimensions
    print("\nStep 2: Defining state and action dimensions...")
    # State dimensions: (load_level, congestion_level, flow_level, health_level)
    state_dim = (4, 4, 3, 3)

    # Action dimensions: (source_site, destination_site, flow_rate_level, app_type, path_index)
    action_dim = (num_sites, num_sites, 3, 5, 4)

    # Step 3: Create tensor-based RL agent
    print("\nStep 3: Creating tensor-based RL agent...")
    components = SDWANComponents()
    agent = SDWANTensorRL(env, components, state_dim, action_dim)

    # Step 4: Train the agent
    print(f"\nStep 4: Training agent for {training_episodes} episodes...")
    training_history = agent.train(num_episodes=training_episodes, max_steps=30)

    # Step 5: Plot and save training history
    print("\nStep 5: Plotting training history...")
    training_plot_path = os.path.join(save_dir, 'training_history.png')
    agent.plot_training_history(save_path=training_plot_path)

    # Step 6: Evaluate the agent
    print("\nStep 6: Evaluating agent performance...")
    evaluation_metrics = agent.evaluate(num_episodes=10)

    # Step 7: Save the trained model
    print("\nStep 7: Saving trained model...")
    model_path = os.path.join(save_dir, 'sdwan_tensor_rl_model.pkl')
    agent.save_model(model_path)

    # Step 8: Create packet visualization
    print("\nStep 8: Creating packet visualization...")
    visualizer = SDWANPacketVisualizer(env, agent)

    # Step 9: Run simulation with visualization
    print("\nStep 9: Running simulation with visualization...")
    animation_path = os.path.join(save_dir, 'sdwan_packet_animation.mp4')
    visualizer.simulate_rl_routing(num_steps=visualization_steps, interval=100, save_path=animation_path)

    print(f"\nVisualization complete! Results saved to {save_dir} directory")

    return {
        'agent': agent,
        'env': env,
        'visualizer': visualizer,
        'training_history': training_history,
        'evaluation_metrics': evaluation_metrics,
        'animation_path': animation_path,
        'training_plot_path': training_plot_path,
        'model_path': model_path
    }


def main():
    """
    Main function to run the SD-WAN visualization with RL-based routing.
    """
    # Create results directory
    results_dir = 'sdwan_results'
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("SD-WAN Visualization with RL-based Routing")
    print("=" * 80)
    print("\nThis script will:")
    print("1. Create a virtual SD-WAN environment")
    print("2. Train a tensor-based RL model for optimal routing")
    print("3. Visualize packet transfers between PCs with RL routing decisions")
    print("4. Save the animation and results")
    print("\nPress Enter to continue...")
    input()

    # Run the visualization with default parameters
    results = visualize_sdwan_network_routing(
        num_sites=6,  # Number of SD-WAN sites
        training_episodes=50,  # Training episodes
        visualization_steps=200,  # Visualization steps
        save_dir=results_dir  # Results directory
    )

    # Print the paths to the results
    print("\nResults:")
    print(f"Animation: {results['animation_path']}")
    print(f"Training plot: {results['training_plot_path']}")
    print(f"Trained model: {results['model_path']}")

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()