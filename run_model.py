
import time
import random
import numpy as np
from predict import predict

# Simulate state input: [latency, packet_loss, jitter, load]
def load_topology():
    """
    Simulate fetching topology from a controller.
    Replace with actual Mininet/Ryu values in production.
    """
    latency = random.uniform(5, 100)         # ms
    packet_loss = random.uniform(0, 5)       # %
    jitter = random.uniform(0, 50)           # ms
    load = random.uniform(0, 100)            # %
    return [latency, packet_loss, jitter, load]

def apply_decision(action):
    """
    Simulate applying action via flow rules.
    Replace with REST API call to controller (e.g., Ryu).
    """
    print(f"Action chosen by model: {action}")
    # Example:
    # requests.post(f"http://controller:8080/path/set", json={"action": action})

def main():
    print("Running RL-based SD-WAN decision engine...\n")
    for _ in range(5):  # Simulate 5 decision-making cycles
        state = load_topology()
        print(f"Network State: {state}")
        action = predict(state)
        apply_decision(action)
        print()
        time.sleep(2)  # Pause between cycles

if __name__ == '__main__':
    main()
