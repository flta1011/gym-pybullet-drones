import time
import numpy as np
from gym_pybullet_drones.envs import HoverAviary

def test_simple_hover():
    """Test a simple hover scenario with visualization."""
    
    # Initialize environment
    env = HoverAviary(
        gui=True,          # Enable GUI
        record=False,      # Disable recording
        freq=240,         # 240Hz simulation frequency
        aggregate_phy_steps=1  # No physics aggregation
    )
    
    time.sleep(1)  # Give PyBullet time to initialize the GUI
    
    try:
        # Reset environment
        obs = env.reset()
        
        # Run simulation for 1000 steps
        for _ in range(1000):
            # Simple hover action (all motors at 50% power)
            action = np.array([0.5, 0.5, 0.5, 0.5])
            
            # Step simulation
            obs, reward, done, info = env.step(action)
            
            # Slow down simulation for better visualization
            time.sleep(1./240.)
            
            # Reset if episode is done
            if done:
                obs = env.reset()
                
        # Test passed if we get here
        assert True
        
    finally:
        # Ensure environment is closed properly
        env.close()

if __name__ == "__main__":
    test_simple_hover() 