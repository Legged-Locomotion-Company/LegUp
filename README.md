# Training legged robot locomotion controllers using reinforcement learning

This project is organized into several modules. `main.py` is the main driver for training a model as of now.
- The `isaac` folder contains everything needed to interface with IsaacGym; it provides an API to interface with IsaacGym easily
- The `agents` folder contains the main code used to train a model. Each agent should really only needs to provide a `reset` function to reset all environments, and a `step` function to do the given actions and return the next observation, rewards, etc. Refer to the OpenAI gym interface for a proper specification. Note, that this implementation relies on the `step` function to automatically reset any environments that have terminated
- The `rl` folder just contains a basic implementation of PPO (that may not be fully correct) for now. This isn't really being used yet.
- The `robots` folder contains the robot model and robot-specific dynamics code for now. This will be replaced by our "robot interface" soon.
