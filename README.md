
# Taxi Q-Learning

A reinforcement learning project using Q-learning to solve the `Taxi-v3` environment from OpenAI's `gym`. This project demonstrates the use of Q-learning to teach an agent how to efficiently pick up and drop off passengers in a grid-based environment.

## Overview

In the `Taxi-v3` environment, the goal is for the taxi to pick up a passenger at one location and drop them off at another. The environment has the following components:
- **5 x 5 grid** representing the city.
- **Passenger spawn points** at fixed locations.
- **Drop-off points** where the passenger must be delivered.
  
Using Q-learning, the agent learns an optimal policy to navigate the grid and complete its objective with minimal steps.

### Parameters

- **Learning Rate (`learning_rate`)**: Determines how much new information overrides old information.
- **Discount Factor (`gamma`)**: Measures how much future rewards are valued.
- **Exploration Rate (`epsilon`)**: Controls the exploration vs. exploitation balance. Starts high to encourage exploration, then decays over time.
- **Epsilon Decay (`epsilon_decay`)**: Factor by which `epsilon` is multiplied at each epoch to gradually reduce exploration.
- **Minimum Epsilon (`min_epsilon`)**: Lower bound for `epsilon` to ensure some exploration even in later stages.

## Training the Agent

The training loop runs through multiple episodes (`epochs`). For each episode, the environment is reset, and the agent repeatedly:
1. Chooses an action based on the exploration-exploitation trade-off (using `epsilon`).
2. Takes the action and observes the reward and next state.
3. Updates the Q-table based on the reward and expected future rewards.
