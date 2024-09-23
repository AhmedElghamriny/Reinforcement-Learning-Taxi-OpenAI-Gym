import numpy as np
import gym
import random

env = gym.make('Taxi-v3')

# Importance percentage of new information relative to old information 
learning_rate = 0.9

# Discount Factor
gamma = 0.95

# Higher epsilon means random choices, Lower epsilon means choices with highest reward in q-table
# 0 is good when we have a solid q-table, 1 is good when we have no information in q-table (random values)
epsilon = 1.0

# Scalar to make epsilon smaller as we gain new information
epsilon_decay = 0.9995

# 1% random chance of taking a random action
min_epsilon = 0.01

epochs = 10000

maxStepsPerEpoch = 100

# 5 x 5 grid = 25 positions for taxi * 5 spawn points for customer * 4 spawn points for hotel = 500 
q_table = np.zeros((env.observation_space.n, env.action_space.n))


def choose_action(state):
    # Generate random number between 0 and 1, if its less than epsilon, take a random action
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state, :])

for epoch in range(epochs):
    # For each epoch reset the environment and store intitial state
    state, _ = env.reset()

    done = False

    for step in range(maxStepsPerEpoch):

        # Choose action from q-table based on the current state
        action = choose_action(state)

        # Perform the action, returns next state + reward
        next_state, reward, done, truncated, info = env.step(action)

        old_value = q_table[state, action]

        # Get the max reward action from the next state
        next_max = np.max(q_table[next_state, :])

        # Update q value of the action we just took by keeping the old value to a certain degree 
        q_table[state, action] = (1 - learning_rate) * old_value + learning_rate * (reward + gamma * next_max)

        state = next_state

        if done or truncated:
            break
    
    # If epsilon * epsilon decay reaches below min epsilon, keep using min epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)


env = gym.make('Taxi-v3', render_mode='human')

for epoch in range(5):
    state, _ = env.reset()

    done = False

    print(f'Epoch: {epoch}')

    for step in range(maxStepsPerEpoch):
        env.render()
        # No need to call choose_action function, since we are testing not training (with random choices)
        action = np.argmax(q_table[state, :])
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state

        if done or truncated:
            env.render()
            print(f'Finished Epoch: {epoch}, with reward: {reward}')
            break

env.close()