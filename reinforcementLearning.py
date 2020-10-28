import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')  # we are going to use the FrozenLake environment
print(env.observation_space.n)  # get number of states
print(env.action_space.n)  # get number of actions

env.reset()  # reset environment to default state
action = env.action_space.sample()  # get a random action
new_state, reward, done, info = env.step(action)  # take action, notice it returns information about the action
env.render()  # render the GUI for the environment

# Building Q-table
env = gym.make('FrozenLake-v0')     # S: start, F: frozen, H: hole, G: goal
STATES = env.observation_space.n
ACTIONS = env.action_space.n

Q = np.zeros((STATES, ACTIONS))  # create a matrix with all 0 values
print(Q)

# Constants
EPISODES = 1500  # how many times to run the environment from the beginning
MAX_STEPS = 100  # max number of steps allowed for each run of environment

LEARNING_RATE = 0.81  # learning rate
GAMMA = 0.96
RENDER = False  # if you want to see training set to true
epsilon = 0.9       # start with a 90% chance of picking a random action

rewards = []
for episode in range(EPISODES):

    state = env.reset()
    for _ in range(MAX_STEPS):

        if RENDER:
            env.render()

        if np.random.uniform(0, 1) < epsilon:       # we will check if a randomly selected value is less than epsilon.
            action = env.action_space.sample()      # take random action
        else:
            action = np.argmax(Q[state, :])     # use Q table to pick best action based on current values

        nextState, reward, done, _ = env.step(action)

        Q[state, action] = Q[state, action] + LEARNING_RATE * (
                reward + GAMMA * np.max(Q[nextState, :]) - Q[state, action])

        state = nextState

        if done:
            rewards.append(reward)
            epsilon -= 0.001
            break  # reached goal

print(Q)
print(f"Average reward: {sum(rewards) / len(rewards)}:")


# and now we can see our Q values!

# we can plot the training progress and see how the agent improved
def get_average(values):
    return sum(values) / len(values)


avgRewards = []
for i in range(0, len(rewards), 100):
    avgRewards.append(get_average(rewards[i:i + 100]))

plt.plot(avgRewards)
plt.ylabel('average reward')
plt.xlabel('episodes (100\'s)')
plt.show()
