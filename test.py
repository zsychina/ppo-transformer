import gym
from agent import Agent
import matplotlib.pyplot as plt

env = gym.make("LunarLander-v2", render_mode="human")


agent = Agent(
    state_dim=env.observation_space.shape[0],
    hidden_dim=128,
    action_dim=env.action_space.n,
    device='mps'
)

agent.load()

for episode_i in range(10):
    state, info = env.reset()
    episode_return = 0

    episode_state = [state]
    done = False
    while not done:
        action = agent.take_action(episode_state)
        next_state, reward, terminated, truncated, info = env.step(action)
    
        if terminated or truncated:
            done = True
        
        state = next_state
        episode_state.append(next_state)
        episode_return += reward
        
    print(f'{episode_i=} {episode_return=}')

env.close()
