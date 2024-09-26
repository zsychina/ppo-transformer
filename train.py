import gym
from agent import Agent
import matplotlib.pyplot as plt

# env = gym.make("LunarLander-v2", render_mode="human")
env = gym.make("LunarLander-v2")


agent = Agent(
    state_dim=env.observation_space.shape[0],
    hidden_dim=128,
    action_dim=env.action_space.n,
    device='cuda',
)

# pretrain
# agent.load()

reward_per_step = []
reward_per_episode = []

for episode_i in range(5000):
    state, info = env.reset()
    episode_return = 0
    done = False
    
    episode_state = [state]
    
    while not done:
        action = agent.take_action(episode_state)
        next_state, reward, terminated, truncated, info = env.step(action)
    
        if terminated or truncated:
            done = True
            
        agent.buffer.states.append(state)
        agent.buffer.actions.append(action)
        agent.buffer.rewards.append(reward)
        agent.buffer.next_states.append(next_state)
        agent.buffer.dones.append(done)
        
        state = next_state
        episode_state.append(next_state)
        
        episode_return += reward
        reward_per_step.append(reward)
        
    print(f'{episode_i=} {episode_return=}')
    reward_per_episode.append(episode_return)
    
    del episode_state[:]
    agent.update()

    if episode_i % 100 == 0:
        agent.save()


agent.save()

plt.plot(reward_per_episode)
plt.show()

env.close()

