import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
import numpy as np
import time

from ddpg_agent import Agent

env = UnityEnvironment(file_name='Reacher.app')

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

agent = Agent(state_size=state_size, action_size=action_size, random_seed=2)

def ddpg(n_episodes=500, max_t=1000, print_every=1, solved_score = 30.0, train_mode = True, consec_episodes = 100,
         actor_path = 'actor_ckpt.pth', critic_path = 'critic_ckpt.pth'):

    mean_scores = []  # list of mean scores from each episode
    min_scores = []  # list of lowest scores from each episode
    max_scores = []  # list of highest scores from each episode
    best_score = -np.inf
    scores_window = deque(maxlen=consec_episodes)  # mean scores from most recent episodes
    moving_avgs = []  # list of moving averages


    for i_episode in range(1, n_episodes + 1):

        env_info = env.reset(train_mode=train_mode)[brain_name]  # reset environment
        states = env_info.vector_observations  # get current state for each agent
        agent.reset()
        scores = np.zeros(num_agents)
        start_time = time.time()

        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations  # get next state
            rewards = env_info.rewards  # get reward
            dones = env_info.local_done
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done, t)
            states = next_states
            scores += rewards
            if np.any(dones):
                break

            #print(rewards)

        duration = time.time() - start_time
        min_scores.append(np.min(scores))
        max_scores.append(np.max(scores))
        mean_scores.append(np.mean(scores))
        scores_window.append(mean_scores[-1])
        moving_avgs.append(np.mean(scores_window))

        if i_episode % print_every == 0:
            print('\rEpisode {} ({} sec)  -- \tMin: {:.1f}\tMax: {:.1f}\tMean: {:.1f}\tMov. Avg: {:.1f}'.format( \
                i_episode, round(duration), min_scores[-1], max_scores[-1], mean_scores[-1], moving_avgs[-1]), flush=True)

        if train_mode and mean_scores[-1] > best_score:
            torch.save(agent.actor_local.state_dict(), actor_path)
            torch.save(agent.critic_local.state_dict(), critic_path)

        if moving_avgs[-1] >= solved_score and i_episode >= consec_episodes:
            print('\nEnvironment SOLVED in {} episodes!\tMoving Average ={:.1f} over last {} episodes'.format( \
                i_episode - consec_episodes, moving_avgs[-1], consec_episodes))
            if train_mode:
                torch.save(agent.actor_local.state_dict(), actor_path)
                torch.save(agent.critic_local.state_dict(), critic_path)
            break

    return mean_scores, moving_avgs


scores, avgs = ddpg()


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores, label='DDPG')
plt.plot(np.arange(len(scores)), avgs, c='r', label='moving avg')
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.legend(loc='upper left')
plt.show()