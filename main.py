import gymnasium as gym
from DQN import DQNAgent
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# Action Space:
# 0 - Do Nothing
# 1 - Shoot
# 2 - Drive Forward
# 3 - Spin Right
# 4 - Spin Left
# 5 - Teleport
# 6 - Up Right
# 7 - Up Left
# 8 - Up Fire
# 9 - Right Fire
# 10 - Left Fire
# 11 - Down Fire
# 12 - Up RIght Fire
# 13 - Up Left Fire


if __name__ == "__main__":
    display = False

    if display:
        env = gym.make('ALE/Asteroids-ram-v5', render_mode="human")
    else:
        env = gym.make('ALE/Asteroids-ram-v5')
    
    state, info = env.reset(seed=7)
    num_episodes = 200
    agent = DQNAgent(env.observation_space.shape[0], env.action_space)
    rewards = []

    if display:
        agent.load_model()
    

    if not display:
        epochs = tqdm(range(num_episodes))

        for epoch in epochs:
            state, info = env.reset(seed=7)
            done = False

            state = agent.preprocess_state(state)
            episode_reward = 0

            while not done:
                # action
                # if epoch % 100 == 0:
                #     # env.render(render_mode='human')
                #     print

                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = agent.preprocess_state(next_state)

                agent.remember(state, action, reward, next_state, terminated, truncated)
                agent.replay()

                state = next_state
                episode_reward += reward

                if terminated or truncated:
                    done = True
            
            rewards.append(episode_reward)
            epochs.set_postfix_str(f"Episodic Reward: {episode_reward}")

        agent.save_model()
        np.savetxt('rewards.txt', np.array(rewards))      
        env.close()
        plt.plot(rewards)
        plt.savefig('rewards.png')


    else:
        state, info = env.reset(seed=7)
        done = False

        state = agent.preprocess_state(state)
        episode_reward = 0
        agent.eval_model()
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = agent.preprocess_state(next_state)

            # agent.remember(state, action, reward, next_state, terminated, truncated)
            # agent.replay()

            state = next_state
            episode_reward += reward

            if terminated or truncated:
                done = True
        
