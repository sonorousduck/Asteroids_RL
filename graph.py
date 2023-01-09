import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    rewards = np.loadtxt('rewards.txt')
    print(rewards)

    avg_rewards = []
    for i in range(len(rewards)):
        avg_rewards.append(np.mean(rewards[i:i+10]))
    
    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.savefig('rewards_DQN.png')