from environment import *
from policy import *

def run_average(n_runs, total_steps, n_arms, epsilon):
     rewards = np.zeros((n_runs, total_steps))
     optimal_actions = np.zeros((n_runs, total_steps))

     for i in range(n_runs):
         bandit = NonStationaryBandit(n_arms)
         Q = np.zeros(n_arms)
         action_counts = np.zeros(n_arms)

         for j in range(total_steps):
             optimal_action = bandit.get_optimal_action()
             action = epsilon_greedy(Q, epsilon)
             reward = bandit.step(action)
             rewards[i, j] = reward
             optimal_actions[i, j] = (action == optimal_action)

             action_counts[action] += 1
             Q[action] = Q[action] + (reward - Q[action]) / action_counts[action]


     return (rewards, optimal_actions)