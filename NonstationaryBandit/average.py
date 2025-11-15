from environment import *
from policy import *
#平均步长更新
def run_average(n_runs, total_steps, n_arms, epsilon):
    #初始化奖励表
     rewards = np.zeros((n_runs, total_steps))
    #保存最优动作
     optimal_actions = np.zeros((n_runs, total_steps))

     for i in range(n_runs):
         bandit = NonStationaryBandit(n_arms)
         Q = np.zeros(n_arms)
         #记录每个动作被选择的次数，用于平均步长更新
         action_counts = np.zeros(n_arms)

         for j in range(total_steps):
             #贪心策略选择最优动作
             optimal_action = bandit.get_optimal_action()
             #epsilon-greedy策略选择动作
             action = epsilon_greedy(Q, epsilon)
             reward = bandit.step(action)
             #第i轮第j次动作的奖励值
             rewards[i, j] = reward
             #如果动作是最优动作，数组中对应值为1，用于记录最优动作选择占比，但其实意义不大，策略已经决定了它的最优动作概率
             optimal_actions[i, j] = (action == optimal_action)
             #被选择的动作计数加一
             action_counts[action] += 1
             #采样平均更新Q值用于下一轮的动作选择，前面记录的动作被选次数起了作用
             Q[action] = Q[action] + (reward - Q[action]) / action_counts[action]


     return (rewards, optimal_actions)