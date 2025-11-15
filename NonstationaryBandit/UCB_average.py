from environment import *
from policy import *

def run_UCB_average(num_runs, total_steps, num_arms, C):  # c 是 UCB 参数
    rewards = np.zeros((num_runs, total_steps))
    optimal_actions = np.zeros((num_runs, total_steps))

    for i in range(num_runs):
        bandit = NonStationaryBandit(num_arms)
        action_counts = np.zeros(num_arms)
        Q = np.zeros(num_arms)
        total_action_counts = 0

        for j in range(total_steps):
            action = UCB_get_action(total_action_counts, n_arms, Q, C, action_counts)

            # 执行动作并获得奖励
            reward = bandit.step(action)
            optimal_action = bandit.get_optimal_action()

            # 更新统计信息
            action_counts[action] += 1
            total_action_counts += 1

            # 更新Q值（增量更新）
            Q[action] = Q[action] + (reward - Q[action]) / action_counts[action]

            # 记录结果
            rewards[i, j] = reward
            optimal_actions[i, j] = (action == optimal_action)

    return rewards, optimal_actions





