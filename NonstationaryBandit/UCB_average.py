from environment import *
from policy import *


def run_UCB_average(num_runs, total_steps, num_arms, C):
    """
    使用采样平均步长的UCB算法在多臂赌博机问题上的完整运行

    Args:
        num_runs: 独立实验运行次数，用于统计平均性能
        total_steps: 每次实验的总步数
        num_arms: 赌博机的臂数（动作空间大小）
        C: UCB探索系数

    Returns:
        rewards: 奖励记录数组，shape=(num_runs, total_steps)
        optimal_actions: 最优动作选择记录，shape=(num_runs, total_steps)

    Algorithm:
        - 动作选择: UCB策略
        - Q值更新: 采样平均方法 (步长=1/n)
        - 适用于平稳环境，能够渐进收敛到真实价值
    """
    rewards = np.zeros((num_runs, total_steps))
    optimal_actions = np.zeros((num_runs, total_steps))

    for i in range(num_runs):
        # 初始化非平稳赌博机环境
        bandit = NonStationaryBandit(num_arms)
        action_counts = np.zeros(num_arms)  # 各动作被选择次数
        Q = np.zeros(num_arms)  # 各动作的价值估计
        total_action_counts = 0  # 总动作执行次数

        for j in range(total_steps):
            # UCB策略选择动作
            action = UCB_get_action(total_action_counts, num_arms, Q, C, action_counts)

            # 执行动作并获得奖励
            reward = bandit.step(action)
            optimal_action = bandit.get_optimal_action()

            # 更新统计信息
            action_counts[action] += 1
            total_action_counts += 1

            # 采样平均更新Q值: α = 1/N(a)
            Q[action] = Q[action] + (reward - Q[action]) / action_counts[action]

            # 记录本轮结果
            rewards[i, j] = reward
            optimal_actions[i, j] = (action == optimal_action)

    return rewards, optimal_actions