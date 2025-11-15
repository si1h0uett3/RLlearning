from environment import *
from policy import *


def run_UCB_constant(num_runs, total_steps, num_arms, alpha, C):
    """
    使用常数步长的UCB算法在多臂赌博机问题上的完整运行

    Args:
        num_runs: 独立实验运行次数
        total_steps: 每次实验的总步数
        num_arms: 赌博机的臂数
        alpha: 常数步长（学习率）
        C: UCB探索系数

    Returns:
        rewards: 奖励记录数组
        optimal_actions: 最优动作选择记录

    Algorithm:
        - 动作选择: UCB策略（基于置信上界）
        - Q值更新: 常数步长方法（固定α）
        - 结合了UCB的智能探索和常数步长的快速适应能力
    """
    rewards = np.zeros((num_runs, total_steps))
    optimal_actions = np.zeros((num_runs, total_steps))

    for i in range(num_runs):
        bandit = NonStationaryBandit(num_arms)
        action_counts = np.zeros(num_arms)
        Q = np.zeros(num_arms)
        total_action_counts = 0

        for j in range(total_steps):
            # UCB策略选择动作
            action = UCB_get_action(total_action_counts, num_arms, Q, C, action_counts)

            # 执行动作并获得奖励
            reward = bandit.step(action)
            optimal_action = bandit.get_optimal_action()

            # 更新动作选择统计
            action_counts[action] += 1
            total_action_counts += 1

            # 常数步长更新Q值: Q = Q + α * (reward - Q)
            Q[action] = Q[action] + alpha * (reward - Q[action])

            # 记录性能指标
            rewards[i, j] = reward
            optimal_actions[i, j] = (action == optimal_action)

    return rewards, optimal_actions