from environment import *
from policy import *


def run_constant(n_runs, total_steps, n_arms, epsilon, alpha):
    """
    使用常数步长的ε-greedy算法在多臂赌博机问题上的完整运行

    Args:
        n_runs: 独立实验运行次数
        total_steps: 每次实验的总步数
        n_arms: 赌博机的臂数
        epsilon: ε-greedy策略的探索概率
        alpha: 常数步长（学习率）

    Returns:
        rewards: 奖励记录数组
        optimal_actions: 最优动作选择记录

    Algorithm:
        - 动作选择: ε-greedy策略
        - Q值更新: 常数步长方法 (固定α)
        - 适用于非平稳环境，能够快速适应环境变化
    """
    rewards = np.zeros((n_runs, total_steps))
    optimal_actions = np.zeros((n_runs, total_steps))

    for i in range(n_runs):
        bandit = NonStationaryBandit(n_arms)
        Q = np.zeros(n_arms)  # 初始化动作价值估计

        for j in range(total_steps):
            # 获取当前最优动作（用于性能评估）
            optimal_action = bandit.get_optimal_action()
            # ε-greedy策略选择动作
            action = epsilon_greedy(Q, epsilon)
            # 执行动作并获得奖励
            reward = bandit.step(action)

            # 记录结果
            rewards[i, j] = reward
            optimal_actions[i, j] = (action == optimal_action)

            # 常数步长更新Q值: Q = Q + α * (reward - Q)
            Q[action] = Q[action] + (reward - Q[action]) * alpha

    return rewards, optimal_actions