import numpy as np


def epsilon_greedy(Q, epsilon):
    """
    ε-greedy策略实现
    以1-ε的概率选择当前最优动作（利用），以ε的概率随机选择动作（探索）

    Args:
        Q: 各动作的当前价值估计数组，shape=(n_arms,)
        epsilon: 探索概率，取值范围[0, 1]

    Returns:
        action: 选择的动作索引
    """
    if np.random.random() > epsilon:
        # 利用：选择当前Q值最大的动作
        return np.argmax(Q)
    else:
        # 探索：随机选择一个动作
        return np.random.randint(len(Q))


def UCB_get_action(total_action_counts, n_arms, Q, C, action_counts):
    """
    UCB（上置信界）策略实现
    基于置信上界平衡探索与利用，优先选择具有高不确定性或高价值的动作

    Args:
        total_action_counts: 总动作执行次数
        n_arms: 动作空间大小
        Q: 各动作的当前价值估计数组
        C: UCB探索系数，控制探索程度
        action_counts: 各动作被选择的次数数组

    Returns:
        action: 选择的动作索引

    Note:
        UCB公式: Q(a) + C * sqrt(ln(t) / N(a))
        其中t为总步数，N(a)为动作a被选择次数
    """
    if total_action_counts < n_arms:
        # 初始阶段：确保每个动作至少被尝试一次
        action = total_action_counts % n_arms
    else:
        # 计算所有动作的UCB值
        ucb_values = Q + C * np.sqrt(np.log(total_action_counts) / (action_counts + 1e-8))
        action = np.argmax(ucb_values)

    return action