import numpy as np
from config import *

def epsilon_greedy(Q, epsilon):
    if np.random.random() >epsilon:
        return np.argmax(Q)
    else:
        return np.random.randint(len(Q))


def UCB_get_action(total_action_counts, n_arms, Q, C, action_counts):
    # UCB动作选择
    if total_action_counts < n_arms:
        # 初始阶段：每个动作至少尝试一次
        action = total_action_counts % n_arms

    else:
        # 计算UCB值并选择动作
        ucb_values = Q + C * np.sqrt(np.log(total_action_counts) / (action_counts + 1e-8))
        action = np.argmax(ucb_values)


    return action