import numpy as np


class NonStationaryBandit:
    """
    非平稳多臂赌博机环境实现

    环境特性:
    - 每个臂的真实价值Q*随时间随机游走
    - 奖励包含高斯噪声
    - 适用于测试算法在动态环境中的适应性
    """

    def __init__(self, n_arms):
        """
        初始化非平稳赌博机环境

        Args:
            n_arms: 臂的数量（动作空间大小）
        """
        self.n_arms = n_arms
        # 初始化各臂的真实价值（均值）
        self.q_star = np.zeros(n_arms)

    def step(self, action):
        """
        执行动作并返回奖励，同时更新环境状态

        Args:
            action: 选择的动作索引

        Returns:
            reward: 获得的奖励值

        Environment Dynamics:
            - 奖励: R = Q*(action) + N(0, 0.01)
            - 环境变化: 所有Q*值随机游走 ΔQ* ~ N(0, 0.01)
        """
        # 生成带噪声的奖励
        reward = self.q_star[action] + np.random.normal(0, 0.01)
        # 非平稳性：所有臂的真实价值随机游走
        self.q_star += np.random.normal(0, 0.01, self.n_arms)

        return reward

    def get_optimal_action(self):
        """
        获取当前时间步的最优动作（用于性能评估）

        Returns:
            optimal_action: 当前具有最高真实价值的动作索引
        """
        return np.argmax(self.q_star)