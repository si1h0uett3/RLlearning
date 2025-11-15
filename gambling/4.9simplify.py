import numpy as np
from typing import Tuple, List


class GamblerProblem:
    """
    赌徒问题MDP实现
    赌徒下注猜测抛硬币结果，目标达到100美元或输光时结束
    """

    def __init__(self, goal: int = 100, p_h: float = 0.4):
        """
        初始化赌徒问题

        Args:
            goal: 目标金额
            p_h: 硬币正面朝上的概率
        """
        self.goal = goal
        self.p_h = p_h
        self.states = np.arange(goal + 1)  # 状态空间: 0到goal
        self.values = np.zeros(goal + 1)  # 状态值函数
        self.policy = np.zeros(goal + 1)  # 策略: 每个状态下应该下注多少

        # 终止状态的值
        self.values[goal] = 1.0  # 达到目标，获胜
        self.values[0] = 0.0  # 输光，失败

    def value_iteration(self, theta: float = 1e-9, max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        值迭代算法

        Args:
            theta: 收敛阈值
            max_iterations: 最大迭代次数

        Returns:
            values: 收敛后的值函数
            policy: 最优策略
        """
        print("开始值迭代...")

        for iteration in range(max_iterations):
            delta = 0
            new_values = self.values.copy()

            # 遍历所有非终止状态
            for state in range(1, self.goal):
                # 可能的动作: 可以下注1到min(state, goal-state)
                possible_actions = range(1, min(state, self.goal - state) + 1)

                if not possible_actions:
                    continue

                # 计算每个动作的期望价值
                action_values = []
                for action in possible_actions:
                    # 正面: 赢得赌注，状态变为 state + action
                    # 反面: 输掉赌注，状态变为 state - action
                    value = (self.p_h * self.values[state + action] +
                             (1 - self.p_h) * self.values[state - action])
                    action_values.append(value)

                # 选择最大价值
                best_value = max(action_values)
                best_action = possible_actions[np.argmax(action_values)]

                # 更新值函数和策略
                delta = max(delta, abs(best_value - new_values[state]))
                new_values[state] = best_value
                self.policy[state] = best_action

            self.values = new_values

            # 检查收敛
            if delta < theta:
                print(f"值迭代在 {iteration + 1} 次迭代后收敛")
                break

            if iteration % 100 == 0:
                print(f"迭代 {iteration}, 最大变化: {delta:.6f}")

        return self.values, self.policy

    def policy_evaluation(self, policy: np.ndarray, theta: float = 1e-9,
                          max_iterations: int = 1000) -> np.ndarray:

        values = np.zeros(self.goal + 1)
        values[self.goal] = 1.0
        values[0] = 0.0

        for iteration in range(max_iterations):
            delta = 0
            new_values = values.copy()

            for state in range(1, self.goal):
                action = int(policy[state])  # 确保action是整数
                if action == 0 or state + action > self.goal or state - action < 0:
                    continue

                # 根据策略计算期望价值
                new_values[state] = (self.p_h * values[state + action] +
                                     (1 - self.p_h) * values[state - action])
                delta = max(delta, abs(new_values[state] - values[state]))

            values = new_values

            if delta < theta:
                print(f"策略评估在 {iteration + 1} 次迭代后收敛")
                break

        return values

    def policy_improvement(self, values: np.ndarray) -> np.ndarray:
        new_policy = np.zeros(self.goal + 1)

        for state in range(1, self.goal):
            possible_actions = range(1, min(state, self.goal - state) + 1)

            if not possible_actions:
                continue

            # 选择贪心动作
            best_value = -float('inf')
            best_action = 0

            for action in possible_actions:
                value = (self.p_h * values[state + action] +
                         (1 - self.p_h) * values[state - action])

                if value > best_value:
                    best_value = value
                    best_action = action

            new_policy[state] = best_action

        return new_policy

    def policy_iteration(self, max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        print("开始策略迭代...")

        # 初始化随机策略
        policy = np.zeros(self.goal + 1)
        for state in range(1, self.goal):
            possible_actions = list(range(1, min(state, self.goal - state) + 1))
            if possible_actions:
                policy[state] = np.random.choice(possible_actions)

        for iteration in range(max_iterations):
            # 策略评估
            values = self.policy_evaluation(policy)

            # 策略改进
            new_policy = self.policy_improvement(values)

            # 检查策略是否稳定
            if np.array_equal(policy, new_policy):
                print(f"策略迭代在 {iteration + 1} 次迭代后收敛")
                self.values = values
                self.policy = new_policy
                return values, new_policy

            policy = new_policy
            print(f"迭代 {iteration + 1} 完成")

        print("达到最大迭代次数")
        self.values = values
        self.policy = policy
        return values, policy

    def simulate_episode(self, initial_capital: int, policy: np.ndarray = None) -> List[Tuple[int, int, int]]:
        """
        模拟一个回合

        Args:
            initial_capital: 初始资本
            policy: 使用的策略，如果为None则使用最优策略

        Returns:
            history: 回合历史 [(状态, 动作, 奖励), ...]
        """
        if policy is None:
            policy = self.policy

        state = initial_capital
        history = []

        while state > 0 and state < self.goal:
            # 选择动作
            action = int(policy[state])  # 确保action是整数
            if action == 0:  # 如果没有有效动作，选择最小下注
                action = 1

            # 模拟抛硬币
            if np.random.random() < self.p_h:
                next_state = state + action  # 赢
                reward = 0
            else:
                next_state = state - action  # 输
                reward = 0

            # 检查终止状态
            if next_state >= self.goal:
                reward = 1
            elif next_state <= 0:
                reward = 0

            history.append((state, action, reward))
            state = next_state

        return history


def compare_probabilities():
    """比较不同概率下的最优策略"""
    print("\n=== 比较不同概率下的策略 ===")
    probabilities = [0.25, 0.4, 0.5, 0.55]

    for p_h in probabilities:
        print(f"\n概率 p_h = {p_h}:")
        gambler = GamblerProblem(goal=100, p_h=p_h)
        values, policy = gambler.value_iteration()

        # 输出关键状态的最优策略
        test_states = [1, 25, 50, 75, 99]
        for state in test_states:
            print(f"  资本 {state:2d}: 最优下注 = {policy[state]:2.0f}")


def main():
    """主函数"""
    # 创建赌徒问题实例
    gambler = GamblerProblem(goal=100, p_h=0.4)

    # 使用值迭代求解
    print("=== 使用值迭代算法 ===")
    values_vi, policy_vi = gambler.value_iteration()

    # 使用策略迭代求解
    print("\n=== 使用策略迭代算法 ===")
    gambler_pi = GamblerProblem(goal=100, p_h=0.4)
    values_pi, policy_pi = gambler_pi.policy_iteration()

    # 比较不同概率
    compare_probabilities()

    # 模拟一些回合
    print("\n=== 模拟回合 ===")
    for i in range(3):
        history = gambler.simulate_episode(initial_capital=50)
        final_state = history[-1][0] if history else 50
        outcome = "获胜" if final_state >= 100 else "失败"
        print(f"回合 {i + 1}: 初始资本50, 最终资本{final_state}, 结果: {outcome}")

    # 输出一些状态的最优策略
    print("\n=== 部分状态的最优下注 ===")
    test_states = [1, 25, 50, 75, 99]
    for state in test_states:
        print(f"资本 {state:2d}: 最优下注 = {policy_vi[state]:2.0f}")

    # 比较两种算法的结果差异
    print("\n=== 算法结果比较 ===")
    value_diff = np.max(np.abs(values_vi - values_pi))
    policy_diff = np.max(np.abs(policy_vi - policy_pi))
    print(f"值函数最大差异: {value_diff:.6f}")
    print(f"策略最大差异: {policy_diff:.0f}")


if __name__ == "__main__":
    main()