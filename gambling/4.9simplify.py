import numpy as np
from typing import Tuple, List


class GamblerProblem:
    """
    赌徒问题（Gambler's Problem）MDP实现

    问题描述：
    一个赌徒有一笔资金，通过下注猜测抛硬币的结果来赢钱。
    如果硬币正面朝上，他赢得下注金额；如果反面朝上，他输掉下注金额。
    游戏在赌徒达到目标金额或输光所有钱时结束。

    这是一个经典的有限MDP问题，常用于演示动态规划和强化学习算法。
    """

    def __init__(self, goal: int = 100, p_h: float = 0.4):
        """
        初始化赌徒问题

        Args:
            goal: 目标金额，游戏达到此金额时获胜
            p_h: 硬币正面朝上的概率，即获胜概率
        """
        self.goal = goal
        self.p_h = p_h
        self.states = np.arange(goal + 1)  # 状态空间: 0到goal，表示当前资金
        self.values = np.zeros(goal + 1)  # 状态值函数：每个状态的期望获胜概率
        self.policy = np.zeros(goal + 1)  # 策略：每个状态下应该下注多少

        # 设置终止状态的值
        self.values[goal] = 1.0  # 达到目标，获胜概率为1
        self.values[0] = 0.0  # 输光资金，获胜概率为0

    def value_iteration(self, theta: float = 1e-9, max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        值迭代算法求解MDP

        算法原理：
        通过贝尔曼最优方程直接迭代更新状态价值函数，直到收敛。
        每次迭代中，对每个状态计算所有可能动作的期望价值，选择最大值更新状态价值。

        Args:
            theta: 收敛阈值，当价值函数的最大变化小于此值时停止迭代
            max_iterations: 最大迭代次数，防止无限循环

        Returns:
            values: 收敛后的最优值函数
            policy: 最优策略，每个状态的最优下注金额
        """
        print("开始值迭代...")

        for iteration in range(max_iterations):
            delta = 0  # 记录本轮迭代中价值函数的最大变化
            new_values = self.values.copy()  # 创建值函数的副本用于更新

            # 遍历所有非终止状态（1到goal-1）
            for state in range(1, self.goal):
                # 可能的动作：可以下注1到min(state, goal-state)
                # 下注不能超过当前资金，也不能超过达到目标所需的金额
                possible_actions = range(1, min(state, self.goal - state) + 1)

                if not possible_actions:
                    continue  # 如果没有有效动作，跳过该状态

                # 计算每个动作的期望价值
                action_values = []
                for action in possible_actions:
                    # 硬币正面：赢得赌注，状态变为 state + action
                    # 硬币反面：输掉赌注，状态变为 state - action
                    value = (self.p_h * self.values[state + action] +
                             (1 - self.p_h) * self.values[state - action])
                    action_values.append(value)

                # 选择所有动作中的最大价值（贝尔曼最优方程）
                best_value = max(action_values)
                best_action = possible_actions[np.argmax(action_values)]

                # 更新值函数和策略
                delta = max(delta, abs(best_value - new_values[state]))
                new_values[state] = best_value
                self.policy[state] = best_action

            # 用新值函数替换旧值函数
            self.values = new_values

            # 检查收敛条件
            if delta < theta:
                print(f"值迭代在 {iteration + 1} 次迭代后收敛")
                break

            # 每100次迭代打印进度
            if iteration % 100 == 0:
                print(f"迭代 {iteration}, 最大变化: {delta:.6f}")

        return self.values, self.policy

    def policy_evaluation(self, policy: np.ndarray, theta: float = 1e-9,
                          max_iterations: int = 1000) -> np.ndarray:
        """
        策略评估：评估给定策略的状态价值函数

        算法原理：
        对于固定的策略，通过迭代计算每个状态在遵循该策略时的期望回报。
        使用贝尔曼期望方程进行迭代更新，直到价值函数收敛。

        Args:
            policy: 要评估的策略，包含每个状态应该执行的动作
            theta: 收敛阈值
            max_iterations: 最大迭代次数

        Returns:
            values: 策略评估后的状态价值函数
        """
        # 初始化值函数：终止状态值固定，其他状态初始为0
        values = np.zeros(self.goal + 1)
        values[self.goal] = 1.0  # 获胜状态价值为1
        values[0] = 0.0  # 失败状态价值为0

        for iteration in range(max_iterations):
            delta = 0
            new_values = values.copy()  # 创建值函数副本用于更新

            # 遍历所有非终止状态
            for state in range(1, self.goal):
                action = int(policy[state])  # 确保action是整数
                # 检查动作有效性
                if action == 0 or state + action > self.goal or state - action < 0:
                    continue

                # 根据策略计算期望价值（贝尔曼期望方程）
                # 只考虑策略指定的动作，不考虑其他可能动作
                new_values[state] = (self.p_h * values[state + action] +
                                     (1 - self.p_h) * values[state - action])
                delta = max(delta, abs(new_values[state] - values[state]))

            # 更新值函数
            values = new_values

            # 检查收敛
            if delta < theta:
                print(f"策略评估在 {iteration + 1} 次迭代后收敛")
                break

        return values

    def policy_improvement(self, values: np.ndarray) -> np.ndarray:
        """
        策略改进：基于当前价值函数改进策略

        算法原理：
        对每个状态，遍历所有可能动作，选择期望价值最大的动作。
        这相当于在当前价值函数下采取贪心策略。

        Args:
            values: 当前状态价值函数

        Returns:
            new_policy: 改进后的策略
        """
        new_policy = np.zeros(self.goal + 1)

        for state in range(1, self.goal):
            # 获取当前状态的所有可能动作
            possible_actions = range(1, min(state, self.goal - state) + 1)

            if not possible_actions:
                continue  # 没有有效动作时跳过

            # 选择贪心动作：期望价值最大的动作
            best_value = -float('inf')
            best_action = 0

            # 遍历所有可能动作，计算每个动作的期望价值
            for action in possible_actions:
                value = (self.p_h * values[state + action] +
                         (1 - self.p_h) * values[state - action])

                # 更新最优动作
                if value > best_value:
                    best_value = value
                    best_action = action

            # 设置新策略：选择期望价值最大的动作
            new_policy[state] = best_action

        return new_policy

    def policy_iteration(self, max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        策略迭代算法求解MDP

        算法原理：
        交替进行策略评估和策略改进，直到策略稳定。
        1. 策略评估：评估当前策略的价值函数
        2. 策略改进：基于价值函数改进策略
        3. 重复直到策略不再改变

        Args:
            max_iterations: 最大迭代次数

        Returns:
            values: 最优状态价值函数
            policy: 最优策略
        """
        print("开始策略迭代...")

        # 初始化随机策略
        policy = np.zeros(self.goal + 1)
        for state in range(1, self.goal):
            possible_actions = list(range(1, min(state, self.goal - state) + 1))
            if possible_actions:
                # 随机选择一个有效动作作为初始策略
                policy[state] = np.random.choice(possible_actions)

        for iteration in range(max_iterations):
            # 步骤1：策略评估 - 计算当前策略的价值函数
            values = self.policy_evaluation(policy)

            # 步骤2：策略改进 - 基于价值函数找到更好的策略
            new_policy = self.policy_improvement(values)

            # 检查策略是否稳定（收敛条件）
            if np.array_equal(policy, new_policy):
                print(f"策略迭代在 {iteration + 1} 次迭代后收敛")
                self.values = values
                self.policy = new_policy
                return values, new_policy

            # 更新策略，继续下一轮迭代
            policy = new_policy
            print(f"迭代 {iteration + 1} 完成")

        # 达到最大迭代次数仍未收敛
        print("达到最大迭代次数")
        self.values = values
        self.policy = policy
        return values, policy

    def simulate_episode(self, initial_capital: int, policy: np.ndarray = None) -> List[Tuple[int, int, int]]:
        """
        模拟一个游戏回合

        Args:
            initial_capital: 初始资本
            policy: 使用的策略，如果为None则使用最优策略

        Returns:
            history: 回合历史，包含每个步骤的(状态, 动作, 奖励)
        """
        if policy is None:
            policy = self.policy

        state = initial_capital
        history = []

        # 游戏继续直到达到终止状态（赢或输）
        while state > 0 and state < self.goal:
            # 根据策略选择动作
            action = int(policy[state])  # 确保action是整数
            if action == 0:  # 如果策略给出无效动作，选择最小下注
                action = 1

            # 模拟抛硬币：根据概率决定胜负
            if np.random.random() < self.p_h:
                next_state = state + action  # 获胜，资金增加
                reward = 0  # 中间步骤奖励为0
            else:
                next_state = state - action  # 失败，资金减少
                reward = 0  # 中间步骤奖励为0

            # 检查是否达到终止状态
            if next_state >= self.goal:
                reward = 1  # 获胜，最终奖励为1
            elif next_state <= 0:
                reward = 0  # 失败，最终奖励为0

            # 记录当前步骤
            history.append((state, action, reward))
            state = next_state

        return history


def compare_probabilities():
    """
    比较不同获胜概率下的最优策略

    展示概率如何影响赌徒的最优下注策略。
    低概率时策略保守，高概率时策略激进。
    """
    print("\n" + "=" * 50)
    print("比较不同概率下的策略")
    print("=" * 50)

    probabilities = [0.25, 0.4, 0.5, 0.55]

    for p_h in probabilities:
        print(f"\n概率 p_h = {p_h}:")
        gambler = GamblerProblem(goal=100, p_h=p_h)
        values, policy = gambler.value_iteration()

        # 输出关键状态的最优策略
        test_states = [1, 25, 50, 75, 99]
        print("  资本状态 | 最优下注")
        print("  ---------|----------")
        for state in test_states:
            print(f"    {state:3d}    |    {policy[state]:2.0f}")


def main():
    """
    主函数：演示赌徒问题的完整解决方案

    包括：
    1. 使用值迭代算法求解
    2. 使用策略迭代算法求解
    3. 比较不同概率下的策略
    4. 模拟游戏回合
    5. 比较两种算法的结果
    """
    print("=" * 60)
    print("赌徒问题（Gambler's Problem）MDP求解")
    print("=" * 60)

    # 使用值迭代求解
    print("\n" + "=" * 40)
    print("使用值迭代算法")
    print("=" * 40)
    gambler = GamblerProblem(goal=100, p_h=0.4)
    values_vi, policy_vi = gambler.value_iteration()

    # 使用策略迭代求解
    print("\n" + "=" * 40)
    print("使用策略迭代算法")
    print("=" * 40)
    gambler_pi = GamblerProblem(goal=100, p_h=0.4)
    values_pi, policy_pi = gambler_pi.policy_iteration()

    # 比较不同概率下的策略
    compare_probabilities()

    # 模拟游戏回合
    print("\n" + "=" * 40)
    print("模拟游戏回合")
    print("=" * 40)
    for i in range(3):
        history = gambler.simulate_episode(initial_capital=50)
        if history:
            final_state = history[-1][0] + history[-1][2]  # 最终状态+最终奖励
        else:
            final_state = 50
        outcome = "获胜" if final_state >= 100 else "失败"
        steps = len(history)
        print(f"回合 {i + 1}: 初始资本50, 步数{steps}, 最终资本{final_state}, 结果: {outcome}")

    # 输出关键状态的最优策略
    print("\n" + "=" * 40)
    print("部分状态的最优下注策略")
    print("=" * 40)
    test_states = [1, 25, 50, 75, 99]
    print("资本状态 | 最优下注")
    print("---------|----------")
    for state in test_states:
        print(f"   {state:3d}   |    {policy_vi[state]:2.0f}")

    # 比较两种算法的结果
    print("\n" + "=" * 40)
    print("算法结果比较")
    print("=" * 40)
    value_diff = np.max(np.abs(values_vi - values_pi))
    policy_diff = np.max(np.abs(policy_vi - policy_pi))
    print(f"值函数最大差异: {value_diff:.8f}")
    print(f"策略最大差异: {policy_diff:.0f}")
    print(f"算法一致性: {'是' if value_diff < 1e-6 and policy_diff == 0 else '否'}")


if __name__ == "__main__":
    main()