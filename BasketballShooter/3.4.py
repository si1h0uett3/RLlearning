import numpy as np

class ShootingRobotMDP:
    def __init__(self):
        # 定义状态
        self.states = [
            'near-hot', 'near-cold',  # 0, 1
            'mid-hot', 'mid-cold',  # 2, 3
            'far-hot', 'far-cold'  # 4, 5
        ]

        # 定义动作
        self.actions = ['shoot', 'move_closer', 'move_farther']  # 0, 1, 2

        # 状态数量和行为数量
        self.n_states = len(self.states)
        self.n_actions = len(self.actions)

        # 折扣因子
        self.gamma = 0.9

        # 初始化转移概率和奖励矩阵
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions, self.n_states))

        # 构建MDP模型
        self._build_mdp()

    def _get_make_probability(self, state_idx):
        """根据状态返回投篮命中概率"""
        if state_idx == 0:  # near-hot
            return 0.9
        elif state_idx == 1:  # near-cold
            return 0.7
        elif state_idx == 2:  # mid-hot
            return 0.7
        elif state_idx == 3:  # mid-cold
            return 0.5
        elif state_idx == 4:  # far-hot
            return 0.4
        elif state_idx == 5:  # far-cold
            return 0.2

    def _get_location(self, state_idx):
        """从状态索引中提取位置信息"""
        return state_idx // 2  # 0:near, 1:mid, 2:far

    def _get_confidence(self, state_idx):
        """从状态索引中提取信心信息"""
        return state_idx % 2  # 0:hot, 1:cold

    def _build_state(self, location, confidence):
        """根据位置和信心构建状态索引"""
        return location * 2 + confidence

    def _build_mdp(self):
        """构建MDP的转移概率和奖励函数"""

        for s in range(self.n_states):
            current_loc = self._get_location(s)
            current_conf = self._get_confidence(s)

            # 动作0: shoot
            make_prob = self._get_make_probability(s)

            # 投篮成功的情况：信心变为hot，位置不变
            s_success = self._build_state(current_loc, 0)  # 变为hot
            self.P[s, 0, s_success] = make_prob
            self.R[s, 0, s_success] = 2.0  # 投进得2分

            # 投篮失败的情况：信心变为cold，位置不变
            s_fail = self._build_state(current_loc, 1)  # 变为cold
            self.P[s, 0, s_fail] = 1 - make_prob
            self.R[s, 0, s_fail] = 0.0  # 投不进得0分

            # 动作1: move_closer (向篮筐移动)
            if current_loc == 0:  # 已经在最近处，保持
                new_loc = 0
            elif current_loc == 1:  # 中距离→近距离
                new_loc = 0
            else:  # 远距离→中距离
                new_loc = 1

            s_new = self._build_state(new_loc, current_conf)  # 信心不变
            self.P[s, 1, s_new] = 1.0
            self.R[s, 1, s_new] = -0.1  # 移动消耗0.1体力

            # 动作2: move_farther (远离篮筐移动)
            if current_loc == 0:  # 近距离→中距离
                new_loc = 1
            elif current_loc == 1:  # 中距离→远距离
                new_loc = 2
            else:  # 已经在最远处，保持
                new_loc = 2

            s_new = self._build_state(new_loc, current_conf)  # 信心不变
            self.P[s, 2, s_new] = 1.0
            self.R[s, 2, s_new] = -0.1  # 移动消耗0.1体力

    def value_iteration(self, theta=1e-6):
        """值迭代算法求解最优值函数和策略"""
        V = np.zeros(self.n_states)
        policy = np.zeros(self.n_states, dtype=int)

        print("开始值迭代...")
        iteration = 0

        while True:
            delta = 0
            # 对每个状态进行更新
            for s in range(self.n_states):
                v_old = V[s]

                # 计算每个动作的Q值
                q_values = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    for s_next in range(self.n_states):
                        if self.P[s, a, s_next] > 0:
                            q_values[a] += self.P[s, a, s_next] * (
                                    self.R[s, a, s_next] + self.gamma * V[s_next]
                            )

                # 选择最大的Q值更新V
                V[s] = np.max(q_values)
                policy[s] = np.argmax(q_values)

                delta = max(delta, abs(v_old - V[s]))

            iteration += 1
            print(f"迭代 {iteration}, 最大变化: {delta:.6f}")

            if delta < theta:
                break

        return V, policy

    def print_policy(self, policy):
        """打印最优策略"""
        print("\n最优策略:")
        print("=" * 40)
        for s in range(self.n_states):
            action_name = self.actions[policy[s]]
            print(f"状态 {self.states[s]:<10} -> 动作: {action_name}")

    def simulate_episode(self, policy, start_state=0, max_steps=10):
        """使用策略模拟一个回合"""
        print(f"\n模拟回合 (初始状态: {self.states[start_state]}):")
        print("=" * 40)

        current_state = start_state
        total_reward = 0
        consecutive_scores = 0  # 连续得分计数

        for step in range(max_steps):
            action = policy[current_state]
            action_name = self.actions[action]

            # 根据策略选择动作，然后根据概率转移到下一个状态
            next_state_probs = self.P[current_state, action, :]
            next_state = np.random.choice(self.n_states, p=next_state_probs)

            # 获取奖励
            reward = self.R[current_state, action, next_state]
            total_reward += reward

            print(f"步骤 {step + 1}: 状态 {self.states[current_state]:<10} -> "
                  f"动作 {action_name:<12} -> "
                  f"新状态 {self.states[next_state]:<10} | "
                  f"奖励: {reward:+.1f}")

            # 更新连续得分计数
            if reward == 2.0:
                consecutive_scores += 1
            else:
                consecutive_scores = 0

            current_state = next_state

            # 如果连续投篮成功3次，提前结束
            if consecutive_scores >= 3:
                print("连续多次投篮成功，提前结束模拟")
                break

        print(f"总奖励: {total_reward:.2f}")
        return total_reward

    def analyze_policy(self, policy):
        """分析策略的合理性"""
        print("\n策略深度分析:")
        print("=" * 50)

        for s in range(self.n_states):
            state_name = self.states[s]
            action = policy[s]
            action_name = self.actions[action]
            make_prob = self._get_make_probability(s)
            loc = self._get_location(s)
            conf = self._get_confidence(s)

            # 计算期望奖励
            expected_reward = 0
            for s_next in range(self.n_states):
                if self.P[s, action, s_next] > 0:
                    expected_reward += self.P[s, action, s_next] * self.R[s, action, s_next]

            rationale = ""
            if action == 0:  # shoot
                if make_prob > 0.6:
                    rationale = "高命中率，直接投篮"
                else:
                    rationale = "虽然命中率不高，但移动成本更高"
            elif action == 1:  # move_closer
                rationale = "移动到更近位置提高命中率"
            else:  # move_farther
                rationale = "特殊情况下的策略选择"

            print(f"{state_name:<10} | 命中率: {make_prob:.1%} | 动作: {action_name:<12} | "
                  f"期望奖励: {expected_reward:+.2f} | {rationale}")


def main():
    # 创建MDP环境
    print("创建投篮机器人MDP模型...")
    mdp = ShootingRobotMDP()

    # 显示状态转移概率示例
    print("\n状态转移概率示例 (状态 mid-cold):")
    print("状态索引 3 对应:", mdp.states[3])
    for a in range(mdp.n_actions):
        print(f"  动作 {mdp.actions[a]}:")
        for s_next in range(mdp.n_states):
            if mdp.P[3, a, s_next] > 0:
                print(f"    -> {mdp.states[s_next]:<10} 概率: {mdp.P[3, a, s_next]:.2f} "
                      f"奖励: {mdp.R[3, a, s_next]:.1f}")

    # 使用值迭代求解最优策略
    V_star, policy_star = mdp.value_iteration()

    # 显示最优值函数
    print("\n最优值函数:")
    print("=" * 40)
    for s in range(mdp.n_states):
        print(f"状态 {mdp.states[s]:<10}: V* = {V_star[s]:.3f}")

    # 显示最优策略
    mdp.print_policy(policy_star)

    # 深度分析策略
    mdp.analyze_policy(policy_star)

    # 模拟几个回合
    print("\n" + "=" * 50)
    print("开始模拟回合")
    print("=" * 50)

    # 从不同初始状态模拟
    start_states = [0, 3, 5]  # near-hot, mid-cold, far-cold
    for start_state in start_states:
        mdp.simulate_episode(policy_star, start_state=start_state, max_steps=8)
        print()

    # 显示一些有趣的策略洞察
    print("\n策略洞察:")
    print("=" * 30)
    print("1. 在near-hot状态直接投篮 (90%命中率)")
    print("2. 在far-cold状态先移动到更近位置")
    print("3. 移动动作有成本，只在必要时使用")
    print("4. 信心状态对决策有重要影响")


if __name__ == "__main__":
    main()