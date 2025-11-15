from MDP import *

def simulate_episode(self, policy, start_state=0, max_steps=10):
    print(f"\n模拟回合 (初始状态: {self.states[start_state]}):")
    print("=" * 40)

    current_state = start_state
    total_reward = 0
    consecutive_scores = 0

    for step in range(max_steps):
        action = policy[current_state]
        action_name = self.actions[action]

        next_state_probs = self.P[current_state, action, :]
        next_state = np.random.choice(self.n_states, p = next_state_probs)

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
                rationale = "命中率不高，综合考虑"
        elif action == 1:  # move_closer
            rationale = "移动到更近位置提高命中率"
        else:  # move_farther
            rationale = "特殊情况下的策略选择"

        print(f"{state_name:<10} | 命中率: {make_prob:.1%} | 动作: {action_name:<12} | "
              f"期望奖励: {expected_reward:+.2f} | {rationale}")