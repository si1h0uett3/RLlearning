import numpy as np

class MDP:
    def __init__(self):
        self.state = [
            'near-hot', 'near-cold',
            'mid-hot', 'mid-cold',
            'far-hot', 'far-cold'
        ]

        self.action = ['shoot', 'move-close', 'move-further']

        self.n_states = len(self.state)
        self.n_actions = len(self.action)

        self.gamma = 0.9

        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions, self.n_states))

        self.build_mdp()


    def get_shoot_probability(self, state_idx):
        if state_idx == 0:
            return 0.9
        elif state_idx == 1:
            return 0.7
        elif state_idx == 2:
            return 0.7
        elif state_idx == 3:
            return 0.5
        elif state_idx == 4:
            return 0.4
        else:
            return 0.2


    def get_location(self, state_idx):
        return state_idx // 2

    def get_confidence(self, state_idx):
        return state_idx % 2

    def get_state(self, location, confidence):
        return location * 2 + confidence

    def build_mdp(self):

        for s in range(self.n_states):
            current_loc = self.get_location(s)
            current_conf = self.get_confidence(s)

            make_prob = self.get_shoot_probability(s)

            s_success = self.get_state(current_loc, 0)
            self.P[s, 0, s_success] = make_prob
            if s >= 4:
                self.R[s, 0, s_success] = 4
            elif s >= 2:
                self.R[s, 0, s_success] = 3
            else:
                self.R[s, 0, s_success] = 2

            s_fail = self.get_state(current_loc, 1)
            self.P[s, 0, s_fail] = 1 - make_prob
            self.R[s, 0, s_fail] = -0.2


            #定义move_close
            if current_loc == 0:
                new_loc = 0
            elif current_loc == 1:
                new_loc = 0
            else:
                new_loc = 1

            s_new = self.get_state(new_loc, current_conf)
            self.P[s, 1, s_new] = 1.0
            self.R[s, 1, s_new] = -1

            #定义move_further
            if current_loc == 0:
                new_loc = 1
            elif current_loc == 1:
                new_loc = 2
            else:
                new_loc = 2

            s_new = self.get_state(new_loc, current_conf)
            self.P[s, 2, s_new] = 1
            self.R[s, 2, s_new] = -0.5

    def value_iteration(self, theta = 1e-6):
        V = np.zeros(self.n_states)
        policy = np.zeros(self.n_states, dtype = int)

        print("开始值迭代")
        iteration = 0

        while True:
            delta = 0
            for s in range(self.n_states):
                v_old = V[s]

                q_values = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    for s_next in range(self.n_states):
                        if self.P[s, a, s_next] > 0:
                            q_values[a] += self.P[s, a, s_next] * (
                                    self.R[s, a, s_next] + self.gamma * V[s_next]
                            )

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
            action_name = self.action[policy[s]]
            print(f"状态 {self.state[s]:<10} -> 动作: {action_name}")

    def simulate_episode(self, policy, start_state=0, max_steps=10):
        print(f"\n模拟回合 (初始状态: {self.state[start_state]}):")
        print("=" * 40)

        current_state = start_state
        total_reward = 0
        consecutive_scores = 0

        for step in range(max_steps):
            action = policy[current_state]
            action_name = self.action[action]

            next_state_probs = self.P[current_state, action, :]
            next_state = np.random.choice(self.n_states, p=next_state_probs)

            reward = self.R[current_state, action, next_state]
            total_reward += reward

            print(f"步骤 {step + 1}: 状态 {self.state[current_state]:<10} -> "
                  f"动作 {action_name:<12} -> "
                  f"新状态 {self.state[next_state]:<10} | "
                  f"奖励: {reward:+.1f}")

            # 更新连续得分计数
            if reward > 0:
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
            state_name = self.state[s]
            action = policy[s]
            action_name = self.action[action]
            make_prob = self.get_shoot_probability(s)
            loc = self.get_location(s)
            conf = self.get_confidence(s)

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
            elif action == 1:  # move_close
                rationale = "移动到更近位置提高命中率"
            else:  # move_farther
                rationale = "特殊情况下的策略选择"

            print(f"{state_name:<10} | 命中率: {make_prob:.1%} | 动作: {action_name:<12} | "
                  f"期望奖励: {expected_reward:+.2f} | {rationale}")