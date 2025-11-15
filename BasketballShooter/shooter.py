import numpy as np
from colorama import Fore, Style, init

# 初始化颜色输出
init(autoreset=True)


class MDP():
    def __init__(self):
        self.states = [
            'near_hot', 'near_cold',
            'mid_hot', 'mid_cold',
            'far_hot', 'far_cold'
        ]

        self.actions = [
            'shoot',
            'move_closer',
            'move_further'
        ]

        self.n_states = len(self.states)
        self.n_actions = len(self.actions)

        self.gamma = 0.9

        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions, self.n_states))

        self.build_mdp()

    def get_shoot_prob(self, curr_state):
        if curr_state == 0:
            return 0.9
        elif curr_state == 1:
            return 0.7
        elif curr_state == 2:
            return 0.6
        elif curr_state == 3:
            return 0.4
        elif curr_state == 4:
            return 0.3
        else:
            return 0.2

    def get_curr_location(self, curr_state):
        return curr_state // 2

    def get_curr_confidence(self, curr_state):
        return curr_state % 2

    def get_curr_state(self, curr_location, curr_confidence):
        return curr_location * 2 + curr_confidence

    def build_mdp(self):
        for s in range(self.n_states):
            curr_location = self.get_curr_location(s)
            curr_confidence = self.get_curr_confidence(s)

            # 选择投篮动作
            make_prob = self.get_shoot_prob(s)
            s_success = self.get_curr_state(curr_location, 0)
            self.P[s, 0, s_success] = make_prob
            self.R[s, 0, s_success] = 5 * (curr_location + 1)

            s_fail = self.get_curr_state(curr_location, 1)
            self.P[s, 0, s_fail] = 1 - make_prob
            self.R[s, 0, s_fail] = -1

            # 选择move_closer动作
            if curr_location == 0:
                new_loc = 0
            elif curr_location == 1:
                new_loc = 0
            else:
                new_loc = 1

            s_new = self.get_curr_state(new_loc, curr_confidence)
            self.P[s, 1, s_new] = 1
            self.R[s, 1, s_new] = -0.5

            # 选择move_further动作
            if curr_location == 0:
                new_loc = 1
            elif curr_location == 1:
                new_loc = 2
            else:
                new_loc = 2

            s_new = self.get_curr_state(new_loc, curr_confidence)
            self.P[s, 2, s_new] = 1
            self.R[s, 2, s_new] = -0.5

    def value_iteration(self, theta=1e-6):
        V = np.zeros(self.n_states)
        policy = np.zeros(self.n_states, dtype=int)

        iteration = 0
        print(f"{Fore.CYAN}开始值迭代...{Style.RESET_ALL}")
        print(f"{Fore.CYAN}目标精度: {theta}{Style.RESET_ALL}")
        print("-" * 50)

        while True:
            delta = 0

            for s in range(self.n_states):
                v_old = V[s]

                q_values = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    for s_next in range(self.n_states):
                        if self.P[s, a, s_next] > 0:
                            q_values[a] += self.P[s, a, s_next] * (self.R[s, a, s_next] + self.gamma * V[s_next])

                V[s] = np.max(q_values)
                policy[s] = np.argmax(q_values)
                delta = max(delta, abs(V[s] - v_old))

            iteration += 1
            if iteration <= 5 or iteration % 5 == 0 or delta < theta:
                color = Fore.GREEN if delta < theta else Fore.YELLOW
                print(f"{color}迭代 {iteration:2d}, 最大变化: {delta:.6f}{Style.RESET_ALL}")

            if delta < theta:
                print(f"{Fore.GREEN}值迭代收敛于第 {iteration} 次迭代{Style.RESET_ALL}")
                break

        return V, policy

    def run(self, policy, start_state=0, max_steps=20):
        print(f"\n{Fore.CYAN}模拟回合开始 (初始状态: {self.states[start_state]}){Style.RESET_ALL}")
        print("=" * 60)

        curr_state = start_state
        total_reward = 0
        keeps = 0

        print(f"{'步骤':<4} {'当前状态':<12} {'动作':<14} {'奖励':<8} {'累计奖励':<10} {'备注'}")
        print("-" * 60)

        for step in range(max_steps):
            action = policy[curr_state]
            next_state_probs = self.P[curr_state, action, :]
            next_state = np.random.choice(self.n_states, p=next_state_probs)
            reward = self.R[curr_state, action, next_state]
            total_reward += reward

            # 状态变化备注
            remark = ""
            if action == 0:  # shoot
                if reward > 0:
                    remark = f"{Fore.GREEN}投篮命中!{Style.RESET_ALL}"
                    keeps += 1
                else:
                    remark = f"{Fore.RED}投篮未中{Style.RESET_ALL}"
                    keeps = 0
            else:
                keeps = 0
                if action == 1:
                    remark = f"{Fore.BLUE}向篮筐移动{Style.RESET_ALL}"
                else:
                    remark = f"{Fore.MAGENTA}远离篮筐移动{Style.RESET_ALL}"

            # 打印步骤信息
            reward_color = Fore.GREEN if reward > 0 else Fore.RED if reward < 0 else Fore.YELLOW
            print(f"{step + 1:<4} {self.states[curr_state]:<12} {self.actions[action]:<14} "
                  f"{reward_color}{reward:>6.2f}{Style.RESET_ALL} {total_reward:>9.2f}   {remark}")

            curr_state = next_state

            # 检查是否提前结束
            if keeps == 5:
                print(f"{Fore.GREEN}连续5次投篮成功，提前结束模拟!{Style.RESET_ALL}")
                break

        total_color = Fore.GREEN if total_reward > 0 else Fore.RED if total_reward < 0 else Fore.YELLOW
        print(f"{Fore.CYAN}模拟结束，累计奖励: {total_color}{total_reward:.2f}{Style.RESET_ALL}")
        print("=" * 60)

        return total_reward

    def policy_analyze(self, policy, V):
        print(f"\n{Fore.CYAN}策略分析结果{Style.RESET_ALL}")
        print("=" * 90)
        print(f"{'状态':<12} {'位置':<6} {'信心':<6} {'命中率':<8} {'最优动作':<14} {'状态价值':<10} {'策略解释'}")
        print("-" * 90)

        for s in range(self.n_states):
            state_name = self.states[s]
            action = policy[s]
            action_name = self.actions[action]
            make_prob = self.get_shoot_prob(s)
            loc = self.get_curr_location(s)
            conf = self.get_curr_confidence(s)
            confidence_text = "高" if conf == 0 else "低"
            location_text = ["近", "中", "远"][loc]

            # 策略解释
            rationale = ""
            if action == 0:  # shoot
                if make_prob > 0.7:
                    rationale = f"{Fore.GREEN}高命中率，直接投篮收益最高{Style.RESET_ALL}"
                elif make_prob > 0.4:
                    rationale = f"{Fore.YELLOW}中等命中率，投篮是最佳选择{Style.RESET_ALL}"
                else:
                    rationale = f"{Fore.RED}低命中率但移动代价更高，仍选择投篮{Style.RESET_ALL}"
            elif action == 1:  # move_closer
                rationale = f"{Fore.BLUE}移动到更近位置提高未来命中率{Style.RESET_ALL}"
            else:  # move_farther
                rationale = f"{Fore.MAGENTA}特殊情况下的策略选择{Style.RESET_ALL}"

            # 根据状态价值着色
            value_color = Fore.GREEN if V[s] > 1.5 else Fore.YELLOW if V[s] > 0 else Fore.RED

            print(f"{state_name:<12} {location_text:<6} {confidence_text:<6} {make_prob:>7.1%} "
                  f"{action_name:<14} {value_color}{V[s]:>9.2f}{Style.RESET_ALL}  {rationale}")

        print("=" * 90)


def main():
    mdp = MDP()

    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'篮球投篮决策MDP分析':^60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")

    V_star, policy_star = mdp.value_iteration()

    # 深度分析策略
    mdp.policy_analyze(policy_star, V_star)

    # 模拟不同起始状态
    start_states = [0, 3, 5]  # near-hot, mid-cold, far-cold
    start_state_names = ["近距高信心", "中距低信心", "远距低信心"]

    print(f"\n{Fore.CYAN}{'模拟不同起始状态的策略执行':^60}{Style.RESET_ALL}")

    for i, start_state in enumerate(start_states):
        print(f"\n{Fore.YELLOW}模拟 {i + 1}: {start_state_names[i]}{Style.RESET_ALL}")
        mdp.run(policy_star, start_state=start_state, max_steps=20)


if __name__ == "__main__":
    main()