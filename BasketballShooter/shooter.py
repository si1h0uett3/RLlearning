import numpy as np
from colorama import Fore, Style, init

# 初始化颜色输出
init(autoreset=True)


class MDP():
    """
    篮球投篮决策的马尔可夫决策过程（MDP）

    模拟一个篮球运动员在不同位置和信心状态下的投篮决策问题
    状态空间：位置（近、中、远） × 信心（热、冷）
    动作空间：投篮、靠近篮筐、远离篮筐
    """

    def __init__(self):
        """
        初始化MDP环境

        定义6个状态：
        - near_hot: 近距高信心
        - near_cold: 近距低信心
        - mid_hot: 中距高信心
        - mid_cold: 中距低信心
        - far_hot: 远距高信心
        - far_cold: 远距低信心
        """
        # 定义状态空间：包含距离和手感两个维度
        self.states = [
            'near_hot', 'near_cold',
            'mid_hot', 'mid_cold',
            'far_hot', 'far_cold'
        ]

        # 定义动作空间：投篮、移动靠近、移动远离
        self.actions = [
            'shoot',  # 投篮动作
            'move_closer',  # 向篮筐移动
            'move_further'  # 远离篮筐移动
        ]

        # 获取状态和动作的数量，用于后续数组初始化
        self.n_states = len(self.states)
        self.n_actions = len(self.actions)

        # 折扣因子：权衡即时奖励与未来奖励的重要性
        self.gamma = 0.9

        # 初始化状态转移概率矩阵 P(s, a, s') 和奖励矩阵 R(s, a, s')
        # P: 在状态s执行动作a后转移到状态s'的概率
        # R: 在状态s执行动作a后转移到状态s'获得的即时奖励
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions, self.n_states))

        # 构建MDP的转移概率和奖励结构
        self.build_mdp()

    def get_shoot_prob(self, curr_state):
        """
        获取指定状态下的投篮命中概率

        Args:
            curr_state: 当前状态索引

        Returns:
            float: 该状态下的投篮命中概率
        """
        # 不同状态的命中率映射
        shoot_probabilities = {
            0: 0.9,  # near_hot: 90%命中率
            1: 0.7,  # near_cold: 70%命中率
            2: 0.6,  # mid_hot: 60%命中率
            3: 0.4,  # mid_cold: 40%命中率
            4: 0.3,  # far_hot: 30%命中率
            5: 0.2  # far_cold: 20%命中率
        }
        return shoot_probabilities[curr_state]

    def get_curr_location(self, curr_state):
        """
        从状态索引获取位置信息

        Args:
            curr_state: 状态索引

        Returns:
            int: 位置索引 (0:近, 1:中, 2:远)
        """
        return curr_state // 2  # 利用整数除法提取位置信息

    def get_curr_confidence(self, curr_state):
        """
        从状态索引获取信心状态

        Args:
            curr_state: 状态索引

        Returns:
            int: 信心状态 (0:热, 1:冷)
        """
        return curr_state % 2  # 利用取模运算提取信心信息

    def get_curr_state(self, curr_location, curr_confidence):
        """
        从位置和信心信息重构状态索引

        Args:
            curr_location: 位置索引 (0:近, 1:中, 2:远)
            curr_confidence: 信心状态 (0:热, 1:冷)

        Returns:
            int: 对应的状态索引
        """
        return curr_location * 2 + curr_confidence

    def build_mdp(self):
        """
        构建MDP的转移概率和奖励结构

        为每个状态和动作组合定义：
        - 转移到下一个状态的概率
        - 对应的即时奖励
        """
        for s in range(self.n_states):
            # 提取当前状态的位置和信心信息
            curr_location = self.get_curr_location(s)
            curr_confidence = self.get_curr_confidence(s)

            # 动作0: shoot (投篮)
            make_prob = self.get_shoot_prob(s)

            # 投篮成功的情况：保持位置，信心变为热状态
            s_success = self.get_curr_state(curr_location, 0)  # 信心变为热
            self.P[s, 0, s_success] = make_prob  # 成功概率
            self.R[s, 0, s_success] = 5 * (curr_location + 1)  # 奖励与位置相关

            # 投篮失败的情况：保持位置，信心变为冷状态
            s_fail = self.get_curr_state(curr_location, 1)  # 信心变为冷
            self.P[s, 0, s_fail] = 1 - make_prob  # 失败概率
            self.R[s, 0, s_fail] = -1  # 失败惩罚

            # 动作1: move_closer (向篮筐移动)
            if curr_location == 0:
                new_loc = 0  # 已经在最近位置，保持不变
            elif curr_location == 1:
                new_loc = 0  # 从中距移动到近距
            else:
                new_loc = 1  # 从远距移动到中距

            # 移动后状态：位置改变，信心保持不变
            s_new = self.get_curr_state(new_loc, curr_confidence)
            self.P[s, 1, s_new] = 1.0  # 确定性转移
            self.R[s, 1, s_new] = -0.5  # 移动消耗

            # 动作2: move_further (远离篮筐移动)
            if curr_location == 0:
                new_loc = 1  # 从近距移动到中距
            elif curr_location == 1:
                new_loc = 2  # 从中距移动到远距
            else:
                new_loc = 2  # 已经在最远位置，保持不变

            # 移动后状态：位置改变，信心保持不变
            s_new = self.get_curr_state(new_loc, curr_confidence)
            self.P[s, 2, s_new] = 1.0  # 确定性转移
            self.R[s, 2, s_new] = -0.5  # 移动消耗

    def value_iteration(self, theta=1e-6):
        """
        值迭代算法求解MDP的最优策略

        通过贝尔曼最优方程迭代更新状态价值函数，直到收敛
        算法原理: V_{k+1}(s) = max_a Σ P(s'|s,a)[R(s,a,s') + γV_k(s')]

        Args:
            theta: 收敛阈值，当价值函数变化小于该值时停止迭代

        Returns:
            V: 最优状态价值函数
            policy: 最优策略（每个状态的最优动作）
        """
        # 初始化状态价值函数和策略
        V = np.zeros(self.n_states)  # 状态价值函数
        policy = np.zeros(self.n_states, dtype=int)  # 策略函数

        iteration = 0
        print(f"{Fore.CYAN}开始值迭代...{Style.RESET_ALL}")
        print(f"{Fore.CYAN}目标精度: {theta}{Style.RESET_ALL}")
        print("-" * 50)

        while True:
            delta = 0  # 记录本轮迭代中价值函数的最大变化

            # 遍历所有状态
            for s in range(self.n_states):
                v_old = V[s]  # 保存旧值用于比较

                # 计算当前状态s下每个动作的Q值
                q_values = np.zeros(self.n_actions)
                for a in range(self.n_actions):
                    # 遍历所有可能的下一状态
                    for s_next in range(self.n_states):
                        if self.P[s, a, s_next] > 0:
                            # Q值计算：转移概率 × (即时奖励 + 折扣的未来价值)
                            q_values[a] += self.P[s, a, s_next] * (
                                    self.R[s, a, s_next] + self.gamma * V[s_next]
                            )

                # 贝尔曼最优方程更新：V(s) = max_a Q(s,a)
                V[s] = np.max(q_values)
                # 策略更新：选择使Q值最大的动作
                policy[s] = np.argmax(q_values)

                # 更新最大变化量
                delta = max(delta, abs(V[s] - v_old))

            iteration += 1

            # 打印迭代进度（前5次、每5次、或收敛时）
            if iteration <= 5 or iteration % 5 == 0 or delta < theta:
                color = Fore.GREEN if delta < theta else Fore.YELLOW
                print(f"{color}迭代 {iteration:2d}, 最大变化: {delta:.6f}{Style.RESET_ALL}")

            # 检查收敛条件
            if delta < theta:
                print(f"{Fore.GREEN}值迭代收敛于第 {iteration} 次迭代{Style.RESET_ALL}")
                break

        return V, policy

    def run(self, policy, start_state=0, max_steps=20):
        """
        使用给定策略运行模拟回合

        Args:
            policy: 策略数组，指定每个状态应该执行的动作
            start_state: 起始状态索引
            max_steps: 最大模拟步数

        Returns:
            total_reward: 本轮模拟的总奖励
        """
        print(f"\n{Fore.CYAN}模拟回合开始 (初始状态: {self.states[start_state]}){Style.RESET_ALL}")
        print("=" * 60)

        curr_state = start_state
        total_reward = 0
        keeps = 0  # 连续投篮成功计数

        # 打印表头
        print(f"{'步骤':<4} {'当前状态':<12} {'动作':<14} {'奖励':<8} {'累计奖励':<10} {'备注'}")
        print("-" * 60)

        for step in range(max_steps):
            # 根据策略选择动作
            action = policy[curr_state]
            # 根据转移概率随机选择下一状态
            next_state_probs = self.P[curr_state, action, :]
            next_state = np.random.choice(self.n_states, p=next_state_probs)
            # 获取即时奖励
            reward = self.R[curr_state, action, next_state]
            total_reward += reward

            # 生成状态变化备注信息
            remark = ""
            if action == 0:  # shoot动作
                if reward > 0:
                    remark = f"{Fore.GREEN}投篮命中!{Style.RESET_ALL}"
                    keeps += 1  # 增加连续成功计数
                else:
                    remark = f"{Fore.RED}投篮未中{Style.RESET_ALL}"
                    keeps = 0  # 重置连续成功计数
            else:
                keeps = 0  # 非投篮动作重置计数
                if action == 1:
                    remark = f"{Fore.BLUE}向篮筐移动{Style.RESET_ALL}"
                else:
                    remark = f"{Fore.MAGENTA}远离篮筐移动{Style.RESET_ALL}"

            # 根据奖励值选择颜色
            reward_color = Fore.GREEN if reward > 0 else Fore.RED if reward < 0 else Fore.YELLOW

            # 打印当前步骤信息
            print(f"{step + 1:<4} {self.states[curr_state]:<12} {self.actions[action]:<14} "
                  f"{reward_color}{reward:>6.2f}{Style.RESET_ALL} {total_reward:>9.2f}   {remark}")

            # 更新当前状态
            curr_state = next_state

            # 检查提前结束条件：连续5次投篮成功
            if keeps == 5:
                print(f"{Fore.GREEN}连续5次投篮成功，提前结束模拟!{Style.RESET_ALL}")
                break

        # 打印最终结果
        total_color = Fore.GREEN if total_reward > 0 else Fore.RED if total_reward < 0 else Fore.YELLOW
        print(f"{Fore.CYAN}模拟结束，累计奖励: {total_color}{total_reward:.2f}{Style.RESET_ALL}")
        print("=" * 60)

        return total_reward

    def policy_analyze(self, policy, V):
        """
        分析并展示最优策略的详细信息

        Args:
            policy: 最优策略
            V: 最优状态价值函数
        """
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

            # 转换编码为可读文本
            confidence_text = "高" if conf == 0 else "低"
            location_text = ["近", "中", "远"][loc]

            # 生成策略解释
            rationale = ""
            if action == 0:  # shoot动作
                if make_prob > 0.7:
                    rationale = f"{Fore.GREEN}高命中率，直接投篮收益最高{Style.RESET_ALL}"
                elif make_prob > 0.4:
                    rationale = f"{Fore.YELLOW}中等命中率，投篮是最佳选择{Style.RESET_ALL}"
                else:
                    rationale = f"{Fore.RED}低命中率但移动代价更高，仍选择投篮{Style.RESET_ALL}"
            elif action == 1:  # move_closer动作
                rationale = f"{Fore.BLUE}移动到更近位置提高未来命中率{Style.RESET_ALL}"
            else:  # move_further动作
                rationale = f"{Fore.MAGENTA}追求更高得分潜力而主动转移{Style.RESET_ALL}"

            # 根据状态价值选择颜色
            value_color = Fore.GREEN if V[s] > 1.5 else Fore.YELLOW if V[s] > 0 else Fore.RED

            # 打印状态分析行
            print(f"{state_name:<12} {location_text:<6} {confidence_text:<6} {make_prob:>7.1%} "
                  f"{action_name:<14} {value_color}{V[s]:>9.2f}{Style.RESET_ALL}  {rationale}")

        print("=" * 90)


def main():
    """
    主函数：运行完整的MDP分析和模拟
    """
    # 创建MDP环境实例
    mdp = MDP()

    # 打印标题
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'篮球投篮决策MDP分析':^60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")

    # 执行值迭代算法求解最优策略
    V_star, policy_star = mdp.value_iteration()

    # 深度分析最优策略
    mdp.policy_analyze(policy_star, V_star)

    # 定义测试的起始状态
    start_states = [0, 3, 5]  # near-hot, mid-cold, far-cold
    start_state_names = ["近距高信心", "中距低信心", "远距低信心"]

    print(f"\n{Fore.CYAN}{'模拟不同起始状态的策略执行':^60}{Style.RESET_ALL}")

    # 对每个起始状态运行模拟
    for i, start_state in enumerate(start_states):
        print(f"\n{Fore.YELLOW}模拟 {i + 1}: {start_state_names[i]}{Style.RESET_ALL}")
        mdp.run(policy_star, start_state=start_state, max_steps=20)


if __name__ == "__main__":
    main()