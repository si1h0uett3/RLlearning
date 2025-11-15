from MDP import *


def main():
    print("创建投篮机器人MDP模型...")
    mdp = MDP()

    print("\n状态转移概率示例 (状态 mid-cold):")
    print("状态索引 3 对应:", mdp.state[3])
    for a in range(mdp.n_actions):
        print(f"  动作 {mdp.action[a]}:")
        for s_next in range(mdp.n_states):
            if mdp.P[3, a, s_next] > 0:
                print(f"    -> {mdp.state[s_next]:<10} 概率: {mdp.P[3, a, s_next]:.2f} "
                      f"奖励: {mdp.R[3, a, s_next]:.1f}")

    V_star, policy_star = mdp.value_iteration()

    print("\n最优值函数:")
    print("=" * 40)
    for s in range(mdp.n_states):
        print(f"状态 {mdp.state[s]:<10}: V* = {V_star[s]:.3f}")

    mdp.print_policy(policy_star)
    mdp.analyze_policy(policy_star)

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