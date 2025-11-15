import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from config import *
from average import *
from constant import *
from UCB_average import *
from UCB_constant import *

if __name__ == "__main__":
    rewards_average, optimal_actions_average = run_average(n_runs, total_steps, n_arms, epsilon)
    rewards_constant, optimal_actions_constant = run_constant(n_runs, total_steps, n_arms, epsilon, alpha)
    rewards_UCB_average, optimal_actions_UCB_average = run_UCB_average(n_runs, total_steps, n_arms, C)
    rewards_UCB_constant, optimal_actions_UCB_constant = run_UCB_constant(n_runs, total_steps, n_arms, alpha, C)

    print("Average method:")
    print(f"Average reward: {rewards_average.mean():.4f}, Optimal action %: {optimal_actions_average.mean():.4f}")

    print("Constant method:")
    print(f"Average reward: {rewards_constant.mean():.4f}, Optimal action %: {optimal_actions_constant.mean():.4f}")

    print("UCB Average method:")
    print(
        f"Average reward: {rewards_UCB_average.mean():.4f}, Optimal action %: {optimal_actions_UCB_average.mean():.4f}")

    print("UCB Constant method:")
    print(
        f"Average reward: {rewards_UCB_constant.mean():.4f}, Optimal action %: {optimal_actions_UCB_constant.mean():.4f}")

    # 创建时间步数组作为横坐标
    steps = np.arange(total_steps)

    # 滑动窗口平均奖励
    window = 100
    smoothed_average = np.convolve(rewards_average.mean(axis=0), np.ones(window) / window, mode='valid')
    smoothed_constant = np.convolve(rewards_constant.mean(axis=0), np.ones(window) / window, mode='valid')
    smoothed_UCB_average = np.convolve(rewards_UCB_average.mean(axis=0), np.ones(window) / window, mode='valid')
    smoothed_UCB_constant = np.convolve(rewards_UCB_constant.mean(axis=0), np.ones(window) / window, mode='valid')
    smoothed_steps = np.arange(len(smoothed_average))

    # 最终性能比较数据
    final_rewards = [
        rewards_average[:, -1000:].mean(),
        rewards_constant[:, -1000:].mean(),
        rewards_UCB_average[:, -1000:].mean(),
        rewards_UCB_constant[:, -1000:].mean()
    ]

    final_optimal = [
        optimal_actions_average[:, -1000:].mean() * 100,
        optimal_actions_constant[:, -1000:].mean() * 100,
        optimal_actions_UCB_average[:, -1000:].mean() * 100,
        optimal_actions_UCB_constant[:, -1000:].mean() * 100
    ]

    final_performance = pd.DataFrame({
        'Method': ['Sample Average', 'Constant Step-size', 'UCB Sample Average', 'UCB Constant Step-size'],
        'Average Reward': final_rewards,
        'Optimal Action %': final_optimal
    })

    # 子图1: 平均奖励对比
    plt.figure(figsize=(12, 8))
    plt.plot(steps, rewards_average.mean(axis=0), label='Sample Average', alpha=0.8, linewidth=2)
    plt.plot(steps, rewards_constant.mean(axis=0), label='Constant Step-size (α=0.1)', alpha=0.8, linewidth=2)
    plt.plot(steps, rewards_UCB_average.mean(axis=0), label='UCB Sample Average', alpha=0.8, linewidth=2)
    plt.plot(steps, rewards_UCB_constant.mean(axis=0), label='UCB Constant Step-size (α=0.1)', alpha=0.8, linewidth=2)
    plt.xlabel('Time Steps')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('average_reward_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: average_reward_comparison.png")
    plt.show()

    # 子图2: 最优动作百分比
    plt.figure(figsize=(12, 8))
    plt.plot(steps, optimal_actions_average.mean(axis=0) * 100, label='Sample Average', alpha=0.8, linewidth=2)
    plt.plot(steps, optimal_actions_constant.mean(axis=0) * 100, label='Constant Step-size (α=0.1)', alpha=0.8,
             linewidth=2)
    plt.plot(steps, optimal_actions_UCB_average.mean(axis=0) * 100, label='UCB Sample Average', alpha=0.8, linewidth=2)
    plt.plot(steps, optimal_actions_UCB_constant.mean(axis=0) * 100, label='UCB Constant Step-size (α=0.1)', alpha=0.8,
             linewidth=2)
    plt.xlabel('Time Steps')
    plt.ylabel('Optimal Action Percentage (%)')
    plt.title('Optimal Action Selection Percentage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('optimal_action_percentage.png', dpi=300, bbox_inches='tight')
    print("Saved: optimal_action_percentage.png")
    plt.show()

    # 子图3: 平滑后的平均奖励
    plt.figure(figsize=(12, 8))
    plt.plot(smoothed_steps, smoothed_average, label='Sample Average', alpha=0.8, linewidth=2)
    plt.plot(smoothed_steps, smoothed_constant, label='Constant Step-size (α=0.1)', alpha=0.8, linewidth=2)
    plt.plot(smoothed_steps, smoothed_UCB_average, label='UCB Sample Average', alpha=0.8, linewidth=2)
    plt.plot(smoothed_steps, smoothed_UCB_constant, label='UCB Constant Step-size (α=0.1)', alpha=0.8, linewidth=2)
    plt.xlabel('Time Steps')
    plt.ylabel('Moving Average Reward (window=100)')
    plt.title('Smoothed Average Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('smoothed_reward.png', dpi=300, bbox_inches='tight')
    print("Saved: smoothed_reward.png")
    plt.show()

    # 子图4: 最终性能比较
    plt.figure(figsize=(12, 8))
    x_pos = np.arange(len(final_performance))

    plt.bar(x_pos - 0.2, final_performance['Average Reward'], width=0.4,
            label='Average Reward', alpha=0.7, color='skyblue')
    plt.bar(x_pos + 0.2, final_performance['Optimal Action %'], width=0.4,
            label='Optimal Action %', alpha=0.7, color='lightcoral')

    plt.xlabel('Method')
    plt.ylabel('Performance Metrics')
    plt.title('Final Performance Comparison (Last 1000 Steps)')
    plt.xticks(x_pos, final_performance['Method'], rotation=15)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for i, v in enumerate(final_performance['Average Reward']):
        plt.text(i - 0.2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(final_performance['Optimal Action %']):
        plt.text(i + 0.2, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('final_performance.png', dpi=300, bbox_inches='tight')
    print("Saved: final_performance.png")
    plt.show()

    print("\nAll plots have been saved successfully!")