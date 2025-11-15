import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
from collections import defaultdict
import time
from tqdm import tqdm
import os
import shutil


class RaceTrack:
    def __init__(self):
        # 赛道参数
        self.width = 21  # x: 0-20
        self.height = 11  # y: 0-10

        # 起点线 (x=0)
        self.start_line = [(0, y) for y in range(2, 9)]  # y=2 to 8

        # 终点线 (x=20)
        self.finish_line = [(20, y) for y in range(2, 9)]

        # 障碍物: [x_min, x_max, y_min, y_max]
        self.obstacles = [
            [8, 12, 0, 4],  # 下方障碍
            [8, 12, 6, 10]  # 上方障碍
        ]

        # 速度范围
        self.min_speed = 0
        self.max_speed = 4

    def is_valid_position(self, x, y):
        """检查位置是否在赛道内"""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False

        # 检查是否在障碍物内
        for obs in self.obstacles:
            x_min, x_max, y_min, y_max = obs
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return False

        return True

    def is_finish_line_crossed(self, x1, y1, x2, y2):
        # 检查是否从左向右穿过x=20
        if not (x1 < 20 <= x2):
            return False

        # 计算在x=20处的y坐标（线性插值）
        t = (20 - x1) / (x2 - x1)
        y_at_20 = y1 + t * (y2 - y1)

        # 检查这个y坐标是否在终点线范围内
        return 2 <= y_at_20 <= 8


class RaceCar:
    def __init__(self, track):
        self.track = track
        self.reset()

    def reset(self):
        """重置到起点"""
        self.x, self.y = random.choice(self.track.start_line)
        self.vx, self.vy = 0, 0
        return (self.x, self.y, self.vx, self.vy)

    def step(self, action, deterministic=False):
        """执行动作"""
        dvx, dvy = action

        # 更新速度（有0.1概率同时变0）
        if not deterministic and random.random() < 0.1:
            self.vx, self.vy = 0, 0
        else:
            self.vx = np.clip(self.vx + dvx, self.track.min_speed, self.track.max_speed)
            self.vy = np.clip(self.vy + dvy, self.track.min_speed, self.track.max_speed)

        # 检查速度不能同时为0（除非在起点）
        if (self.vx, self.vy) == (0, 0) and (self.x, self.y) not in self.track.start_line:
            self.vx = 1  # 给一个最小速度

        # 计算新位置
        new_x = self.x + self.vx
        new_y = self.y + self.vy

        # 检查是否到达终点
        if self.track.is_finish_line_crossed(self.x, self.y, new_x, new_y):
            self.x, self.y = new_x, new_y
            return (self.x, self.y, self.vx, self.vy), 0, True  # 到达终点

        # 简化碰撞检测：只检查新位置是否有效
        if not self.track.is_valid_position(new_x, new_y):
            # 碰撞了，重置到起点
            state = self.reset()
            return state, -1, False

        # 有效移动
        self.x, self.y = new_x, new_y
        return (self.x, self.y, self.vx, self.vy), -1, False


class MonteCarloAgent:
    def __init__(self, track, epsilon=0.1):
        self.track = track
        self.epsilon = epsilon
        self.actions = [(dvx, dvy) for dvx in [-1, 0, 1] for dvy in [-1, 0, 1]]

        # Q值表和计数
        self.Q = defaultdict(lambda: np.zeros(len(self.actions)))
        self.returns = defaultdict(lambda: np.zeros(len(self.actions)))
        self.visits = defaultdict(lambda: np.zeros(len(self.actions)))

        # 策略
        self.policy = {}

        # 训练历史
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = []

    def state_to_key(self, state):
        """将状态转换为可哈希的key"""
        x, y, vx, vy = state
        return (x, y, vx, vy)

    def get_action(self, state, deterministic=False):
        """根据当前策略选择动作"""
        state_key = self.state_to_key(state)

        if deterministic:
            # 确定性策略：选择Q值最大的动作
            return self.actions[np.argmax(self.Q[state_key])]

        # ε-贪婪策略
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return self.actions[np.argmax(self.Q[state_key])]

    def update_policy(self, episode):
        """使用蒙特卡洛方法更新策略"""
        states, actions, rewards = episode
        G = 0

        # 反向遍历episode
        for t in range(len(states) - 1, -1, -1):
            state = states[t]
            action = actions[t]
            reward = rewards[t]

            G = reward + G  # 累积回报

            state_key = self.state_to_key(state)
            action_idx = self.actions.index(action)

            # 每次访问MC
            self.returns[state_key][action_idx] += G
            self.visits[state_key][action_idx] += 1
            self.Q[state_key][action_idx] = (
                    self.returns[state_key][action_idx] / self.visits[state_key][action_idx]
            )

            # 更新策略
            self.policy[state_key] = self.actions[np.argmax(self.Q[state_key])]
                    # ==================== 改为首次访问MC ====================
        # 在循环开始前添加：
        # first_visit = {}
        #
        # 在这里添加检查：
        # state_action_pair = (state_key, action_idx)
        # if state_action_pair not in first_visit:
        #     first_visit[state_action_pair] = True
        #     
        #     # 将上面的更新代码移到这个if语句内：
        #     self.returns[state_key][action_idx] += G
        #     self.visits[state_key][action_idx] += 1
        #     self.Q[state_key][action_idx] = (
        #         self.returns[state_key][action_idx] / self.visits[state_key][action_idx]
        #     )
        #     self.policy[state_key] = self.actions[np.argmax(self.Q[state_key])]
        # ======================================================

    def train(self, num_episodes=10000, eval_interval=100):
        """训练代理"""
        car = RaceCar(self.track)

        print(f"Training for {num_episodes} episodes...")
        start_time = time.time()

        for episode in tqdm(range(num_episodes), desc="Training Progress"):
            states = []
            actions = []
            rewards = []

            state = car.reset()
            done = False
            episode_reward = 0

            # 生成一个episode
            while not done and len(states) < 1000:  # 防止无限循环
                action = self.get_action(state)
                next_state, reward, done = car.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                episode_reward += reward

                state = next_state

            # 记录训练历史
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(len(states))
            self.success_rate.append(1 if rewards[-1] == 0 else 0)  # 最后奖励为0表示成功

            # 更新策略
            self.update_policy((states, actions, rewards))

            # 定期评估和显示进度
            if (episode + 1) % eval_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-eval_interval:])
                avg_length = np.mean(self.episode_lengths[-eval_interval:])
                success_ratio = np.mean(self.success_rate[-eval_interval:])

                tqdm.write(f"Episode {episode + 1}/{num_episodes} | "
                           f"Avg Reward: {avg_reward:.2f} | "
                           f"Avg Length: {avg_length:.1f} | "
                           f"Success Rate: {success_ratio:.2f}")

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Final success rate: {np.mean(self.success_rate[-100:]):.3f}")

    def get_optimal_trajectory(self, start_state=None):
        """获取最优轨迹（无随机性）"""
        car = RaceCar(self.track)
        if start_state:
            car.x, car.y, car.vx, car.vy = start_state
        else:
            car.reset()

        states = [(car.x, car.y, car.vx, car.vy)]
        actions = []

        done = False
        step = 0
        max_steps = 50  # 防止无限循环

        while not done and step < max_steps:
            state = (car.x, car.y, car.vx, car.vy)
            action = self.get_action(state, deterministic=True)
            next_state, reward, done = car.step(action, deterministic=True)

            states.append(next_state)
            actions.append(action)
            step += 1

        return states, actions


class ResultSaver:
    def __init__(self, output_dir="race_track_results"):
        self.output_dir = output_dir
        self.setup_output_directory()

    def setup_output_directory(self):
        """设置输出目录，如果存在则清空"""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        print(f"Output directory created: {self.output_dir}")

    def save_plot(self, filename, dpi=300, bbox_inches='tight'):
        """保存当前图表到文件"""
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Saved: {filepath}")


def plot_training_progress(agent, saver):
    """绘制训练进度并保存"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 平滑函数
    def smooth(data, window=100):
        return np.convolve(data, np.ones(window) / window, mode='valid')

    # 绘制奖励
    ax1.plot(agent.episode_rewards, alpha=0.3, color='blue')
    ax1.plot(smooth(agent.episode_rewards), color='blue', linewidth=2)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True, alpha=0.3)

    # 绘制回合长度
    ax2.plot(agent.episode_lengths, alpha=0.3, color='green')
    ax2.plot(smooth(agent.episode_lengths), color='green', linewidth=2)
    ax2.set_title('Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.grid(True, alpha=0.3)

    # 绘制成功率
    success_rate_smooth = smooth(agent.success_rate, window=50)
    ax3.plot(success_rate_smooth, color='red', linewidth=2)
    ax3.set_title('Success Rate (Smoothed)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Success Rate')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    # 绘制Q值大小
    q_values = []
    for state_key in agent.Q:
        q_values.append(np.max(agent.Q[state_key]))

    if q_values:
        ax4.hist(q_values, bins=50, alpha=0.7, color='purple')
        ax4.set_title('Distribution of Max Q-Values')
        ax4.set_xlabel('Q-Value')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    saver.save_plot("training_progress.png")
    plt.close()


def plot_track_and_trajectory(track, trajectory, saver, title="Race Track - Optimal Trajectory",
                              filename="optimal_trajectory.png"):
    """绘制赛道和轨迹并保存"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制赛道边界
    ax.add_patch(Rectangle((0, 0), track.width - 1, track.height - 1,
                           fill=False, edgecolor='black', linewidth=2))

    # 绘制障碍物
    for obs in track.obstacles:
        x_min, x_max, y_min, y_max = obs
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        ax.add_patch(Rectangle((x_min, y_min), width, height,
                               facecolor='red', alpha=0.3, label='Obstacles'))

    # 绘制起点和终点
    start_x = [0] * len(track.start_line)
    start_y = [y for _, y in track.start_line]
    ax.scatter(start_x, start_y, color='green', s=100, label='Start', zorder=5)

    finish_x = [20] * len(track.finish_line)
    finish_y = [y for _, y in track.finish_line]
    ax.scatter(finish_x, finish_y, color='blue', s=100, label='Finish', zorder=5)

    # 绘制轨迹
    traj_x = [state[0] for state in trajectory]
    traj_y = [state[1] for state in trajectory]
    ax.plot(traj_x, traj_y, 'o-', color='purple', linewidth=2, markersize=6, label='Optimal Trajectory')

    # 标记起点和终点
    ax.scatter([traj_x[0]], [traj_y[0]], color='green', s=200, marker='*', zorder=6, label='Start Point')
    ax.scatter([traj_x[-1]], [traj_y[-1]], color='blue', s=200, marker='*', zorder=6, label='End Point')

    # 添加速度箭头
    for i, state in enumerate(trajectory[:-1]):
        x, y, vx, vy = state
        if vx != 0 or vy != 0:  # 如果有速度
            ax.arrow(x, y, vx * 0.3, vy * 0.3, head_width=0.3, head_length=0.2,
                     fc='orange', ec='orange', alpha=0.7)

    ax.set_xlim(-1, track.width)
    ax.set_ylim(-1, track.height)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')

    plt.tight_layout()
    saver.save_plot(filename)
    plt.close()


def save_trajectory_details(trajectory, actions, saver):
    """保存轨迹详细信息到文本文件"""
    filename = "trajectory_details.txt"
    filepath = os.path.join(saver.output_dir, filename)

    with open(filepath, 'w') as f:
        f.write("Optimal Trajectory Details\n")
        f.write("=" * 50 + "\n")
        f.write(f"Trajectory length: {len(trajectory)} steps\n\n")

        f.write("Trajectory states:\n")
        f.write("Step | Position (x, y) | Velocity (vx, vy)\n")
        f.write("-" * 50 + "\n")
        for i, state in enumerate(trajectory):
            f.write(f"{i:4d} | ({state[0]:2d}, {state[1]:2d})        | ({state[2]:2d}, {state[3]:2d})\n")

        f.write("\nActions taken:\n")
        f.write("Step | Action (dvx, dvy)\n")
        f.write("-" * 30 + "\n")
        for i, action in enumerate(actions):
            f.write(f"{i:4d} | {action}\n")

    print(f"Saved trajectory details: {filepath}")


def main():
    # 创建输出目录管理器
    saver = ResultSaver("race_track_results")

    # 创建赛道和代理
    track = RaceTrack()
    agent = MonteCarloAgent(track, epsilon=0.1)

    print("Training Monte Carlo agent...")
    agent.train(num_episodes=10000, eval_interval=500)

    print("\nGetting optimal trajectory...")
    trajectory, actions = agent.get_optimal_trajectory()

    # 保存轨迹详细信息
    save_trajectory_details(trajectory, actions, saver)

    # 绘制训练进度
    print("\nPlotting and saving training progress...")
    plot_training_progress(agent, saver)

    # 绘制最优轨迹
    print("Plotting and saving optimal trajectory...")
    plot_track_and_trajectory(track, trajectory, saver,
                              "Race Track - Optimal Trajectory",
                              "optimal_trajectory.png")

    # 测试多个起点
    print("\nTesting multiple starting positions...")
    test_starts = [(0, 2, 0, 0), (0, 5, 0, 0), (0, 8, 0, 0)]

    for i, start in enumerate(test_starts):
        trajectory, _ = agent.get_optimal_trajectory(start)
        plot_track_and_trajectory(track, trajectory, saver,
                                  f"Optimal Trajectory from Start Position {i + 1}",
                                  f"optimal_trajectory_start_{i + 1}.png")

    # 保存训练摘要
    summary_file = os.path.join(saver.output_dir, "training_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Training Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Final success rate: {np.mean(agent.success_rate[-100:]):.3f}\n")
        f.write(f"Average episode length: {np.mean(agent.episode_lengths):.1f}\n")
        f.write(f"Average episode reward: {np.mean(agent.episode_rewards):.2f}\n")
        f.write(f"Number of states visited: {len(agent.Q)}\n")

    print(f"\nAll results saved to: {saver.output_dir}")
    print("Generated files:")
    for file in os.listdir(saver.output_dir):
        print(f"  - {file}")


if __name__ == "__main__":
    main()
