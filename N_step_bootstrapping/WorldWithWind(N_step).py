import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyArrow
import random
import os


class WorldWithWind:
    def __init__(self):
        self.m = 7
        self.n = 10
        self.wind_strength = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

        self.actions = [
            (0, 1),  # right
            (0, -1),  # left
            (-1, 0),  # up
            (1, 0),  # down
            (1, 1),  # down-right
            (-1, 1),  # up-right
            (1, -1),  # down-left
            (-1, -1),  # up-left
            (0, 0)  # stay
        ]

        self.start_state = (4, 0)
        self.end_state = (2, 7)

        self.Q_table1 = np.zeros((self.m, self.n, len(self.actions)))  # SARSA
        self.Q_table2 = np.zeros((self.m, self.n, len(self.actions)))  # Q-learning

    def is_valid_position(self, state):
        x, y = state
        return 0 <= x < self.m and 0 <= y < self.n

    def wind(self, state):
        x, y = state
        windstrength = self.wind_strength[y]
        new_x = max(0, x - windstrength)
        return (new_x, y)

    def step(self, state, action):
        x, y = state
        vx, vy = self.actions[action]

        new_x = max(0, min(self.m - 1, x + vx))
        new_y = max(0, min(self.n - 1, y + vy))
        new_state = (new_x, new_y)

        new_state = self.wind(new_state)

        if new_state == self.end_state:
            reward = 0
        else:
            reward = -1

        return new_state, reward

    def choose_action(self, Q_table, state, epsilon):
        if np.random.random() < epsilon:
            action = random.randint(0, len(self.actions) - 1)
        else:
            x, y = state
            action = np.argmax(Q_table[x, y, :])
        return action

    def sarsa_agent(self, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
        steps_per_episode = []
        success_rate = []

        for i in range(episodes):
            state = self.start_state
            action = self.choose_action(self.Q_table1, state, epsilon)
            step = 0
            done = False

            # 存储轨迹用于n步更新
            trajectory = []

            while not done and step < 1000:
                next_state, reward = self.step(state, action)
                next_action = self.choose_action(self.Q_table1, next_state, epsilon)

                next_x, next_y = next_state

                # 存储当前步的经验
                trajectory.append((state, action, reward))

                # 如果轨迹足够长或者回合结束，进行更新
                if len(trajectory) >= 3 or done:
                    # 计算n步回报
                    n = len(trajectory)
                    G = 0
                    # 计算前n-1步的折扣回报
                    for j in range(n - 1):
                        G += (gamma ** j) * trajectory[j][2]

                    # 加上最后一步的Q值估计
                    if done:
                        G += (gamma ** (n - 1)) * trajectory[-1][2]
                    else:
                        G += (gamma ** (n - 1)) * self.Q_table1[next_x, next_y, next_action]

                    # 更新第一个状态
                    first_state, first_action, _ = trajectory[0]
                    x, y = first_state
                    td_error = G - self.Q_table1[x, y, first_action]
                    self.Q_table1[x, y, first_action] += alpha * td_error

                    # 移除已更新的经验
                    trajectory.pop(0)

                state, action = next_state, next_action
                step += 1

                if next_state == self.end_state:
                    done = True

            steps_per_episode.append(step)
            success_rate.append(1 if done else 0)

            # Progress display
            if i % 100 == 0:
                recent_success = np.mean(success_rate[-100:]) if i > 0 else success_rate[0]
                print(
                    f"SARSA Episode {i}: Steps = {step}, Success = {done}, Recent Success Rate = {recent_success:.2f}")

        return steps_per_episode, success_rate

    def Q_learning_agent(self, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
        steps_per_episode = []
        success_rate = []

        for i in range(episodes):
            state = self.start_state
            step = 0
            done = False

            trajectory = []

            while not done and step < 1000:
                # Choose current action
                action = self.choose_action(self.Q_table2, state, epsilon)
                next_state, reward = self.step(state, action)

                next_x, next_y = next_state

                trajectory.append((state, action, reward))

                if len(trajectory) >= 3 or done:
                    n = len(trajectory)
                    G = 0

                    for j in range(n - 1):
                        G += (gamma ** j) * trajectory[j][2]

                    if done:
                        G += (gamma ** (n - 1)) * trajectory[-1][2]

                    else:
                        max_next_q = np.max(self.Q_table2[next_x, next_y, :])
                        G += (gamma ** (n-1)) * max_next_q


                    first_state, first_action, _ = trajectory[0]
                    x, y = first_state
                    td_error = G - self.Q_table2[x, y, first_action]
                    self.Q_table2[x, y, first_action] += alpha * td_error

                    trajectory.pop(0)

                # Move to next state
                state = next_state
                step += 1

                if state == self.end_state:
                    done = True

            steps_per_episode.append(step)
            success_rate.append(1 if done else 0)

            # Progress display
            if i % 100 == 0:
                recent_success = np.mean(success_rate[-100:]) if i > 0 else success_rate[0]
                print(
                    f"Q-learning Episode {i}: Steps = {step}, Success = {done}, Recent Success Rate = {recent_success:.2f}")

        return steps_per_episode, success_rate

    def analyze_results(self, sarsa_steps, sarsa_success, qlearning_steps, qlearning_success):
        """Analyze training results"""
        print("\n" + "=" * 50)
        print("Training Results Analysis")
        print("=" * 50)

        # SARSA results
        sarsa_final_success = np.mean(sarsa_success[-100:])
        sarsa_avg_steps = np.mean(sarsa_steps[-100:])
        print(f"SARSA:")
        print(f"  - Final Success Rate: {sarsa_final_success:.2%}")
        print(f"  - Average Steps: {sarsa_avg_steps:.1f}")
        print(f"  - Total Success: {sum(sarsa_success)}/{len(sarsa_success)}")

        # Q-learning results
        qlearning_final_success = np.mean(qlearning_success[-100:])
        qlearning_avg_steps = np.mean(qlearning_steps[-100:])
        print(f"Q-learning:")
        print(f"  - Final Success Rate: {qlearning_final_success:.2%}")
        print(f"  - Average Steps: {qlearning_avg_steps:.1f}")
        print(f"  - Total Success: {sum(qlearning_success)}/{len(qlearning_success)}")

        # Comparison
        if sarsa_final_success > qlearning_final_success:
            print(f"SARSA performs better, success rate higher by {sarsa_final_success - qlearning_final_success:.2%}")
        elif qlearning_final_success > sarsa_final_success:
            print(
                f"Q-learning performs better, success rate higher by {qlearning_final_success - sarsa_final_success:.2%}")
        else:
            print("Both algorithms perform similarly")

    def plot_learning_curves(self, sarsa_steps, sarsa_success, qlearning_steps, qlearning_success, save_dir='results'):
        """Plot learning curves and save separately"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 1. Steps comparison
        plt.figure(figsize=(10, 6))
        window = 50
        sarsa_steps_smooth = [np.mean(sarsa_steps[i:i + window]) for i in range(len(sarsa_steps) - window + 1)]
        qlearning_steps_smooth = [np.mean(qlearning_steps[i:i + window]) for i in
                                  range(len(qlearning_steps) - window + 1)]

        plt.plot(range(window - 1, len(sarsa_steps)), sarsa_steps_smooth, 'b-', label='SARSA', linewidth=2)
        plt.plot(range(window - 1, len(qlearning_steps)), qlearning_steps_smooth, 'r-', label='Q-learning', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Steps (Moving Average)')
        plt.title('Learning Curves - Steps Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_dir}/learning_curves_steps.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Success rate comparison
        plt.figure(figsize=(10, 6))
        sarsa_success_smooth = [np.mean(sarsa_success[i:i + window]) for i in range(len(sarsa_success) - window + 1)]
        qlearning_success_smooth = [np.mean(qlearning_success[i:i + window]) for i in
                                    range(len(qlearning_success) - window + 1)]

        plt.plot(range(window - 1, len(sarsa_success)), sarsa_success_smooth, 'b-', label='SARSA', linewidth=2)
        plt.plot(range(window - 1, len(qlearning_success)), qlearning_success_smooth, 'r-', label='Q-learning',
                 linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Success Rate (Moving Average)')
        plt.title('Learning Curves - Success Rate Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_dir}/learning_curves_success.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Cumulative success rate
        plt.figure(figsize=(10, 6))
        sarsa_cumulative = np.cumsum(sarsa_success) / (np.arange(len(sarsa_success)) + 1)
        qlearning_cumulative = np.cumsum(qlearning_success) / (np.arange(len(qlearning_success)) + 1)

        plt.plot(sarsa_cumulative, 'b-', label='SARSA', alpha=0.7)
        plt.plot(qlearning_cumulative, 'r-', label='Q-learning', alpha=0.7)
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Success Rate')
        plt.title('Cumulative Success Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_dir}/cumulative_success.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_policy_comparison(self, save_dir='results'):
        """Plot policy comparison heatmap"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.figure(figsize=(10, 8))

        # Get final policies
        sarsa_policy = np.argmax(self.Q_table1, axis=2)
        qlearning_policy = np.argmax(self.Q_table2, axis=2)

        # Create combined policy map
        combined_policy = np.zeros((self.m, self.n))
        for i in range(self.m):
            for j in range(self.n):
                if (i, j) == self.start_state:
                    combined_policy[i, j] = -2  # Start
                elif (i, j) == self.end_state:
                    combined_policy[i, j] = -1  # Goal
                elif sarsa_policy[i, j] == qlearning_policy[i, j]:
                    combined_policy[i, j] = 1  # Same policy
                else:
                    combined_policy[i, j] = 0  # Different policy

        # Plot heatmap
        im = plt.imshow(combined_policy, cmap='coolwarm', aspect='auto')

        # Add text annotations
        for i in range(self.m):
            for j in range(self.n):
                if (i, j) == self.start_state:
                    plt.text(j, i, 'S', ha='center', va='center', fontweight='bold', fontsize=12)
                elif (i, j) == self.end_state:
                    plt.text(j, i, 'G', ha='center', va='center', fontweight='bold', fontsize=12)
                elif combined_policy[i, j] == 1:
                    plt.text(j, i, '✓', ha='center', va='center', fontweight='bold', fontsize=10)
                elif combined_policy[i, j] == 0:
                    plt.text(j, i, '✗', ha='center', va='center', fontweight='bold', fontsize=10)

        # Add wind background
        for j in range(self.n):
            wind_str = self.wind_strength[j]
            if wind_str > 0:
                plt.gca().add_patch(Rectangle((j - 0.5, -0.5), 1, self.m,
                                              alpha=0.1, color='gray'))
                plt.text(j, -0.5, f'W{wind_str}', ha='center', va='top', fontsize=8)

        plt.colorbar(im, label='Policy Comparison')
        plt.title('Final Policy Comparison\n(✓: Same policy, ✗: Different policy)')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.savefig(f'{save_dir}/policy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_trajectories(self, save_dir='results'):
        """Plot trajectories for both algorithms"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Get optimal paths
        sarsa_path = self.get_optimal_path('sarsa')
        qlearning_path = self.get_optimal_path('qlearning')

        # Plot SARSA trajectory
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        self._plot_single_trajectory(sarsa_path, 'SARSA Optimal Trajectory')
        plt.subplot(1, 2, 2)
        self._plot_single_trajectory(qlearning_path, 'Q-learning Optimal Trajectory')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/trajectories_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Also save individual trajectory plots
        plt.figure(figsize=(8, 6))
        self._plot_single_trajectory(sarsa_path, 'SARSA Optimal Trajectory')
        plt.savefig(f'{save_dir}/sarsa_trajectory.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(8, 6))
        self._plot_single_trajectory(qlearning_path, 'Q-learning Optimal Trajectory')
        plt.savefig(f'{save_dir}/qlearning_trajectory.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_single_trajectory(self, path, title):
        """Plot single trajectory"""
        # Create grid
        plt.imshow(np.zeros((self.m, self.n)), cmap='Greys', alpha=0.3, aspect='auto')

        # Plot wind zones
        for j in range(self.n):
            wind_str = self.wind_strength[j]
            if wind_str > 0:
                plt.gca().add_patch(Rectangle((j - 0.5, -0.5), 1, self.m,
                                              alpha=0.2, color='blue', label=f'Wind {wind_str}'))

        # Plot trajectory
        if len(path) > 1:
            y_coords = [p[1] for p in path]  # columns as x
            x_coords = [p[0] for p in path]  # rows as y

            plt.plot(y_coords, x_coords, 'o-', linewidth=3, markersize=8,
                     color='red', alpha=0.7, label='Path')

            # Mark start and end
            plt.plot(y_coords[0], x_coords[0], 's', markersize=12,
                     color='green', label='Start')
            plt.plot(y_coords[-1], x_coords[-1], 's', markersize=12,
                     color='blue', label='Goal')

            # Add step numbers
            for i, (x, y) in enumerate(path):
                if i == 0 or i == len(path) - 1:
                    plt.text(y, x, f'{i}', ha='center', va='center',
                             fontweight='bold', fontsize=8, color='white')

        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Set limits
        plt.xlim(-0.5, self.n - 0.5)
        plt.ylim(self.m - 0.5, -0.5)

    def get_optimal_path(self, algorithm='sarsa'):
        """Get optimal path for given algorithm"""
        Q_table = self.Q_table1 if algorithm == 'sarsa' else self.Q_table2

        state = self.start_state
        path = [state]
        steps = 0
        max_steps = 50

        while state != self.end_state and steps < max_steps:
            x, y = state
            action = np.argmax(Q_table[x, y, :])
            next_state, _ = self.step(state, action)

            if next_state in path:  # Avoid cycles
                break

            path.append(next_state)
            state = next_state
            steps += 1

        print(f"\n{algorithm.upper()} Optimal Path ({len(path)} steps):")
        for i, (x, y) in enumerate(path):
            print(f"  Step {i}: ({x}, {y})")

        return path

    def save_all_plots(self, sarsa_steps, sarsa_success, qlearning_steps, qlearning_success, save_dir='results'):
        """Save all plots to directory"""
        print(f"\nSaving all plots to '{save_dir}' directory...")

        self.plot_learning_curves(sarsa_steps, sarsa_success, qlearning_steps, qlearning_success, save_dir)
        self.plot_policy_comparison(save_dir)
        self.plot_trajectories(save_dir)

        print("All plots saved successfully!")


def main():
    env = WorldWithWind()

    print("Training SARSA...")
    sarsa_steps, sarsa_success = env.sarsa_agent(episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1)

    print("\nTraining Q-learning...")
    qlearning_steps, qlearning_success = env.Q_learning_agent(episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1)

    # Analyze results
    env.analyze_results(sarsa_steps, sarsa_success, qlearning_steps, qlearning_success)

    # Save all plots
    env.save_all_plots(sarsa_steps, sarsa_success, qlearning_steps, qlearning_success, 'results')

    # Show optimal paths
    sarsa_path = env.get_optimal_path('sarsa')
    qlearning_path = env.get_optimal_path('qlearning')


if __name__ == "__main__":
    main()