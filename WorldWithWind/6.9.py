import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyArrow
import random


class WindyGridWorld:
    def __init__(self, n=7, wind_strength=None):
        self.n = n  # rows
        self.m = 10  # columns
        # Wind strength: number of cells blown upward in each column
        self.wind_strength = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

        # 8 actions: up, down, left, right, and 4 diagonals
        self.actions = {
            0: (-1, 0),  # up
            1: (1, 0),  # down
            2: (0, -1),  # left
            3: (0, 1),  # right
            4: (-1, -1),  # up-left
            5: (-1, 1),  # up-right
            6: (1, -1),  # down-left
            7: (1, 1),  # down-right
            8: (0, 0)
        }

        self.start_state = (3, 0)  # start at row 3, column 0
        self.goal_state = (3, 7)  # goal at row 3, column 7

        # Initialize Q-tables for SARSA and Q-learning
        self.q_sarsa = np.zeros((self.n, self.m, len(self.actions)))
        self.q_qlearning = np.zeros((self.n, self.m, len(self.actions)))

    def is_valid_state(self, state):
        row, col = state
        return 0 <= row < self.n and 0 <= col < self.m

    def apply_wind(self, state):
        """Apply wind effect to the state - this happens EVERY step in windy columns"""
        row, col = state
        wind = self.wind_strength[col]
        new_row = max(0, row - wind)  # wind blows upward
        return (new_row, col)

    def step(self, state, action):
        row, col = state
        dr, dc = self.actions[action]

        # Apply movement first
        new_row = max(0, min(self.n - 1, row + dr))
        new_col = max(0, min(self.m - 1, col + dc))
        new_state = (new_row, new_col)

        # THEN apply wind effect - this happens regardless of action in windy columns
        new_state = self.apply_wind(new_state)

        # Reward: -1 for each step, 0 for reaching goal
        reward = -1
        if new_state == self.goal_state:
            reward = 0

        return new_state, reward

    def choose_action(self, q_table, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, len(self.actions) - 1)
        else:
            row, col = state
            return np.argmax(q_table[row, col, :])

    def sarsa_learning(self, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
        steps_per_episode = []
        success_rate = []

        for episode in range(episodes):
            state = self.start_state
            action = self.choose_action(self.q_sarsa, state, epsilon)
            steps = 0
            done = False

            while not done and steps < 1000:
                next_state, reward = self.step(state, action)
                next_action = self.choose_action(self.q_sarsa, next_state, epsilon)

                # SARSA update
                row, col = state
                next_row, next_col = next_state

                td_target = reward + gamma * self.q_sarsa[next_row, next_col, next_action]
                td_error = td_target - self.q_sarsa[row, col, action]
                self.q_sarsa[row, col, action] += alpha * td_error

                state, action = next_state, next_action
                steps += 1

                if state == self.goal_state:
                    done = True

            steps_per_episode.append(steps)
            success_rate.append(1 if done else 0)

            if episode % 100 == 0:
                success_percent = np.mean(success_rate[-100:]) * 100
                print(f"SARSA Episode {episode}, Steps: {steps}, Success: {success_percent:.1f}%")

        return steps_per_episode, success_rate

    def q_learning(self, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
        steps_per_episode = []
        success_rate = []

        for episode in range(episodes):
            state = self.start_state
            steps = 0
            done = False

            while not done and steps < 1000:
                action = self.choose_action(self.q_qlearning, state, epsilon)
                next_state, reward = self.step(state, action)

                # Q-learning update
                row, col = state
                next_row, next_col = next_state

                td_target = reward + gamma * np.max(self.q_qlearning[next_row, next_col, :])
                td_error = td_target - self.q_qlearning[row, col, action]
                self.q_qlearning[row, col, action] += alpha * td_error

                state = next_state
                steps += 1

                if state == self.goal_state:
                    done = True

            steps_per_episode.append(steps)
            success_rate.append(1 if done else 0)

            if episode % 100 == 0:
                success_percent = np.mean(success_rate[-100:]) * 100
                print(f"Q-learning Episode {episode}, Steps: {steps}, Success: {success_percent:.1f}%")

        return steps_per_episode, success_rate

    def get_optimal_path(self, q_table):
        """Get the optimal path using the learned policy"""
        path = [self.start_state]
        state = self.start_state
        visited = set()

        while state != self.goal_state and len(path) < 50:
            if state in visited:  # Prevent infinite loops
                break
            visited.add(state)

            row, col = state
            action = np.argmax(q_table[row, col, :])
            next_state, _ = self.step(state, action)

            path.append(next_state)
            state = next_state

        return path

    def analyze_wind_effect(self):
        """Analyze how wind affects movement from different positions"""
        print("\nWind Effect Analysis:")
        for col in range(self.m):
            wind = self.wind_strength[col]
            if wind > 0:
                print(f"Column {col}: Wind strength {wind}")
                for row in range(min(3, self.n)):
                    original = (row, col)
                    after_wind = self.apply_wind(original)
                    print(f"  From ({row},{col}) -> ({after_wind[0]},{after_wind[1]})")

    def plot_results(self, sarsa_steps, qlearning_steps, sarsa_success, qlearning_success, sarsa_path, qlearning_path):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Plot learning curves
        ax1.plot(sarsa_steps, label='SARSA', alpha=0.7, color='blue')
        ax1.plot(qlearning_steps, label='Q-learning', alpha=0.7, color='red')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Steps per Episode')
        ax1.set_title('Learning Curves: SARSA vs Q-learning')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot success rates
        window = 50
        sarsa_success_smooth = np.convolve(sarsa_success, np.ones(window) / window, mode='valid')
        qlearning_success_smooth = np.convolve(qlearning_success, np.ones(window) / window, mode='valid')

        ax2.plot(sarsa_success_smooth, label='SARSA', color='blue')
        ax2.plot(qlearning_success_smooth, label='Q-learning', color='red')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate (Smoothed)')
        ax2.set_title('Success Rates Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot SARSA path
        self._plot_path(ax3, sarsa_path, 'SARSA Optimal Path with Wind Effect')

        # Plot Q-learning path
        self._plot_path(ax4, qlearning_path, 'Q-learning Optimal Path with Wind Effect')

        plt.tight_layout()
        plt.show()

    def _plot_path(self, ax, path, title):
        # Create grid
        for i in range(self.n):
            for j in range(self.m):
                # Color based on wind strength
                wind = self.wind_strength[j]
                if wind == 0:
                    color = 'white'
                elif wind == 1:
                    color = 'lightblue'
                else:
                    color = 'deepskyblue'

                if (i, j) == self.start_state:
                    color = 'lightgreen'
                elif (i, j) == self.goal_state:
                    color = 'lightcoral'

                ax.add_patch(Rectangle((j, i), 1, 1, fill=True, color=color, alpha=0.7))
                ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='black', alpha=0.3))

        # Plot wind strength indicators
        for j, wind in enumerate(self.wind_strength):
            if wind > 0:
                ax.text(j + 0.5, -0.3, f'Wind: +{wind}', ha='center', va='center',
                        fontsize=8, color='blue', fontweight='bold')

        # Plot path with arrows showing movement + wind effect
        for k in range(len(path) - 1):
            i1, j1 = path[k]
            i2, j2 = path[k + 1]

            # Convert to center coordinates
            x1, y1 = j1 + 0.5, i1 + 0.5
            x2, y2 = j2 + 0.5, i2 + 0.5

            # Draw arrow for the actual movement
            ax.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0.1,
                     head_length=0.1, fc='red', ec='red', alpha=0.8)

            # If wind affected this move, show wind effect
            if self.wind_strength[j1] > 0:
                wind_effect = self.wind_strength[j1]
                ax.text(x1, y1 - 0.2, f'+{wind_effect}W', ha='center', va='top',
                        fontsize=6, color='blue', fontweight='bold')

        # Mark start and goal
        start_i, start_j = self.start_state
        goal_i, goal_j = self.goal_state
        ax.text(start_j + 0.5, start_i + 0.5, 'START', ha='center', va='center',
                fontweight='bold', fontsize=10)
        ax.text(goal_j + 0.5, goal_i + 0.5, 'GOAL', ha='center', va='center',
                fontweight='bold', fontsize=10)

        ax.set_xlim(-0.5, self.m + 0.5)
        ax.set_ylim(-0.5, self.n + 0.5)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel('Columns')
        ax.set_ylabel('Rows')

        # Add grid lines
        ax.set_xticks(np.arange(0, self.m + 1))
        ax.set_yticks(np.arange(0, self.n + 1))
        ax.grid(True, alpha=0.3)


def main():
    # Initialize the windy grid world with correct wind strengths
    wind_strength = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]  # Wind strength for each column
    env = WindyGridWorld(n=7, wind_strength=wind_strength)

    # Analyze wind effect
    env.analyze_wind_effect()

    print("\nTraining SARSA...")
    sarsa_steps, sarsa_success = env.sarsa_learning(episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1)

    print("\nTraining Q-learning...")
    qlearning_steps, qlearning_success = env.q_learning(episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1)

    # Get optimal paths
    sarsa_path = env.get_optimal_path(env.q_sarsa)
    qlearning_path = env.get_optimal_path(env.q_qlearning)

    print(f"\nSARSA optimal path length: {len(sarsa_path)}")
    print(f"Q-learning optimal path length: {len(qlearning_path)}")

    # Plot results
    env.plot_results(sarsa_steps, qlearning_steps, sarsa_success, qlearning_success, sarsa_path, qlearning_path)

    # Print policy analysis
    print("\nPolicy Analysis:")
    print("Both algorithms learn to navigate the wind effects:")
    print("- In columns with wind=1, agent might move downward to compensate")
    print("- In columns with wind=2, agent needs more strategic positioning")
    print("- The optimal path utilizes wind to reach the goal faster")


if __name__ == "__main__":
    main()