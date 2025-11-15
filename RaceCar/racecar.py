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
        self.width = 21
        self.height = 11

        self.start_line = [(0, y) for y in range(2, 9)]
        self.finish_line = [(20, y) for y in range(2, 9)]

        self.obstacles = [
            [8, 12, 0, 4],
            [6, 12, 6, 10]
        ]

        self.min_speed = 0
        self.max_speed = 4


    def is_valid_position(self, x, y):
        if not (0 <= x <= self.width and 0 <= y <= self.height):
            return False

        for obs in self.obstacles:
            x_min, x_max, y_min, y_max = obs
            if (x_min <= x <= x_max) and (y_min <= y <= y_max):
                return False

        return  True

    def is_finish_line_crossed(self, x1, y1, x2, y2):
        if not (x1 <= 20 <= x2):
            return  False

        k = (20 - x1) / (x2 - x1)

        return 2 <= y1 + k * (y2 - y1) <= 9



class RaceCar:
    def __init__(self, track):
        self.track = track
        self.reset()


    def reset(self):
        self.x, self.y = random.choice(self.track.start_line)
        self.vx, self.vy = 0, 0

        return (self.x, self.y, self.vx, self.vy)


    def step(self, action, deterministic=False):
        dvx, dvy = action

        if not deterministic and random.random() < 0.1:
            self.vx, self.vy = 0, 0
        else:
            self.vx = np.clip(self.vx + dvx, self.track.min_speed, self.track.max_speed)
            self.vy = np.clip(self.vy + dvy, self.track.min_speed, self.track.max_speed)

        if (self.vx, self.vy) == (0, 0) and (self.x, self.y) not in self.track.start_line:
            self.vx = 1


        new_x = self.x + self.vx
        new_y = self.y + self.vy

        if self.track.is_finish_line_crossed(self.x, self.y, new_x, new_y):
            self.x, self.y = new_x, new_y
            return (self.x, self.y, self.vx, self.vy), 0, True

        if not self.track.is_valid_position(new_x, new_y):
            state = self.reset()

            return state, -1, False

        self.x, self.y = new_x, new_y
        return (self.x, self.y, self.vx, self.vy), -1, False



class MonteCarloAgent:
    def __init__(self, track, epsilon=0.1):
            self.track = track
            self.epsilon = epsilon
            self.actions = [(dvx, dvy) for dvx in [-1, 0, 1] for dvy in [-1, 0, 1]]

            self.Q = defaultdict(lambda: np.zeros(len(self.actions)))
            self.returns = defaultdict(lambda: np.zeros(len(self.actions)))
            self.visits = defaultdict(lambda: np.zeros(len(self.actions)))
            self.policy = {}

            self.episode_rewards = []      # 添加这行
            self.episode_lengths = []      # 添加这行
            self.success_rate = []



    def state_to_key(self, state):
            x, y, vx, vy = state
            return (x, y, vx, vy)

    def get_action(self, state, deterministic=False):
            state_key = self.state_to_key(state)

            if deterministic:
                return self.actions[np.argmax(self.Q[state_key])]

            if random.random() < self.epsilon:
                return random.choice(self.actions)
            else:
                return self.actions[np.argmax(self.Q[state_key])]


    def update_policy(self, episode):
            states, actions, rewards = episode
            G = 0

            for t in range(len(states) - 1, -1, -1):
                state = states[t]
                action = actions[t]
                reward = rewards[t]

                G = reward + G

                state_key = self.state_to_key(state)
                action_idx = self.actions.index(action)

                self.returns[state_key][action_idx] += G
                self.visits[state_key][action_idx] += 1
                self.Q[state_key][action_idx] = (
                        self.returns[state_key][action_idx] / self.visits[state_key][action_idx]
                )

                self.policy[state_key] = self.actions[np.argmax(self.Q[state_key])]

    def train(self, num_episodes):
        car = RaceCar(self.track)

        for episode in range(num_episodes):
            states = []
            actions = []
            rewards = []

            state = car.reset()
            done = False
            episode_reward = 0

            while not done and len(states) < 1000:
                action = self.get_action(state)
                next_state, reward, done = car.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                episode_reward += reward

                state = next_state

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(len(states))
            self.success_rate.append(1 if rewards[-1] == 0 else 0)

            self.update_policy((states, actions, rewards))


def main():
    track = RaceTrack()
    agent = MonteCarloAgent(track, epsilon=0.1)
    agent.train(num_episodes=10000)

if __name__ == "__main__":
    main()



