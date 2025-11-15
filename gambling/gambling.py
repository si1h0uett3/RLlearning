import numpy as np
from typing import *

class gambler_model:
    def __init__(self, goal=100, p_h=0.4):
        self.goal = goal
        self.p_h = p_h

        self.states = np.arange(goal+1)
        self.values = np.zeros(goal+1)
        self.policy = np.zeros(goal + 1)
        self.values[goal] = 1.0
        self.values[0] = 0.0



    def value_iteration(self, theta=1e-9, max_iteration=100):

        for iteration in range(max_iteration):
            delta = 0
            values = self.values.copy()

            for state in range(1, self.goal):
                possible_actions = range(1,  min(state, self.goal - state) + 1)

                action_values = []

                for action in possible_actions:
                    value = self.p_h * self.values[state + action] + (1 - self.p_h) * self.values[state - action]
                    action_values.append(value)

                best_action = possible_actions[np.argmax(action_values)]
                values[state] = max(action_values)
                delta = max(delta, abs(self.values[state] - values[state]))
                self.policy[state] = best_action

            self.values = values

            if delta < theta:
                break
        return self.values, self.policy


    def policy_evaluation(self, policy, theta=1e-9, max_iterations=100):
        values = np.zeros(self.goal+1, dtype=float)
        values[self.goal] = 1.0
        values[0] = 0.0

        for iteration in range(max_iterations):
            delta = 0
            new_values = values.copy()

            for state in range(1, self.goal):
                action = int(policy[state])
                if action == 0 or state + action > self.goal or state - action < 0:
                    continue

                new_values[state] = self.p_h * values[state + action] + (1 - self.p_h) * values[state - action]
                diff = np.abs(new_values[state] - values[state])
                delta = max(delta, diff)

            values = new_values

            if delta < theta:
                break

        return  values

    def policy_improvement(self, values):
        new_policy = np.zeros(self.goal+1)

        for state in range(1, self.goal):
            possible_actions = range(1, min(state, self.goal-state)+1)

            if not possible_actions:
                continue

            best_value = -float('inf')
            best_action = 0

            for action in possible_actions:
                value = self.p_h * values[state + action] + (1 - self.p_h) * values[state - action]

                if value > best_value:
                    best_value = value
                    best_action = action


            new_policy[state] = best_action

        return  new_policy

    def policy_iteration(self, max_iterations=100):
        policy = np.zeros(self.goal+1)

        for state in range(1, self.goal):
            possible_actions = list(range(1, min(state, self.goal-state)+1))
            if possible_actions:
                policy[state] = np.random.choice(possible_actions)

        for iteration in range(max_iterations):
            values = self.policy_evaluation(policy)
            new_policy = self.policy_improvement(values)

            if np.array_equal(policy, new_policy):
                print("策略已收敛")
                self.values = values
                self.policy = new_policy
                return values, new_policy

            policy = new_policy

        self.values = values
        self.policy = policy
        return values, policy

    def simulate_episode(self, initial_capital, policy):
        if policy is None:
            policy = self.policy

        state = initial_capital
        history = []

        while state > 0 and state < self.goal:
            action = int(policy[state])
            if action == 0:
                action = 1

            if np.random.random() < self.p_h:
                next_state = state + action
                reward = 0

            else:
                next_state = state - action
                reward = 0

            if next_state >= self.goal:
                reward = 1
            elif next_state <= 0:
                reward = 0

            history.append((state, action, reward))
            state = next_state

        return history









