import numpy as np

class NonStationaryBandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.q_star = np.zeros(n_arms)

    def step(self, action):
        reward = self.q_star[action] + np.random.normal(0, 0.01)
        self.q_star += np.random.normal(0, 0.01, self.n_arms)

        return reward

    def get_optimal_action(self):
        return  np.argmax(self.q_star)