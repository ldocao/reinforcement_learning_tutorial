import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import gym
from gym import wrappers


class OpenAIGym(object):
    MONITOR = "./frozenlake-experiment-1"
    API_KEY = "sk_7VlNAYVTXGVHRTY3Dk8Q"

    def __init__(self, game):
        self.game = game
        self.N_STATES = self.game.observation_space.n
        self.N_ACTIONS = self.game.action_space.n



    
class Sarsa(OpenAIGym):
    """SARSA on-policy learning algorithm"""

    DISCOUNT_FACTOR = 0.9 #importance of future reward [0,1], aka gamma parameter


    def __init__(self, game):
        super().__init__(game)
        self.q_table = np.zeros([self.N_STATES, self.N_ACTIONS]) #values in this matrix represent the potential reward for a given (state, action) couple
        self.action_counter = np.zeros([self.N_STATES, self.N_ACTIONS]) #number of times an action is selected so that damping factor is used the same way for every actions
        #for dev purposes
        self.q_history = []
        self.reward_history = []
        self.status_history = []
        
    def _update_q_table(self):
        """Update a single value in the Q-table because we only know what happened for a given (state, action) event, following SARSA algorithm. In this algorithm, the alpha parameter (aka learning rate) controls the importance of past learning vs recent learning. If alpha=0, then the algorithm will not learn anything. On the other hand, if alpha=1, the algorithm will consider the most recent information."""
        a = self.action
        s = self.current_state
        alpha = self.learning_rate(s, a)
        gamma = self.__class__.DISCOUNT_FACTOR

        instantaneous_reward = self.reward
        maximal_future_reward = np.max(self.q_table[self.new_state, :])
        future_reward = gamma * maximal_future_reward 
        total_reward = instantaneous_reward + future_reward

        self.q_table[s, a] = (1-alpha)*self.q_table[s, a] + alpha*total_reward
        
        self.q_history.append(self.q_table)
        self.reward_history.append(total_reward)


    def learning_rate(self, state, action):
        """Returns the learning rate for a given (state, action). This allows for an adaptive learning rate consistently for each different action available"""
        learning_rate = 1. / (1+self.action_counter[state, action])
        self.action_counter[state, action] += 1
        return learning_rate



class EpsilonGreedy(object):
    EXPLORATION = 0.1 #fraction of time the algorithm we'll choose a random action instead of the greedy one
    
    def choose_action(self):
        """Returns the action to take following epsilon-greedy algorithm"""
        if self._is_greedy():
            self.action = np.argmax(self.q_table[self.current_state, :])
        else:
            self.action = random.randint(0, self.N_ACTIONS-1)
            
        return self.action

    @classmethod
    def _is_greedy(cls):
        xi = random.random()
        is_greedy = xi > cls.EXPLORATION
        if is_greedy:
            return True
        else:
            return False




        
        
class Agent(Sarsa, EpsilonGreedy):
    N_EPISODES = 10000
    N_STEPS = 200 #max number of steps to take within a game
        
    def learn_to_play(self):
        n_episodes = self.__class__.N_EPISODES
        
        for _ in range(n_episodes):
            self._play_and_update_q_table()

    
    def _play_and_update_q_table(self):
        self._initialize_game()
        
        for _ in range(self.__class__.N_STEPS):
            self.choose_action()
            self._game_continues()
            self._update_q_table()
            self.current_state = self.new_state

            if self.is_dead:
                break


    def _initialize_game(self):
        self.current_state = self.game.reset()
        self.is_dead = False

    
    def _game_continues(self):
        new_state, reward, is_dead, _ = self.game.step(self.action)
        reward = self._customized_reward(new_state, is_dead)
        self.new_state = new_state
        self.is_dead = is_dead
        self.reward = reward
        self.status_history.append(self.current_state)

        
    def _customized_reward(self, new_state, is_dead):
        """Initially, reward is 1.0 if goal is reached, otherwise 0. We change this behavior"""
                
        GOAL = 15
        
        if is_dead:
            return -1
        elif new_state == GOAL:
            return 10.
        elif new_state == self.current_state:
            return -0.1
        else:
            distance_traveled = new_state - self.current_state
            getting_closer = max(0, distance_traveled) #we don't want to penalize if we go back, or at least it shouldn't be worse than falling in a hole
            return getting_closer
            

            
frozen_lake = gym.make('FrozenLake-v0')

f = 1.
h = -1.
n = 0.
g = 10.
optimal_answer = [(n, f, f, n),
                  (n, h, n, f),
                  (n, f, f, n),
                  (n, f, h, n),
                  (n, n, n, n),#row2
                  (h, f, h, n),
                  (n, n, n, n),
                  (n, h, f, n),
                  (n, f, f, n),#row3
                  (n, f, f, h),
                  (n, f, h, n),
                  (n, n, n, n),
                  (n, n, n, n),#row4
                  (h, n, f, n),
                  (n, n, g, n),
                  (n, n, n, n)]


#frozen_lake = wrappers.Monitor(frozen_lake, OpenAIGym.MONITOR, force=True)
ai = Agent(frozen_lake)
ai.learn_to_play()

#frozen_lake.close()
#gym.upload(OpenAIGym.MONITOR, api_key=OpenAIGym.API_KEY)

np.set_printoptions(precision=3)
print("left       down        right        up")
print(ai.q_table)
