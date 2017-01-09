import gym
import numpy as np
import time
from gym import wrappers


class OpenAI(object):
    MONITOR = "./frozenlake-experiment-1"
    API_KEY = "sk_7VlNAYVTXGVHRTY3Dk8Q"
    
class ReinforcementLearning(OpenAI):
    LEARNING_RATE = 0.8 
    DISCOUNT_FACTOR = 0.99 #importance of future reward [0,1]
    N_EPISODES = 10000

    def __init__(self):
        self.q_table = np.zeros([self.N_STATES, self.N_ACTIONS])

        
        
class FrozenLake(ReinforcementLearning):
    N_STEPS = 100 #number of steps to take within a game

    def __init__(self):
        self.game = gym.make('FrozenLake-v0')
        self.game = wrappers.Monitor(self.game, self.MONITOR, force=True)
        self.N_STATES = self.game.observation_space.n
        self.N_ACTIONS = self.game.action_space.n
        super().__init__()
        
    def learn_to_play(self):
        n_episodes = self.__class__.N_EPISODES
        
        for i in range(n_episodes):
            self.current_episode = i
            self._play_and_update_q_table()


        self.game.close()
        gym.upload(self.MONITOR, api_key=self.API_KEY)
        
        return self.q_table

    
    def _play_and_update_q_table(self):
        self._initialize_game()
        
        for _ in range(self.__class__.N_STEPS):
            self._render_game()
            self._choose_action()
            self._game_continues()
            self._update_q_table()
            self.current_state = self.new_state

            if self.is_dead:
                break



        
    def _initialize_game(self):
        self.current_state = self.game.reset()
        self.is_dead = False
        self.estimated_future_reward = 0

        
    def _choose_action(self):
        """Returns the best action to take following epsilon-greedy algorithm
        
        Explanation
        -----------
        We look up the q_table to choose which action is the best to take given the current state. A priori, it is thus the argmax. But, we alter these values with a random probability (epsilon-greedy algorithm) to explore other possible actions. This randomness decreases over time using a damping factor.
        """
        exploitation_probability = self.q_table[self.current_state, :]
        
        time_damping = 1./(self.current_episode+1)
        random_probability = np.random.randn(1, self.game.action_space.n)
        exploration_probability = random_probability * time_damping
        
        self.action = np.argmax(exploitation_probability + exploration_probability)
        return self.action

    
    def _game_continues(self):
        """Reward is 1.0 if goal is reached, otherwise 0
        
        Source
        ------
        https://gym.openai.com/envs/FrozenLake-v0
        """
        new_state, reward, is_dead, _ = self.game.step(self.action)
        self.new_state = new_state
        self.reward = self._customize_reward(new_state)
        self.is_dead = is_dead
        self.estimated_future_reward +=  reward

        
    def _update_q_table(self):
        """Update a single value in the Q-table because we only know what happened for a given (state, action) event"""
        learning_rate = self.__class__.LEARNING_RATE
        discount_factor = self.__class__.DISCOUNT_FACTOR

        instantaneous_reward = self.reward
        future_reward = discount_factor*np.max(self.q_table[self.new_state, :]) - self.q_table[self.current_state, self.action]
        total_reward = instantaneous_reward + future_reward
        
        self.q_table[self.current_state, self.action] +=  learning_rate*total_reward


    @staticmethod
    def _customize_reward(state):
        custom_reward = {5:-0.1,
                         7:-0.1,
                         11:-0.1,
                         12:-0.1,
                         15:100.}

        try:
            return custom_reward[state]
        except KeyError:
            return 0.

    def _render_game(self):
        if self.current_episode%100 == 0:
            self.game.render()


q_table = FrozenLake().learn_to_play()
