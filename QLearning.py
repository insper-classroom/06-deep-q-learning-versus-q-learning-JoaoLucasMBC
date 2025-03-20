import gymnasium as gym
import numpy as np
import os
import datetime


class QLearningMountain:

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.8,
        alpha=0.6,
        epsilon=0.8,
        eps_dec=0.9999,
        eps_min=0.05,
    ):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min

        # Discritize state space
        self.n_states = (env.observation_space.high - env.observation_space.low) * np.array([10, 100])
        self.n_states = np.round(self.n_states, 0).astype(int) + 1
        self.n_actions = env.action_space.n
    
        self.q_table = np.zeros((self.n_states[0], self.n_states[1], self.n_actions))
        self.rewards_per_episode = []
    

    def select_action(self, state: np.ndarray) -> int:
        rv = np.random.random()
        
        if rv < self.epsilon:
            return np.random.choice([i for i in range(self.n_actions)])
        
        return np.argmax(self.q_table[state[0]][state[1]])


    def log(self, folder: str = "logs", prefix="log") -> None:
        # Create logs directory if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        rewards_file_name = f"{folder}/" + prefix + "_rewards_" + date + ".csv"
        q_table_file_name = f"{folder}/" + prefix + "_q_table_" + date

        with open(rewards_file_name, "w") as file:
            file.write("episode,reward\n")
            for i, reward in enumerate(self.rewards_per_episode):
                file.write(f"{i},{reward}\n")
            
        np.save(q_table_file_name, self.q_table)
    

    def transform_state(self, state: np.ndarray) -> np.ndarray:
        state_adj = (state - self.env.observation_space.low)*np.array([10, 100])
        return np.round(state_adj, 0).astype(int)


    def fit(self, episodes: int = 1000, max_steps: int = 1000, verbose: bool = False) -> None:

        self.reset()
        
        for i in range(episodes):
            state = self.env.reset()[0]
            state = self.transform_state(state)

            T = 0
            acc_reward = 0

            # Arbritary max steps
            while T < max_steps:
                action = self.select_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state = self.transform_state(next_state)
                acc_reward += reward
                
                # Bellman's equation
                self.q_table[state[0]][state[1]][action] = self.q_table[state[0]][state[1]][action] + self.alpha * (reward + self.gamma * max(self.q_table[next_state[0]][next_state[1]]) - self.q_table[state[0]][state[1]][action])

                state = next_state

                if done or truncated: break

                T += 1

            self.rewards_per_episode.append(acc_reward)
            
            if verbose and i % 100 == 0:
                print(f"Episódio {i}, Recompensa acumulada média dos últimos 100 episódios: {np.mean(self.rewards_per_episode[-100:])}")
        
            # Update epsilon apenas se não bateu no mínimo
            if self.epsilon > self.eps_min:
                self.epsilon *= self.eps_dec
            
        return self.q_table, self.rewards_per_episode


    def load_q_table(self, file_name: str) -> None:
        self.q_table = np.load(file_name)
    

    def transform(self, testing_env: gym.Env, max_steps: int = 1000, verbose: bool = False) -> None:
        state = testing_env.reset()[0]
        state = self.transform_state(state)
        done = False
        total_reward = 0
        steps = 0

        # Loopa pegando apenas o max sem atualizar a Q-table
        while not done and steps < max_steps:
            action = np.argmax(self.q_table[state[0]][state[1]])
            next_state, reward, done, _, _ = testing_env.step(action)
            next_state = self.transform_state(next_state)
            total_reward += reward
            steps += 1
            state = next_state
        
        if verbose:
            print(f"Total de passos: {steps}, Recompensa acumulada: {total_reward}")

        return steps, total_reward
    

    def close(self) -> None:
        self.env.close()


    def reset(self) -> None:
        self.q_table = np.zeros(self.q_table.shape)
        self.rewards_per_episode = []