import numpy as np
import random
import gc

class DeepQLearning:

    #
    # Implementacao do algoritmo proposto em 
    # Playing Atari with Deep Reinforcement Learning, Mnih et al., 2013
    # https://arxiv.org/abs/1312.5602
    #

    def __init__(self, env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, batch_size, memory, model, max_steps):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
        self.batch_size = batch_size
        self.memory = memory
        self.model = model
        self.max_steps = max_steps

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.env.action_space.n)
        action = self.model.predict(state, verbose=0)
        return np.argmax(action[0])

    # cria uma memoria longa de experiencias
    def experience(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal)) 

    def experience_replay(self):
        # soh acontece o treinamento depois da memoria ser maior que o batch_size informado
        if len(self.memory) <= self.batch_size:
            return
    
        batch = random.sample(self.memory, self.batch_size) #escolha aleatoria dos exemplos
        states = np.array([i[0] for i in batch], dtype=np.float32)
        actions = np.array([i[1] for i in batch])
        rewards = np.array([i[2] for i in batch], dtype=np.float32)
        next_states = np.array([i[3] for i in batch], dtype=np.float32)
        terminals = np.array([i[4] for i in batch], dtype=np.float32)

        # np.squeeze(): Remove single-dimensional entries from the shape of an array.
        # Para se adequar ao input
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        # usando o modelo para selecionar as melhores acoes
        next_max = np.amax(self.model.predict_on_batch(next_states), axis=1)
        
        targets = rewards + self.gamma * (next_max) * (1 - terminals)
        targets_full = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(self.batch_size)])

        # usando os q-valores para atualizar os pesos da rede
        targets_full[[indexes], [actions]] = targets
        self.model.fit(states, targets_full, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_dec

    def train(self):
        rewards = []
        for i in range(self.episodes+1):
            (state,_) = self.env.reset()
            state = np.reshape(state, (1, self.env.observation_space.shape[0])).astype(np.float32)
            score = 0
            steps = 0
            #
            # no caso do Cart Pole eh possivel usar a condicao de done do episodio
            #
            done = False
            while not done:
                steps += 1
                action = self.select_action(state)
                next_state, reward, terminal, truncated, _ = self.env.step(action)
                if terminal or (steps>self.max_steps):
                    done = True          
                score += reward
                next_state = np.reshape(next_state, (1, self.env.observation_space.shape[0])).astype(np.float32)
                self.experience(state, action, reward, next_state, terminal)
                state = next_state
                self.experience_replay()
                if done:
                    print(f'Episódio: {i+1}/{self.episodes}. Score: {score}')
                    break
            rewards.append(score)
            
            if i % 10 == 0:
                gc.collect()
            #keras.backend.clear_session()

        return rewards
