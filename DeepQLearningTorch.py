import numpy as np
import random
import gc
import torch

class DeepQLearning:

    #
    # Implementacao do algoritmo proposto em 
    # Playing Atari with Deep Reinforcement Learning, Mnih et al., 2013
    # https://arxiv.org/abs/1312.5602
    #

    def __init__(
            self, 
            env,
            gamma, 
            epsilon, 
            epsilon_min, 
            epsilon_dec, 
            episodes, 
            batch_size, 
            memory, 
            model, 
            max_steps,
            device,
            loss_fn,
            optimizer,
            reward_avg_tol=450
        ):

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

        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.reward_avg_tol = reward_avg_tol

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.env.action_space.n)
        
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    # cria uma memoria longa de experiencias
    def experience(self, state, action, reward, next_state, terminal):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.int64, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        terminal = torch.tensor(terminal, dtype=torch.float32, device=self.device)

        self.memory.append((state, action, reward, next_state, terminal))


    def experience_replay(self):
        # soh acontece o treinamento depois da memoria ser maior que o batch_size informado
        if len(self.memory) <= self.batch_size:
            return
    
        batch = random.sample(self.memory, self.batch_size) #escolha aleatoria dos exemplos
        
        states, actions, rewards, next_states, terminals = zip(*batch)
        states = torch.cat(states)  # No need for np.vstack, tensors are already stacked
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.cat(next_states)
        terminals = torch.stack(terminals)
      
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            next_q_values *= (1 - terminals)
            target_q_values = rewards + self.gamma * next_q_values

        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_dec

    def train(self):
        rewards = []
        for i in range(self.episodes+1):
            (state,_) = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
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
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
                self.experience(state, action, reward, next_state, terminal)
                state = next_state
                self.experience_replay()
                if done:
                    print(f'Episódio: {i+1}/{self.episodes}. Score: {score}')
                    break
            rewards.append(score)
            
            if i % 10 == 0:
                gc.collect()
            
            # If the average of the last 15 rewards is greater than 450, the environment is considered solved
            if np.mean(rewards[-15:]) > self.reward_avg_tol:
                print(f'Solved in {i} episodes!')
                break

        return rewards
