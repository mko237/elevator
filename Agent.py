from collections import namedtuple
from time import sleep
from copy import deepcopy
import random,math
#basing logic on https://gist.github.com/n1try/2a6722407117e4d668921fce53845432#file-dqn_cartpole-py

import os
import numpy as np
#import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
class Agent():
    def __init__(self, building_height, elevator_nums, actions,weights_file=None,
        gamma=.90, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.9995,
        alpha=0.01, alpha_decay=0.01, batch_size=64, monitor=False, quiet=False):

        self.building_height = building_height
        self.elevator_nums = elevator_nums
        self.weights_file = weights_file
        self.actions = actions
        self.memory = ReplayMemory(200000)
        self.discount_factor = .97
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.batch_size = batch_size

        self.build_model()
#    @profile
    def get_action(self, state,step,epsilon_off=False):
        #random action for each elevator
        #return self.env`.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.model.predict(state))
        if epsilon_off:
            epsilon = .5
            self.epsilon=epsilon
        else:
            epsilon = self.get_epsilon(step)
        if np.random.random() <= epsilon:
            action = np.random.randint(0,self.actions, (self.elevator_nums))
        else:
            action_q_val = self.predict(state)
            action = np.argmax(action_q_val, axis=1)
            action = action.squeeze()
        return action

    def get_epsilon(self, t,):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log((t + 1) * self.epsilon_decay,500)))

    def update_memory(self, states, actions, rewards):
        batch_size = len(states) - 1

        # propagate rewards backwards to previous states
        full_reward = None
        full_reward_exp = 1
        for idx in reversed(range(batch_size)):
            if rewards[idx] != 0:
                full_reward = rewards[idx]
                full_reward_exp = 1
            else:
                if full_reward is not None:
                    rewards[idx] = full_reward*self.gamma**full_reward_exp
                    full_reward_exp += 1

        #if np.array(rewards).sum() != 0:
        if True:
            # store states into agent memory
            for idx in range(batch_size):
                s = states[idx]
                a = actions[idx]
                r = rewards[idx]
                ns = states[idx+1]
                self.memory.push(s,a,ns,r)

        #    print(np.array(rewards).sum())
#    @profile
    def predict(self, state):
        if isinstance(state, list):
            state = np.vstack(state).T
        return self.model.predict(state)

    def update_network(self, states, actions, rewards, counter):
        self.update_memory(states, actions, rewards, )

        x_batch, y_batch = [], []
        minibatch = self.memory.sample(
            min(len(self.memory), self.batch_size))
        elevator_action_index = range(self.elevator_nums)
        for state, action, next_state, reward in minibatch:
            y_target = self.predict(state)
            #action = self.get_action(state,9999999999999999999999999999)
            #y_target_old = deepcopy(y_target)
            y_target[0][elevator_action_index, action] = reward + self.gamma * np.max(self.predict(next_state)[0],axis=1)
            #print(y_target[0]-y_target_old[0])
            #print(action)
            #sleep(5)
            x_batch.append(state)
            y_batch.append(y_target[0])

        # Format batches for model fit
        x_batch = np.vstack(x_batch)
        y_batch = np.array(y_batch)

        self.model.fit(x_batch, y_batch, batch_size=len(x_batch), epochs=1, verbose=1,callbacks=self.callbacks_list)
        if self.epsilon > self.epsilon_min:
            #self.epsilon *= self.epsilon_decay
            pass


    def build_model(self):
        # Init model
        self.model = Sequential()
        self.model.add(Dense(128, input_dim=self.building_height+2*self.elevator_nums, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(48, activation='relu'))
        self.model.add(Dense(48, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.elevator_nums*self.actions, activation='linear'))
        self.model.add(Reshape((self.elevator_nums,self.actions)))
        checkpoint = ModelCheckpoint("best_weights_128_6.hdf5",verbose=1,save_best_only=False,mode='max',period=100)
        self.callbacks_list=[checkpoint]
        if self.weights_file:
            self.model.load_weights(self.weights_file)
        self.model.compile(loss='mse',metrics=['mse','mae'], optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))


    def build_loss(self):
        pass

    def save(self, num):
        pass

    def reload(self):
        pass


# From https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
