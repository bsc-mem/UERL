# SPDX-License-Identifier: BSD-3-Clause
# 
# This file is part of UERL project (https://github.com/bsc-mem/UERL).
# 
# Copyright (c) 2024 Isaac Boixaderas Coderch
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the conditions stated in the BSD-3-Clause License are met.
# 
# For more information, see the LICENSE file at the root of this project or visit
# https://opensource.org/licenses/BSD-3-Clause.


import pandas as pd
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler


class SumTree(object):
    """
    This SumTree code is modified version of Morvan Zhou: 
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py

    The SumTree class implements a binary tree data structure used in 
    prioritized experience replay for reinforcement learning. 
    It efficiently stores experiences and their priorities, allowing 
    for quick access and updating of priorities. This class enables the 
    implementation of prioritized replay mechanisms, enhancing the 
    learning efficiency of reinforcement learning algorithms.
    """
    data_pointer = 0
    
    def __init__(self, capacity):
        """
        Initialize the tree with all nodes = 0,
        and initialize the data with all values = 0

        Args:
            capacity (int): Integer indicating the capacity of the SumTree.
        """
        # Number of leaf nodes (final nodes) that contains experiences
        self.capacity = capacity
        
        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children),
        # so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)
        
        """ tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0
        [Size: capacity] priorities score
        """
        
        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)
    
    
    def add(self, priority, data):
        """
        Add our priority score in the sumtree leaf and add the experience.

        Args:
            priority (float): Priority score to be added to the 
                              SumTree leaf.
            data (tuple): Experience associated with the priority score.
        """
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1
        
        """ tree:
            0
           / \
          0   0
         / \ / \
tree_index  0 0  0
        We fill the leaves from left to right.
        """
        
        # Update data frame
        self.data[self.data_pointer] = data
        
        # Update the leaf
        self.update(tree_index, priority)
        
        # Add 1 to data_pointer
        self.data_pointer += 1
        
        # If above capacity, go back to first index (overwrite)
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
            
    
    def update(self, tree_index, priority):
        """
        Update the leaf priority score and propagate 
        the change through tree.

        Args:
            tree_index (int): Index of the SumTree leaf to be updated.
            priority (float): New priority score.
        """
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        # Propagate the change through tree
        while tree_index != 0:
            """
            This method is faster than the recursive loop in 
            the reference code. Access the line above.
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES
            
                0
               / \
              1   2
             / \ / \
            3  4 5  [6] 
            
            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
    
    
    def get_leaf(self, v):
        """
        Get the leaf_index, priority value of that leaf and experience 
        associated with that index.
        
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]

        Args:
            v (float): Value used for searching the SumTree to find 
                       the leaf index.
        """
        parent_index = 0
        
        while True:
            # This loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            
            else:
                # Downward search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
            
        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    @property
    def total_priority(self):
        # Return the root node
        return self.tree[0]


class PER(object):
    """
    This PER code is modified version of the original code:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py

    The PER (Prioritized Experience Replay) class implements prioritized 
    experience replay for reinforcement learning. It uses a SumTree data 
    structure to store experiences based on their priorities. This class 
    offers methods to store, sample, and update experiences efficiently, 
    improving the learning process by prioritizing important experiences.
    """
    
    PER_b_increment_per_sampling = 0.001
    # Clipped abs error
    absolute_error_upper = 1.

    def __init__(self, capacity, burn_in=10, PER_a=0.6, PER_b=0.4):
        """
        Making the tree.

        Remember that our tree is composed of a sum tree that contains 
        the priority scores at his leaf and also a data array.
        We don't use deque because it means that at each timestep our 
        experiences change index by one.
        We prefer to use a simple array and to overwrite when the 
        memory is full.

        Args:
            capacity (int): Capacity of the SumTree.
            burn_in (int, optional): Burn-in period before sampling 
                                     begins. Defaults to 10.
            PER_a (float, optional): Tradeoff parameter for prioritized 
                                     sampling. Defaults to 0.6.
            PER_b (float, optional): Importance-sampling parameter. 
                                     Defaults to 0.4.
        """
        self.tree = SumTree(capacity)
        self.burn_in = burn_in
        # Avoid some experiences to have 0 probability of being taken
        self.PER_e = 0.01
        # Make a tradeoff between taking only exp with high priority 
        # and sampling randomly
        self.PER_a = PER_a
        # Importance-sampling, from initial value increasing to 1
        self.PER_b = PER_b
    
    def burn_in_capacity(self):
        """
        Check burn in capacity
        """
        return len(self.tree.tree) / self.burn_in
        
    def store(self, experience):
        """
        Store a new experience in our tree
        Each new experience have a score of max_prority 
        (it will be then improved when we use this exp to train our DDQN)

        Args:
            experience (tuple): Experience tuple containing 
                                (state, action, reward, next_state).
        """
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        
        # If the max priority = 0 we can't put priority = 0 since this 
        # exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        
        # Set the max p for new p
        self.tree.add(max_priority, experience)

    def sample_batch(self, n):
        """
        - First, to sample a minibatch of k size, 
        the range [0, priority_total] is / into k ranges.
        - Then a value is uniformly sampled from each range
        - We search in the sumtree, the experience where priority score 
        correspond to sample values are retrieved from.
        - Then, we calculate IS weights for each minibatch element

        Args:
            n (int): Size of the minibatch to sample.

        Returns:
            tuple: A tuple containing:
                - b_idx (np.array): Indices of the sampled 
                                    experiences in the SumTree.
                - memory_b (list): Sampled experiences.
                - b_ISWeights (np.array): Importance-sampling weights
                                          for each sampled experience.
        """
        # Create a sample array that will contains the minibatch
        memory_b = []
        
        b_idx = np.empty((n,), dtype=np.int32)
        b_ISWeights = np.empty((n, 1), dtype=np.float32)
        
        # Calculate the priority segment
        # Divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n
    
        # Increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1, self.PER_b + self.PER_b_increment_per_sampling])
        
        for i in range(n):
            # A value is uniformly sample from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            
            # Experience that correspond to each value is retrieved
            index, priority, data = self.tree.get_leaf(value)
            
            b_idx[i]= index
            
            experience = [data]
            
            memory_b.append(experience)
        
        return b_idx, memory_b, b_ISWeights
    
    def batch_update(self, tree_idx, abs_errors):
        """
        Update the priorities on the tree.

        Args:
            tree_idx (np.array): Indices of the experiences in the SumTree.
            abs_errors (np.array): Absolute errors corresponding to 
                                   each experience.
        """
        # Convert to abs and avoid 0
        abs_errors += self.PER_e
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class duelingDQN(torch.nn.Module):
    """
    The duelingDQN class implements a dueling deep Q-network (DQN) for 
    reinforcement learning. It consists of a neural network architecture 
    that incorporates both value-based and advantage-based components to 
    estimate the Q-values of actions in a given state.
    """

    def __init__(self, env, device='cpu'):
        """
        Initializes the duelingDQN model.

        Args:
            env (gym.Env): Environment for which the model is being 
                           created.
            device (str, optional): Device on which the model will be 
                                    trained ('cpu' or 'cuda').
                                    Defaults to 'cpu'.
        """
        super(duelingDQN, self).__init__()
        self.device = device
        obs = env.reset().flatten()
        self.n_inputs = len(obs)
        self.n_outputs = env.action_space.n
        # Available actions
        self.actions = list(range(self.n_outputs))
        
        # Build neural networks. Common net
        self.fc1 = nn.Linear(self.n_inputs, 256, bias=True)
        self.fc2 = nn.Linear(256, 256, bias=True)
        self.fc3 = nn.Linear(256, 128, bias=True)

        # Subnet for value function
        self.fc_value = nn.Linear(128, 64, bias=True)
        self.value = nn.Linear(64, 1, bias=True)
        
        # Sub-xarxa for advantage: A(s,a)
        self.fc_adv = nn.Linear(128, 64, bias=True)
        self.adv = nn.Linear(64, self.n_outputs, bias=True)
    
    def forward(self, state):
        """
        Defines the forward pass of the duelingDQN model.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Q-values estimated by the model for each 
                          action in the input state.
        """
        # Common connection between layers of subnets
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
            
        # Connection between layers of subnets value
        value = F.relu(self.fc_value(x))
        value = self.value(value)
        
        # Connection between layers of subnets advantage
        adv = F.relu(self.fc_adv(x))
        adv = self.adv(adv)
        
        # Add both subnets: Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
        Q = value + adv - adv.mean()
        
        if self.device == 'cuda':
            self.model.cuda()
        return Q
    
    def get_action(self, state, epsilon=0.05):
        """
        Selects an action using an epsilon-greedy policy.

        Args:
            state (numpy.ndarray): Input state.
            epsilon (float, optional): Probability of selecting a 
                                       random action. Defaults to 0.05.

        Returns:
            int: Selected action.
        """
        # e-greedy method
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions)  
        else:
            qvals = self.get_qvals(state)
            action= torch.max(qvals, dim=-1)[1].item()
        return action
    
    def get_qvals(self, state):
        if type(state) is tuple:
            state = np.array([np.ravel(s) for s in state])
        state_t = torch.FloatTensor(np.array(state)).to(device=self.device)
        return self.forward(state_t)


class duelingDQNAgent:
    """
    The duelingDQNAgent class represents an agent that interacts with an 
    environment using a dueling deep Q-network (DQN) to learn an optimal 
    policy. It includes methods for training the DQN, taking actions 
    based on an epsilon-greedy policy, and updating the DQN parameters 
    using experience replay.
    """
       
    def __init__(self, env, dueling_dnnetwork, buffer, learning_rate,
                 epsilon=0.1, epsilon_min=0.05, epsilon_decay=0.99,
                 batch_size=32):
        """
        Initializes the duelingDQNAgent.

        Args:
            env (gym.Env): Environment for which the agent is being 
                           created.
            dueling_dnnetwork (duelingDNNetwork): Dueling DNNetwork.
            buffer (PER): Replay buffer for storing experiences.
            learning_rate (float): Learning rate for the optimizer.
            epsilon (float, optional): Initial value of epsilon for 
                                       epsilon-greedy exploration. 
                                       Defaults to 0.1.
            epsilon_min (float, optional): Minimum value of epsilon.
                                           Defaults to 0.05.
            epsilon_decay (float, optional): Decay rate for epsilon.
                                             Defaults to 0.99.
            batch_size (int, optional): Batch size for training.
                                        Defaults to 32.
        """
        self.env = env
        self.dueling_dnnetwork = dueling_dnnetwork
        self.target_network = deepcopy(dueling_dnnetwork)
        self.buffer = buffer
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.epsilon_min = epsilon_min
        # Adam optimizer
        self.optimizer = torch.optim.Adam(dueling_dnnetwork.parameters(),
                                          lr=learning_rate)
        # Scaler
        self.scaler = MinMaxScaler(feature_range=(1, 2))
        # Generate random job sizes
        # TODO Take care! Custom maximum job size of 50,000 node-hours
        job_sizes = np.append(
            np.random.uniform(0, 50_000, size=len(self.env.envs[0].full_df)-1),
            0
        ).reshape(-1, 1)
        self.scaler.fit(np.append(self.env.envs[0].full_df, job_sizes, axis=1))
        self.initialize()

    def initialize(self):
        self.sync_eps = []
        self.total_reward = 0
        self.step_count = 0
        self.training_epsilons = []
        self.training_rewards = []
        self.update_loss = []
        self.training_loss = []
        self.episode_actions = []

    def take_step(self, eps, mode='train'):
        """
        Takes a step in the environment.

        Args:
            eps (float): Epsilon value for epsilon-greedy exploration.
            mode (str, optional): Mode of operation ('train' or 'explore').
                                  Defaults to 'train'.

        Returns:
            bool: True if the episode is the last one, False otherwise.
        """
        if mode == 'explore':             
            # Burn-in random action
            action = np.random.choice([0, 1], p=[0.95, 0.05])
        else:
            # Chose best action based on best Q
            action = self.dueling_dnnetwork.get_action(self.state0, epsilon=eps)
            self.step_count += 1

        self.episode_actions.append(action)
            
        # Take step with action and get new state and reward
        new_state, reward, done, _ = self.env.step([action])
        new_state = new_state.flatten()
        new_state = self.scaler.transform(new_state.reshape(1, -1))[0]
        self.total_reward += reward
        # Store experience to the buffer
        experience = (self.state0, action, reward, done, new_state)
        self.buffer.store(experience)
        # logging.debug(f'Step experience: {experience}')
        self.state0 = new_state.copy()
        
        if done:
            self.state0 = self.env.reset().flatten()
            self.state0 = self.scaler.transform(self.state0.reshape(1, -1))[0]
        return done
    
    def train(self, gamma, max_episodes,
              dnn_update_frequency, dnn_sync_frequency):
        """
        Trains the duelingDQNAgent.

        Args:
            gamma (float): Discount factor. Defaults to 0.99.
            max_episodes (int): Maximum number of episodes for training.
                                Defaults to 50,000.
            dnn_update_frequency (int): Frequency of updating the DNN 
                                        parameters. Defaults to 4.
            dnn_sync_frequency (int): Frequency of synchronizing the 
                                      target network with the main network.
                                      Defaults to 2000.
        """
        self.gamma = gamma

        # Fill replay buffer with random experiences
        logging.debug("Filling replay buffer...")
        while self.buffer.burn_in_capacity() < 1:
            self.take_step(self.epsilon, mode='explore')

        episode = 0
        training = True
        while training:
            self.state0 = self.env.reset().flatten()
            self.state0 = self.scaler.transform(self.state0.reshape(1, -1))[0]
            self.total_reward = 0
            gamedone = False
            while gamedone == False:
                gamedone = self.take_step(self.epsilon, mode='train')
                logging.debug(f'Episode: {episode}')
               
                # Update main network with stablished frequency
                if self.step_count % dnn_update_frequency == 0:
                    self.update()
                
                # Synch main net with objective net with stablished freq
                if self.step_count % dnn_sync_frequency == 0:
                    ddn_state = self.dueling_dnnetwork.state_dict()
                    self.target_network.load_state_dict(ddn_state)
                    self.sync_eps.append(episode)
                
                if gamedone:
                    episode += 1
                    # Store epsilon, training rewards and losses
                    self.training_epsilons.append(self.epsilon)
                    self.training_rewards.append(self.total_reward)
                    self.training_loss.append(self.update_loss)
                    
                    self.update_loss = []
                    
                    actions_dict = pd.Series(self.episode_actions) \
                                     .value_counts().to_dict()
                    if 0 not in actions_dict.keys():
                        actions_dict[0] = 0
                    if 1 not in actions_dict.keys():
                        actions_dict[1] = 0
                    logging.debug((
                        f'Reward: {self.total_reward}; '
                        f'Action0: {actions_dict[0]}; '
                        f'Action1: {actions_dict[1]}; '
                        f'Epsilon: {round(self.epsilon, 5)}'
                    ))
                    self.episode_actions = []
                    
                    # Finnish training if max_episodes is reached
                    if episode >= max_episodes:
                        training = False
                        logging.debug('Episode limit reached.')
                        break
                        
                    # Update epsilon based on decay speed and epsilon_min
                    if self.epsilon > self.epsilon_min:
                        new_epsilon = self.epsilon * self.epsilon_decay
                        self.epsilon = max(new_epsilon, self.epsilon_min)

    def calculate_loss_n_abs_errors(self, batch):
        """
        Calculates the loss and absolute errors for a batch of experiences.

        Args:
            batch (list): Batch of experiences.

        Returns:
            tuple: Contains the loss and absolute errors.
        """
        # There's a bug where the last batch instance is a [0], remove it
        if len(batch[-1]) == 1:
            batch = batch[:-1]
        # Separate experience variables and cast them to tensors
        states = np.array([each[0][0] for each in batch])
        actions = np.array([each[0][1] for each in batch])
        rewards = np.array([each[0][2] for each in batch])
        dones = np.array([each[0][3] for each in batch])
        next_states = np.array([each[0][4] for each in batch])
        rewards_vals = torch.FloatTensor(rewards) \
                            .to(device=self.dueling_dnnetwork.device)\
                            .reshape(-1,1)
        actions_vals = torch.LongTensor(np.array(actions)) \
                            .reshape(-1,1) \
                            .to(device=self.dueling_dnnetwork.device)
        dones_t = torch.BoolTensor(dones) \
                       .to(device=self.dueling_dnnetwork.device)
        
        # Get Q values of the main network
        qvals = torch.gather(
            self.dueling_dnnetwork.get_qvals(states), 1, actions_vals)
        
        # Get action with the maximum Q of the main network
        next_actions = torch.max(
            self.dueling_dnnetwork.get_qvals(next_states), dim=-1)[1]
        next_actions_vals = torch.LongTensor(next_actions) \
                                 .reshape(-1,1) \
                                 .to(device=self.dueling_dnnetwork.device)
        
        # Get values of Q from the objective network
        target_qvals = self.target_network.get_qvals(next_states)
        qvals_next = torch.gather(target_qvals, 1, next_actions_vals).detach()
        qvals_next[dones_t] = 0
        
        # Calculate Bellman equation
        expected_qvals = self.gamma * qvals_next + rewards_vals
        # Calculate absolute errors
        abs_errors = torch.abs(qvals - expected_qvals).data.numpy()
        # Calculate loss
        loss = F.mse_loss(qvals, expected_qvals.reshape(-1,1))

        return loss, abs_errors
    
    def update(self):
        """
        Updates the parameters of the duelingDQNAgent.
        """
        # Remove any previous gradient
        self.optimizer.zero_grad()
        # Obtain random mini-batch from memory
        tree_idx, batch, _ = self.buffer.sample_batch(self.batch_size)
        # Calculate loss and absolute errors
        loss, absolute_errors = self.calculate_loss_n_abs_errors(batch)
        # Difference for obtaining gradients
        loss.backward()
        # Update priority
        self.buffer.batch_update(tree_idx, absolute_errors)
        # Apply gradients
        self.optimizer.step()
        # Store loss values
        if self.dueling_dnnetwork.device == 'cuda':
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())

