from collections import defaultdict
import sys
import numpy as np 
import pickle
import random

class Agent:

    def __init__(self, env, alpha, dir, gamma=1.0, eps_start=1.0, eps_decay=0.9999, eps_min=0.05):
        self.env = env
        self.eps_start = eps_start
        self.gamma = gamma
        self.alpha = alpha
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.nA = env.ACTION_SPACE
        self.dir = dir

    def interact(self, final_episode, initial_episode = 1, Q = None):
        '''
            interact with the environment and learn
        '''
        if Q is None:
            Q = defaultdict( lambda: np.zeros(self.nA))  #initialise Q-table
        epsilon = self.eps_start
        
        # loop over episodes
        for i_episode in range(initial_episode, final_episode+1):
            # monitor progress
            if i_episode % 100 == 0:
                print("\rEpisode {}/{}".format(i_episode, final_episode), end="")
                sys.stdout.flush() 
            #calculating epsilon
            epsilon = max(epsilon*self.eps_decay, self.eps_min)
            
            #observing state s0 and taking action
            state = self.env.reset()  
            action = self.act(state, Q, epsilon)
            while True:
                #rendering the environment
                self.env.render(action, i_episode, epsilon, self.alpha, self.gamma)
                next_state, reward, done = self.env.step(action)
                Q[state][action] += self.alpha*(reward + self.gamma*np.max(Q[next_state]) - Q[state][action])
                if done:
                    break
                #update state and action
                state = next_state
                action = self.act(state, Q, epsilon)
            
            if i_episode % 100 == 0:
                self.save_policy(Q, i_episode, name = 'Q_table')

        # Form a policy
        self.policy = self.get_Policy(Q)

    def continue_learning(self, Q, initial_episode, final_episode, eps_decay, eps_min=0.05):
        epsilon = 1
        for i in range(initial_episode):
            epsilon = max(epsilon*eps_decay, eps_min)
        
        self.eps_start = epsilon
        self.eps_decay = eps_decay
        self.interact(final_episode, initial_episode, Q)

    def act(self, state, Q = None, epsilon = 0, test = False):
        '''
            take action,with probability of random action being epsilon , in train_phase
        '''
        if not test:
            if random.random() > epsilon:
                action = np.argmax(Q[state])
            else:
                action = random.randint(0, self.nA-1)
        else:
            #if in test phase, choose best action from the policy learned
            action = self.policy[state]  
        return action
    
    def get_Policy(self, Q):
        '''
            returns optimal policy using the Q-table
        '''
        policy = defaultdict(lambda: 0)
        for state, action in Q.items():
            policy[state] = np.argmax(action)
        return policy    

    def set_policy(self, directory):
        '''
            To be used while importing experience from colab
        '''
        with open(directory, 'rb') as f:
            policy_new = pickle.load(f)
        self.policy = defaultdict(lambda:0, policy_new)  #saved as defaultdict
        print('policy Loaded')        

    
    def save_policy(self,policy, episode, name):
        try:
            policy = dict(policy)
            with open(f'{self.dir}/{name}_{episode}.pickle','wb') as f:
                pickle.dump(policy, f)
        except :
            print('not saved')