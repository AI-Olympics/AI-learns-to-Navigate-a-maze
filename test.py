import pickle
import pygame
from Agent import Agent
from Environment import Maze_Env
import numpy as np
import time
import random

#colors
white = (255,255,255)
black = (0, 0, 0)

display_width, display_height = 600, 600

#initialising the maze
maze = np.load('maze2/maze_2.npy')
rows, cols = maze.shape

MAX_MOVES = 500
env = Maze_Env(maze, display_width, display_height, MAX_MOVES)

#initialising agent
directory = 'maze2'
agent = Agent(env, alpha = 0, dir = None)

#load the final policy
file = 'maze2/policy_final.pickle'
agent.set_policy(file)

#or load an intermediate Q-table
#with open('maze2/Q_table_1000.pickle', 'rb') as f:
#   Q = pickle.load(f)

#testing the agent
for i in range(10):  #running for 10 times
    state = env.reset()
    total_reward = 0
    while True:
        action = agent.act(state, test = True)  
        #rendering the environment
        env.render(action)
        time.sleep(0.3)       
        state, reward, done = env.step(action)
        total_reward += reward
        if done:
            env.render(action)
            time.sleep(2)
            print(total_reward)
            break 
