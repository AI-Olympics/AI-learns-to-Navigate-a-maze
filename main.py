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
rows, cols = 20, 20
maze = np.ones((rows, cols))

#---------------------------Adding walls to the maze---------------------------------#
walls = []
pygame.init()
screen = pygame.display.set_mode((display_width, display_height))
screen.fill(white)

block_width, block_height = display_width/cols, display_height/rows
quit = False
while not quit:
    pygame.display.update()  #updating the display
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN: 
            x, y = pygame.mouse.get_pos()
            X, Y = int(x/block_width), int(y/block_height)            
            if event.button == 1:   # if left mouse button clicked, add it to walls
                if (X,Y) not in walls:
                    walls.append((X,Y))
                    pygame.draw.rect(screen, black, [X*block_width, Y*block_height, block_width, block_height])  #draw a black rectangle
            elif event.button == 3:  # if right mouse button clicked, remove it from walls
                try: walls.remove((X,Y))
                except: pass
                pygame.draw.rect(screen, white, [X*block_width, Y*block_height, block_width, block_height])  #draw a black rectangle
        
        elif event.type == pygame.QUIT:
            quit = True
            pygame.quit()

for x,y in walls:
    maze[y, x] = 0  # y, x is right because rows --> vertical, column--> horizontal
#adding boundaries
maze = np.pad(maze, [(1,1),(1,1)], mode='constant')

#---------------------------------------Train the agent-------------------------------------#

MAX_MOVES = 500
env = Maze_Env(maze, display_width, display_height, MAX_MOVES)

#initialising agent
directory = 'maze2'
EPS_DECAY = 0.999
agent = Agent(env = env, alpha = 0.1, dir = directory, eps_decay=EPS_DECAY)

NUM_EPISODES = 1000
agent.interact(NUM_EPISODES)

#---------------------Use this if continuing the training of Agent-----------------------------#
'''
#initialising the maze
maze = np.load('maze2/maze_2.npy')
rows, cols = maze.shape

MAX_MOVES = 500
env = Maze_Env(maze, display_width, display_height, MAX_MOVES)

#initialising agent
directory = 'maze2'
agent = Agent(env, alpha = 0.1, dir = directory)

#set a policy
#file = 'maze2/policy_final.pickle'
#agent.set_policy(file)

#or load a Q-table
with open('maze2/Q_table_1000.pickle', 'rb') as f:
   Q = pickle.load(f)

#training the agent via interaction with env
EPS_DECAY = 0.999
agent.continue_learning(Q, initial_episode = 1000, final_episode = 1500, eps_decay = EPS_DECAY)

'''
#---------------------------------Finally save the policy for testing the agent---------------------------------------------#

policy = agent.policy
agent.save_policy(policy, episode = 'final', name='policy')
np.save(f'{directory}/maze_2', maze)  #save maze
