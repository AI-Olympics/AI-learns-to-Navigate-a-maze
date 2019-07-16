#Environment
import cv2
import numpy as np 
import random
import time
from collections import deque 
from PIL import Image

#colours - b, g, r
WHITE = (255,255,255)
RED = (0,0,255)
BLACK = (0,0,0)
BLUE = (255,0,0)
GREEN = (0,255,0)
LIGHT_GOLDEN = (50,235,255)

class Maze_Env():

    def __init__(self, maze, display_width, display_height, max_moves, start_position = (1,1)):
        
        self.MAZE = maze
        self.HEIGHT, self.WIDTH = maze.shape
        self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT = display_width, display_height

        self.STATE_SPACE = self.WIDTH * self.HEIGHT  #since agent can be in any one of the coordinates
        self.ACTION_SPACE = 4  #four directions
        self.MAX_MOVES = max_moves
        
        #positions
        self.START_POSITION = start_position
        goal_x, goal_y = self.WIDTH-2, self.HEIGHT-2
        self.SUB_GOALS = [(goal_x-1, goal_y), (goal_x, goal_y-1), (goal_x-1,goal_y-1), (goal_x-2,goal_y-1)]
        self.GOAL_X, self.GOAL_Y = goal_x, goal_y

#--------------------------------------------Reset the environment--------------------------------------------------#
    
    def reset(self):
        goal_x, goal_y = self.GOAL_X, self.GOAL_Y
        self.MOVES = self.MAX_MOVES  #resetting the moves to 100
        self.SUB_GOALS = [(goal_x-1, goal_y), (goal_x, goal_y-1), (goal_x-1,goal_y-1), (goal_x,goal_y-2)]
        self.STATE = self.START_POSITION  #setting the agent to start position
        return self.STATE

#-----------------------------------------------Render the enviroment-----------------------------------------------#    

    def render(self, action, episode=-1, epsilon=-1, alpha=-1, gamma=-1):
        '''
            rendering the environment
        '''
        extra_width = 5
        display_matrix = np.zeros((self.HEIGHT, self.WIDTH + extra_width, 3), dtype = np.uint8) #basically an image with 3 channels RGB
        #copy the maze matrix 3 times, and save it in display_matrix
        for i in range(3):
            display_matrix[:,:self.WIDTH,i] = self.MAZE*255
      
        agent_x, agent_y = self.STATE  
        display_matrix[agent_y, agent_x] = RED  #paint the agent's position
        display_matrix[self.GOAL_Y, self.GOAL_X] = GREEN   #paint the goal's position
        for x,y in self.SUB_GOALS:                          #pain sub_goals
            display_matrix[y, x] = LIGHT_GOLDEN 
        display_matrix[:,self.WIDTH:, :] = WHITE  #adding white on right side
        display_matrix = self.display_action(display_matrix, action)

        img = Image.fromarray(display_matrix, 'RGB')
        img = np.array(img.resize((self.DISPLAY_WIDTH + extra_width*30,  self.DISPLAY_HEIGHT)))
        #self.display_action(img, action)
        
        if episode > 0:
            cv2.putText(img, f'Episode : {episode}', (30, self.DISPLAY_HEIGHT - 30), 
                                cv2.FONT_HERSHEY_COMPLEX , thickness = 2, fontScale = 0.8, color = BLUE, lineType = 2)
            cv2.putText(img, 'Epsilon : {:.3f}'.format(epsilon), (self.DISPLAY_WIDTH + 10, self.DISPLAY_HEIGHT - 80), 
                                cv2.FONT_HERSHEY_COMPLEX ,  fontScale = 0.5, color = BLACK, lineType = 2)
            cv2.putText(img, f'Alpha : {alpha}', (self.DISPLAY_WIDTH + 10, self.DISPLAY_HEIGHT - 60), 
                                cv2.FONT_HERSHEY_COMPLEX ,  fontScale = 0.5, color = BLACK, lineType = 2)
            cv2.putText(img, f'Gamma : {gamma}', (self.DISPLAY_WIDTH + 10, self.DISPLAY_HEIGHT - 40), 
                                cv2.FONT_HERSHEY_COMPLEX ,  fontScale = 0.5, color = BLACK, lineType = 2)

        cv2.imshow("Maze Game", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  #when Q is pressed
            print('Stoping Execution')
            cv2.destroyAllWindows()
            quit()

#--------------------------------------Handy function, to show action taken by agent--------------------------------#
    
    def display_action(self, matrix, action):
        
        mid_height = int(self.WIDTH/2)
        matrix[mid_height,-4,:] = BLUE   #left button
        matrix[mid_height,-2,:] = BLUE   #right button
        matrix[mid_height-1,-3,:] = BLUE  #up button
        matrix[mid_height+1,-3,:] = BLUE  #down button
       
        if action == 0:
            matrix[mid_height,-4,:] = RED   #left 
        elif action == 1:
            matrix[mid_height,-2,:] = RED   #right
        elif action == 2:
            matrix[mid_height-1,-3,:] = RED  #up 
        elif action ==3:
            matrix[mid_height+1,-3,:] = RED  #down 

        return matrix
#--------------------------------------Agent takes step and the environment changes------------------------------------#

    def step(self,action):

        reward = -1  #negative reward for each step
        done = False

        #decreasing the no. of moves
        self.MOVES -= 1
        #done if moves =0
        if self.MOVES==0:
            done = True

        x_change, y_change = 0, 0
        #decide action
        if action == 0:
            x_change = -1  #moving left
        elif action == 1:
            x_change = 1   #moving right
        elif action == 2:
            y_change = -1 #moving upwards
        elif action ==3:
            y_change = 1  #moving downwards

        agent_x, agent_y = self.STATE

        #get the new position
        new_x = agent_x + x_change
        new_y = agent_y + y_change
        
        # If agent hits the maze walls, do not update it's position
        if self.MAZE[new_y, new_x] == 0:
            pass 
        else:
            agent_x, agent_y = new_x, new_y

        #reached subgoals
        reached = []
        for x, y in self.SUB_GOALS:
            if agent_x == x and agent_y == y:
                reward = 20
                reached.append((x, y))
        for pos in reached:
            self.SUB_GOALS.remove(pos)

        #reached goal
        if agent_x == self.GOAL_X and agent_y == self.GOAL_Y:
            reward = 50  #positive reward
            done = True

        self.STATE = (agent_x, agent_y)

        return self.STATE, reward, done
