import pygame
import sys
import math
from pygame.locals import*  # type: ignore
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import time

class Arm():
    def __init__(self):
        self.window_width = 800
        self.window_height = 600
        self.center_x = self.window_width // 2
        self.center_y = self.window_height // 2
        self.x = 0
        self.y = 0
        self.angle = 89.552000
        
       
        
        # Initialize Pygame
        pygame.init()


        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Arm")
        
     


    
    def simulate(self,action):
        reward = 0
        next_state = 0
        done = False
        global counter
        trigger = False
        trigger1 = False
        
        target = [(386.74233688942786, 223.86240646589232),(397.1889167450849, 166.11547199180603),(384.3007218182302, 273.75623854543323)]
        
        if action == '0' and self.x >=  396.1991091206932:#400.99651654160004:    #340.6506737489858:   down
            self.angle = 90.51199999999999
            
        if action == '1' and  self.x >=  396.1991091206932: #340.6506737489858 : 
            counter=1
            pygame.draw.circle(self.window, (0, 255, 0), (target[counter]),9.5 , 100)
            self.angle = 89.712
            trigger = True

            
        if action == '2' and  self.x >=  396.1991091206932: #340.6506737489858 : 
            
            pygame.draw.circle(self.window, (0, 255, 0), (target[counter]),9.5 , 100)    
            self.angle = 90.91199999999998
            trigger1 = True
            
              
          
        self.x = self.center_x -  60* math.cos(self.angle)
        self.y = self.center_y - 60 * math.sin(self.angle)
        end_x = self.x +  76* math.cos(self.angle)
        end_y = self.y - 76 * math.sin(self.angle)
    
        next_state = (end_x,end_y)
        next_state = torch.tensor( next_state, dtype=torch.float)
        next_state = next_state.view(-1)
        
      
      
                
        self.window.fill((0, 0, 0))
            
        pygame.draw.line(self.window, (255, 255, 255), (self.center_x, self.center_y), (self.x, self.y), 5)
        pygame.draw.line(self.window, (255, 255, 255), (self.x,self.y), (end_x, end_y), 5)
    
            # Draw the circle / joints
        pygame.draw.circle(self.window, (0, 0, 255), (self.center_x, self.center_y),8 , 100)
        pygame.draw.circle(self.window, (0, 0, 255), (self.x, self.y),8 , 100)

      
        pygame.draw.circle(self.window, (0, 255, 0), (target[counter]),9.5 , 100)

        pygame.draw.circle(self.window, (255, 0, 0), (end_x, end_y),8 , 100)
        
      
      
        # target check 
        if (end_x,end_y) == target[counter]:
           
            if counter == 0:
                reward = 100
                counter+=1
                pygame.draw.circle(self.window, (0, 255, 0), (target[counter]),9.5 , 100)

            if counter == 1 and trigger == True:
                reward = 200
                counter+=1
                pygame.draw.circle(self.window, (0, 255, 0), (target[counter]),9.5 , 100)
                
               
         
            if counter == 2 and trigger1 == True:
                reward = 300
                counter+=1
            
          
            if counter == 3:    
                counter = 0
                done = True
             
        
       
        pygame.display.flip()  
        pygame.time.Clock().tick(0.95)
        
        return reward , next_state, done

          
    def reset(self):
        self.angle = 89.552000
       

simulation = Arm()

    
# Neural Network
class PolicyNet(nn.Module):
    def __init__(self,input_size,output_size):
        super(PolicyNet,self).__init__()
        self.l1 = nn.Linear(input_size,256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, output_size)

    def forward(self,x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


#hyperparameters
input_size = 2 
output_size = 3

policy = PolicyNet(input_size,output_size)
policy.load_state_dict(torch.load('Trained_models/model1.pth'))
policy.eval()  # Set the model to evaluation mode

counter = 0
# Training loop

simulation.reset()
state = (399.7342622555733,164.01875881450317)
state = torch.tensor( state , dtype=torch.float)
state = state.view(-1)
done = False
while True:   
    for event in pygame.event.get():
        if event.type == QUIT or event.type == KEYUP and event.key == K_ESCAPE:
            pygame.quit()
            sys.exit()
      
      
    with torch.no_grad():
        q_values = policy(state)
    action = torch.argmax(q_values).item()          
    reward,next_state, done = simulation.simulate(str(action))
    state = next_state        
    if done:     
        simulation.reset()
        state = (399.7342622555733,164.01875881450317)
        state = torch.tensor( state , dtype=torch.float)
        state = state.view(-1)
        
 