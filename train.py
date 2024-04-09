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
        
        target = [(386.74233688942786, 223.86240646589232),(397.1889167450849, 166.11547199180603),(384.3007218182302, 273.75623854543323)]

  
        if action == '0' and self.x >=  396.1991091206932:#400.99651654160004:    #340.6506737489858:   down
            self.angle = 90.51199999999999
           
        if action == '1' and  self.x >=  396.1991091206932: #340.6506737489858 :     
            self.angle = 89.712
            
        if action == '2' and  self.x >=  396.1991091206932: #340.6506737489858 :     
            self.angle = 90.91199999999998
            
              
          
        self.x = self.center_x -  60* math.cos(self.angle)
        self.y = self.center_y - 60 * math.sin(self.angle)
        end_x = self.x +  76* math.cos(self.angle)
        end_y = self.y - 76 * math.sin(self.angle)
    
        next_state = (end_x,end_y)
        next_state = torch.tensor( next_state, dtype=torch.float)
        next_state = next_state.view(-1)
        
      
        # print(end_x,end_y)
        # print(self.x,self.angle)
        
        # termination check
        # if self.x ==  459.609265017363 or self.x == 396.1991091206932: #self.x == 340.6506737489858:  
        #     reward = -200
        #     done = True
                
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
                reward = 200
               
           
        
            if counter == 1:
                reward = 100
                
               
         
            if counter == 2:
                reward = 100
                
            counter += 1     
            if counter == 3:    
                counter = 0
                done = True
             
        
       
        pygame.display.flip()  
        pygame.time.Clock().tick(120)
        
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


# Experience Replay
class Experience():
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


#hyperparameters
input_size = 2 
output_size = 3
gamma = 0.9
epsilon = 1
epsilon_min = 0.001
epsilon_decay = 0.999965
learning_rate = 0.001
replay_capacity = 100000
batch_size = 1


policy = PolicyNet(input_size,output_size)
target = PolicyNet(input_size,output_size)
target.load_state_dict(policy.state_dict())  # Copy policy weights to target network
target.eval()  # Set target network to evaluation mode

replay = Experience(replay_capacity)

# loss $ optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)










counter = 0
# Training loop
for episodes in range(16000):
    simulation.reset()
    state = (399.7342622555733,164.01875881450317)
    state = torch.tensor( state , dtype=torch.float)
    state = state.view(-1)
    done = False
    trigger = False
    for episode in range(10000):
        for event in pygame.event.get():
            if event.type == QUIT or event.type == KEYUP and event.key == K_ESCAPE:
                pygame.quit()
                sys.exit()
      
        if random.random() < epsilon:
            action = random.randint(0,2)
        else:
            with torch.no_grad():
                q_values = policy(state)
                action = torch.argmax(q_values).item()
        
        
        reward,next_state, done = simulation.simulate(str(action))
        
        # print('')        
        # print(f'states:{state}')
        # print(f'actions:{action}')
        # print(f'rewards:{reward}')
        # print(f'next_states:{next_state}')
        # print('')
        
        
        replay.push(state,torch.tensor([action]),torch.tensor([reward]), next_state) 
        
        batch = replay.sample(1)
        
        if len(replay.memory) >= batch_size:
            batch = replay.sample(1)
          
            
        states,actions,rewards,next_states = zip(*batch) # unpacking
           
           
        states = torch.stack(states).squeeze(1)
        actions = torch.cat(actions).unsqueeze(1)
        rewards = torch.cat(rewards)
        next_states = torch.stack(next_states).squeeze(1)
        
        q_values = policy(states).gather(1, actions)
        q_values =  q_values.squeeze(1)
       
           
           
        max_q_values = target(next_states).max(1)[0]
        target_q_values = (rewards + gamma * max_q_values)
       

        
      
        loss = criterion(q_values,  target_q_values)
        optimizer.zero_grad()    
        loss.backward()
        optimizer.step()
        print(f'loss:{loss}')       
        epsilon = max(epsilon * epsilon_decay, epsilon_min) 
        state = next_state
        if episode % 25 == 0:
            target.load_state_dict(policy.state_dict())
        
        if done:
            break
       
torch.save(policy.state_dict(), 'Trained_models/model1.pth')     

pygame.quit()