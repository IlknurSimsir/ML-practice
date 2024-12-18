import gym 
import numpy as np

enviroment = gym.make('FrozenLake-v1', is_slippery=False,render_mode="ansi")
enviroment.reset()

nb_states= enviroment.observation_space.n
nb_actions= enviroment.action_space.n
qtable = np.zeros((nb_states,nb_actions))

print(qtable)

action=enviroment.action_space.sample()
"""
left:0
down:1
right:2
up:3
"""
new_state,reward,done,info, _= enviroment.step(action)


#%%
import gym 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

enviroment = gym.make('FrozenLake-v1', is_slippery=False,render_mode="ansi")
enviroment.reset()

nb_states= enviroment.observation_space.n
nb_actions= enviroment.action_space.n
qtable = np.zeros((nb_states,nb_actions))

print(qtable)
episodes=100
alpha=0.5#learnin rate
gama=0.5#discount factor

outcomes=[]

#training
for _ in tqdm(range(episodes)):
    state,_=enviroment.reset()# _ değer probability kullanmacayağımız için o işareti kullandık
    done=False
    outcomes.append("Failure")
    while not done:#ajan başarılı olana kadar state de hareket et 
        if np.max(qtable[state])>0:
            action=np.argmax(qtable[state])#öğrenilen bilgiler doğrultusunda hareket
        else:
            action=enviroment.action_space.sample()#rasgele hareket

        new_state,reward,done,info, _= enviroment.step(action)
        
        #update q table
        qtable[state,action]=qtable[state,action]+alpha*(reward+gama*np.max(qtable[new_state])-qtable[state,action])
        state=new_state
        if reward:
            outcomes[-1]="Success"

print("Q table after training")
print(qtable)

plt.bar(range(episodes),outcomes)

episodes=100
nb_success=0
#test
for _ in tqdm(range(episodes)):
    state,_=enviroment.reset()# _ değer probability kullanmacayağımız için o işareti kullandık
    done=False
    
    while not done:#ajan başarılı olana kadar state de hareket et 
        if np.max(qtable[state])>0:
            action=np.argmax(qtable[state])#öğrenilen bilgiler doğrultusunda hareket
        else:
            action=enviroment.action_space.sample()#rasgele hareket

        new_state,reward,done,info, _= enviroment.step(action)
        nb_success+=reward
        state=new_state
print("success rate: ",nb_success)

#%%
import random
import gym
import numpy as np
from tqdm import tqdm


env=gym.make('Taxi-v3',render_mode="ansi")
env.reset()
"""
0: Move south (down)

1: Move north (up)

2: Move east (right)

3: Move west (left)

4: Pickup passenger

5: Drop off passenger

"""
action_space = env.action_space.n
state_space= env.observation_space.n

q_table=np.zeros((state_space,action_space))

alpha=0.1
gama=0.6
epsilon=0.1

for i in tqdm(range(1,100001)):
    state,_=env.reset()
    done=False
    while not done:#explore %10
        if random.uniform(0, 1)<epsilon :
            action=env.action_space.sample()
            
        else:#exploit
            action=np.argmax(q_table[state])
        next_state,reward,done,info,_=env.step(action)
        q_table[state,action]=q_table[state,action]+alpha*(reward+gama*np.max(q_table[next_state])-q_table[state,action])

        state=next_state

#test
total_epoch,total_penalties=0,0
episodes=100

for i in tqdm(range(episodes)):
    state,_=env.reset()
    epochs,penalties,reward=0,0,0
    done=False
    while not done:#explore %10
        
        action=np.argmax(q_table[state])
        next_state,reward,done,info,_=env.step(action)
        state=next_state
        if reward ==-10:
            penalties +=1
        epochs+=1
    total_epoch+=epochs
    total_penalties+=penalties
    
print("epidode",format(episodes))
print("av timestamp",total_epoch/episodes)
print("av penalty",total_penalties/penalties)










