# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 18:41:42 2017

@author: Elias
"""

import numpy as np
import random
import ataxx_game
from DQN import BrainDQN0,BrainDQN1

##1 stands for black, -1 stands for white
actions=7*7*7*7
epsilon=0.1
def playchess():
    #initial DQN
    brain0=BrainDQN0(actions)
    brain1=BrainDQN1(actions)
    #initial GAME
    game=ataxx_game.ataxx()
    observation0=game.board
    #reward0=[0,0]
    terminal=False
    brain0.setInitState(observation0.reshape([7,7,1]))
    brain1.setInitState(observation0.reshape([7,7,1]))
    cout=1
    while 1!=0:
        if cout%2==1:
            #black's move
            qvalue=brain1.getQValue()
            avaliable=[]
            for i in range(7):
                for j in range(7):
                    if game.board[i,j]==1:
                        for a in [-2,-1,0,1,2]:
                            for b in [-2,-1,0,1,2]:
                                try:
                                    if 6>=i+a>=0 and 6>=j+b>=0 and game.board[i+a,j+b]==0:
                                        avaliable.append([i,j,i+a,j+b])
                                except:
                                    pass
            values=0
            action=[]
            action_index=0
            for x in avaliable:
                index=0
                for i in range(4):
                    index+=x[i]*(7**(3-i))
                if qvalue[0,index]>values:
                    values=qvalue[0,index]
                    action=x.copy()
                    action_index=index
            if random.random()<=epsilon:
                action = avaliable[random.randrange(len(avaliable))]
                action_index=0
                for i in range(4):
                    action_index+=action[i]*(7**(3-i))   
            action_vector=np.zeros(2401)
            action_vector[action_index]=1
            nextObservation,reward,terminal = game.put_action(action,1)
            nextObservation=np.reshape(nextObservation,[7,7,1])
            brain1.setPerception(nextObservation,action_vector,reward[1],terminal)
            if terminal:
                cout=1
                brain0.currentState=observation0.reshape([7,7,1])
                brain1.currentState=observation0.reshape([7,7,1])
            else:
                cout+=1
                brain0.currentState=nextObservation
        else:
            #white's move
            qvalue=brain0.getQValue()
            avaliable=[]
            for i in range(7):
                for j in range(7):
                    if game.board[i,j]==-1:
                        for a in [-2,-1,0,1,2]:
                            for b in [-2,-1,0,1,2]:
                                try:
                                    if 6>=i+a>=0 and 6>=j+b>=0 and game.board[i+a,j+b]==0:
                                        avaliable.append([i,j,i+a,j+b])
                                except:
                                    pass
            values=0
            action=[]
            action_index=0
            for x in avaliable:
                index=0
                for i in range(4):
                    index+=x[i]*(7**(3-i))
                if qvalue[0,index]>values:
                    values=qvalue[0,index]
                    action=x.copy()
                    action_index=index
            if random.random()<=epsilon:
                action = avaliable[random.randrange(len(avaliable))]
                action_index=0
                for i in range(4):
                    action_index+=action[i]*(7**(3-i))  
            action_vector=np.zeros(2401)
            action_vector[action_index]=1
            nextObservation,reward,terminal = game.put_action(action,-1)
            nextObservation=np.reshape(nextObservation,[7,7,1])
            brain0.setPerception(nextObservation,action_vector,reward[0],terminal)
            if terminal:
                cout=1
                brain0.currentState=observation0.reshape([7,7,1])
                brain1.currentState=observation0.reshape([7,7,1])
            else:
                cout+=1
                brain1.currentState=nextObservation
def main():
    playchess()
if __name__=='__main__':
    main()