# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 18:41:42 2017

@author: Elias
"""
import json
import numpy as np
import ataxx_game
from DQN import BrainDQN
##1 stands for black, -1 stands for white
actions=7*7*7*7
def playchess():
    #initial DQN
    brain0=BrainDQN(actions)
    brain1=BrainDQN(actions)
    #initial GAME
    game=ataxx_game.ataxx()
    observation0=game.board
    #reward0=[0,0]
    terminal=False
    brain0.setInitState(observation0)
    brain1.setInitState(observation0)
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
            action_vector=np.zeros(2401)
            action_vector[action_index]=1
            nextObservation,reward,terminal = game.put_action(action,1)
            nextObservation=np.reshape(nextObservation,[7,7,1])
            brain1.setPerception(nextObservation,action_vector,reward[1],terminal)
            if terminal:
                cout=1
            else:
                cout+=1
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
            action_vector=np.zeros(2401)
            action_vector[action_index]=1
            nextObservation,reward,terminal = game.put_action(action,-1)
            nextObservation=np.reshape(nextObservation,[7,7,1])
            brain0.setPerception(nextObservation,action_vector,reward[0],terminal)
            if terminal:
                cout=1
            else:
                cout+=1
def contest():
    brain=...
    my_color=
    game=ataxx_game.ataxx()
    while 1!=0:
        action_enemy=input
        observation,reward,terimnal=game.put_action(action_enemy,-my_color)
        observation=np.reshape(observation,[7,7,1])
        brain.setPerception(observation,action_vector,reward[0],terminal)
        qvalue=brain.getQValue()
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
        my_action=[]
        for x in avaliable:
            index=0
            for i in range(4):
                index+=x[i]*(7**(3-i))
            if qvalue[0,index]>values:
                values=qvalue[0,index]
                my_action=x.copy()
        response=my_action
        
def main1():
    playchess()
def main2():
    contest()
if __name__=='__main__':
    main1()