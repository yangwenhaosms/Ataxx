# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 19:04:02 2017

@author: Elias
"""

import numpy as np

class ataxx:
    def __init__(self):
        self.board=np.zeros([7,7],dtype='int64')
        self.board[0,0]=1
        self.board[6,6]=1
        self.board[0,6]=-1
        self.board[6,0]=-1
    def put_action(self,action,role):
        action_take=action[0:2]
        action_put=action[2:4]
        terminal=False
        reward=[0,0]
        ##role=1 means black, role=-1 means white
        if action==[]:
            terminal=True
            self.__init__()
            reward[(role+1)//2]=-100
            return(self.board,reward,terminal)
        if self.board[action_take[0],action_take[1]]==0:
            terminal=True
            self.__init__()
            reward[(role+1)//2]=-100
            return(self.board,reward,terminal)
        if self.board[action_take[0],action_take[1]]!=role:
            terminal=True
            self.__init__()
            reward[(role+1)//2]=-100
            return(self.board,reward,terminal)
        if self.board[action_put[0],action_put[1]]!=0:
            terminal=True
            self.__init__()
            reward[(role+1)//2]=-100
            return(self.board,reward,terminal)
        if action_put[0]==action_take[0] and action_put[1]==action_take[1]:
            terminal=True
            self.__init__()
            reward[(role+1)//2]=-100
            return(self.board,reward,terminal)
        if abs(action_take[0]-action_put[0])<=1 and abs(action_take[1]-action_put[1])<=1:
            self.board[action_put[0],action_put[1]]=role
            reward[(role+1)//2]+=1
        elif abs(action_take[0]-action_put[0])==2 or abs(action_take[1]-action_put[1])==2:
            self.board[action_take[0],action_take[1]]=0
            self.board[action_put[0],action_put[1]]=role
        else:
            terminal=True
            self.__init__()
            reward[(role+1)//2]=-100
            return(self.board,reward,terminal)
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                try:
                    if action_put[0]+i>=0 and action_put[1]+j>=0 and self.board[action_put[0]+i,action_put[1]+j]==-role:
                        self.board[action_put[0]+i,action_put[1]+j]=role
                        reward[(role+1)//2]+=1
                        reward[1-(role+1)//2]-=1
                except:
                    pass
        check0=0
        check1=0
        for i in range(7):
            for j in range(7):
                if self.board[i,j]==1:
                    check1+=1
                elif self.board[i,j]==-1:
                    check0+=1
        if check0==0:
            terminal=True
            self.__init__()
            reward[0]=-100
        elif check1==0:
            terminal=True
            self.__init__()
            reward[1]=-100
        elif check0+check1==49:
            terminal=True
            self.__init__()
            reward[0]+=check0-check1
            reward[1]+=check1-check0
        return(self.board,reward,terminal)