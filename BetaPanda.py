# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:37:39 2017

@author: peter
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 21:11:18 2017

@author: Elias
"""

import json
import tensorflow as tf 
import numpy as np 
import random
from collections import deque 

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
        check0=0
        check1=0
        ##role=1 means black, role=-1 means white
        if action==[]:
            terminal=True
            self.__init__()
            for i in range(7):
                for j in range(7):
                    if self.board[i,j]==1:
                        check1+=1
                    elif self.board[i,j]==-1:
                        check0+=1
            reward[0]+=check0-check1
            reward[1]+=check1-check0
            return(self.board,reward,terminal)
        if self.board[action_take[0],action_take[1]]==0:
            terminal=True
            self.__init__()
            reward[(role+1)//2]=-50
            return(self.board,reward,terminal)
        if self.board[action_take[0],action_take[1]]!=role:
            terminal=True
            self.__init__()
            reward[(role+1)//2]=-50
            return(self.board,reward,terminal)
        if self.board[action_put[0],action_put[1]]!=0:
            terminal=True
            self.__init__()
            reward[(role+1)//2]=-50
            return(self.board,reward,terminal)
        if action_put[0]==action_take[0] and action_put[1]==action_take[1]:
            terminal=True
            self.__init__()
            reward[(role+1)//2]=-50
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
            reward[(role+1)//2]=-50
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
        white_ava=0
        black_ava=0
        for i in range(7):
            for j in range(7):
                if self.board[i,j]==1:
                    for a in [-2,-1,0,1,2]:
                        for b in [-2,-1,0,1,2]:
                            try:
                                if 6>=i+a>=0 and 6>=j+b>=0 and self.board[i+a,b+j]==0:
                                    black_ava+=1
                            except:
                                pass
                elif self.board[i,j]==-1:
                    for a in [-2,-1,0,1,2]:
                        for b in [-2,-1,0,1,2]:
                            try:
                                if 6>=i+a>=0 and 6>=j+b>=0 and self.board[i+a,b+j]==0:
                                    white_ava+=1
                            except:
                                pass
        for i in range(7):
            for j in range(7):
                if self.board[i,j]==1:
                    check1+=1
                elif self.board[i,j]==-1:
                    check0+=1
        if white_ava==0 or black_ava==0:
            terminal=True
            self.__init__()
            reward[0]+=check0-check1
            reward[1]+=check1-check0
        if check0==0:
            terminal=True
            self.__init__()
            reward[0]=-50
        elif check1==0:
            terminal=True
            self.__init__()
            reward[1]=-50
        elif check0+check1==49:
            terminal=True
            self.__init__()
            reward[0]+=check0-check1
            reward[1]+=check1-check0

        return(self.board,reward,terminal)

##parameters
# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.8 # decay rate of past observations
OBSERVE = 1000. # timesteps to observe before training
EXPLORE = 200000. # frames over which to anneal epsilon
FINAL_EPSILON = 0#0.001 # final value of epsilon
INITIAL_EPSILON = 0#0.01 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH_SIZE = 500 # size of minibatch
UPDATE_TIME = 100

##
class BrainDQN0:
    def __init__(self,actions):
        # init replay memory
        self.replayMemory = deque()
        # init some parameters
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions
        # init Q network
        self.stateInput,self.QValue,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.createQNetwork()
        # init Target Q Network
        self.stateInputT,self.QValueT,self.W_conv1T,self.b_conv1T,self.W_conv2T,self.b_conv2T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T = self.createQNetwork()
        self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1),self.b_conv1T.assign(self.b_conv1),self.W_conv2T.assign(self.W_conv2),self.b_conv2T.assign(self.b_conv2),self.W_fc1T.assign(self.W_fc1),self.b_fc1T.assign(self.b_fc1),self.W_fc2T.assign(self.W_fc2),self.b_fc2T.assign(self.b_fc2)]
        self.createTrainingMethod()
        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.saver.restore(self.session, "/data/network0")
            
    def createQNetwork(self):
        # network weights
        W_conv1 = self.weight_variable([4,4,1,32])
        b_conv1 = self.bias_variable([32])

        W_conv2 = self.weight_variable([3,3,32,64])
        b_conv2 = self.bias_variable([64])

        W_fc1 = self.weight_variable([64,256])
        b_fc1 = self.bias_variable([256])

        W_fc2 = self.weight_variable([256,self.actions])
        b_fc2 = self.bias_variable([self.actions])

        # input layer

        stateInput = tf.placeholder("float",[None,7,7,1])

        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(stateInput,W_conv1,1) + b_conv1)
  
        h_conv2 = tf.nn.relu(self.conv2d(h_conv1,W_conv2,1) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
              
        h_conv2_flat = tf.reshape(h_pool2,[-1,64])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat,W_fc1) + b_fc1)

        # Q Value layer
        QValue = tf.matmul(h_fc1,W_fc2) + b_fc2

        return stateInput,QValue,W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2

    def copyTargetQNetwork(self):
        self.session.run(self.copyTargetQNetworkOperation)

    def createTrainingMethod(self):
        self.actionInput = tf.placeholder("float",[None,self.actions])
        self.yInput = tf.placeholder("float", [None]) 
        Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)


    def trainQNetwork(self):

		
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

        # Step 2: calculate y 
        y_batch = []
        QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT:nextState_batch})
        for i in range(0,BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        self.trainStep.run(feed_dict={self.yInput : y_batch,self.actionInput : action_batch,self.stateInput : state_batch})

        # save network every 100000 iteration
        if self.timeStep % 10000 == 0:
            self.saver.save(self.session, 'C:/Users/peter/Desktop/Ataxx/saved_networks0/' + 'network' + '-dqn', global_step = self.timeStep)

        if self.timeStep % UPDATE_TIME == 0:
            self.copyTargetQNetwork()


    def setPerception(self,nextObservation,action,reward,terminal):
        #newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
        newState = nextObservation
        self.replayMemory.append((self.currentState,action,reward,newState,terminal))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE:
            # Train the network
            self.trainQNetwork()

        # print info
        state = ""
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        ##print ("TIMESTEP", self.timeStep, "/ STATE", state, \
        ##"/ EPSILON", self.epsilon)

        self.currentState = newState
        self.timeStep += 1

    def getQValue(self):
        QValue = self.QValue.eval(feed_dict= {self.stateInput:[self.currentState]})
#        action = np.zeros([self.actions,4],dtype='int64')
#        for i in range(1,self.actions):
#            action[i]=action[i-1].copy()
#            action[i,0]+=1
#            for j in range(4):
#                if action[i,j]==7 and j!=3:
#                    action[i,j]=0
#                    action[i,j+1]+=1
#            
#        action_index = 0
#        if random.random() <= self.epsilon:
#            action_index = random.randrange(self.actions)
#        else:
#            action_index = np.argmax(QValue)

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE

        return(QValue)

    def setInitState(self,observation):
        self.currentState = observation

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)

    def conv2d(self,x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 1, 1, 1], padding = "VALID")

##
class BrainDQN1:
    def __init__(self,actions):
        # init replay memory
        self.replayMemory = deque()
        # init some parameters
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions
        # init Q network
        self.stateInput,self.QValue,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.createQNetwork()
        # init Target Q Network
        self.stateInputT,self.QValueT,self.W_conv1T,self.b_conv1T,self.W_conv2T,self.b_conv2T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T = self.createQNetwork()
        self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1),self.b_conv1T.assign(self.b_conv1),self.W_conv2T.assign(self.W_conv2),self.b_conv2T.assign(self.b_conv2),self.W_fc1T.assign(self.W_fc1),self.b_fc1T.assign(self.b_fc1),self.W_fc2T.assign(self.W_fc2),self.b_fc2T.assign(self.b_fc2)]
        self.createTrainingMethod()
        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.saver.restore(self.session,"/data/network1")
        
    def createQNetwork(self):
        # network weights
        W_conv1 = self.weight_variable([4,4,1,32])
        b_conv1 = self.bias_variable([32])

        W_conv2 = self.weight_variable([3,3,32,64])
        b_conv2 = self.bias_variable([64])

        W_fc1 = self.weight_variable([64,256])
        b_fc1 = self.bias_variable([256])

        W_fc2 = self.weight_variable([256,self.actions])
        b_fc2 = self.bias_variable([self.actions])

        # input layer

        stateInput = tf.placeholder("float",[None,7,7,1])

        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(stateInput,W_conv1,1) + b_conv1)
  
        h_conv2 = tf.nn.relu(self.conv2d(h_conv1,W_conv2,1) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)
              
        h_conv2_flat = tf.reshape(h_pool2,[-1,64])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat,W_fc1) + b_fc1)

        # Q Value layer
        QValue = tf.matmul(h_fc1,W_fc2) + b_fc2

        return stateInput,QValue,W_conv1,b_conv1,W_conv2,b_conv2,W_fc1,b_fc1,W_fc2,b_fc2

    def copyTargetQNetwork(self):
        self.session.run(self.copyTargetQNetworkOperation)

    def createTrainingMethod(self):
        self.actionInput = tf.placeholder("float",[None,self.actions])
        self.yInput = tf.placeholder("float", [None]) 
        Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)


    def trainQNetwork(self):

		
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

        # Step 2: calculate y 
        y_batch = []
        QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT:nextState_batch})
        for i in range(0,BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        self.trainStep.run(feed_dict={self.yInput : y_batch,self.actionInput : action_batch,self.stateInput : state_batch})

        # save network every 100000 iteration
        if self.timeStep % 10000 == 0:
            self.saver.save(self.session, 'C:/Users/peter/Desktop/Ataxx/saved_networks1/' + 'network' + '-dqn', global_step = self.timeStep)

        if self.timeStep % UPDATE_TIME == 0:
            self.copyTargetQNetwork()


    def setPerception(self,nextObservation,action,reward,terminal):
        #newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
        newState = nextObservation
        self.replayMemory.append((self.currentState,action,reward,newState,terminal))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > OBSERVE:
            # Train the network
            self.trainQNetwork()

        # print info
        state = ""
        if self.timeStep <= OBSERVE:
            state = "observe"
        elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

#        print ("TIMESTEP", self.timeStep, "/ STATE", state, \
#        "/ EPSILON", self.epsilon)

        self.currentState = newState
        self.timeStep += 1

    def getQValue(self):
        QValue = self.QValue.eval(feed_dict= {self.stateInput:[self.currentState]})
#        action = np.zeros([self.actions,4],dtype='int64')
#        for i in range(1,self.actions):
#            action[i]=action[i-1].copy()
#            action[i,0]+=1
#            for j in range(4):
#                if action[i,j]==7 and j!=3:
#                    action[i,j]=0
#                    action[i,j+1]+=1
#            
#        action_index = 0
#        if random.random() <= self.epsilon:
#            action_index = random.randrange(self.actions)
#        else:
#            action_index = np.argmax(QValue)

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE

        return(QValue)

    def setInitState(self,observation):
        self.currentState = observation

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)

    def conv2d(self,x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 1, 1, 1], padding = "VALID")

def main():
    actions=7*7*7*7
    json_information=input()
    information=eval(json_information)
    request=information['requests']
    response=information['responses']
    game=ataxx()
    observation=game.board.copy()
    
    if request[0]=={'x0':-1,'x1':-1,'y0':-1,'y1':-1}:
        color=1
        brain=BrainDQN1(actions)
    else:
        color=-1
        brain=BrainDQN0(actions)
        current=request[0]
        action_update=[current['y0'],current['x0'],current['y1'],current['x1']]
        observation,reward,terminal=game.put_action(action_update,-color)
    if len(response)>=1:
        for i in range(len(response)):
            current=response[i]
            action_update=[current['y0'],current['x0'],current['y1'],current['x1']]
            observation,reward,terminal=game.put_action(action_update,color)
            current=request[i+1]
            action_update=[current['y0'],current['x0'],current['y1'],current['x1']]
            observation,reward,terminal=game.put_action(action_update,-color)
    brain.setInitState(observation.reshape([7,7,1]))
#        observation=np.reshape(observation,[7,7,1])
#        brain.setPerception(observation,action_update,reward[(color+1)//2],terminal)
    qvalue=brain.getQValue()
    avaliable=[]
    for i in range(7):
        for j in range(7):
            if game.board[i,j]==color:
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
    
    output=json.dumps({'response':{'y0':my_action[0],'x0':my_action[1],'y1':my_action[2],'x1':my_action[3]}})
    print(output)

if __name__=='__main__':
    main()
        