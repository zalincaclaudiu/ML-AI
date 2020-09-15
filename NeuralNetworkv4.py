import numpy as np
import csv
np.random.seed(1)



class NeuralNetwork(object):
    def __init__(self,nrInput,nrHidden,nrOutputs):
        self.n_inputs=nrInput
        self.n_hidden=nrHidden
        self.n_outputs=nrOutputs
        self.weights=[]
        synaptic_weights1 = 2 * np.random.random((n_inputs,n_hidden)) - 1
        synaptic_weights2 = 2 * np.random.random((n_hidden,n_outputs)) - 1

        self.weights.append(synaptic_weights1)
        self.weights.append(synaptic_weights2)
        print(self.weights[0])
        print(self.weights[1])
        self.training_inputs=np.array([
                        [0,0,1,0],
                        [1,1,1,0],
                        [1,0,1,0],
                        [0,1,1,0],
                        [1,1,0,1],
                        [0,0,0,0],
                        ])
        self.nrOfTrainings=self.training_inputs.shape[0]
        self.training_outputs=np.array([[0,0],[1,1],[1,0],[0,1],[1,1],[0,0]])

    def forwardPropagation(self):
        self.h=np.dot(self.training_inputs,self.weights[0])
        self.outh=self.sigmoid(self.h)
        self.y=np.dot(self.outh,self.weights[1])
        self.outy=self.sigmoid(self.y)
    
    def forwardPropagationPredict(self,array):
        h=np.dot(array,self.weights[0])
        outh=self.sigmoid(h)
        y=np.dot(outh,self.weights[1])
        outy=self.sigmoid(y)
        print('Predict: ',outy)

    def backPropagation(self):
        self.error=self.training_outputs-self.outy
        self.calculateError()
        self.adjustments=self.error*self.sigmoid_derivative(self.outy)
        self.weights[1]+=np.dot(self.outh.T,self.adjustments)

        self.adjustments2=self.sigmoid_derivative(self.outh)*np.dot(self.adjustments,self.weights[1].T)
        self.weights[0]+=np.dot(self.training_inputs.T,self.adjustments2)

    def calculateError(self):
        k=self.training_inputs.shape[0]
        n=self.n_outputs
        self.MSE=np.sum(np.multiply(self.error,self.error))/(k*n)


    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivative(self,x):
        return x*(1-x)

    def train(self):
        for i in range(5000):
            self.forwardPropagation()
            self.backPropagation()
            if(i%1000==0):
                print(self.MSE)

n_inputs=int(input('Enter the number of inputs:'))
n_hidden=int(input('Enter the number of hidden:'))
n_outputs=int(input('Enter the number of outputs:'))
neuralNetwork=NeuralNetwork(n_inputs,n_hidden,n_outputs)
neuralNetwork.train()
# 0 0 - expected
# x3 x2 
neuralNetwork.forwardPropagationPredict(np.array([[0,0,1,1]]))