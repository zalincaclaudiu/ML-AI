import numpy as np
import random
np.random.seed(1)



class NeuralNetwork(object):
    def __init__(self):
        self.training_inputs=np.loadtxt("E:\Facultate\An 3\Practica\Laborator5 - implementare retea neuronala\iris.data",delimiter=",",usecols=(0,1,2,3))
        training_strings=np.loadtxt("E:\Facultate\An 3\Practica\Laborator5 - implementare retea neuronala\iris.data",delimiter=",",usecols=4,dtype='U')
        self.training_outputs=np.empty([0,3])
        for x in training_strings:
            if(x=='Iris-setosa'):
                self.training_outputs=np.vstack((self.training_outputs,np.array([1,0,0])))
            if(x=='Iris-versicolor'):
                self.training_outputs=np.vstack((self.training_outputs,np.array([0,1,0])))
            if(x=='Iris-virginica'):
                self.training_outputs=np.vstack((self.training_outputs,np.array([0,0,1])))
        self.n_inputs=self.training_inputs.shape[1]
        self.n_hidden=10
        self.n_outputs=self.training_outputs.shape[1]
        self.weights=[]
        synaptic_weights1 = 2 * np.random.random((self.n_inputs,self.n_hidden)) - 1
        synaptic_weights2 = 2 * np.random.random((self.n_hidden,self.n_outputs)) - 1

        self.weights.append(synaptic_weights1)
        self.weights.append(synaptic_weights2)
        self.nrOfTrainings=self.training_inputs.shape[0]
        self.trial=np.zeros(self.n_inputs+self.n_hidden)
        self.limit=2
        self.MIN=-1
        self.MAX=1
        self.forwardPropagation()

    def forwardPropagation(self):
        self.h=np.dot(self.training_inputs,self.weights[0])
        self.outh=self.sigmoid(self.h)
        self.y=np.dot(self.outh,self.weights[1])
        self.outy=self.sigmoid(self.y)
        self.calculateError()

    def forwardPropagationPredict(self,array):
        h=np.dot(array,self.weights[0])
        outh=self.sigmoid(h)
        y=np.dot(outh,self.weights[1])
        outy=self.sigmoid(y)
        print('Predict: ',outy)
        if(outy[0,0]>0.5):
            print('Iris Secora')
        if(outy[0,1]>0.5):
            print('Iris Versicolor')
        if(outy[0,2]>0.5):
            print('Iris Virginica')
            
    def backPropagation(self):
        self.error=self.training_outputs-self.outy
        self.calculateError()
        self.adjustments=self.error*self.sigmoid_derivative(self.outy)
        self.weights[1]+=np.dot(self.outh.T,self.adjustments)

        self.adjustments2=self.sigmoid_derivative(self.outh)*np.dot(self.adjustments,self.weights[1].T)
        self.weights[0]+=np.dot(self.training_inputs.T,self.adjustments2)

    def geneticTraining(self):
        for i in range(7000):
            self.WorkerbeesPhase()
            if(i%1000==0):
                print(self.MSE)

    def createNewChar(self,pChar,fChar):
        alpha=random.random()*2-1
        newChar=fChar+alpha*(fChar-pChar)
        if(newChar<self.MIN):
            newChar=self.MIN
        if(newChar>self.MAX):
            newChar=self.MAX
        return newChar

    def calculateError(self):
        self.error=self.training_outputs-self.outy
        k=self.training_inputs.shape[0]
        n=self.n_outputs
        self.MSE=np.sum(np.multiply(self.error,self.error))/(k*n)

    def WorkerbeesPhase(self):
        for i in range(self.n_inputs):
            self.forwardPropagation()
            oldMSE=self.MSE
            temp=np.copy(self.weights[0][i])
            newFood=np.copy(self.weights[0][i])
            partener=random.randint(0,self.n_inputs-1)
            while(i==partener):
                partener=random.randint(0,self.n_inputs-1)
            crts=random.randint(0,self.n_hidden-1)
            newFood[crts]=self.createNewChar(self.weights[0][partener,crts],newFood[crts])
            self.weights[0][i]=newFood
            self.forwardPropagation()
            if(self.MSE>=oldMSE):
                self.weights[0][i]=temp
                self.trial[i]+=1
                #print('nu am gasit')
            else:
                #print('am gasit')
                self.trial[i]=0
            self.forwardPropagation()

        for i in range(self.n_hidden):
            self.forwardPropagation()
            oldMSE=self.MSE
            temp=np.copy(self.weights[1][i])
            newFood=np.copy(self.weights[1][i])
            partener=random.randint(0,self.n_hidden-1)
            while(i==partener):
                partener=random.randint(0,self.n_hidden-1)
            crts=random.randint(0,self.n_outputs-1)
            newFood[crts]=self.createNewChar(self.weights[1][partener,crts],newFood[crts])
            self.weights[1][i]=newFood
            self.forwardPropagation()
            if(self.MSE>oldMSE):
                self.weights[1][i]=temp
                self.trial[i+self.n_inputs]+=1
            else:
                self.trial[i+self.n_inputs]=0         
            self.forwardPropagation()
            
    def OnlookerbeesPhase(self):
        self.initializeProbabilities()
        foodIndex=0
        lookBee=0
        while(lookBee<self.nrOnlookerBees):
            pFoodIndex=random.random()
            if(pFoodIndex<self.p[foodIndex]):
                newFood=self.foodSources[foodIndex][:]
                partener=random.randint(0,self.nrFoodSources-1)
                while(foodIndex==partener):
                    partener=random.randint(0,self.nrFoodSources-1)
                crts=random.randint(0,self.nrCharFood-1)
                newFood[crts]=self.createNewChar(self.foodSources[partener][crts],newFood[crts])
                fNewFood=self.f(newFood)
                newFit=self.fitness(fNewFood)
                if(self.maximizeFit(newFit,self.fit[foodIndex])):
                    self.foodSources[foodIndex]=newFood
                    self.fValues[foodIndex]=fNewFood
                    self.fit[foodIndex]=newFit
                    self.trial[foodIndex]=0
                    lookBee+=1
                else:
                    self.trial[foodIndex]=self.trial[foodIndex]+1
            foodIndex=(foodIndex+1)%self.nrFoodSources

    def ScoutbeesPhase(self):
            for i in range(self.n_inputs):
                if(self.trial[i]>self.limit):
                    self.weights[0][i]=2*np.random.random_sample((self.n_hidden,))-1
                    self.forwardPropagation()
                    self.trial[i]=0
            for i in range(self.n_hidden):
                if(self.trial[i+self.n_inputs]>self.limit):
                    self.weights[1][i]=2*np.random.random_sample((self.n_outputs,))-1
                    self.forwardPropagation()
                    self.trial[i]=0
    

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivative(self,x):
        return x*(1-x)

    def train(self):
        for i in range(100000):
            self.forwardPropagation()
            self.backPropagation()
            if(i%1000==0):
                print(self.MSE)

neuralNetwork=NeuralNetwork()
neuralNetwork.geneticTraining()
#iris-virginica
neuralNetwork.forwardPropagationPredict(np.array([[5.8,2.7,5.1,1.9]]))
#iris-versicolor
neuralNetwork.forwardPropagationPredict(np.array([[6.6,3.0,4.4,1.4]]))
#iris-secora 
neuralNetwork.forwardPropagationPredict(np.array([[5.1,3.3,1.7,0.5]]))