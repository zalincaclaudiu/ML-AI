import random

class ArtificialBeeColony(object):
    def __init__(self):
        self.colonySize=10
        self.limit=1
        self.nrWorkerBees=self.colonySize//2
        self.nrFoodSources=self.colonySize//2
        self.nrOnlookerBees=self.colonySize//2
        self.nrCharFood=2
        self.MIN=-5
        self.MAX=5
        self.maxIterations=60
        self.trial=[]
        self.fit=[]
        self.fValues=[]
        self.foodSources=[]
        self.p=[]
        self.bestFood=[]
        self.bestFit=-1
        self.bestFValue=-1

    def initialize(self):
        for i in range(self.nrFoodSources):
            food=self.createFoodSource()
            self.foodSources.append(food)
            self.trial.append(0)
            self.p.append(0)
            fx=self.f(food)
            self.fValues.append(fx)
            self.fit.append(self.fitness(fx))

    def f(self,food):
        return food[0]*food[0]-food[0]*food[1]+food[1]*food[1]+2*food[0]+4*food[1]+3

    def fitness(self,value):
        if(value>=0):
            return 1/(1+value)
        else:
            return 1+(value*-1)
    
    def WorkerbeesPhase(self):
        for i in range(self.nrWorkerBees):
            newFood=self.foodSources[i][:]
            partener=random.randint(0,self.nrFoodSources-1)
            while(i==partener):
                partener=random.randint(0,self.nrFoodSources-1)
            crts=random.randint(0,self.nrCharFood-1)
            newFood[crts]=self.createNewChar(self.foodSources[partener][crts],newFood[crts])
            fNewFood=self.f(newFood)
            newFit=self.fitness(fNewFood)
            if(self.minimizeFit(newFit,self.fit[i])):
                self.foodSources[i]=newFood
                self.fValues[i]=fNewFood
                self.fit[i]=newFit
                self.trial[i]=0
            else:
                self.trial[i]=self.trial[i]+1
            
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
                if(self.minimizeFit(newFit,self.fit[foodIndex])):
                    self.foodSources[foodIndex]=newFood
                    self.fValues[foodIndex]=fNewFood
                    self.fit[foodIndex]=newFit
                    self.trial[foodIndex]=0
                    lookBee+=1
                else:
                    self.trial[foodIndex]=self.trial[foodIndex]+1
            foodIndex=(foodIndex+1)%self.nrFoodSources

    def ScoutbeesPhase(self):
        scout=self.exceedLimit()
        if(scout!=-1):
            for i in range(scout,self.nrFoodSources):
                if(self.trial[i]>self.limit):
                    self.foodSources[i]=self.createFoodSource()
                    self.fValues[i]=self.f(self.foodSources[i])
                    self.fit[i]=self.fitness(self.fValues[i])
                    self.trial[i]=0

    def minimizeFit(self,newFit,oldFit):
        if(newFit<oldFit):
            return True
        else:
            return False
    
    def maximizeFit(self,newFit,oldFit):
        if(newFit>oldFit):
            return True
        else:
            return False

    def createNewChar(self,pChar,fChar):
        alpha=random.random()*2-1
        newChar=fChar+alpha*(fChar-pChar)
        if(newChar<self.MIN):
            newChar=self.MIN
        if(newChar>self.MAX):
            newChar=self.MAX
        return newChar

    def initializeProbabilities(self):
        for i in range(self.nrFoodSources):
            self.p[i]=self.fit[i]/self.sumFit()

    def sumFit(self):
        sum=0
        for i in range(self.nrFoodSources):
            sum=sum+self.fit[i]
        return sum

    def exceedLimit(self):
        for i in range(self.nrFoodSources):
            if(self.trial[i]>self.limit):
                return i
        return -1

    def createFoodSource(self):
        food=[]
        for i in range(self.nrCharFood):
            k=random.uniform(self.MIN,self.MAX)
            food.append(k)
        return food

    def maximumValue(self):
        iMax=0
        max=self.fValues[0]
        for i in range(1,self.nrFoodSources):
            if(self.fValues[i]>max):
                iMax=i
                max=self.fValues[i]
        return iMax

    def minimumValue(self):
        iMin=0
        min=self.fValues[0]
        for i in range(1,self.nrFoodSources):
            if(self.fValues[i]<min):
                iMin=i
                min=self.fValues[i]
        return iMin
    
    def memorizeBestValue(self):
        indBest=self.maximumValue()
        self.bestFood=self.foodSources[indBest]
        self.bestFit=self.fit[indBest]
        self.bestFValue=self.fValues[indBest]

    def ABCalgorithm(self):
        self.initialize()
        for i in range(self.maxIterations):
            self.WorkerbeesPhase()
            self.OnlookerbeesPhase()
            self.ScoutbeesPhase()
            self.memorizeBestValue()
            self.printSolution(i)

    def printSolution(self,ind): 
        print('***ITERATIA '+str(ind+1)+'***')
        print('\n')
        print('\tfood Source','f(x)','fit','trial',sep='\t\t')
        for i in range(0,self.nrFoodSources):
            self.printVector(self.foodSources[i])
            print("%+03.4f" % self.fValues[i],'\t',end='')
            print("%+03.4f" % self.fit[i],'\t',end='')
            print("%d" % self.trial[i])
        print('\n\n')

    def printVector(self,v):
        print('    | ',end='')
        for i in range(len(v)):
            print("%+03.4f" % v[i],end='   ')
        print('| ',end='\t')    

ABC=ArtificialBeeColony()
ABC.ABCalgorithm()