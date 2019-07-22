import numpy as np 
      

X=np.array(([0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]), dtype=float)

y=np.array(([0],[1],[1],[0]), dtype=float)


def sigmoid_function(s):
    return 1/(1+np.exp(-s))


def sigmoid_function_derivative(d):
    return d * (1 - d)


class NeuralNetwork:
    def __init__(neural, x,y):

        np.random.seed(1)

        neural.input = x
        neural.weights1= np.random.rand(neural.input.shape[1],4) 
        neural.weights2 = np.random.rand(4,1)
        neural.y = y
        neural.output = np. zeros(y.shape)
        
    def feedforward(neural):
        neural.layer1 = sigmoid_function(np.dot(neural.input, neural.weights1))
        neural.layer2 = sigmoid_function(np.dot(neural.layer1, neural.weights2))
        return neural.layer2
        
    def backpropagation(neural):
        d_weights2 = np.dot(neural.layer1.T, 2*(neural.y -neural.output)*sigmoid_function_derivative(neural.output))
        d_weights1 = np.dot(neural.input.T, np.dot(2*(neural.y -neural.output)*sigmoid_function_derivative(neural.output), neural.weights2.T)*sigmoid_function_derivative(neural.layer1))
    
        neural.weights1 += d_weights1
        neural.weights2 += d_weights2

    def train(neural, X, y):
        neural.output = neural.feedforward()
        neural.backpropagation()
        

NN = NeuralNetwork(X,y)
for i in range(20000): 
    if i % 100 ==0: 
        print ("for iteration # " + str(i) + "\n")
        print ("Input : \n" + str(X))
        print ("Actual Output: \n" + str(y))
        print ("Predicted Output: \n" + str(NN.feedforward()))
        print ("Loss: \n" + str(np.mean(np.square(y - NN.feedforward())))) 
        print ("\n")
  
    NN.train(X, y)
