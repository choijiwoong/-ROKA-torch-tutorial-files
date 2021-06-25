import numpy as np
import matplotlib.pyplot as plt

#The feature of activation function-Nonlinear function->if activation function is linear, then it means  just updating of weight(it's called by projection layer or linear layer)

#Sigmoid Function
def sigmoid(x):
    return 1/(1+np.exp(-x))

x=np.arange(-5.0,5.0,0.1)
y=sigmoid(x)

plt.plot(x,y)
plt.plot([0,0],[1.0,0.0],':')
plt.title('Sigmoid Function')
plt.show()

#that's why we don't use Sigmoid. because of Vanishing Gradient. not update forward weight

#Hyperbolic tangent function
x=np.arange(-5.0,5.0,0.1)
y=np.tanh(x)

plt.plot(x,y)
plt.plot([0,0],[-1.0,1.0],':')
plt.axhline(y=0,color='orange',linestyle='--')
plt.title('Tanh Function')
plt.show()
#Hyperbolic tangent is well used than Sigmoid function because of low Vanishing Gradient than Sigmoid

#ReLU Function
def relu(x):
    return np.maximum(0,x)

x=np.arange(-5.0,5.0,0.1)
y=relu(x)

plt.plot(x,y)
plt.plot([0,0],[5.0,0.0],':')
plt.title('Relu Function')
plt.show()
#problem->if input is - then weight is 0 too. it calls dying ReLU

#Leaky ReLU
a=0.1#(Leaky rate)
def leaky_relu(x):
    return np.maximum(a*x,x)

x=np.arange(-5.0,5.0,0.1)
y=leaky_relu(x)

plt.plot(x,y)
plt.plot([0,0],[5.0,0.0],':')
plt.title('Leaky ReLU Function')
plt.show()

#Softmax Function
x=np.arange(-5.0,5.0,0.1)
y=np.exp(x)/np.sum(np.exp(x))

plt.plot(x,y)
plt.title('Softmax Function')
plt.show()
#Sigmoid Function is well used at Binary Classification, Softmax function is well used at MultiClass Classification

#Theory
#Binary Classification->Sigmoid_active F, nn.BCELoss()_cost F
#MultiClass Classificationi->Softmax_active F, nn.CrossEntropyLoss()_cost F (p.s)nn.CrossEntropyLoss has already Softmax Function
#Regressive->none, MSE_cost F
