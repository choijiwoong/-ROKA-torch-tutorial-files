#Single-Layer Perceptron: input layer, output layer (step is called layer)

def AND_gate(x1,x2):
    w1, w2, b=0.5, 0.5, -0.7
    result=x1*w1+x2*w2+b
    if result<=0:
        return 0
    else:
        return 1

def NAND_gate(x1,x2):
    w1, w2, b=-0.5, -0.5, +0.7
    result=x1*w1+x2*w2+b
    if result<=0:
        return 0
    else:
        return 1

def OR_gate(x1,x2):
    w1, w2, b=0.6, 0.6, -0.5
    result=x1*w1+x2*w2+b
    if result<=0:
        return 0
    else:
        return 1
    
#MultiLayer Perceptron, MLP: input layer, hidden layer, output layer(more than 1 hidden layer)
#XOR
def XOR_gate(x1,x2):
    s1=NAND_gate(x1,x2)
    s2=OR_gate(x1,x2)
    return AND_gate(s1,s2)

print(XOR_gate(0,0),XOR_gate(0,1),XOR_gate(1,0),XOR_gate(1,1))
    

#Deep Neural Network, DNN: more than 2 hidden layer

#if Artificial Network in training is Deep Neural Network, it calls Deep Learning.
