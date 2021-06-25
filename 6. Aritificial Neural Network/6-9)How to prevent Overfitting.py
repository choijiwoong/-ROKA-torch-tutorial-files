#Overfitting: too much training
#Data Augmentation: make data larger forcely whrn data's amount is so small

#1. Reduce complexity of model
#example
import torch
import torch.nn as nn

class Architecture1(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Architecture1, self).__init__()
        self.fc1=nn.Linear(input_size, hidden_size)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(hidden_size, num_classes)
        self.relu=nn.ReLU()
        self.fc3=nn.Kinear(hidden_size, num_classes)

    def forward(self, x):
        out= self.fc1(x)
        out=self.relu(out)
        out=self.fc2(out)
        out=self.relu(out)
        out=self.fc3(out)
        return out
#can reduce complexity
class Architecture1(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Architecture1, self).__init__()
        self.fc1=nn.Linear(input_size, hidden_size)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out=self.fc1(x)
        out=self=relu(out)
        out=self.fc2(out)
        return out

#2. Apply Regularization
#L1 regularization and L2 regularization(weight decay)
model=Architecture1(10,20,2)
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4, weight_decay=1e-5)

#3. dROPOUT
