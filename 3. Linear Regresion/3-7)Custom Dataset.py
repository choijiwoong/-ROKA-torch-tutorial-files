import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomDataset(Dataset):#inheritance Dataset
    def __init__(self):
        self.x_data=[[73,80,75],
                [93,88,93],
                [89,91,90],
                [96,98,100],
                [73,66,70]]
        self.y_data=[[152],[185],[180],[196],[142]]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x=torch.FloatTensor(self.x_data[idx])
        y=torch.FloatTensor(self.y_data[idx])
        return x, y

dataset=CustomDataset()
dataloader=DataLoader(dataset,batch_size=2, shuffle=True)

model=torch.nn.Linear(3,1)
optimizer=torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs=20
for epoch in range(nb_epochs+1):
    for batch_idx, samples in enumerate(dataloader):
        #print(batch_idx)
        #print(samples)
        x_train, y_train=samples
        prediction=model(x_train)
        cost=F.mse_loss(prediction,y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print("Epoch {:4d}/{} batch {}/{} cost: {:.6f}".format(epoch, nb_epochs, batch_idx+1,len(dataloader),cost.item()))
