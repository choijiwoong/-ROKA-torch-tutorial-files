import torch
import torch.nn.functional as F
torch.manual_seed(1)

z=torch.rand(3,5,requires_grad=True)#make random tensor
hypothesis=F.softmax(z,dim=1)#hypothesis set softmax's output about random tensor

y=torch.randint(5,(3,)).long()#y's label making.
cost=F.cross_entropy(z,y)
print(cost)

#F.softmax(z,dim=0)_softmax function with input 'z'
#torch.log(hypothesis)_calculation of cost function's meterial_logical meterial_use F.softmax as input_*need ont_hot vectorize*

#F.log_softmax()=log(F.softmax())
#(y_one_hot**-F.log_softmax(z,dim=1)).sum(dim=1).mean()-> F.nll_loss(F.log_softmax(z,dim=1),y) *not needed ont_hot vectorizing*
#F.log_softmax()+F.nll_loss()=F.cross_entropy

