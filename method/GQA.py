import torch 
import torch.nn as nn


drop = nn.Dropout(p = 0.5)

t1 = torch.tensor([1, 2, 3], dtype=torch.float32)
layer = nn.Linear(in_features =3 , out_features = 5,bias =True)
ot = layer(t1)
t2 = drop(ot)


print (t2)
