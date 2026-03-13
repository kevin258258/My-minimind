import torch
x = torch.tensor([1,2,3,4,5,6,7])
y = torch.tensor([0,0,0,0,0,0,0])
condition = x > 4
result = torch.where(condition,x,y)
print(result)
# 这里是where方法
t1 = torch.tensor([[1,2],[3,4]])
t2 = torch.tensor([[1,2],[3,4]])
t3 = torch.cat((t1,t2),dim =0)
print(t3.shape)
# unsqueeze 用来添加维度，通常为了对齐
t4 = t3.unsqueeze(0)
print(t4.shape  )