from sklearn.linear_model import LinearRegression
import numpy as np

X=np.array([[2,1],[4,2],[6,3],[8,4]])
y=np.array([20,40,60,80])

model=LinearRegression()
model.fit(X,y)

prediction=model.predict([[10,5]])

print(prediction)

from torchvision import datasets

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True
)

print("Total Images:", len(train_data))


import matplotlib.pyplot as plt
from torchvision import datasets

train_data = datasets.MNIST(root="data",train=True,download=True)
image,label = train_data[0]

plt.imshow(image,cmap="gray")
plt.title(f"Label:{label}")
plt.show()

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transform = transforms. ToTensor()

train_data = datasets.MNIST(root="data",train=True,download=True,transform=transform)
#first image
image,label = train_data[0]

plt.imshow(image.squeeze(),cmap="gray")
plt.title(f"Label:{label}")
plt.show()


import torch.nn as nn

model = nn.Sequential(nn.Flatten(),nn.Linear(28*28,128),nn.ReLU(),nn.Linear(128,10))
print(model)


import torch
import torch.optim as optim

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

image,label = train_data[0]

image = image.view(1,28*28)
label = torch.tensor([label])

#forward
output = model(image)
loss = loss_fn(output, label)

#backward
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Loss:", loss.item())


import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

tranform = transforms.ToTensor()

train_data = datasets.MNIST(root="data",train=True,download=True,transform=transform)
model = nn.Sequential(nn.Flatten(),nn.Linear(28*28,128),nn.ReLU(),nn.Linear(128,10))

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

for i in range(100):
    image.label = train_data[i]

    image = image.view(1,28*28)
    label = torch.tensor([label])
    output =model(image)
    loss = loss_fn(output,label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("Finall Loss:",loss.item())


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

transform = transforms. ToTensor()
train_data=  datasets.MNIST(root="data",train=True,download=True,transform=transform)

trian_loader = DataLoader(train_data,batch_size=64,shuffle=True)

model = nn.Sequential(nn.Flatten(),nn.Linear(28*28,128),nn.ReLU(),nn.Linear(128,10))

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

epochs = 3
for epoch in range(epochs):
     total_loss = 0

     for images,labels in trian_loader:

         outputs = model(images)

         loss = loss_fn(outputs, labels)

         optimizer.zero_grad()
         loss.backward()
         optimizer.step()

         total_loss += loss.item()

     print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")




