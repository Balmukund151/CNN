import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms


xy = pd.read_csv('fashion-mnist_train.csv')
x_data = Variable(torch.tensor(xy.iloc[:, 1:].values))
y_data = Variable(torch.tensor(xy.iloc[:, [0]].values))
#num = xy.shape[0]

#print(x_data.data.shape)
#print(y_data.data.shape)

xy_test = pd.read_csv('fashion-mnist_test.csv')

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1,6,5)
        self.batch_Norm1 = torch.nn.BatchNorm2d(6)
        self.pool =  torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.batch_Norm2 = torch.nn.BatchNorm2d(16)
        self.fc1 =   torch.nn.Linear(16 * 4 * 4, 120)
        self.fc2 =   torch.nn.Linear(120, 84)
        self.fc3 =   torch.nn.Linear(84, 10)
        self.relu =  torch.nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.batch_Norm1(self.conv1(x))))
        x = self.pool(self.relu(self.batch_Norm2(self.conv2(x))))
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x;
    
class MyDataset(Dataset):
    
    def __init__(self, data, transform = None):
        self.transform = transform
        self.label, self.img = xy.iloc[:, [0]].values, xy.iloc[:, 1:].values
        self.img = self.img.reshape(-1, 28, 28, 1).astype('float32')
        
        
    def __getitem__(self, index):
        label, img = self.label[index], self.img[index]
        if self.transform is not None:
            img = self.transform(img)
        return label, img

    def __len__(self):
        return len(self.label)

my_transform = transforms.Compose([transforms.ToTensor()])    
x_train = MyDataset(xy, transform=my_transform)
x_test = MyDataset(xy_test, transform=my_transform)

train_dataloader = DataLoader(dataset=x_train,batch_size=100, shuffle=False)
test_dataloader = DataLoader(dataset=x_test,batch_size=100, shuffle=False)

model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
epoch_loss = []
batch_ids = []
count = 0

for epoch in range(20):
    total_loss = 0.0
    for batch_id, (label, image) in enumerate(train_dataloader):
        y_pred = model(image)
        loss = criterion(y_pred,label.squeeze_())
        total_loss += loss.item()
        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        if batch_id % 100 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_id + 1, loss.item()))
            running_loss = 0.0
            batch_ids.append(count)
            epoch_loss.append(loss.item())
            count +=1
            
print(batch_ids)
print(epoch_loss)
    
plt.plot(batch_ids, epoch_loss)
plt.xlabel('bactch No.')
plt.ylabel('loss')
plt.title('Loss v/s batch')
plt.grid(True)
plt.show()

with torch.no_grad():
    correct = 0
    total = 0
    for i, (label, image) in enumerate(test_dataloader):
        y_pred_test = model(image)
        predicted = torch.argmax(y_pred_test,dim=1)
        total += label.size(0)
        correct += (predicted == label.squeeze_()).sum().item()
    print('Test Accuracy : %.3f' % (100 * correct / total))
