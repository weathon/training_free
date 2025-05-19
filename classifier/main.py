import wandb
import torch
import torch.nn as nn
import os
import random
import numpy as np


class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, root_dir):
        self.data = data
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        latent = torch.load(os.path.join(self.root_dir, self.data[idx]), map_location='cpu')
        if type(latent) == dict:
            latent = latent['ldist'][0,:12]
        else:
            latent = latent[0]
            
        noise = torch.randn_like(latent)
        t = random.random() 
        latent = latent * (1 - t) + noise * t
        label = 0 if not self.data[idx].endswith('.txt.pt') else 1
        return latent.to(torch.float32), label
    

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.pooling1 = lambda x: torch.mean(x, dim=2)
        self.conv1 = nn.Conv2d(12, 48, kernel_size=3)
        self.pooling2 = nn.AvgPool2d(2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(48, 64, kernel_size=3)
        self.pooling3 = lambda x: torch.mean(x, dim=(2, 3))
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.pooling1(x)
        x = self.conv1(x)
        x = self.pooling2(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.pooling3(x)
        x = self.relu2(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x

classifier = Classifier()
files = os.listdir('res')
random.shuffle(files)
train = files[:int(len(files) * 0.8)]
val = files[int(len(files) * 0.8):]

train_dataset = LatentDataset(train, train, 'res')
val_dataset = LatentDataset(val, val, 'res')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
print(len(train_loader), len(val_loader))
print(train_dataset[0][0].shape)
print(classifier(train_dataset[0][0].unsqueeze(0)).shape)

wandb.init(project="latent_classifier")
criterion = nn.BCELoss()

classifier = classifier.to('cuda')
optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.001)
for epoch in range(100):
    classifier.train()
    for i, (data, label) in enumerate(train_loader):
        data = data.to('cuda')
        label = label.to('cuda')
        optimizer.zero_grad()
        output = classifier(data)
        loss = criterion(output.squeeze(), label.float())
        loss.backward()
        optimizer.step()
        wandb.log({"loss": loss.item()})
    
    classifier.eval()
    with torch.no_grad():
        val_loss = 0
        for data, label in val_loader:
            data = data.to('cuda')
            label = label.to('cuda')
            output = classifier(data)
            loss = criterion(output.squeeze(), label.float())
            val_loss += loss.item()
    print(f'Validation Loss: {val_loss / len(val_loader)}')
    wandb.log({"val_loss": val_loss / len(val_loader)})
