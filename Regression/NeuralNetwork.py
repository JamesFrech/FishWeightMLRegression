import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from NeuralNetworkModel import TorchDataset, NeuralNetwork
import matplotlib.pyplot as plt
from datetime import datetime

import torch
torch.manual_seed(42)

# Read in the data
data = pd.read_csv('../data/Fish_dataset.csv')
data.drop('Species',axis=1,inplace=True) # Dont use species

# Select model inputs
target='Weight'
inputs=['Length1','Length2','Length3','Height','Width']

# Split the training and testing data.
X_train, X_test, y_train, y_test = train_test_split(
    data[inputs], data[target], test_size=0.2, random_state=42)

# Scale the data for faster convergence
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

# convert pandas dataframes and numpy arrays into PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values.squeeze(), dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values.squeeze(), dtype=torch.float32)

# Initialize model
model = NeuralNetwork(n_inputs=len(inputs),n_outputs=1)

# Create torch datasets and dataloaders
train_data = TorchDataset(X_train, y_train)
test_data = TorchDataset(X_test, y_test)

trainloader = DataLoader(train_data, batch_size=64, shuffle=False)
testloader = DataLoader(test_data, batch_size=64, shuffle=False)

# Define loss function and optimizer
# MSE loss
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)
#scheduler = StepLR(optimizer, step_size=300, gamma=0.1)

# Training the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Choose number epochs
n_epoch=600

# Initialize loss and mae lists
loss_epoch=[]
train_mae_epoch=[]
test_mae_epoch=[]

for epoch in range(n_epoch):  # Loop over the dataset multiple times
    start=datetime.now()
    print(f'epoch {epoch}')
    running_loss = 0.0

    mae_train = 0.0
    mae_test = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, targets = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Calculate the loss
        running_loss += loss.item()
        mae_train += sum(abs(outputs - targets))

    mae_train = mae_train/X_train.shape[0]

    # Testing the model
    with torch.no_grad():
        for data in testloader:
            inputs, targets = data[0].to(device), data[1].to(device)
            outputs = model(inputs).squeeze()

            mae_test += sum(abs(outputs - targets))

    mae_test=mae_test/X_test.shape[0] # Take the mean of the summed absolute errors
    print(f'Train mae: {mae_train}')
    print(f'Test mae: {mae_test}')

    loss_epoch.append(running_loss)
    train_mae_epoch.append(mae_train.detach().to('cpu'))
    test_mae_epoch.append(mae_test.detach().to('cpu'))
    print(datetime.now()-start)

    #scheduler.step()

    # Delete loss and outputs to stop RAM from overloading
    del running_loss, mae_train, mae_test, outputs

print('Finished Training')

# Loss
plt.plot([i for i in range(len(loss_epoch))],loss_epoch)
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('images/NN_loss.png',bbox_inches='tight')
plt.close()

# Plot mae
train_mae_epoch = [i.mean() for i in train_mae_epoch]
test_mae_epoch = [i.mean() for i in test_mae_epoch]

plt.plot([i for i in range(n_epoch)],train_mae_epoch,label='train')
plt.plot([i for i in range(n_epoch)],test_mae_epoch,label='validation')
plt.legend()
plt.title('Train and Validate MAE per Epoch')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.savefig('images/NN_MAE.png',bbox_inches='tight')
plt.close()

# Final train predictions
pred_train = torch.tensor([])
with torch.no_grad():
    model.eval()  # Set the model to evaluation mode
    for i, data in enumerate(trainloader):
            inputs, targets = data[0].to(device), data[1].to(device)
            prediction = model(inputs).squeeze()
            pred_train = torch.cat([pred_train,prediction],axis=0)

# Final test predictions
# Pass the test point through the model
pred_test = torch.tensor([])
with torch.no_grad():
    model.eval()  # Set the model to evaluation mode
    for i, data in enumerate(testloader):
            inputs, targets = data[0].to(device), data[1].to(device)
            prediction = model(inputs).squeeze()
            pred_test = torch.cat([pred_test,prediction],axis=0)

# Save the model weights
torch.save(model.state_dict(),'NN_weights.pth')

# Calculate the RMSE
train_rmse=np.sqrt(np.mean((pred_train.numpy()-y_train.numpy())**2))
test_rmse=np.sqrt(np.mean((pred_test.numpy()-y_test.numpy())**2))

print('Train RMSE:',train_rmse)
print('Test RMSE:',test_rmse)

metrics = pd.DataFrame([['NeuralNet',train_rmse,test_rmse]],
                        columns=['Model','TrainRMSE','TestRMSE'])
metrics.to_csv('metrics/NeuralNetwork.csv',index=False)
