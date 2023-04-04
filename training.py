#%%
import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sklearn
from sklearn.metrics import classification_report

def train_model(project_path, dataset_path):
    repo = 'OxWearables/ssl-wearables'
    harnet30 = torch.hub.load(repo, 'harnet30', class_num=5, pretrained=True)

    classification_head = torch.nn.Sequential(torch.nn.Linear(1024,1024),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(1024,1024),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(1024, 512),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(512, 7))

    # Replace classifier by appropriate size one
    my_model = torch.nn.Sequential(*list(harnet30.children())[:-1], torch.nn.Flatten(), classification_head)
    # Freeze ResNet model weights
    list(my_model.children())[0].requires_grad_(False)
    list(my_model.children())[1].requires_grad_(True)
    list(my_model.children())[2].requires_grad_(True)

    print(my_model)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    #%%
    # Load data
    class SleepStagesDataset(Dataset):
        def __init__(self, annotations_csv, data_csv, transform=None):
            # annotations_csv is expected to contain filename and label for every data point
            self.annotations = pd.read_csv(annotations_csv)
            self.data = pd.read_csv(data_csv)
            self.transform = transform

        def __len__(self):
            return int(self.annotations.shape[0])

        def __getitem__(self, idx):
            label = self.annotations.iloc[idx, 1] + 1  # Add 1 so we do not have negative labels
            sample_start_index = self.annotations.iloc[idx, 0]
            data = self.data.iloc[sample_start_index:(sample_start_index+900),:].to_numpy().transpose()
            if self.transform:
                data = self.transform(data)
            return data, label


    index_path = os.path.join(dataset_path, 'index_array.csv')
    data_path = os.path.join(dataset_path, 'motion_data_downsampled_all.csv')

    dataset = SleepStagesDataset(index_path, data_path, transform=torch.FloatTensor)
    # Split data into train and test
    generator = torch.Generator().manual_seed(42)
    train, test = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=generator)
    train_dataloader = DataLoader(train, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=64, shuffle=True)

    print(dataset.__getitem__(0)[0].shape)
    #%%


    def train_loop(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def test_loop(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        pred_df = []

        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                # Per class predictions
                pred_df.extend(pred.argmax(1).type(torch.int).tolist())





        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


        pred_df = pd.DataFrame(data=pred_df, columns=['Predicted Class'])
        pred_df['count'] = 1
        pred_df = pred_df.groupby('Predicted Class').count()
        print(pred_df)


    learning_rate = 1e-3
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)

    my_model.to(device)

    epochs = 250
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, my_model, loss_fn, optimizer)
        test_loop(test_dataloader, my_model, loss_fn)
        if t % 20 == 0:
            print('Saving model...')
            filename = "model"+str(t)+".pth"
            save_path = os.path.join(dataset_path, filename)
            print(save_path)
            torch.save(my_model.state_dict(), save_path)
            # Test per class accuracy of my_model on test set
            with torch.no_grad():
                complete_test_dataloader = DataLoader(test, batch_size=1024, shuffle=True)
                i=0
                for X, y in complete_test_dataloader:
                    X = X.to(device)
                    y = y.to(device)
                    pred = my_model(X)
                    pred = pred.argmax(1).type(torch.int).tolist()
                    y = y.type(torch.int).tolist()
                    print(classification_report(y, pred, digits=4))
                    i = i+1
                    if i>1:
                        break

    print("Done!")
