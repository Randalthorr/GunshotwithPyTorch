import AudioUtil
import pandas
from pathlib import Path
import os
from DataLoader import SoundDS
from torch.utils.data import random_split
import torch
from AudioClassifier import AudioClassifier
import Train
import Inference
from torch import nn


if __name__ == '__main__':

    mode = input("Enter Mode: (0: Load mode, 1: Save Mode, 2: Dont Save)")
    if mode == '1' or mode == '2':
        dataset_csv = AudioUtil.askFile()
        df = pandas.read_csv(dataset_csv[0])

        df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)

        df = df[['relative_path', 'classID']]

        continue_selecting = 'y'
        while continue_selecting == 'y':
            dataset_csv = AudioUtil.askFile()
            dftemp = pandas.read_csv(dataset_csv[0])

            df = df[['relative_path', 'classID']]
            df = pandas.concat([df, dftemp], ignore_index=True)
            df.head()
            continue_selecting = input('Continue selecting? (y/n): ')

        # dataset_csv = AudioUtil.askFile()
        #
        # df_val = pandas.read_csv(dataset_csv[0])
        # df_val.head()
        #
        # df_val = df_val[['relative_path', 'classID']]
        # df.head()
        data_path = AudioUtil.askDir()
        # val_data_path = AudioUtil.askDir()

        myds = SoundDS(df, data_path)
        # val_ds = SoundDS(df_val, val_data_path)

        # Random split of 80:20 between training and validation
        # num_items = len(train_ds)
        # num_train = num_items
        # num_val = num_items - num_train
        # train_ds, dummy = random_split(train_ds, [num_train, num_val])
        # del dummy
        # num_items = len(val_ds)
        # num_train = num_items
        # num_val = num_items - num_train
        # val_ds, dummy = random_split(val_ds, [num_train, num_val])
        # del dummy
        # Trying process
        num_items = len(myds)
        num_train = round(num_items * 0.9)
        num_val = num_items - num_train
        train_ds, val_ds = random_split(myds, [num_train, num_val])

        # print(type(val_ds), type(train_ds), type(myds))
        # input()
        # Create training and validation data loaders7
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

        # Create the model and put it on the GPU if available // AUDIO CLASSIFIER
        myModel = AudioClassifier()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 0:
                dev = torch.cuda.current_device()
                print(torch.cuda.get_device_name(dev))
        myModel = nn.DataParallel(myModel)
        myModel = myModel.to(device)
        # Check that it is on Cuda
        next(myModel.parameters()).device

        # Training
        num_epochs = 20 # Just for demo, adjust this higher.
        Train.training(myModel, train_dl, num_epochs)
        Inference.inference(myModel, val_dl, device)
        if mode == '1':
            torch.save(myModel.state_dict(), r"C:\Users\furka\Desktop\Temp\fmin=1000_withAugmentfiles.pth")
            print("saved")
            input()
    elif mode == '0':
        model = AudioClassifier()
        model.load_state_dict(torch.load(r"C:\Users\furka\Desktop\Temp\fmin=1000_withAugmentfiles.pth"))
        model.eval()

        dataset_csv = AudioUtil.askFile()
        df_val = pandas.read_csv(dataset_csv[0])
        df_val.head()
        # df_val['relative_path'] = '/fold' + df_val['fold'].astype(str) + '/' + df_val['slice_file_name'].astype(str)
        df_val = df_val[['relative_path', 'classID']]
        val_data_path = AudioUtil.askDir()

        val_ds = SoundDS(df_val, val_data_path)
        num_items = len(val_ds)
        num_train = num_items
        num_val = num_items - num_train
        val_ds, dummy = random_split(val_ds, [num_train, num_val])
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        Inference.inference(model, val_dl, device)
