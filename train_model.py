#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse

# Other Dependencies
import boto3


#TODO: Import dependencies for Debugging andd Profiling
import smdebug.pytorch as smd

def test(model, test_loader, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    for data, label in test_loader:
        # send data, model and label to use gpu device or cpu device - pass that in as the parameter
        model.eval()
        # work only when grads are set to zero
        output = model(data)
        cost = criterion(label, ouput)
        
        # calculate accuracy and loss etc
    pass

def train(model, train_loader, validation_loader, device, criterion, optimizer):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    for data, label in train_loader:
        # send data, model and label to use gpu device or cpu device - pass that in as the parameter
        model.train()
        optimizer.zero_grad()
        output = model(data)
        cost = criterion(label, ouput)
        optimizer.backward()
        optimizer.step()
        
        # calculate accuracy and loss etc
        
        # calculate on validation data set and think about early stopping etc!!
        model.eval()
    
    pass
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    # Seems like another version of pytorch does it differently, where this parameter is deprecated
    model = models.resnet50(pretrained=True)
    number_of_classes = 4
    fc_input_features_count = model.fc.in_features
    last_fc = nn.Sequential(nn.linear(fc_input_features, number_of_classes)
    model.fc = last_fc
    
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    
    # 320 * 240 Image shape
    
    
    return train_loader, validation_loader, test_loader


class BloodCellDataset(torch.utils.data.Dataset):
    def __init__(self, s3_dataset_path, local_data_dir):
        self.s3_dataset_path = s3_dataset_path
        self.local_data_dir = local_data_dir
        
        # Must I do some of this work like create .lst files etc outside and place in s3 already, while training job just uses that or must that work also be performed here
        # or is it fine I have not annotation - metadata - csv file etc and just do it form the images right here
        # i.e is pattern to do E of ETL - extraction outside the training script, in the submission script or the jupyter notebook?
        # What about the costs involved???
        # Try to name variables just like other dataset class and class we inherting from
        # conventions - to be easy to use
        # pickeled file?
        # downloading one file at a time using boto 3 client may not be great!!!
        # this means best option would be to sync using cli - prior to trainign script!!!
    
        s3 = boto3.client('s3')
        s3.download_file
        self.images = 
        
        self.data = 
        self.target
        


def main(args):
                            
    hook = smd.get_hook(create_if_not_exists=True)
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
                            
    hook.register_module(model)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = torch.NNLoss()
    # Is this the correct place to do this?
    hook.register_loss_(loss_criterion)
    # Some are parameters
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    device = ""
    if asdfas:
       pass
    else:
        pass
    
                            
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    hook.set_mode(ModeKeys.TRAIN)
    model=train(model, train_loader, validation_loader, device, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    hook.set_mode(ModeKeys.EVAL)
    test(model, test_loader, criterion)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    # Hyperparametrs received below etc
    args.add_argument()
    args=parser.parse_args()
    
    main(args)
