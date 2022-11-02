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

def test(model, test_loader):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    pass

def train(model, train_loader, criterion, optimizer):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    pass
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    pass

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    
    # 320 * 240 Image shape
    
    
    pass


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
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = None
    optimizer = None
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, train_loader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
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
    
    args=parser.parse_args()
    
    main(args)
