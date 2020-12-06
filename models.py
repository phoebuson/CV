## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # input image is 224 x 224, output size is 224-5+1 = 220; 
        self.conv1 = nn.Conv2d(1, 32, 5)    
        
        # maxpooling layers; output size is 32*110*110
        self.pool = nn.MaxPool2d(2, 2)
                
        # convolutional layer 2; output size is 64*106*106 (110-5+1 = 106); after pooling: 64*53*53
        self.conv2 = nn.Conv2d(32,64,5)
        
        # convolutional layer 3; output 128*51*51;  (53-3+1 = 51); after pooling: output size is 128*25*25 (51/2 round down to 25) 
        self.conv3 = nn.Conv2d(64,128,3)
        
        # convolutional layer 4; output: 256*23*23 (25-3+1 = 23); after pooling: output size is after pooling 256*11*11
        self.conv4 = nn.Conv2d(128,256,3)
        
        # convolutional layer 5; output: 512*11*11 (11-1+1 = 11); after pooling: output size is after pooling 512*5*5
        self.conv5 = nn.Conv2d(256,512,1)
        
        # fully connected layer
        self.fc1 = nn.Linear(512*5*5, 4000)
        self.fc2 = nn.Linear(4000, 4000)
        self.fc3 = nn.Linear(4000, 4000)
        self.fc4 = nn.Linear(4000, 68*2)
        
        # drop out layer
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        # convolutional layers   
        x = self.pool(F.relu(self.conv1(x)))
        #x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        #x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        #x = self.dropout(x)
        x = self.pool(F.relu(self.conv4(x)))
        #x = self.dropout(x)
        x = self.pool(F.relu(self.conv5(x)))
        #x = self.dropout(x)
        
        # prep for linear layer by flattening the feature maps into feature vectors
        x = x.view(x.size(0), -1)
        
        # dense layers 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        
        return x
