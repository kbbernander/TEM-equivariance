#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 2024

@author: karl bengtsson bernander
"""

import torch
import matplotlib.pyplot as pyplot
import pickle

from e2cnn import gspaces
from e2cnn import nn

print("E2 CNN, VGG16 simplified version")

class EquivariantCNN(torch.nn.Module):
    
    def __init__(self, n_classes=8):
        	
        super(EquivariantCNN, self).__init__()
        
        # Use TrivialOnR2(fibergroup=None) and Trivial representations for ordinary convolutions. FlipRot2DOnR2 uses 90 degree rotations and flips.
        self.r2_act = gspaces.FlipRot2dOnR2(N=4)
        
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        
        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 16 feature fields, each transforming under the regular representation of D8
        # Batch normalization is turned off
        out_type = nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr])
        self.block1 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=1, padding=1, maximum_offset=0),
            #nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 16 regular feature fields of D8
        out_type = nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr])
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=1, padding=1, maximum_offset=0),
            #nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool1 = nn.SequentialModule(
            nn.PointwiseMaxPool(out_type, kernel_size=2, stride=2, padding=0)
        )
        
        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the third convolution layer are 32 regular feature fields of D8
        out_type = nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr])
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, maximum_offset=0),
            #nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fourth convolution layer are 32 regular feature fields of D8
        out_type = nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr])
        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, maximum_offset=0),
            #nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool2 = nn.SequentialModule(
            nn.PointwiseAvgPool(out_type, kernel_size=2, stride=2, padding=0)
        )
        
        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block4.out_type
        # the output type of the fifth convolution layer are 64 regular feature fields of D8
        out_type = nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, maximum_offset=0),
            #nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 6
        # the old output type is the input type to the next layer
        in_type = self.block5.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields of D8
        out_type = nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.block6 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, maximum_offset=0),
            #nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 7
        # the old output type is the input type to the next layer
        in_type = self.block6.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields of D8
        out_type = nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.block7 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, maximum_offset=0),
            #nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool3 = nn.SequentialModule(
            nn.PointwiseMaxPool(out_type, kernel_size=2, stride=2, padding=0)
        )
        
        # convolution 8
        # the old output type is the input type to the next layer
        in_type = self.block7.out_type
        # the output type of the sixth convolution layer are 128 regular feature fields of D8
        out_type = nn.FieldType(self.r2_act, 128*[self.r2_act.regular_repr])
        self.block8 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, maximum_offset=0),
            #nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 9
        # the old output type is the input type to the next layer
        in_type = self.block8.out_type
        # the output type of the sixth convolution layer are 128 regular feature fields of D8
        out_type = nn.FieldType(self.r2_act, 128*[self.r2_act.regular_repr])
        self.block9 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, maximum_offset=0),
            #nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )

        # convolution 10
        # the old output type is the input type to the next layer
        in_type = self.block9.out_type
        # the output type of the sixth convolution layer are 128 regular feature fields of D8
        out_type = nn.FieldType(self.r2_act, 128*[self.r2_act.regular_repr])
        self.block10 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, maximum_offset=0),
            #nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool4 = nn.SequentialModule(
           nn.PointwiseMaxPool(out_type, kernel_size=2, stride=2, padding=0)
       )
        
        # convolution 11
        # the old output type is the input type to the next layer
        in_type = self.block10.out_type
        # the output type of the sixth convolution layer are 128 regular feature fields of D8
        out_type = nn.FieldType(self.r2_act, 128*[self.r2_act.regular_repr])
        self.block11 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=1, padding=0, maximum_offset=0),
            #nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 12
        # the old output type is the input type to the next layer
        in_type = self.block11.out_type
        # the output type of the sixth convolution layer are 128 regular feature fields of D8
        out_type = nn.FieldType(self.r2_act, 128*[self.r2_act.regular_repr])
        self.block12 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=1, padding=0, maximum_offset=0),
            #nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 13
        # the old output type is the input type to the next layer
        in_type = self.block12.out_type
        # the output type of the sixth convolution layer are 128 regular feature fields of D8
        out_type = nn.FieldType(self.r2_act, 128*[self.r2_act.regular_repr])
        self.block13 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=1, padding=0, maximum_offset=0),
            #nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool5 = nn.SequentialModule(
            nn.PointwiseMaxPool(out_type, kernel_size=2, stride=1, padding=0)
        )
        in_type = self.block13.out_type
        self.gpool = nn.GroupPooling(in_type)
        
        # number of output channels
        c = self.gpool.out_type.size
        print(c)
        
        # Fully Connected
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(c, 4096),
            #torch.nn.BatchNorm1d(4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, 4096),
            #torch.nn.BatchNorm1d(4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, n_classes),
        )
    
    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = nn.GeometricTensor(input, self.input_type)
        
        # apply each equivariant block
        
        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
        
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.pool3(x)

        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.pool4(x)

        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.pool5(x)
        
        # pool over the group. Remove this for ordinary CNNs.
        x = self.gpool(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor
        
        # classify with the final fully connected layers)
        #print(x.shape)
        x = self.fully_net(x.reshape(x.shape[0], -1))
        
        return x
    
from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation
from torchvision.transforms import Pad
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose

import torchvision.transforms.functional as TF
import random


import numpy as np

from PIL import Image

import time

current_GMT = time.time()
print("Time: ", current_GMT)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    print ("cuda to the rescue")
else:
    print ("no cuda :(")

class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

# For the MNIST dataset, if available
class MnistRotDataset(Dataset):
    
    def __init__(self, mode, transform=None):
        assert mode in ['train', 'test']
            
        if mode == "train":
            file = "mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat"
        else:
            file = "mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat"
        
        self.transform = transform

        data = np.loadtxt(file, delimiter=' ')
            
        self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32) 
        self.labels = data[:, -1].astype(np.int64)
        self.num_samples = len(self.labels)
    
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.labels)

totensor = ToTensor()

class BloodDataset(Dataset):
    
    def __init__(self, mode, transform=None):
        assert mode in ['train', 'test', 'validation']
            
        if mode == "train":
            file = "/mimer/NOBACKUP/groups/naiss2023-22-69/data/bloodMNIST/Cells_train_aug.txt" 
        elif mode == "test": 
            file = "/mimer/NOBACKUP/groups/naiss2023-22-69/data/bloodMNIST/Cells_test_noaug.txt"
        elif mode == "validation":
            file = "/mimer/NOBACKUP/groups/naiss2023-22-69/data/bloodMNIST/Cells_val_noaug.txt"    
        
        self.transform = transform

        data = np.loadtxt(file, delimiter=' ')

        if mode == "train":
            self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
            self.labels = data[:, -1].astype(np.int64)
            self.num_samples = len(self.labels)
        elif mode == "test":
            self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
            self.labels = data[:, -1].astype(np.int64)
            self.num_samples = len(self.labels)
        elif mode == "validation":
            self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
            self.labels = data[:, -1].astype(np.int64)
            self.num_samples = len(self.labels)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.labels)                     

totensor = ToTensor()

model = EquivariantCNN().to(device)

def test_model(model: torch.nn.Module, x: Image):
    # evaluate the `model` on 4 rotated versions of the input image `x`
    model.eval()
    
    print()
    print('##########################################################################################')
    header = 'angle |  ' + '  '.join(["{:6d}".format(d) for d in range(10)])
    print(header)
    with torch.no_grad():
        for r in range(4):
            x_transformed = totensor(x.rotate(r*90., Image.BILINEAR)).reshape(1, 1, 28, 28)
            x_transformed = x_transformed.to(device)

            y = model(x_transformed)
            
            angle = r * 90
            print("{:5d} : {}".format(angle, y))
    print('##########################################################################################')
    print()

    
# build the test set    
#mnist_test = MnistRotDataset(mode='test')
oral_test = BloodDataset(mode='test')

# retrieve the first image from the test set
x, y = next(iter(oral_test))

#print(model)

# evaluate the model
test_model(model, x)

rotation_transform = MyRotationTransform(angles=[0, 90, 180, 270])

train_transform = Compose([
    totensor,
])

validation_transform = Compose([
    totensor,
])

#mnist_train = MnistRotDataset(mode='train', transform=train_transform)
oral_train = BloodDataset(mode='train', transform=train_transform)
train_loader = torch.utils.data.DataLoader(oral_train, batch_size=1, shuffle=True)

oral_validation = BloodDataset(mode='validation', transform=validation_transform)
validation_loader = torch.utils.data.DataLoader(oral_validation, batch_size=1, shuffle=True)

test_transform = Compose([
    totensor,
])
#mnist_test = MnistRotDataset(mode='test', transform=test_transform)
oral_test = BloodDataset(mode='test', transform=test_transform)
test_loader = torch.utils.data.DataLoader(oral_test, batch_size=1, shuffle=True)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0)

print(model)

#for parameter in model.parameters():
#    print(parameter)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)

current_GMT = time.time()
print("Time: ", current_GMT)

nr_epochs = 100
xb = np.linspace(1 , nr_epochs, nr_epochs)
yb = np.linspace(1 , nr_epochs, nr_epochs)
zb = np.linspace(1 , nr_epochs, nr_epochs)

times = np.linspace(1 , nr_epochs, nr_epochs)

#conf_train = np.linspace(1 , nr_epochs, nr_epochs)
#conf_test = np.linspace(1 , nr_epochs, nr_epochs)

conf_train = np.zeros([8, 8])
conf_test = np.zeros([8, 8])
conf_validation = np.zeros([8, 8])

i_best=0
for epoch in range(nr_epochs):
    model.train()
    current_GMT = time.time()
    times[int(epoch)]=current_GMT
    print("Time: ", current_GMT)
    for i, (x, t) in enumerate(train_loader):

        optimizer.zero_grad()

        x = x.to(device)
        t = t.to(device)
        y = model(x)

        loss = loss_function(y, t)

        loss.backward()

        optimizer.step()
    
    if epoch % 1 == 0:
        confusion_matrix_test = torch.zeros(8, 8)
        total = 0
        correct = 0
        with torch.no_grad():
            model.eval()
            for i, (x, t) in enumerate(test_loader):

                x = x.to(device)
                t = t.to(device)
                y = model(x)

                _, prediction = torch.max(y.data, 1)
                total += t.shape[0]
                correct += (prediction == t).sum().item()
                
                for a, b in zip(t.view(-1), prediction.view(-1)):
                    confusion_matrix_test[a.long(), b.long()] += 1
        print(f"epoch {epoch} | test accuracy: ")
        print(f" {correct/total*100.}")
        #print(f"sensitivity: {confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])}")
        #print(f"specificity: {confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])}")
        #print(confusion_matrix)
        yb[int(epoch)]=(correct/total*100)
        if((correct/total*100)>i_best):
            i_best = correct/total*100
        confusion_matrix_train = torch.zeros(8, 8)
        total = 0
        correct = 0
        with torch.no_grad():
            model.eval()
            for i, (x, t) in enumerate(train_loader):

                x = x.to(device)
                t = t.to(device)
                y = model(x)

                _, prediction = torch.max(y.data, 1)
                total += t.shape[0]
                correct += (prediction == t).sum().item()
                
                for a, b in zip(t.view(-1), prediction.view(-1)):
                    confusion_matrix_train[a.long(), b.long()] += 1
        print(f"epoch {epoch} | train accuracy:")
        print(f" {correct/total*100.}")
        #print(f"sensitivity: {confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])}")
        #print(f"specificity: {confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])}")
        #print(confusion_matrix)
        xb[int(epoch)]=(correct/total*100)

        confusion_matrix_validation = torch.zeros(8, 8)
        total = 0
        correct = 0
        with torch.no_grad():
            model.eval()
            for i, (x, t) in enumerate(validation_loader):

                x = x.to(device)
                t = t.to(device)
                y = model(x)

                _, prediction = torch.max(y.data, 1)
                total += t.shape[0]
                correct += (prediction == t).sum().item()
            
                for a, b in zip(t.view(-1), prediction.view(-1)):
                    confusion_matrix_validation[a.long(), b.long()] += 1
        print(f"epoch {epoch} | validation accuracy:")
        print(f" {correct/total*100.}")
        #print(f"sensitivity: {confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])}")
        #print(f"specificity: {confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])}")
        #print(confusion_matrix)
        zb[int(epoch)]=(correct/total*100)

        conf_train = np.concatenate((conf_train,confusion_matrix_train.detach().cpu().numpy()))
        conf_test = np.concatenate((conf_test,confusion_matrix_test.detach().cpu().numpy()))
        conf_validation = np.concatenate((conf_validation,confusion_matrix_validation.detach().cpu().numpy()))


current_GMT = time.time()
print("Time: ", current_GMT)

print(f"this was the best result {i_best}")

# build the test set    
#other_mnist_test = MnistRotDataset(mode='test')
other_oral_test = BloodDataset(mode='test')

# retrieve the first image from the test set
x, y = next(iter(other_oral_test))


# evaluate the model
test_model(model, x)

#with open('results/GCNN_BN0_accuracy_run1.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#    pickle.dump([xb, yb], f)

#with open('results/GCNN_BN0_confusionmatrixtrain_run1.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#            pickle.dump(conf_train, f)
    
#with open('results/GCNN_BN0_confusionmatrixtest_run1.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#            pickle.dump(conf_test, f)  

#with open('results/GCNN_BN0_confusionmatrixtimes_run1.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#            pickle.dump(times, f)  

print(confusion_matrix_test)