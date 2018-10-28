# Base class for all neural network modules.
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision
import time
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)



 # Example implementaion of Simple CNN on CIFAR Dataset
 # (recognize hand-written digits)

 #1) depth of generated activation map depends on how many
 # filters we are going to use

 # convolutional layers, maxpooling (avg pooling) - locally connected
 # layers is just feature extraction, whereas FC layers is classification

# Data is feature engineered using the SimpleCNN
class SimpleCNN(nn.Module):


 #Our batch shape for input x is (3, 32, 32)
 # define constructor
    def __init__(self):
     #inherit from Net
     super(SimpleCNN, self).__init__()
     #Input channels = 3(RGB), output channels = 18(# of filters)
     self.conv1 = nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
     self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
     # Applies a linear transformation to
     # the incoming data: :math:y = xA^T + b
     #params: input features size, output features size, bias
     # in our case 2nd param = 10, because 10 digits
     #self.fc = nn.Linear(320,10)#instead of 320 we can add any number, and then recompute

     #4608 input features, 64 output features (see sizing flow below)
     self.fc1=torch.nn.Linear(18 * 16 * 16, 64)

     #64 input features, 10 output features for our 10 defined classes
     self.fc2=torch.nn.Linear(64, 10)



    def forward(self,x):
        #Computes the activation of the first convolution
        #Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))

        #Size changes from (18, 32, 32) to (18, 16, 16)
        #(the first dimension, or number of feature maps,
        #remains unchanged during any pooling operation)
        x = self.pool(x)

        #Reshape data to input to the input layer of the neural net
        #Size changes from (18, 16, 16) to (1, 4608)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 18 * 16 *16)

        #Computes the activation of the first fully connected layer
        #Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))

        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        return(x)



def export_data(path):
    #The compose function allows for multiple transforms
    #1) transforms.ToTensor() converts our PILImage to a tensor of
    # shape (C x H x W) in the range [0,1]
    #2) transforms.Normalize(mean,std) normalizes a tensor to a
    # (mean, std) for (R, G, B)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform)

    test_set = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform)

    return(train_set, test_set)



def outputSize(in_size, kernel_size, stride, padding):

    output = int((in_size - kernel_size + 2*(padding)) / stride) + 1

    return(output)

#DataLoader takes in a dataset and a sampler for loading (num_workers deals with system level memory)
#If your model and data is small,
# it shouldn’t be a problem.
# Otherwise I would rather use the DataLoader
# to load and push the samples onto the GPU than to make my model smaller.
# Data loader. Combines a dataset and a sampler,
# and provides single- or multi-process iterators over the dataset.
def get_train_loader(batch_size, train_set):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           sampler=train_sampler, num_workers=2)
    return(train_loader)



def createLossAndOptimizer(net, learning_rate=0.001):
    #Loss function
    loss = torch.nn.CrossEntropyLoss()

    #Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    return(loss, optimizer)


def trainNet(net, batch_size, n_epochs, learning_rate,
            test_loader, val_loader,train_sampler,test_sampler,val_sampler):

    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    #export data
    train_set, test_set = export_data('./cifardata')

     #Get training data
    train_loader = get_train_loader(batch_size,train_set)
    n_batches = len(train_loader)

    #Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)

    #Time for printing
    training_start_time = time.time()

    #Loop for n_epochs
    for epoch in range(n_epochs):
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0

        for i, data in enumerate(train_loader, 0):
            #Get inputs
            inputs, labels = data

            #Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)

            #Set the parameter gradients to zero
            optimizer.zero_grad()

            #Forward pass, backward pass, optimize
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()


             #Print statistics
            running_loss += loss_size.data[0]
            total_train_loss += loss_size.data[0]

            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))

                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()


        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        for inputs, labels in val_loader:

            #Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)

            #Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.data[0]



def main():
    #Training
    n_training_samples = 20000
    train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))

    #Validation
    n_val_samples = 5000
    val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))

    #Test
    n_test_samples = 5000
    test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))

    #Test and validation loaders have constant batch sizes, so we can define them directly
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, sampler=test_sampler, num_workers=2)
    val_loader = torch.utils.data.DataLoader(train_set, batch_size=128, sampler=val_sampler, num_workers=2)

    CNN = SimpleCNN()
    trainNet(CNN, batch_size=32, n_epochs=5, learning_rate=0.001,test_loader, val_loader,train_sampler,test_sampler,val_sampler)
