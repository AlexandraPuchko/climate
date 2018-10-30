# Base class for all neural network modules.
import torch
import numpy as np
import torch.nn as nn
import time
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from utils import plot_images


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assume that we are on a CUDA machine, then this should print a CUDA device:
print(device)



 # Example implementaion of Simple CNN on CIFAR Dataset

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
     #Input channels = 3(RGB), output channels = 18(# of filters) (18 feature maps)
     self.conv1 = nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
     self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

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



# def outputSize(in_size, kernel_size, stride, padding):
#     output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
#
#     return(output)


# Data loader. Combines a dataset and a sampler,
# and provides single- or multi-process iterators over the dataset.
def get_train_val_loader(batch_size, set, sampler, show_sampler):
    train_loader = torch.utils.data.DataLoader(set, batch_size=batch_size,
                                           sampler=sampler, num_workers=2)


    if show_sampler :
        data_iter = iter(train_loader)
        images, labels = data_iter.next()
        X = images.numpy().transpose([0, 2, 3, 1])
        plot_images(X, labels)
    return(train_loader)



def createLossAndOptimizer(net, learning_rate=0.001):
    loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    return(loss, optimizer)


def trainNet(net, train_loader, val_loader,train_batch_size, n_epochs, learning_rate):
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", train_batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)


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

            #inputs, labels = inputs.to(device), labels.to(device) //if with cuda

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
    train_set, test_set = export_data('./cifardata')

    #Training, CIFAR has 60.000 training samples
    n_training_samples = 20000
    # define how much we want to sample out of a dataset
    train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))

    #Validation
    n_val_samples = 5000
    val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))

    #Test
    n_test_samples = 5000
    test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))

    show_train_sampler = False
    train_loader = get_train_val_loader(32,train_set,train_sampler,show_train_sampler)

    show_val_sampler = False
    val_loader = get_train_val_loader(128,train_set,val_sampler,show_val_sampler)
    #test_loader = get_loader(4,test_set,test_sampler)

    CNN = SimpleCNN()

    #net.to(device)  //if with cuda
    trainNet(CNN, train_loader, val_loader,train_batch_size=32, n_epochs=5, learning_rate=0.001)

# Run
main()
