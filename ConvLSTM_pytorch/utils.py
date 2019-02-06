import netCDF4 as n
import argparse
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from convLSTM import ConvLSTM
import matplotlib.pyplot as plt
from train import trainNet
import chocolate as choco
import random




def parse_all_args():
     """
     Parses arguments

     :return: Argument object
     """
     parser = argparse.ArgumentParser()

     # Positional Arguments
     parser.add_argument("netcdf",
                         help="Data set(a nc)")

    #  # Model Flags
     parser.add_argument("-patience",
                         help="How many epochs to continue training without improving\
         dev accuracy (int)",
                         type=int,
                         default=10)
     parser.add_argument("-lr",
                         type=float,
                         help="The learning rate (a float) [default=0.001]",
                         default=0.001)
     parser.add_argument("-mb",
                         type=int,
                         help="The minibatch size (an int) [default=1]",
                         default=1)
     parser.add_argument("-num-steps",
                         type=int,
                         help="The number of steps to unroll for Truncated BPTT (an int) [default=20]",
                         default=20)
     parser.add_argument("-max-len",
                         type=int,
                         help="The maximum length of a sequence (an int) [default=12]",
                         default=24)#2 * 12 months = 2 years
     parser.add_argument("-epochs",
                         type=int,
                         help="The number of epochs to train for (an int) [default=20]",
                         default=20)
     parser.add_argument("-normalize",
                         type=str,
                         choices=["log"],
                         help="Set normalization scheme. Choice must be in the set {log}")
    #  parser.add_argument("-area_weighted",
    #                      action='store_true',
    #                      help="Train using area-weighted MSE loss function.")

     # Output Flags
     parser.add_argument("-dev_preds", type=str,
                         help="Model's predictions on the dev set, exported as a NetCDF",
                         default="../../outputs/dev_predictions.nc")
     parser.add_argument("-dev_truths", type=str,
                         help="Grounds truths of the dev set, exported to NetCDF",
                         default="../../outputs/dev_truths.nc")
     parser.add_argument("-test_preds", type=str,
                         help="Model's predictions on the test set, exported as a NetCDF",
                         default="../../outputs/test_predictions.nc")
     parser.add_argument("-model",
                         type=str,
                         help="Save the best model with this prefix (string)",
                         default="/tmp/model.ckpt")

     return parser.parse_args()




def split_data(pr, nc_time, norm_type, max_len):
     ins, times = ([] for i in range(2))

     # Compute bounds for train/dev/test
     num_seqs = (len(pr)) // max_len
     train_len = int(round(0.7 * num_seqs))

     if ((num_seqs - train_len) % 2) != 0:
         train_len += 1
     test_len = dev_len = int((num_seqs - train_len) / 2)

     assert ((train_len + test_len + dev_len) == num_seqs)

     # Normalize before split
     if norm_type == "log":
         pr = log_normalize(pr, train_len)

     # Create sequences.
     # TODO: Create 'lens' list to pass to 'sequence_length' parameter in dynamic_rnn to handle the last few data points.
     num_seqs = (len(pr)) // max_len
     for i in range(num_seqs):
         ins.append(pr[i*max_len:(i+1)*max_len])
         times.append(nc_time[i * max_len:(i + 1) * max_len])

     inputs = np.asarray(ins, dtype=np.float32)# Precipitation
     inputs = inputs[:, :, np.newaxis,:, :]  # Adding 'channel' dimension to conform to ConvLSTM cell.
     times = np.asarray(times, dtype=np.float32)

     # train (70%)
     train_seqs = inputs[0:train_len]
     train_times = times[0:train_len]

     # dev (15%)-> concatenate all inputs into one long dev
     dev_ub = (train_len + dev_len)
     dev_seqs = inputs[train_len:dev_ub]
    #  dev_merged = []
    #  for i in range(0,len(dev_seqs)):
    #      dev_merged.extend(dev_seqs[i])
    #  dev_seqs = dev_merged

     dev_times = times[train_len:dev_ub]

     # test (15%)
     test_seqs = inputs[dev_ub:(num_seqs*max_len)] # Double check that this is the appropriate index
     test_times = times[dev_ub:(num_seqs*max_len)]

     return train_seqs, dev_seqs, test_seqs



def log_normalize(pr, train_len):
     global TRAIN_MAX, TRAIN_MIN, MEDIAN_SHIFT_FACTOR

     pr += ZERO_SHIFT_FACTOR
     pr = np.log2(pr)
     TRAIN_MIN = pr[0:train_len].min()
     pr -= TRAIN_MIN
     TRAIN_MAX = pr[0:train_len].max()
     pr /= TRAIN_MAX
     pr *= 2
     MEDIAN_SHIFT_FACTOR = (pr.max() - pr.min()) / 2
     pr -= MEDIAN_SHIFT_FACTOR

     return pr



def log_denormalize(values):
     values += MEDIAN_SHIFT_FACTOR
     values /= 2
     values *= TRAIN_MAX
     values += TRAIN_MIN
     values = np.power(2, values)
     values -= ZERO_SHIFT_FACTOR

     return values


def export_netCDF(z, nc, filename, devtime):
     dataset = n.Dataset(filename, 'w', format='NETCDF4_CLASSIC')
     latD = dataset.createDimension('lat', 64)
     lonD = dataset.createDimension('lon', 128)
     timeD = dataset.createDimension('time', None)

     # create netCDF output
     #  A variable represents an array of values of the same type.
     latOut = dataset.createVariable('latitude', np.float32, ('lat',))
     longOut = dataset.createVariable('longitude', np.float32, ('lon',))
     timeOut = dataset.createVariable('time', np.float64, ('time',))
     # variables may be multidimensional
     prOut = dataset.createVariable('pr', np.float32, ('time', 'lat', 'lon'))

     # Variable Attributes
     # Attributes are used to store data about the data (ancillary data or metadata)
     latOut.units = nc.variables['lat'].units
     longOut.units = nc.variables['lon'].units
     timeOut.units = nc.variables['time'].units
     prOut.units = nc.variables['pr'].units

     # test write
     latOut[:] = nc.variables['lat'][:]
     longOut[:] = nc.variables['lon'][:]
     timeOut[:] = devtime

     # writing out one month
     prOut[:] = z
     dataset.close()

def createLossAndOptimizer(net, learning_rate):

    loss = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), learning_rate)
    return(loss, optimizer)

def generate_params():
    layer = random.randint(2, 30)
    epoch = random.randint(50, 150)
    hidden_dim_param = []
    start_pow = random.choice([1, 2, 3, 4])#do not include 32 for start, otherwise
                                           # all of the values in the sequence will have to be 32
    end_pow = 5
    hidden_dim_param.append(2 ** start_pow)
    layer = 4
    # randomly generate increasing sequence of hidden_dim size
    # based on the value of layer
    for i in range(layer - 1):
        rand_pow = random.randint(start_pow, end_pow)
        start_pow = rand_pow
        hidden_dim_param.append(2 ** rand_pow)


    return layer, hidden_dim_param, epoch



def main():

    # parse arguments
    args = parse_all_args()

     # parse netCDF data
    nc = n.Dataset(args.netcdf)
    time = nc.variables['time'][:]
    pr = nc.variables['pr'][:]
    channels = 1 #precipitation value


    '''
    Hyper-parameters:
    -arg.patience : 10
    -hidden_dim_param[2,4,8,16,32]
    -num of layers[2:20]
    -num of epochs[50-150]
    '''


    #Load sequences
    train_seqs, dev_seqs, test_seqs = split_data(pr, time, args.normalize, args.max_len)
    print('Finished loading and splitting data.')

    #run 50 experiments
    for exp_id in range(0, 50):
        layer, hidden_dim_param, epochs = generate_params()
        print(layer, hidden_dim_param, epochs)

        convLSTM = ConvLSTM(input_size=(64, 128),input_dim=channels,hidden_dim=hidden_dim_param,kernel_size=(3, 3),num_layers=layer)
        #use GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        convLSTM = convLSTM.to(device)
        loss, optimizer = createLossAndOptimizer(convLSTM, learning_rate=args.lr)
        trainNet(exp_id, convLSTM, loss, optimizer,train_seqs, dev_seqs, test_seqs,args, device, epochs, plot=False)


if __name__ == "__main__":
     main()
