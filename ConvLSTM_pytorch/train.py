import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
import pdb
import logging
import matplotlib.pyplot as plt


# NOTE: These constants assume the model converges around epoch 20.0
LIN_DECAY_CONST = (-1.0 / 20.0)


#Static decay functions
def linear(epoch):
    return max(0, 1 + (LIN_DECAY_CONST * epoch))

def update_epsilon(epoch):
    return linear(epoch)

def compute_decay_constants(epochs):
    """
    Computes the decay constants used for computing scheduled sampling epsilon value

    : param epochs: (int) NUmber of iterations over the training set.
    """
    global LIN_DECAY_CONST

    LIN_DECAY_CONST
    LIN_DECAY_CONST = -1.0/float(epochs)


def plotMAE(seq_len, mae, std):
    plt.plot(seq_len, mae, 'r--', seq_len, std, 'g^')
    plt.xlabel('Sequence length')
    plt.ylabel('Mean, Std')
    plt.show()



def evaluateNet(net, loss, dev_x, dev_y, prev_hidden_states, device):
    print('Evaluating on dev set...')
    #1) feed model with a hidden states from the training mode
    #2) do pass through all data in a dev set
    #seq_len = len(dev_x)
    seq_len = dev_x.size(1)
    next_hidden_state = prev_hidden_states

    #create matrix of losses and first init all values to 0
    losses = [[-1 for x in range(seq_len)] for y in range(seq_len)]
    mae = []
    std = []

    for step in range(0, seq_len):
        #get new hidden states on every pass through the sequence
        seq_outputs, next_hidden_state = net.evaluate(dev_x, next_hidden_state, step,seq_len, device)
        dev_y = torch.squeeze(torch.tensor(dev_y),0)
        current_dev = dev_y[step:]

        for t in range(len(seq_outputs)):
            #compute loss for one datapoint in a sequence
            running_loss = loss(seq_outputs[t], current_dev[t,:,:,:])
            losses[step][t] = running_loss.data

    #compute mean and std
    mae = []
    std = []

    for col in range(seq_len):
        curr_std = []
        sum = 0
        for row in range(seq_len):
            if losses[row][col] != -1:
                curr_std.append(losses[row][col])
                sum += losses[row][col]
            else:
                break

        mae.append(sum / row)
        std.append(np.std(np.array(curr_std), axis = 0))


    return mae, std




def trainNet(net, loss, optimizer,train_seqs, dev_seqs, test_seqs,args, device, plot=False):

        print('Training started...')

        mb_row = 0
        row_start = mb_row*args.mb
        row_end = np.min([(mb_row+1)*args.mb, len(dev_seqs)]) # Last minibatch might be partial
        dev_x = torch.from_numpy(dev_seqs[row_start:row_end, 0:args.max_len-1])
        dev_y = torch.from_numpy(dev_seqs[row_start:row_end, 1:args.max_len]).to(device)
        #TODO: concatenate several dev_y to one
        for mb_row in range(1, int(np.floor(len(dev_seqs) / args.mb))):
            row_start = mb_row*args.mb
            row_end = np.min([(mb_row+1)*args.mb, len(dev_seqs)]) # Last minibatch might be partial
            dev_x_curr = torch.from_numpy(dev_seqs[row_start:row_end, 0:args.max_len-1])
            dev_x = torch.cat((dev_x, dev_x_curr), 1)#concatenate across time
            dev_y_curr = torch.from_numpy(dev_seqs[row_start:row_end, 1:args.max_len]).to(device)
            dev_y = torch.cat((dev_y, dev_y_curr), 1)

        dev_y = torch.squeeze(torch.tensor(dev_y),0)



        best_dev_err = float('inf')
        bad_count = 0
        num_seqs = len(train_seqs)
        print("Number of sequences in a training set: %d" % int(np.floor(num_seqs / args.mb)))

        #for scheduled sampling
        epsilon = 1.0
        compute_decay_constants(args.epochs)

        for epoch in range(args.epochs):
            print("Epoch %d" % epoch)

	    #TODO: do not shuffle, do smth else
            # shuffle data once per epoch
            idx = np.random.permutation(num_seqs)
            train_seqs = train_seqs[idx]
            hidden_states = None


            #do first forward on a first sequence, then do k = len(sequence) shift
            # and apply hidden states and memory cell states from the last forward to a new image
            for mb_row in range(int(np.floor(num_seqs / args.mb))):
                row_start = mb_row*args.mb
                row_end = np.min([(mb_row+1)*args.mb, num_seqs]) # Last minibatch might be partial


                mb_x = train_seqs[row_start:row_end, 0:args.max_len-1]
                mb_y = train_seqs[row_start:row_end, 1:args.max_len]
                mb_y = torch.squeeze(torch.tensor(mb_y),0).to(device)

                # training
                optimizer.zero_grad()
                train_outputs, prev_hidden_states = net.forward(mb_x, hidden_states, epsilon, device)

                #update hidden_state to next sequence
                hidden_states = prev_hidden_states

                train_loss = loss(train_outputs, mb_y)
                print("Train loss = %.7f" % train_loss.data)
                train_loss.backward()
                optimizer.step()



            # NOTE: Recompute epsilon for scheduled sampling each epoch (check it out, evaulate on dev set each epoch?)
            epsilon = update_epsilon(epoch)
            print("Linear decay applied. epsilon=%.5f" % epsilon)

            mae, std = evaluateNet(net, loss, dev_x, dev_y, prev_hidden_states, device)
            x_axes = [0 for i in range(0, dev_x.size(1))]
            print("MAE")
            print(mae)
            #print std too
            if plot:
                plotMAE(x_axes, mae, std)

                #
                # bad_count += 1
                # if my_dev_err < best_dev_err:
                #     bad_count = 0
                #     best_dev_err = my_dev_err
                # # saver.save(sess, args.model) save model
                #
                # if bad_count > args.patience:
                #     print('Converged due to early stopping...')
                #     break
                #
                #
                # # Reshape output for writing to netCDF
                # dev_outputs = dev_outputs.reshape(-1, 64, 128)
                # dev_y = dev_y.reshape(-1, 64, 128)
                # dev_time_y = dev_times[:, 1:args.max_len]
                # dev_time_y = dev_time_y.flatten()
                #
                # # Denormalize output
                # if args.normalize == "log":
                #     dev_outputs = log_denormalize(dev_outputs)
                #     dev_y = log_denormalize(dev_y)
                # my_dev_err = compute_mse(dev_y, dev_outputs)
                # #
                # # # Exporting the predicted precipitation maps as well as the true maps, for visual comparison.
                # export_netCDF(dev_outputs, nc, args.dev_preds, dev_time_y)
                # export_netCDF(dev_y, nc, args.dev_truths, dev_time_y)
                # #
                # # # Reporting Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Percent Error (MAPE),
                # # # Mean Absolute Error.
                # dev_mape = 100 * np.sum(np.absolute(np.divide(np.subtract(dev_y, dev_outputs), dev_y.mean()))) / dev_y.size
                # dev_mae = np.sum(np.absolute(np.subtract(dev_y, dev_outputs))) / dev_y.size
                # dev_rmse = np.sqrt(my_dev_err)
                # print("After denormalization:\n dev_mse = %.10f\ndev_rmse = %.10f\ndev_mape = "
                # "%.10f\ndev_mae = %.10f" %
                # (my_dev_err, dev_rmse, dev_mape, dev_mae))
