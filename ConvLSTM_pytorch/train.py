import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
import pdb
import logging


# NOTE: These constants assume the model converges around epoch 100
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



def trainNet(net, loss, optimizer,train_seqs, dev_seqs, test_seqs,args):

        print('Training started...')

        best_dev_err = float('inf')
        bad_count = 0
        num_seqs = len(train_seqs)

        #for scheduled sampling
        epsilon = 1.0
        compute_decay_constants(args.epochs)

        for epoch in range(args.epochs):

            #TODO: do not shuffle, do smth else
            # shuffle data once per epoch
            idx = np.random.permutation(num_seqs)
            train_seqs = train_seqs[idx]
            hidden_state = None
            print("Number of sequences in a training set: %d" % int(np.floor(num_seqs / args.mb)))#98

            #do first forward on a first sequence, then do k = len(sequence) shift
            # and apply hidden states and memory cell states from the last forward to a new image
            for mb_row in range(int(np.floor(num_seqs / args.mb))):
                row_start = mb_row*args.mb
                row_end = np.min([(mb_row+1)*args.mb, num_seqs]) # Last minibatch might be partial


                mb_x = train_seqs[row_start:row_end, 0:args.max_len-1]
                mb_y = train_seqs[row_start:row_end, 1:args.max_len]
                mb_y= torch.squeeze(torch.tensor(mb_y),0)

                # training
                optimizer.zero_grad()
                train_outputs, last_layer_hidden_states = net.forward(mb_x, hidden_state, epsilon)

                #update hidden_state to next sequence
                hidden_state = last_layer_hidden_states

                train_loss = loss(train_outputs, mb_y)
                print("Train loss = %.7f" % train_loss.data)
#                train_loss.backward(retain_graph=True)
                train_loss.backward()
                optimizer.step()


            # NOTE: Recompute epsilon for scheduled sampling each epoch
            epsilon = update_epsilon(epoch)
            print("Linear decay applied. epsilon=%.2f" % epsilon)



                #TODO 'Evaluating on dev set...'
                # print('Evaluating on dev set...')
                # dev_x = dev_seqs[:, 0:args.max_len-1]
                # dev_y = dev_seqs[:, 1:args.max_len]
                # # dev_h0 = np.zeros(shape=(len(dev_y), 64, 128, 1), dtype=np.float32)
                # # dev_c0 = np.zeros(shape=(len(dev_y), 64, 128, 1), dtype=np.float32)
                # dev_y_tensor = torch.squeeze(torch.tensor(dev_y),0)
                # print("dev_x: " + str(dev_x.shape))
                # print("mb_y_tensor:  " + str(dev_y_tensor.shape))
                #
                # dev_outputs = net.forward(dev_x, None)
                # print("dev_outputs: " + str(dev_outputs.shape))
                # dev_loss = loss(dev_outputs, dev_y_tensor)



                # Compute and print error metrics
                # dev_mape = 100 * np.sum(np.absolute(np.divide(np.subtract(dev_y, outputs), dev_y.mean()))) / dev_y.size
                # dev_mae = np.sum(np.absolute(np.subtract(dev_y, outputs))) / dev_y.size
                # dev_rmse = np.sqrt(dev_loss)
                #
                # print(
                # "Epoch %d: dev_mse=%.10f dev_rmse=%.10f dev_mape=%.10f dev_mae=%.10f bad_count=%d"
                # % (epoch, my_dev_err, dev_rmse, dev_mape, dev_mae, bad_count))
                #
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
