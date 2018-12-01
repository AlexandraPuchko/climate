import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
import pdb
import logging


# NOTE: These constants assume the model converges around epoch 50
LIN_DECAY_CONST = (-1.0 / 50.0)


#Static decay functions
def linear(epoch):
    return maximum(0, 1 + (LIN_DECAY_CONST * epoch))

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





class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + 2*self.hidden_dim,
                              out_channels=4 * self.hidden_dim,#because we have 4 gates
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)


    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state

        #apply convolution
        combined = torch.cat([input_tensor, h_cur,c_cur], dim=1) # concatenate along channel axis
        combined_conv = self.conv(combined)

        #split along channel axis
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    # (B*Cin*H*W)
    def init_hidden(self, batch_size):
        #init tensor with zeros
        # return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
        #         Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())

        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)))


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.decay_func = "linear" #might be changed to exp or negative sigmoid

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        #module list is like a Python list. It is similar to forward, but forward has its embedded forward method,
        # whereas we should redefine our own in ModuleList
        self.cell_list = nn.ModuleList(cell_list)
        self._hidden = self._init_hidden(1)

    def forward(self, input_x, hidden_state, epsilon, mode):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        train_y_vals, last_layer_hidden_states
        """
        #use GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_x = input_tensor.permute(1, 0, 2, 3, 4)
        # else:
        #     input_tensor = input_tensor.permute(1, 0, 3, 4, 2)

        input_x = torch.from_numpy(input_x)
        #.float().to(device)

        if hidden_state is None:
            #TODO:learnable weights
            hidden_state = self._hidden
        else:
            hidden_state = [(h.detach(),c.detach()) for h,c in hidden_state]

        ## of months in a sequence
        seq_len = input_x.size(1)
        #contains  maps to input for each cell month by month
        cur_layer_input = input_x

        #from t = 0 to T
        one_timestamp_output = []

        #first assume train_x is a true input map and true_y is train_x
        train_x = cur_layer_input[:, 0, :, :, :]
        train_y = train_x

        #take hidden states for first layer
        hidden_states = hidden_state
        # save all predicted maps to compute the loss
        train_y_vals = []
        for t in range(seq_len):
            #NOTE: This is where we use scheduled sampling to set up our next input.
            #      Flip biased coin, take ground truth with a probability of 'epsilon'
            #      ELSE take model output.
            #print("binomial: ", np.random.binomial(1, epsilon, 1)[0])
            if mode == "train":
                if np.random.binomial(1, epsilon, 1)[0]:
                    train_x = cur_layer_input[:, t, :, :, :]
                else:
                    train_x = train_y
            elif mode == "eval":
                if t == 0:
                    train_x = cur_layer_input[:, t, :, :, :]
                else:#TODO:change
                    train_x = train_y


            for layer_idx in range(self.num_layers):
                h, c = self.cell_list[layer_idx](input_tensor=train_x,
                                                 cur_state=hidden_states[layer_idx])
                train_x = h
                one_timestamp_output.append([h, c])

            # save all pairs (c_i, h_i) to feed the next timestep
            hidden_states = one_timestamp_output


            #get predicted value of h from the last layer for t = i
            last_hidden_state = one_timestamp_output[-1][0]
            in_channels_to_conv = last_hidden_state.size(1)
            padding_size = self.kernel_size[0][0] // 2, self.kernel_size[0][1] // 2
            conv_h = nn.Conv2d(in_channels=in_channels_to_conv,
                                  out_channels=1,# precipitation value
                                  kernel_size=(3,3),
                                  padding=padding_size,
                                  bias=self.bias)

            train_y = conv_h(last_hidden_state)
            train_y_vals.append(torch.squeeze(train_y, 0))
            #empty array of (h_i,c_i)
            one_timestamp_output = []
            last_layer_hidden_states = hidden_states

        #convert all outputs from the current sequence to tensor (stack along feature axes)
        train_y_vals = torch.stack(train_y_vals,dim=0)
        return train_y_vals, last_layer_hidden_states


    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    #apply same kernel for every layer, if we haven;t define different kernels per each layer
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param



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
