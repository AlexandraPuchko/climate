import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
from cell import ConvLSTMCell

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


    def evaluate(self, input_x, hidden_states, step, seq_len, device):

        input_x = input_x.float().to(device)

        #hidden_state = [(h.detach(),c.detach()) for h,c in hidden_state]
        #contains  maps to input for each cell month by month
        cur_layer_input = input_x

        #from t = 0 to T
        one_timestamp_output = []


        #feed true image for the first timestamp, then feed only predicted
        dev_x = cur_layer_input[:, step, :, :, :]
        dev_y = dev_x


        # save all predicted maps to compute the loss
        eval_outputs = []
        for t in range(step, seq_len):
            dev_x = dev_y

            for layer_idx in range(self.num_layers):
                h, c = self.cell_list[layer_idx](input_tensor=dev_x,
                                                 cur_state=hidden_states[layer_idx])
                dev_x = h
                one_timestamp_output.append([h, c])

            # save all pairs (c_i, h_i) to feed the next timestep
            hidden_states = one_timestamp_output
            if t == step:
                #save next_hidden state for step = step + 1
                next_hidden_state = hidden_states

            #get predicted value of h from the last layer for t = i
            last_hidden_state = one_timestamp_output[-1][0]
            in_channels_to_conv = last_hidden_state.size(1)
            padding_size = self.kernel_size[0][0] // 2, self.kernel_size[0][1] // 2
            conv_h = self.dynamic_conv_h(in_channels_to_conv, padding_size, device).to(device)
            dev_y = conv_h(last_hidden_state)
            eval_outputs.append(torch.squeeze(dev_y, 0))

            #empty array of (h_i,c_i)
            one_timestamp_output = []

        #convert all outputs from the current sequence to tensor (stack along feature axes)
        eval_outputs = torch.stack(eval_outputs,dim=0)
        return eval_outputs, next_hidden_state




    def forward(self, input_x, hidden_state, epsilon, device):
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

        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_x = input_tensor.permute(1, 0, 2, 3, 4)
        # else:
        #     input_tensor = input_tensor.permute(1, 0, 3, 4, 2)

        input_x = torch.from_numpy(input_x).float().to(device)

        if hidden_state is None:
            #TODO:learnable weights
            hidden_state = self._hidden
        else:
            hidden_state = [(h.detach(),c.detach()) for h,c in hidden_state]

        # #of months in a sequence
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
            if np.random.binomial(1, epsilon, 1)[0]:
                train_x = cur_layer_input[:, t, :, :, :]
            else:
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

            #TODO think how to dynamically update weights to cuda
            conv_h = self.dynamic_conv_h(in_channels_to_conv, padding_size, device).to(device)

            train_y = conv_h(last_hidden_state)

            train_y_vals.append(torch.squeeze(train_y, 0))
            #empty array of (h_i,c_i)
            one_timestamp_output = []
            last_layer_hidden_states = hidden_states

        #convert all outputs from the current sequence to tensor (stack along feature axes)
        train_y_vals = torch.stack(train_y_vals,dim=0)
        return train_y_vals, last_layer_hidden_states



    def dynamic_conv_h(self, in_channels_to_conv, padding_size, device):

        conv_h = nn.Conv2d(in_channels=in_channels_to_conv,
                              out_channels=1,# precipitation value
                              kernel_size=(3,3),
                              padding=padding_size,
                              bias=self.bias)
        #use GPU
        self.to(device)
        return conv_h



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
