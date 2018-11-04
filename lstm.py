# Pytorch’s LSTM expects all of its inputs to be 3D tensors.
# The first axis: the sequence itself,
#     second: indexes instances in the mini-batch,
#     third: indexes elements of the input.

# EXAMPLE: AN LSTM FOR PART-OF-SPEECH TAGGING
 # Author: Robert Guthrie
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# w_1,…,w_M  - input sentence, where w_i ∈ V, our Vocab.
# T  -  tag set
# y_i - the tag of word w
# ŷ - prediction of the tag of word wi
# output is a sequence ŷ_1,…,ŷ_M, where ŷ i ∈ T

# ŷ_i=argmax_j(logSoftmax(Ah_i+b))_j (predicted tag is the tag that has the maximum value in this vector.)


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # in our case vocab size = 9 (all words in two senteces)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
    # Before we've done anything, we dont have any hidden state.
    # Refer to the Pytorch documentation to see exactly
    # why they have this dimensionality.
    # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
        torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def my_word_embeddings(training_data):
    word_to_ix = {}
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:
                # assign index to every word (word embeddings):  map word from the V to real number
                word_to_ix[word] = len(word_to_ix)
    tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
    return word_to_ix,tag_to_ix


def createLossAndOptimizer(net, learning_rate):
    # The negative log likelihood loss. It is useful to train a classification problem with C classes.
    loss = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), learning_rate)
    return(loss, optimizer)


def trainLSTM(net,training_data,word_to_ix,tag_to_ix,loss_function,optimizer,n_epochs):
# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()

#torch.no_grad() is used when you evaluate your model
# and don’t need to call backward() to calculate
# the gradients and update the corresponding parameters.
    with torch.no_grad():
        #ntraining_data[0][0] - parsed sentence
        inputs = prepare_sequence(training_data[0][0], word_to_ix)
        tag_scores = net(inputs)
        print(tag_scores)

    for epoch in range(n_epochs):
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            net.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            net.hidden = net.init_hidden()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)

            # Step 3. Run our forward pass.
            tag_scores = net(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

    # See what the scores are after training
    with torch.no_grad():
        inputs = prepare_sequence(training_data[0][0], word_to_ix)
        tag_scores = net(inputs)

        # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
        # for word i. The predicted tag is the maximum scoring tag.
        # Here, we can see the predicted sequence below is 0 1 2 0 1
        # since 0 is index of the maximum value of row 1,
        # 1 is the index of maximum value of row 2, etc.
        # Which is DET NOUN VERB DET NOUN, the correct sequence!
        print(tag_scores)


def main():
    # These will usually be more like 32 or 64 dimensional.
    # We will keep them small, so we can see how the weights change as we train.
    EMBEDDING_DIM = 6
    HIDDEN_DIM = 6

    # 1. Prepare data
    training_data = [
        ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
    ]
    word_to_ix,tag_to_ix = my_word_embeddings(training_data)

    #word_to_ix - indexed Vocabulary, tag_to_ix - indexed tags
    lstm = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    loss, optimizer = createLossAndOptimizer(lstm, learning_rate=0.1)

    # Normally we would NOT do 300 epochs, but we have toy data
    trainLSTM(lstm,training_data,word_to_ix,tag_to_ix,loss, optimizer,n_epochs=300)

main()
