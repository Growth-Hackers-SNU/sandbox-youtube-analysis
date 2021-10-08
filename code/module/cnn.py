from tqdm import tqdm_notebook
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kiwipiepy import Kiwi
from sklearn.model_selection import train_test_split
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler,
                              SequentialSampler)
import torch.optim as optim
import random
import time

class CNN_NLP(nn.Module):
    def __init__(self,
                 pretrained_embedding = None,
                 freeze_embedding = False,
                 vocab_size = None,
                 embed_dim = 300,
                 filter_sizes = [3, 4, 5],
                 num_filters = [100, 100, 100],
                 num_classes = 7,
                 dropout = 0.5,
                 ):
        super(CNN_NLP, self).__init__()

        if pretrained_embedding is not None :
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze = freeze_embedding)

        else :
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(num_embeddings = vocab_size,
                                          embedding_dim = self.embed_dim,
                                          padding_idx = 0,
                                          max_norm = 5.0)
            
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels = self.embed_dim,
                      out_channels = num_filters[i],
                      kernel_size = filter_sizes[i])
            for i in range(len(filter_sizes))
        ])

        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, input_idx) :
        input_embed = self.embedding(input_idx).float()

        input_reshape = input_embed.permute(0, 2, 1)

        input_conv_list = [F.relu(conv1d(input_reshape)) for conv1d in self.conv1d_list]

        input_pool_list = [F.max_pool1d(input_conv, kernel_size = input_conv.shape[2]) for input_conv in input_conv_list]

        input_fc = torch.cat([input_pool.squeeze(dim = 2) for input_pool in input_pool_list], dim = 1)

        logits = self.fc(self.dropout(input_fc))

        return logits

def CNN_E2E(df,
            num_classes = 7,
            epochs = 10,
            how = 'basic',
            fasttext_dir = "/content/drive/Shareddrives/[GH x Sandbox]/reference_data/cc.ko.300.vec"):
    USE_GPU = True
    dtype = torch.float32

    if USE_GPU and torch.cuda.is_available() :
        device = torch.device('cuda')
    else :
        device = torch.device('cpu')

    labels = df['class']

    ########## Token for cnn ##########
    max_len = 0
    tokenized_texts = []
    word2idx = {}

    word2idx['<pad>'] = 0
    word2idx['<unk>'] = 1

    idx = 2
    kiwi = Kiwi()
    kiwi.prepare()
    
    for cnt, sent in enumerate(df['comment']):
        analyzed_sent = kiwi.analyze(str(sent))

        tokenized_sent = []

        for i in range(len(analyzed_sent[0][0])) :
            tokenized_sent.append(analyzed_sent[0][0][i][0])

        tokenized_texts.append(tokenized_sent)

    ########## Making Vocab ##########
    
        for token in tokenized_sent :
            if token not in word2idx :
                word2idx[token] = idx
                idx += 1
        
        max_len = max(max_len, len(tokenized_sent))
    
    ########## Encoding ##########
    input_idx = []

    for tokenized_sent in tokenized_texts :
        if len(tokenized_sent) < 100 :
            tokenized_sent += ['<pad>'] * (100 - len(tokenized_sent))
        else :
            tokenized_sent = tokenized_sent[:100]

        input_id = [word2idx.get(token) for token in tokenized_sent]
        input_idx.append(input_id)

    input_idx = np.array(input_idx)

    ########## loading FastTest ##########
    fin = open(fasttext_dir, 'r', encoding = 'utf-8', newline = '\n', errors = 'ignore')
    n, d = map(int, fin.readline().split())

    embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), d))
    embeddings[word2idx['<pad>']] = np.zeros((d,))

    count = 0 
    for line in tqdm_notebook(fin):
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        if word in word2idx:
            count += 1
            embeddings[word2idx[word]] = np.array(tokens[1:], dtype=np.float32)

    print(f"There are {count} / {len(word2idx)} pretrained vectors found.")

    embeddings = torch.tensor(embeddings)

    ########## CNN Modeling ##########
    CNN = CNN_NLP(pretrained_embedding = embeddings,
                  freeze_embedding = True,
                  num_classes = num_classes).to(device)
    
    ########## Train ##########

    train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_idx, list(labels), test_size=0.1, random_state=25)
    train_dataloader, val_dataloader = data_loader(train_inputs, val_inputs, train_labels, val_labels, batch_size=30)   

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(CNN.parameters(),
                               lr = 0.25, rho = 0.95)
    
    best_acc = 0
    
    print("Start training...\n")
    print(f"{'Epoch' :^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
    print("-"*60)
    

    for epoch_i in range(epochs) :
        CNN.train()
        total_loss = 0.0
        
        correct = 0
        total = 0
        t0_epoch = time.time()

        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_labels = tuple(t.to(device) for t in batch)

            CNN.zero_grad()

            logits = CNN(b_input_ids)
            loss = loss_fn(logits, b_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.detach()

        avg_train_loss = total_loss / len(train_dataloader)

        if val_dataloader is not None:
            val_loss, val_accuracy = evaluate(CNN, val_dataloader, device, loss_fn)

            if val_accuracy > best_acc:
                best_acc = val_accuracy

            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            
    print("\n")
    print(f"Training complete! Best accuracy: {best_acc:.2f}%.")
    

    return CNN, word2idx



def CNN_predict(CNN, test_df, word2idx):
    tokenized_texts = []
    kiwi = Kiwi()
    kiwi.prepare()
    for cnt, sent in enumerate(test_df['comment']):
        analyzed_sent = kiwi.analyze(str(sent))

        tokenized_sent = []

        for i in range(len(analyzed_sent[0][0])) :
            tokenized_sent.append(analyzed_sent[0][0][i][0])

        tokenized_texts.append(tokenized_sent)

    input_idx = []

    for tokenized_sent in tokenized_texts :
        if len(tokenized_sent) < 100 :
            tokenized_sent += ['<pad>'] * (100 - len(tokenized_sent))
        else :
            tokenized_sent = tokenized_sent[:100]

        input_id = [word2idx.get(token) for token in tokenized_sent]
        input_idx.append(input_id)

    input_idx = np.array(input_idx)

    dataloader = DataLoader(input_idx, batch_size = 50)

    CNN.eval()
    total_predicted = torch.tensor([]).to(torch.device('cuda'))
    with torch.no_grad() :
        for batch in dataloader:
            b_input_ids = batch.to(torch.device('cuda'))

            logits = CNN(b_input_ids)
            _, predicted = torch.max(logits.data, 1)

            total_predicted = torch.cat( [total_predicted, predicted] )
    return total_predicted

def evaluate(CNN, val_dataloader, device, loss_fn):
    CNN.eval()
    
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = CNN(b_input_ids)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy


    
def data_loader(train_inputs, val_inputs, train_labels, val_labels,
            batch_size=50):

    # Convert data type to torch.Tensor
    train_inputs, val_inputs, train_labels, val_labels = tuple(torch.tensor(data) for data in
            [train_inputs, val_inputs, train_labels, val_labels])

    # Specify batch_size
    batch_size = 50

    # Create DataLoader for training data
    train_data = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create DataLoader for validation data
    val_data = TensorDataset(val_inputs, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader


def mixup_loss(logits, labels_1, labels_2, lam):
    loss = lam *criterion(logits, labels_1) + (1 - lam) * criterion(logits, labels_2)
    return loss 

def input_mixup(inputs, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(inputs.shape[0]).cuda()
    mixup_data = lam * inputs + (1 - lam) * inputs[index, :]
    labels_1, labels_2 = labels, labels[index]

    return mixup_data, labels_1, labels_2, lam