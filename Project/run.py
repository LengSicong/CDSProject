import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from argparse import ArgumentParser
import numpy as np
import csv

import matplotlib.pyplot as plt

from model import ClassifierNetwork

def load_data():
    import pickle
    import numpy as np

    audio_pkl = "./dataset/mosi/raw/audio.pickle"
    text_pkl = "./dataset/mosi/raw/text.pickle"
    video_pkl = "./dataset/mosi/raw/video.pickle"

    audio_file = open(audio_pkl,'rb')
    audio_dict = pickle.load(audio_file, encoding="bytes")
    audio_data = np.asarray(audio_dict)

    text_file = open(text_pkl,'rb')
    text_dict = pickle.load(text_file, encoding="bytes")
    text_data = np.asarray(text_dict)

    video_file = open(video_pkl,'rb')
    video_dict = pickle.load(video_file, encoding="bytes")
    video_data = np.asarray(video_dict)

    audio_file.close()
    text_file.close()
    video_file.close()

    return audio_data, text_data, video_data

def process_data(audio_data, text_data, video_data):

    # unpack the data
    (a_train, a_train_label, a_test, a_test_label, a_maxlen, a_train_length, a_test_length) = audio_data
    (t_train, t_train_label, t_test, t_test_label, t_maxlen, t_train_length, t_test_length) = text_data
    (v_train, v_train_label, v_test, v_test_label, v_maxlen, v_train_length, v_test_length) = video_data

    # transfer to torch tensor
    a_train = torch.Tensor(a_train)
    t_train = torch.Tensor(t_train)
    v_train = torch.Tensor(v_train)

    a_test = torch.Tensor(a_test)
    t_test = torch.Tensor(t_test)
    v_test = torch.Tensor(v_test)

    train_label = torch.Tensor(a_train_label)
    test_label = torch.Tensor(a_test_label)

    # train mask and task mask 
    # to be done!
    train_mask = np.zeros((a_train.shape[0], a_train.shape[1]))
    for i in range(a_train.shape[0]):
        train_mask[i][:a_train_length[i]] = 1

    test_mask = np.zeros((a_test.shape[0], a_test.shape[1]))
    for i in range(a_test.shape[0]):
        test_mask[i][:a_test_length[i]] = 1
    
    train_mask = torch.Tensor(train_mask)
    test_mask = torch.Tensor(test_mask) 

    return a_train, v_train, t_train, a_test, t_test, v_test, train_label, test_label, train_mask, test_mask

def train_epoch(model, opt, criterion, batch_size=1):
    model.train()
    losses = 0
    label_pred = []
    for i in range(a_train.shape[0]):
        a_batch = a_train[i:i+batch_size]
        t_batch = t_train[i:i+batch_size]
        v_batch = v_train[i:i+batch_size]
        label_batch = train_label[i:i+batch_size]
        mask_batch = train_mask[i:i+batch_size]

        a_batch = Variable(a_batch)
        t_batch = Variable(t_batch)
        v_batch = Variable(v_batch)
        label_batch = Variable(label_batch)
        mask_batch = Variable(mask_batch)

        opt.zero_grad()
        label_predicted = model(t_batch, a_batch, v_batch, mask_batch)
        label_pred.append(label_predicted)

        # mask the output
        label_predicted = label_predicted.view(-1)[:int(sum(train_mask[i]))]
        label_batch = label_batch.view(-1)[:int(sum(train_mask[i]))]

        loss = criterion(label_predicted,label_batch)

        loss.backward()
        opt.step()
        losses += loss.data.numpy()
    # print(label_pred)
    return losses/a_train.shape[0], label_predicted, label_batch

def evaluate(model, criterion, data_a, data_t, data_v,label,mask, batch_size=1):
    model.eval()
    losses = 0
    label_pred = []
    label_original = []
    for i in range(data_a.shape[0]):
        a_batch = data_a[i:i+batch_size]
        t_batch = data_t[i:i+batch_size]
        v_batch = data_v[i:i+batch_size]
        label_batch = label[i:i+batch_size]
        mask_batch = mask[i:i+batch_size]
    
        label_predicted = model(t_batch, a_batch, v_batch, mask_batch)
        label_predicted = label_predicted.view(-1)[:int(sum(mask[i]))]
        label_batch = label_batch.view(-1)[:int(sum(mask[i]))]
        label_pred.append(label_predicted)
        label_original.append(label_batch)
        loss = criterion(label_predicted,label_batch)
        losses += loss.data.numpy()
    return losses/a_test.shape[0], label_pred, label_original

def binary_acc(y_pred, y_test):
    acc= 0.0
    for i in range(len(y_pred)):

        # y_pred_tag = torch.round(torch.sigmoid(y_pred[i]))
        y_pred_tag = torch.round(y_pred[i])

        correct_results_sum = (y_pred_tag == y_test[i]).sum().float()
        acc2 = correct_results_sum/y_test[i].shape[0]
        acc += acc2.data.numpy()
    
    return acc/len(y_pred)
if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--text_encoder', type=str, default='lstm', help='[lstm, cnn_rnn, transformer, cnn]')
    parser.add_argument('--audio_encoder', type=str, default='lstm', help='[lstm, cnn_rnn, transformer, cnn]')
    parser.add_argument('--video_encoder', type=str, default='lstm', help='[lstm, cnn_rnn, transformer, cnn]')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden size of the model')
    parser.add_argument('--audio_dim', type=int, default=73, help='audio feature dimension of the data')
    parser.add_argument('--video_dim', type=int, default=100, help='video feature dimension of the data')
    parser.add_argument('--text_dim', type=int, default=100, help='text feature dimension of the data')
    parser.add_argument('--drop_rate', type=int, default=0.5, help='drop out rate of the model')
    parser.add_argument('--attention_heads', type=int, default=4, help='the number of attention heads in multi-head attention')
    parser.add_argument('--transformer_layers', type=int, default=1, help='the number of transformer layers stacked together')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate when training the model')

    cfgs = parser.parse_args()
    print(cfgs)

    # Load Data from pickle file
    audio_data, text_data, video_data = load_data()

    # Process data to proper format
    a_train, v_train, t_train, a_test, t_test, v_test, train_label, test_label, train_mask, test_mask = process_data(audio_data, text_data, video_data)

    # Model Construction
    model = ClassifierNetwork(cfgs)

    opt = optim.Adam(model.parameters(), lr=cfgs.learning_rate)

    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    # train the model
    e_losses = []
    num_epochs = 20
    print("Training......")
    for e in range(num_epochs):
        epoch_loss,label_predicted,label_batch = train_epoch(model, opt, criterion)
        print("epoch ", e)
        print("loss " , epoch_loss)
        train_loss, label_pred, label_original = evaluate(model, criterion,a_train, t_train, v_train,train_label,train_mask)
        train_acc = binary_acc(label_pred,label_original)
        print("Train accuracy ", train_acc)
        e_losses.append(epoch_loss)
    #print(e_losses)

    # validate model on test set
    test_loss, label_pred, label_original = evaluate(model, criterion,a_test, t_test, v_test,test_label,test_mask)
    print(test_loss)
    #print(label_pred)
    print("*************")
    #print(label_original)
    test_acc = binary_acc(label_pred,label_original)
    print("Test accuracy ", test_acc)
    
    # append the results to a csv file
    with open('results.csv', 'a') as f:
        writer = csv.writer(f)
        # writer.writerow({"configs: ": str(cfgs), "train_acc: ": str(train_acc), "test_acc": str(test_acc)})
        writer.writerow([str(cfgs.text_encoder), str(cfgs.video_encoder), str(cfgs.audio_encoder), str(train_acc), str(test_acc)])

    plt.plot(e_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.savefig('./figures/'+cfgs.text_encoder+'_'+cfgs.video_encoder+'_'+cfgs.audio_encoder)
    # plt.show()

