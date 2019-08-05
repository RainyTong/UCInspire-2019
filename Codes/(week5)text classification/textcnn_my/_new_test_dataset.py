import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
import random

import rootpath

rootpath.append()
from yutong_nlp.textcnn_mine.CNN_text1 import CNN_Text
from backend.data_preparation.connection import Connection


def handle_args():
    parser = argparse.ArgumentParser(description='CNN twitter classifier')
    # learning
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    # parser.add_argument('-weight-decay', type=float, default=0.0001, help='weight decay, L2 penalty [default: 0.0001]')
    parser.add_argument('-epochs', type=int, default=4, help='number of epochs for train [default: 256]')
    parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
    parser.add_argument('-log-interval', type=int, default=1,
                        help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-test-interval', type=int, default=100,
                        help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-save-interval', type=int, default=500,
                        help='how many steps to wait before saving [default: 500]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-early-stop', type=int, default=1000,
                        help='iteration numbers to stop without performance increasing')
    parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
    # parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
    parser.add_argument('-stride', type=int, default=1, help='stride for conv2d')
    parser.add_argument('-glove-embed', type=bool, default=True,
                        help='whether to use the glove twitter embedding or not')
    parser.add_argument('-glove-embed-train', type=bool, default=True,
                        help='whether to train the glove embedding or not')
    parser.add_argument('-multichannel', type=bool, default=True, help='multiple channel of input')

    # model
    parser.add_argument('-dropout', type=float, default=0.7, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
    parser.add_argument('-embed-dim', type=int, default=100, help='number of embedding dimension [default: 128]')
    parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('-kernel-sizes', type=str, default='2,2,2',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')

    # device
    parser.add_argument('-device', type=int, default=-1,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')

    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
    parser.add_argument('-test', action='store_true', default=False, help='train or test')

    # workflow
    parser.add_argument('-read-path', type=str, default='./dataset/',
                        help='the path to the labeled data')
    parser.add_argument('-pos-num', type=int, default=1000, help='the number of positive tweets [default: 5000]')
    parser.add_argument('-neg-num', type=int, default=1000, help='the number of negative tweets [default: 5000]')
    parser.add_argument('-workflow', type=str, default='w2', help='which workflow to analysis [default: w2]')
    parser.add_argument('-keywords', type=str, default='cc', help='the keywords of the workflow [default: cc]')
    parser.add_argument('-pad-len', type=int, default=64, help='the length of padding [default: 64]')
    parser.add_argument('-worker-num', type=int, default=2, help='the number of workers')
    parser.add_argument('-pos-weight', type=float, default=1.0, help='the pos class penalty')

    args = parser.parse_args()
    return args


def read_test_data():
    tweets_neg = []
    labels_neg = []
    tweets_pos = []
    labels_pos = []
    with Connection() as conn:
        cur = conn.cursor()
        # FOR label2 data:
        cur.execute("SELECT text from records where label1 is null and label2 = 0 ")
        text_label2_0 = cur.fetchmany(264)

        for record in text_label2_0:
            a = str(record[0])
            a = a.encode('ascii', 'ignore').decode('ascii')
            tweets_pos.append(a.strip().replace('\n', '. '))
            labels_pos.append(0)  # 0 for true(wildfire), 1 for false

        # FOR label1 data:
        cur.execute("SELECT text from records where label1 = 0 and label2 is null")
        text_label1_0 = cur.fetchmany(264)

        for record in text_label1_0:
            a = str(record[0])
            a = a.encode('ascii', 'ignore').decode('ascii')
            tweets_pos.append(a.strip().replace('\n', '. '))
            labels_pos.append(0)  # 0 for true(wildfire), 1 for false


        # FOR label2 data:
        cur.execute("SELECT text from records where label1 is null and label2 = 1 ")
        text_label2_1 = cur.fetchmany(264)

        for record in text_label2_1:
            a = str(record[0])
            a = a.encode('ascii', 'ignore').decode('ascii')
            tweets_neg.append(a.strip().replace('\n', '. '))
            labels_neg.append(1)  # 0 for true(wildfire), 1 for false

        # FOR label1 data:
        cur.execute("SELECT text from records where label1 = 1 and label2 is null")
        text_label1_1 = cur.fetchmany(264)

        for record in text_label1_1:
            a = str(record[0])
            a = a.encode('ascii', 'ignore').decode('ascii')
            tweets_neg.append(a.strip().replace('\n', '. '))
            labels_neg.append(1)  # 0 for true(wildfire), 1 for false

        random.seed(1)
        random.shuffle(tweets_pos)
        random.seed(1)
        random.shuffle(tweets_neg)

        tweets_test = tweets_pos + tweets_neg
        labels_test = labels_pos + labels_neg
        tweet_label_pair_test = list(zip(tweets_test, labels_test))
        random.seed(1)
        random.shuffle(tweet_label_pair_test)
        tweet_texts_Test, tweet_labels_Test = zip(*tweet_label_pair_test)

        print("tweet_test = " + str(len(tweet_texts_Test)) + " label_test = " + str(len(tweet_labels_Test)))

        return tweet_texts_Test, tweet_labels_Test


def data_processing(tweet_texts, vocab):
    text_feature = []
    for tweets in tweet_texts:
        features = [vocab[w].index for w in tweets.split() if w in vocab]
        text_feature.append(features)
    text_feature_padding = pad_sequences(text_feature, maxlen=64, padding='post')
    return text_feature_padding


def get_dataset_loader(feature_padding_test, tweet_labels_Test):
    test_dataset = TensorDataset(torch.LongTensor(feature_padding_test), torch.LongTensor(tweet_labels_Test))
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                             num_workers=4)
    return test_loader


def get_test_loader(vocab):
    tweet_texts_Test, tweet_labels_Test = read_test_data()
    feature_padding_test = data_processing(tweet_texts_Test, vocab)
    test_loader = get_dataset_loader(feature_padding_test, tweet_labels_Test)
    return test_loader


def test(test_loader, model):
    data_size, corrects = 0, 0
    TP, TN = 0, 0
    for i, batch in enumerate(test_loader, 0):
        texts, labels = batch
        texts, labels = Variable(texts), Variable(labels)  # .cuda() .cuda()
        logit = model(texts)
        #####
        corrects += (torch.max(logit, 1)[1].view(labels.size()).data == labels.data).sum()
        # ===>
        if labels.data[0] == 0 and torch.max(logit, 1)[1].view(labels.size()).data == labels.data:
            TP += 1
        elif labels.data[0] == 1 and torch.max(logit, 1)[1].view(labels.size()).data == labels.data:
            TN += 1

        #####

        data_size += len(texts)
    accuracy = 100.0 * corrects / data_size

    FP = (data_size / 2) - TP
    FN = (data_size / 2) - TN

    recall = TP / (TP + FN)
    FPR = FP / (FP + TN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    F1 = 2 * precision * recall / (precision + recall)

    print('Evaluation - acc: {:.4f}%({}/{})'.format(accuracy, corrects, data_size))
    print("TP: {} TN: {} FP: {} FN: {}".format(TP, TN, FP, FN))
    print("recall: {:.4f}  FPR: {:.4f}  specificity: {:.4f}  precision: {:.4f}  F1: {:.4f}".format(recall, FPR,
                                                                                                   specificity,
                                                                                                   precision, F1))


if __name__ == "__main__":
    args = handle_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available();
    del args.no_cuda

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    # set the random seed
    torch.manual_seed(3)
    torch.cuda.manual_seed_all(3)

    gensim_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',
                                                     binary=True)
    vocab = gensim_model.vocab
    vocab_len = len(vocab)
    weights = gensim_model.vectors

    model = CNN_Text(args, vocab_len, weights)

    model.load_state_dict(torch.load('../Checkpoint_2.ckpt'))

    model.eval()

    test_loader = get_test_loader(vocab)

    test(test_loader, model)
