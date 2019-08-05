import numpy
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

def train(train_loader, validate_loader, test_loader, model, my_loss, weight, args):
    if args.cuda:
        model.cuda()
        my_loss.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps, best_reduction, last_step = 0, 0, 0
    model.train()
    for epoch in range(1, args.epochs+1):
        size, corrects, avg_loss = 0, 0, 0
        for i, batch in enumerate(train_loader, 0):
            texts, labels = batch
            size += len(texts)
            if args.cuda:
                texts, labels = Variable(texts.cuda()), Variable(labels.cuda())
            else:
                texts, labels = Variable(texts), Variable(labels)

            optimizer.zero_grad()
            output = model(texts)
            loss = F.cross_entropy(output, labels, weight=weight)

            # add l1 regularization
            # l1_crit = nn.MSELoss(size_average=False)
            # reg_loss = 0
            # for param in model.parameters():
            #     reg_loss += l1_crit(param, target=torch.zeros_like(param))
            # loss += 0.0005 * reg_loss

            loss.backward()
            optimizer.step()

            steps += 1
            avg_loss+=loss.item()*len(texts)
            corrects += (torch.max(output, 1)[1].view(labels.size()).data == labels.data).sum()
            if steps % 100 == 0:
                eval(validate_loader, model, my_loss, weight)
        if epoch % 5 == 0:
            validate_avg_loss, test_avg_loss, final_accuracy, reduction = test(validate_loader, test_loader, model,
                                                                                   my_loss, weight=weight)
        accuracy = 100.0 * corrects / size
        avg_loss /= size
        print('\tEpoch[{}]-loss:{:.6f} acc:{:.4f}%({}/{})'.format(epoch, avg_loss, accuracy, corrects, size))


def eval(data_loader, model, my_loss, weight):
    model.eval()
    data_size, corrects, avg_loss = 0, 0, 0
    for i, batch in enumerate(data_loader, 0):
        texts, labels = batch
        texts, labels = Variable(texts.cuda()), Variable(labels.cuda())
        logit = model(texts)
        # loss=F.cross_entropy(logit, labels, weight=weight, size_average=False)
        # loss=F.cross_entropy(logit, labels, size_average=False)
        loss = my_loss(logit, labels, weight)
        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)[1].view(labels.size()).data == labels.data).sum()
        data_size += len(texts)
    avg_loss /= data_size
    accuracy = 100.0 * corrects / data_size
    print('Evaluation - loss: {:.6f} acc: {:.4f}%({}/{})'.format(avg_loss, accuracy, corrects, data_size))
    return accuracy



class My_loss(torch.nn.Module):

    def __init__(self):
        super(My_loss, self).__init__()

    def forward(self, output, target, weight):
        loss1 = F.cross_entropy(output, target, weight=weight)
        return loss1
        # pos_prob=output[target==1][:,1]
        # sort_pos_prob=pos_prob.sort(descending=True)[0]
        # min_accuracy = sort_pos_prob[(sort_pos_prob.size(0)*torch.Tensor([0.9])).long()-torch.Tensor([1]).item()]
        # loss2=1-min_accuracy
        # return loss1+loss2

def test(validate_loader, test_loader, model, my_loss, weight):
    validate_avg_loss, validate_acc, validate_probs, validate_labels = predict_prob(validate_loader, model, my_loss, weight)
    pos_proba = validate_probs[validate_labels == 1][:, 1]
    pos_proba_sorted = -numpy.sort(-pos_proba)
    target_accuracy = [1.0, 0.98, 0.9]
    test_avg_loss, test_acc, test_probs, test_labels = predict_prob(test_loader, model, my_loss, weight)
    denominator_accuracy = numpy.sum(test_labels == 1)
    final_accuracy = []
    reduction = []
    for accuracy in target_accuracy:
        min_accuracy = pos_proba_sorted[int(pos_proba_sorted.shape[0] * accuracy) - 1]
        test_probs_copy = numpy.copy(test_probs)
        test_pos_probs = test_probs_copy[:, 1]
        test_pos_probs[test_pos_probs >= min_accuracy] = 1
        test_pos_probs[test_pos_probs < min_accuracy] = 0
        denominator_reduction = test_pos_probs.shape[0]
        nominator_reduction = numpy.sum(test_pos_probs == 0)
        test_pos_probs_1 = test_pos_probs[test_labels == 1]
        nominator_accuracy = numpy.sum(test_pos_probs_1 == 1)
        final_accuracy.append(nominator_accuracy / denominator_accuracy)
        reduction.append(nominator_reduction / denominator_reduction)
    print(
        "\nTest-Phase: validate-avg-loss: {:.6f} validate-acc: {:.4f}% test-avg-loss: {:.6f} test-acc: {:.4f}%".format(
            validate_avg_loss, validate_acc, test_avg_loss, test_acc))
    print("final-accuracy: {} reduction: {}".format(final_accuracy, reduction))
    return validate_avg_loss, test_avg_loss, final_accuracy, reduction

def predict_prob(data_loader, model, my_loss, weight):
    model.eval()
    data_size, corrects, avg_loss = 0, 0, 0
    pred_probs, true_labels = None, None
    for i, batch in enumerate(data_loader, 0):
        texts, labels = batch
        texts, labels = Variable(texts.cuda()), Variable(labels.cuda())
        batch_probs = model(texts)
        # avg_loss+=F.cross_entropy(batch_probs, labels, weight=weight, size_average=False).item()
        # avg_loss+=F.cross_entropy(batch_probs, labels, size_average=False).item()
        avg_loss += my_loss(batch_probs, labels, weight).item()
        corrects += (torch.max(batch_probs, 1)[1].view(labels.size()).data == labels.data).sum()
        data_size += len(texts)
        if i == 0:
            true_labels = labels.cpu().numpy()
        else:
            true_labels = numpy.concatenate((true_labels, labels.cpu().numpy()))
        batch_probs_array = batch_probs.cpu().detach().numpy()
        if i == 0:
            pred_probs = batch_probs_array
        else:
            pred_probs = numpy.concatenate((pred_probs, batch_probs_array), axis=0)
    accuracy = 100.0 * corrects / data_size
    avg_loss /= data_size
    return avg_loss, accuracy, pred_probs, true_labels