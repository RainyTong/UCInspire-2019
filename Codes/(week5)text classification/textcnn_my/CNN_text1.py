import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Text(nn.Module):

    def __init__(self, args, vocab_len, weights):
        super(CNN_Text, self).__init__()
        self.args = args

        # parameters
        V = vocab_len
        D = args.embed_dim
        C = 2
        Ci = 1  # conv layer input dimention
        Co = args.kernel_num  # conv layer output dimention
        Ks_str = args.kernel_sizes
        Ks = [int(k) for k in Ks_str.split(',')]

        # embedding layer, input_size = V, output_size = D
        self.embed = nn.Embedding(V, D)

        if args.glove_embed:
            self.embed.weight = nn.Parameter(torch.FloatTensor(weights), requires_grad=args.glove_embed_train)
        if args.multichannel:
            self.embed1 = nn.Embedding(V, D)
            self.embed1.weight = nn.Parameter(torch.FloatTensor(weights), requires_grad=False)
            Ci = 2

        # self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (1, 300)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def forward(self, input):
        x = self.embed(input)
        x = x.unsqueeze(1)

        if self.args.multichannel:
            x1 = self.embed1(input)
            x1 = x1.unsqueeze(1)
            x = torch.cat((x, x1), 1)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        x = torch.cat(x, 1)

        x = self.dropout(x)

        logit = F.softmax(self.fc1(x), dim=-1)

        return logit
