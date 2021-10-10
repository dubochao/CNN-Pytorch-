
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args

        class_num = args.class_num
        chanel_num = 1
        filter_num = args.filter_num
        filter_sizes = args.filter_sizes

        vocabulary_size = args.vocabulary_size
        embedding_dimension = args.embedding_dim
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        if args.static:
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.non_static)
        if args.multichannel:
            self.embedding2 = nn.Embedding(vocabulary_size, embedding_dimension).from_pretrained(args.vectors)
            chanel_num += 1
        else:
            self.embedding2 = None
        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, filter_num, (size, embedding_dimension)) for size in filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.hiddens=nn.Linear(len(filter_sizes) * filter_num, len(filter_sizes) * filter_num*2)
        self.hiddens1 = nn.Linear(len(filter_sizes) * filter_num* 2, len(filter_sizes) * filter_num )
        self.hiddens2 = nn.Linear(len(filter_sizes) * filter_num, len(filter_sizes) * filter_num //2)
        self.fc = nn.Linear(len(filter_sizes) * filter_num//2, class_num)
        self.relu=nn.LeakyReLU(negative_slope=0.01, inplace=False)
    def forward(self, x):
        if self.embedding2:
            x = torch.stack([self.embedding(x), self.embedding2(x)], dim=1)
        else:
            x = self.embedding(x)
            x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.hiddens(x)
        x = self.relu(x)
        x = self.hiddens1(x)
        x = self.relu(x)
        x = self.hiddens2(x)
        x = self.relu(x)
        logits = self.fc(x)
        return logits
