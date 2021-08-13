import torch

# From https://github.com/xiaxin1998/DHCN
def SSL(embed1, embed2):
    def row_shuffle(embedding):
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
        return corrupted_embedding
    def row_column_shuffle(embedding):
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
        corrupted_embedding = corrupted_embedding[:,torch.randperm(corrupted_embedding.size()[1])]
        return corrupted_embedding
    def score(x1, x2):
        return torch.sum(torch.mul(x1, x2), 1)

    pos = score(embed1, embed2)
    neg1 = score(embed2, row_column_shuffle(embed1))
    one = torch.cuda.FloatTensor(neg1.shape[0]).fill_(1)
    con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos))-torch.log(1e-8 + (one - torch.sigmoid(neg1))))
    return con_loss
