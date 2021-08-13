import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GCN_multilabels(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super(GCN_multilabels, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__


class GCN_concat(torch.nn.Module):
    def __init__(self, dataset, dataset2, num_layers, hidden):
        super(GCN_concat, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)

        self.conv2 = GCNConv(dataset2.num_features, hidden)
        self.convs2 = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs2.append(GCNConv(hidden, hidden))
        self.lin2 = Linear(hidden, hidden)

        self.lin3 = Linear(hidden*2, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.conv2.reset_parameters()
        for conv in self.convs2:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, data1, data2):
        x1, edge_index1, batch1 = data1.x, data1.edge_index, data1.batch
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch
        x1 = F.relu(self.conv1(x1, edge_index1))
        for conv in self.convs:
            x1 = F.relu(conv(x1, edge_index1))
        x1 = global_mean_pool(x1, batch1)
        x1 = F.relu(self.lin1(x1))

        x2 = F.relu(self.conv2(x2, edge_index2))
        for conv in self.convs2:
            x2 = F.relu(conv(x2, edge_index2))
        x2 = global_mean_pool(x2, batch2)
        x2 = F.relu(self.lin2(x2))

        combined = torch.cat((x1.view(x1.size(0), -1), x2.view(x2.size(0), -1)), dim=1) #将多维的tensor转成1维的
        combined = F.dropout(combined, p=0.5, training=self.training)
        x = self.lin3(combined)
        return x1, x2, F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GCN_concat_multilabels(torch.nn.Module):
    def __init__(self, dataset, dataset2, num_layers, hidden):
        super(GCN_concat_multilabels, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)

        self.conv2 = GCNConv(dataset2.num_features, hidden)
        self.convs2 = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs2.append(GCNConv(hidden, hidden))
        self.lin2 = Linear(hidden, hidden)

        self.lin3 = Linear(hidden*2, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.conv2.reset_parameters()
        for conv in self.convs2:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, data1, data2):
        x1, edge_index1, batch1 = data1.x, data1.edge_index, data1.batch
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch
        x1 = F.relu(self.conv1(x1, edge_index1))
        for conv in self.convs:
            x1 = F.relu(conv(x1, edge_index1))
        x1 = global_mean_pool(x1, batch1)
        x1 = F.relu(self.lin1(x1))

        x2 = F.relu(self.conv2(x2, edge_index2))
        for conv in self.convs2:
            x2 = F.relu(conv(x2, edge_index2))
        x2 = global_mean_pool(x2, batch2)
        x2 = F.relu(self.lin2(x2))

        combined = torch.cat((x1.view(x1.size(0), -1), x2.view(x2.size(0), -1)), dim=1) #将多维的tensor转成1维的
        combined = F.dropout(combined, p=0.5, training=self.training)
        x = self.lin3(combined)
        return x1, x2, x

    def __repr__(self):
        return self.__class__.__name__
  
    
class GCN_add(torch.nn.Module):
    def __init__(self, dataset, dataset2, num_layers, hidden):
        super(GCN_add, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)

        self.conv2 = GCNConv(dataset2.num_features, hidden)
        self.convs2 = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs2.append(GCNConv(hidden, hidden))
        self.lin2 = Linear(hidden, hidden)
        
        self.lin3 = Linear(hidden*2, hidden)

        self.lin4 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.conv2.reset_parameters()
        for conv in self.convs2:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.lin4.reset_parameters()

    def forward(self, data1, data2):
        x1, edge_index1, batch1 = data1.x, data1.edge_index, data1.batch
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch
        x1 = F.relu(self.conv1(x1, edge_index1))
        for conv in self.convs:
            x1 = F.relu(conv(x1, edge_index1))
        x1 = global_mean_pool(x1, batch1)
        x1 = F.relu(self.lin1(x1))

        x2 = F.relu(self.conv2(x2, edge_index2))
        for conv in self.convs2:
            x2 = F.relu(conv(x2, edge_index2))
        x2 = global_mean_pool(x2, batch2)
        x2 = F.relu(self.lin2(x2))

        combined = torch.cat((x1.view(x1.size(0), -1), x2.view(x2.size(0), -1)), dim=1)
        combined = F.dropout(combined, p=0.5, training=self.training)
        combined = F.relu(self.lin3(combined))
        x = self.lin4(combined)
        return x1, x2, F.log_softmax(x, dim=-1)


class GCN_add_multilabels(torch.nn.Module):
    def __init__(self, dataset, dataset2, num_layers, hidden):
        super(GCN_add_multilabels, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)

        self.conv2 = GCNConv(dataset2.num_features, hidden)
        self.convs2 = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs2.append(GCNConv(hidden, hidden))
        self.lin2 = Linear(hidden, hidden)
        
        self.lin3 = Linear(hidden*2, hidden)

        self.lin4 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.conv2.reset_parameters()
        for conv in self.convs2:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.lin4.reset_parameters()

    def forward(self, data1, data2):
        x1, edge_index1, batch1 = data1.x, data1.edge_index, data1.batch
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch
        x1 = F.relu(self.conv1(x1, edge_index1))
        for conv in self.convs:
            x1 = F.relu(conv(x1, edge_index1))
        x1 = global_mean_pool(x1, batch1)
        x1 = F.relu(self.lin1(x1))

        x2 = F.relu(self.conv2(x2, edge_index2))
        for conv in self.convs2:
            x2 = F.relu(conv(x2, edge_index2))
        x2 = global_mean_pool(x2, batch2)
        x2 = F.relu(self.lin2(x2))

        combined = torch.cat((x1.view(x1.size(0), -1), x2.view(x2.size(0), -1)), dim=1) #将多维的tensor转成1维的
        combined = F.dropout(combined, p=0.5, training=self.training)
        combined = F.relu(self.lin3(combined))
        x = self.lin4(combined)
        return x1, x2, x