import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.pointnet import PointNetFeature, STN3d, STN64d, STN6d


def prepare_input_first_level(tensor_x, tensor_quantiles):
    x = tensor_x.data.cpu().numpy()
    quantiles = tensor_quantiles.numpy()

    base = 0

    stack_quantiles = quantiles.reshape(-1)
    nonzero_quantiles = stack_quantiles[np.nonzero(stack_quantiles)]  # get the nonzero entries
    max_len = np.max(nonzero_quantiles)
    num_seq = nonzero_quantiles.size

    res_seq = np.zeros((num_seq, max_len, 3))

    stack_data = x.reshape(-1, 3)
    for i in range(num_seq):
        num = nonzero_quantiles[i]
        res_seq[i, 0:num, :] = stack_data[base:base + num, :]
        base = base + num

    # sort the data by sequence length
    sort_index = np.argsort(nonzero_quantiles)[::-1]  # sort in descending order according to length
    inverse_index = np.argsort(sort_index)

    sort_data = res_seq[sort_index, :, :]
    sort_len = nonzero_quantiles[sort_index]

    return sort_data, sort_len, inverse_index


class RNNUnit(nn.Module):
    def __init__(self, device, hidden_size):
        super(RNNUnit, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.rnn = nn.GRU(input_size=3, hidden_size=self.hidden_size,
                          num_layers=2, bidirectional=False, batch_first=True).to(device)

    def forward(self, x, quantiles, seq_data, seq_len, inverse_index, items_indices):
        packed_seq_data = pack_padded_sequence(seq_data, seq_len, batch_first=True)
        packed_output, _ = self.rnn(packed_seq_data)

        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # get the last time step for each sequence
        seq_len = torch.LongTensor(seq_len)
        idx = (seq_len - 1).view(-1, 1).expand(output.size(0), output.size(2)).unsqueeze(1)
        decoded = output.gather(1, idx.to(self.device)).squeeze()

        # unsort the input sequence
        inverse_index = torch.LongTensor(inverse_index)
        odx = inverse_index.view(-1, 1).expand(seq_len.size(0), output.size(-1))
        decoded = decoded.gather(0, odx.to(self.device))

        # finally, prepare for the data for following convolution
        batch_num = quantiles.shape[0]
        plane_num = quantiles.shape[1]

        res_image = torch.zeros(batch_num * plane_num * plane_num, output.size(2)).to(self.device)

        num_items = items_indices.shape[0]
        items_indices = torch.LongTensor(items_indices).view(-1, 1).expand(num_items, output.size(2))
        res_image = res_image.scatter_(0, items_indices.to(self.device), decoded)

        res_image = res_image.view(batch_num, plane_num, plane_num, output.size(2))
        res_image = res_image.permute(0, 3, 1, 2).contiguous()

        return res_image


class RCNet(nn.Module):
    def __init__(self, device, k=40, npoints=2048):
        super(RCNet, self).__init__()
        self.k = k
        self.npoints = npoints
        self.device = device

        self.rnn = RNNUnit(device, hidden_size=128)

        self.trans_6d = STN6d()
        self.trans_64d = STN64d()

        self.conv1 = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=False)  # this layer can be removed to reduce computational cost a bit

        self.fc1 = nn.Linear(8192, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.bn0 = nn.BatchNorm2d(128)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(512)

        # fc for segmentation
        self.seg_fc1 = torch.nn.Conv1d(1216, 512, 1)
        # self.seg_fc1 = torch.nn.Conv1d(1088, 512, 1)
        self.seg_fc2 = torch.nn.Conv1d(512, 256, 1)
        self.seg_fc3 = torch.nn.Conv1d(256, 128, 1)
        self.seg_fc4 = torch.nn.Conv1d(128, 128, 1)

        # self.seg_fc2 = torch.nn.Conv1d(1024, 256, 1)
        # self.seg_fc3 = torch.nn.Conv1d(512, 128, 1)
        # self.seg_fc4 = torch.nn.Conv1d(384, 128, 1)
        self.seg_fc5 = torch.nn.Conv1d(128, k, 1)

        self.seg_bn1 = nn.BatchNorm1d(512)
        self.seg_bn2 = nn.BatchNorm1d(256)
        self.seg_bn3 = nn.BatchNorm1d(128)
        self.seg_bn4 = nn.BatchNorm1d(128)

        # lift 3d feature into higher space
        self.pn_conv1 = torch.nn.Conv1d(6, 64, 1)
        self.pn_conv2 = torch.nn.Conv1d(64, 64, 1)
        self.pn_bn1 = nn.BatchNorm1d(64)
        self.pn_bn2 = nn.BatchNorm1d(64)

    def forward(self, x, quantiles, seq_data, seq_len, inverse_index, items_indices, gather_idx, ori_point_idx):
        # -- apply stn
        y = x
        trans = self.trans_6d(y)
        y = y.transpose(2, 1)
        y = torch.bmm(y, trans)
        y = y.transpose(2, 1)

        x_trans = F.relu(self.pn_bn1(self.pn_conv1(x)))
        x_trans = F.relu(self.pn_bn2(self.pn_conv2(x_trans)))

        trans = self.trans_64d(x_trans)
        x_trans = x_trans.transpose(2, 1)
        x_trans = torch.bmm(x_trans, trans)
        x_trans = x_trans.transpose(2, 1)
        # --

        # x = torch.transpose(x, 2, 1).contiguous()

        x = x.transpose(2, 1).contiguous()
        res_image = self.rnn(x, quantiles, seq_data, seq_len, inverse_index, items_indices)
        res_image = self.bn0(res_image)

        point_feats_1 = get_point_local_features(res_image, gather_idx, ori_point_idx, self.device)  # improves the performance but not too much. Delete it to futher improve efficiency

        x = F.relu(self.bn1(self.conv1(res_image)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        # point_feats_2 = F.interpolate(x, scale_factor=2, mode='nearest')
        # point_feats_2 = get_point_local_features(point_feats_2, gather_idx, ori_point_idx, self.device)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)

        # point_feats_3 = F.interpolate(x, scale_factor=4, mode='nearest')
        # point_feats_3 = get_point_local_features(point_feats_3, gather_idx, ori_point_idx, self.device)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 8192)
        x = F.dropout(x, 0.8)
        x = F.relu(self.fc_bn1(self.fc1(x)))

        x = x.view(-1, 1024, 1).repeat(1, 1, self.npoints)
        x_combined = torch.cat((x_trans, x, point_feats_1), 1)
        # x_combined = torch.cat((x_trans, x, point_feats_1, point_feats_2, point_feats_3), 1)
        # x_combined = torch.cat((x_trans, x), 1)
        x_combined = F.dropout(x_combined)

        x_combined = F.relu(self.seg_bn1(self.seg_fc1(x_combined)))  # seems like it is unnecessary to use so many fc layers. 3 fc layers are good as well.
        # x_combined = F.dropout(x_combined, 0.8)

        x_combined = F.relu(self.seg_bn2(self.seg_fc2(x_combined)))

        x_combined = F.relu(self.seg_bn3(self.seg_fc3(x_combined)))

        x_combined = F.relu(self.seg_bn4(self.seg_fc4(x_combined)))
        x_combined = self.seg_fc5(x_combined)

        x_combined = x_combined.transpose(2, 1).contiguous()
        x_combined = F.log_softmax(x_combined.view(-1, self.k), dim=1)
        x_combined = x_combined.view(-1, self.npoints, self.k)

        return x_combined


class EnsembleRCNet(nn.Module):
    def __init__(self, device, which_dir, k=40, npoints=1024):
        super(EnsembleRCNet, self).__init__()
        self.which_dir = which_dir

        if which_dir == 1:
            self.RCNet1 = RCNet(device, k, npoints)
        elif which_dir == 2:
            self.RCNet2 = RCNet(device, k, npoints)
        else:
            self.RCNet3 = RCNet(device, k, npoints)

    def forward(self, x, quantiles, seq_data, seq_len, inverse_index, items_indices, gather_idx, ori_point_idx):
        if self.which_dir == 1:
            x = self.RCNet1(x, quantiles, seq_data, seq_len, inverse_index, items_indices, gather_idx, ori_point_idx)
        elif self.which_dir == 2:
            x = self.RCNet2(x, quantiles, seq_data, seq_len, inverse_index, items_indices, gather_idx, ori_point_idx)
        else:
            x = self.RCNet3(x, quantiles, seq_data, seq_len, inverse_index, items_indices, gather_idx, ori_point_idx)

        return x


def get_point_local_features(img, gather_idx, ori_point_idx, device):
    batch_size = img.size(0)
    feature_size = img.size(1)
    img = img.view(batch_size, feature_size, -1)

    gather_idx = gather_idx.numpy()
    gather_idx = torch.LongTensor(gather_idx).unsqueeze(1).expand(-1, feature_size, -1)
    gather_idx = gather_idx.to(device)
    gather_feats = torch.gather(img, 2, gather_idx)

    # point_num = ori_point_idx.shape[1]
    # res_point_feats = torch.zeros(batch_size, feature_size, point_num).to(device)
    # ori_point_idx = torch.LongTensor(ori_point_idx).unsqueeze(1).expand(-1, feature_size, -1)
    # res_point_feats = res_point_feats.scatter_(2, ori_point_idx.to(device), gather_feats)

    return gather_feats


if __name__ == '__main__':
    a = np.array([[1,1,1],[2,2,2], [3,3,3], [4,4,4]])
    q1 = [2,0,2]
    b = np.array([[4,5,6],[7,8,9],[9,9,9], [10,10,10]])
    q2 = [0,3,1]
    xx = np.array([a,b])
    quant = np.array([q1,q2])
