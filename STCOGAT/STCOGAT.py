import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    sequential,
    GATConv,
    InnerProductDecoder,
    TransformerConv,
    SAGEConv,
)
from torch_geometric.utils import negative_sampling
from sklearn.metrics import average_precision_score, roc_auc_score
import math
import numpy as np
import pandas as pd

EPS = 1e-15
MAX_LOGSTD = 10


class MutualEncoder(torch.nn.Module):
    def __init__(self, col_dim, row_dim, num_layers=3, drop_p=0.2):
        super(MutualEncoder, self).__init__()
        self.col_dim = col_dim
        self.row_dim = row_dim
        self.num_layers = num_layers
        self.rows_layers = nn.ModuleList(
            [
                sequential.Sequential(
                    "x,edge_index",
                    [
                        (SAGEConv(self.row_dim, self.row_dim), "x, edge_index -> x1"),
                        (nn.Dropout(drop_p, inplace=False), "x1 -> x2"),
                        nn.LeakyReLU(inplace=True),
                    ],
                )
                for _ in range(num_layers)
            ]
        )

        self.cols_layers = nn.ModuleList(
            [
                sequential.Sequential(
                    "x,edge_index",
                    [
                        (SAGEConv(self.col_dim, self.col_dim), "x, edge_index -> x1"),
                        nn.LeakyReLU(inplace=True),
                        (nn.Dropout(drop_p, inplace=False), "x1 -> x2"),
                    ],
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, knn_edge_index, genet_edge_index):
        embbded = x.clone()
        for i in range(self.num_layers):
            embbded = self.cols_layers[i](embbded.T, knn_edge_index).T
            embbded = self.rows_layers[i](embbded, genet_edge_index)
        return embbded


class TransformerConvReducrLayer(TransformerConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        dropout=0,
        add_self_loops=True,
        scale_param=2,
        **kwargs
    ):

        super().__init__(
            in_channels, out_channels, heads, dropout, add_self_loops, **kwargs
        )
        self.treshold_alpha = None
        self.scale_param = scale_param

    def message(self, query_i, key_j, value_j, edge_attr, index, ptr, size_i):
        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key_j += edge_attr
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        if not self.scale_param is None:
            alpha = alpha - alpha.mean()
            alpha = alpha / ((1 / self.scale_param) * alpha.std())
            alpha = F.sigmoid(alpha)
        else:
            alpha = softmax(alpha, index, ptr, size_i)
        self.treshold_alpha = alpha
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = value_j
        if edge_attr is not None:
            out += edge_attr
        out *= alpha.view(-1, self.heads, 1)
        return out


class DimEncoder(torch.nn.Module):
    def __init__(
        self, feature_dim, inter_dim, embd_dim, reducer=False, drop_p=0.2, scale_param=3
    ):
        super(DimEncoder, self).__init__()
        self.reducer = reducer
        self.gcn = GATConv(feature_dim, inter_dim)
        self.res_proj1 = nn.Linear(feature_dim, inter_dim, bias=False)
        self.norm1 = nn.BatchNorm1d(inter_dim)

        if self.reducer:
            self.trans = TransformerConvReducrLayer(
                inter_dim,
                embd_dim,
                dropout=drop_p,
                add_self_loops=False,
                heads=1,
                scale_param=scale_param,
            )
        else:
            self.trans = GATConv(
                inter_dim, embd_dim, heads=1, concat=False, dropout=drop_p
            )

        self.res_proj2 = nn.Linear(inter_dim, embd_dim, bias=False)
        self.norm2 = nn.BatchNorm1d(embd_dim)

        self.project = nn.Linear(inter_dim + embd_dim, embd_dim)
        self.contrastive_projection = nn.Linear(embd_dim, embd_dim)
        self.atten_map = None
        self.atten_weights = None
        self.plot_count = 0

    def forward(self, x, edge_index, infrance=False):
        h0 = x  
        h1 = self.gcn(h0, edge_index)  
        res1 = self.res_proj1(h0)  
        h1 = self.norm1(h1 + res1)  
        h1 = torch.cat([h1 + res1], dim=1)  

        h2, atten_map = self.trans(h1, edge_index, return_attention_weights=True)
        res2 = self.res_proj2(h1)  
        h2 = self.norm2(h2 + res2)  
        h2 = torch.cat([h2 + res2], dim=1)  

        h_cat = torch.cat([h1, h2], dim=1)  
        out = F.elu(self.project(h_cat))  
        if self.reducer and not infrance:
            if self.atten_map is None:
                self.atten_map = atten_map[0].detach()
                self.atten_weights = atten_map[1].detach()
            else:
                self.atten_map = torch.concat(
                    [self.atten_map.T, atten_map[0].detach().T]
                ).T
                self.atten_weights = torch.concat(
                    [self.atten_weights, atten_map[1].detach()]
                )
        return out

    def contrastive_loss(self, z, pos_edge_index, neg_edge_index):
        pos_z_i = z[pos_edge_index[0]]
        pos_z_j = z[pos_edge_index[1]]
        neg_z_i = z[neg_edge_index[0]]
        neg_z_j = z[neg_edge_index[1]]
        pos_sim = F.cosine_similarity(
            self.contrastive_projection(pos_z_i), self.contrastive_projection(pos_z_j)
        ).mean()

        neg_sim = F.cosine_similarity(
            self.contrastive_projection(neg_z_i), self.contrastive_projection(neg_z_j)
        ).mean()
        return -pos_sim + neg_sim

    def reduce_network(self, threshold=0.1, min_connect=6):
        self.plot_count += 1
        graph = self.atten_weights.cpu().detach().numpy()
        threshold_bound = np.percentile(graph, 10)
        threshold = min(threshold, threshold_bound)
        df = pd.DataFrame(
            {
                "v1": self.atten_map[0].cpu().detach().numpy(),
                "v2": self.atten_map[1].cpu().detach().numpy(),
                "atten": graph.squeeze(),
            }
        )

        saved_edges = df.groupby("v1")["atten"].nlargest(min_connect).index.values
        saved_edges = [v2 for _, v2 in saved_edges]
        df.iloc[saved_edges, 2] = threshold + EPS
        indexs = list(df.loc[df.atten >= threshold].index)
        atten_map = self.atten_map[:, indexs]
        self.atten_map = None
        self.atten_weights = None
        return atten_map, df


class FeatureDecoder(torch.nn.Module):
    def __init__(self, feature_dim, embd_dim, inter_dim, drop_p=0.0):
        super(FeatureDecoder, self).__init__()
        self.feature_dim = feature_dim
        self.embd_dim = embd_dim
        self.inter_dim = inter_dim
        self.decoder = nn.Sequential(
            nn.Linear(embd_dim, inter_dim),
            nn.Dropout(drop_p),
            nn.ELU(),
            nn.Linear(inter_dim, inter_dim),
            nn.Dropout(drop_p),
            nn.ELU(),
            nn.Linear(inter_dim, feature_dim),
            nn.Dropout(drop_p),
        )

    def forward(self, z):
        out = self.decoder(z)
        return out


class STCOGAT(torch.nn.Module):
    def __init__(
        self,
        col_dim,
        row_dim,
        inter_row_dim,
        embd_row_dim,
        inter_col_dim,
        embd_col_dim,
        lambda_rows=1,
        lambda_cols=1,
        pre_emb_dim=16,
        lambda_cluster=0.1,
        lambda_smooth=0.1,
        lambda_contra=0.1,
        num_clusters=5,
        alpha=3.0,
        num_layers=3,
        drop_p=0.2,
        use_rows_encoder=False,
    ):

        super().__init__()
        self.col_dim = col_dim
        self.row_dim = row_dim
        self.lambda_rows = lambda_rows
        self.lambda_cols = lambda_cols
        self.lambda_cluster = lambda_cluster
        self.lambda_smooth = lambda_smooth
        self.lambda_contra = lambda_contra
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.use_rows_encoder = use_rows_encoder
        self.cluster_layer = nn.Parameter(
            torch.randn(self.num_clusters, embd_col_dim + pre_emb_dim)
        )
        self.encoder = MutualEncoder(col_dim, row_dim, num_layers, drop_p)
        if self.use_rows_encoder:
            self.rows_encoder = DimEncoder(
                row_dim,
                inter_row_dim,
                embd_row_dim,
                reducer=False,
                drop_p=drop_p,
                scale_param=None,
            )
        self.cols_encoder = DimEncoder(
            col_dim, inter_col_dim, embd_col_dim, reducer=True, drop_p=drop_p
        )

        self.knn_pre_mlp = nn.Sequential(
            nn.Linear(col_dim, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(drop_p),
            nn.Linear(64, pre_emb_dim),
            nn.BatchNorm1d(pre_emb_dim),
            nn.ELU(),
            nn.Dropout(drop_p),
        )

        self.feature_decoder = FeatureDecoder(
            col_dim, embd_col_dim + pre_emb_dim, inter_col_dim, drop_p=0.0
        )

        self.ipd = InnerProductDecoder()
        self.feature_criterion = nn.MSELoss()

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None, sig=False):
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        if not sig:
            embd = torch.corrcoef(z)
            pos = torch.sigmoid(embd[pos_edge_index[0], pos_edge_index[1]])
            neg = torch.sigmoid(embd[neg_edge_index[0], neg_edge_index[1]])
            pos_loss = -torch.log(pos + EPS).mean()
            neg_loss = -torch.log(1 - neg + EPS).mean()
        else:
            pos_loss = -torch.log(self.ipd(z, pos_edge_index, sigmoid=sig) + EPS).mean()
            neg_loss = -torch.log(
                1 - self.ipd(z, neg_edge_index, sigmoid=sig) + EPS
            ).mean()

        return pos_loss + neg_loss

    def test(self, z, pos_edge_index, neg_edge_index):
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)
        pos_pred = self.ipd(z, pos_edge_index, sigmoid=True)
        neg_pred = self.ipd(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)

    def calculate_loss(
        self,
        x,
        knn_edge_index,
        genet_edge_index,
        highly_variable_index,
        spa_edge_index,
        coords,
    ):

        embbed = self.encoder(x, knn_edge_index, genet_edge_index)
        embbed_cols = self.cols_encoder(embbed.T, knn_edge_index)
        knn_pre_emb = self.knn_pre_mlp(embbed.T)
        combined_cols = torch.cat([embbed_cols, knn_pre_emb], dim=1)
        out_features = self.feature_decoder(combined_cols)
        out_features = (out_features - out_features.mean(0)) / (out_features.std(0) + EPS)
        reg = self.recon_loss(out_features.T, genet_edge_index, sig=False)
        col_loss = self.feature_criterion(
            x[highly_variable_index.values].T,
            out_features.T[highly_variable_index.values].T,
        )
        recon_total = self.lambda_cols * (col_loss + reg)
        if self.use_rows_encoder:
            row_loss = self.recon_loss(
                self.rows_encoder(embbed, genet_edge_index), genet_edge_index, sig=True
            )
            recon_total = self.lambda_rows * row_loss + recon_total
        contrastive_loss = self.cols_encoder.contrastive_loss(
            embbed_cols,
            knn_edge_index,
            negative_sampling(knn_edge_index, embbed_cols.size(0)),
        )

        z = combined_cols
        dist = torch.sum((z.unsqueeze(1) - self.cluster_layer) ** 2, dim=2)
        q = 1.0 / (1.0 + dist / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / q.sum(1)).t()
        p = (q**2) / q.sum(0)
        p = (p.t() / p.sum(1)).t()
        loss_cluster = F.kl_div(q.log(), p, reduction="batchmean")
        loss_cluster = self.lambda_cluster * loss_cluster
        row = spa_edge_index[0]
        col = spa_edge_index[1]
        dists = torch.norm(coords[row] - coords[col], dim=1)
        sigma = torch.median(dists)
        sigma = sigma.item() if sigma.item() > EPS else EPS
        dist_sq = ((coords[row] - coords[col]) ** 2).sum(dim=1)
        w = torch.exp(-dist_sq / (2 * sigma**2))
        diff = (z[row] - z[col]).pow(2).sum(dim=1)
        edge_term = w * diff
        loss_smooth = edge_term.sum() / (w.sum() + 1e-8)
        loss = (
            recon_total
            + self.lambda_smooth * loss_smooth
            + self.lambda_contra * contrastive_loss
        )
        return loss, recon_total, loss_cluster

    def forward(self, x, knn_edge_index, genet_edge_index):
        embbed = self.encoder(x, knn_edge_index, genet_edge_index)
        if self.use_rows_encoder:
            embbed_rows = self.rows_encoder(embbed, genet_edge_index)
        else:
            embbed_rows = None
        embbed_cols = self.cols_encoder(embbed.T, knn_edge_index, infrance=True)
        knn_pre_emb = self.knn_pre_mlp(embbed.T)
        combined_cols = torch.cat([embbed_cols, knn_pre_emb], dim=1)
        out_features = self.feature_decoder(combined_cols)
        return embbed_rows, combined_cols, out_features
