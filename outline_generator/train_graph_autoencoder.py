import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import Adam

import twpgen_config as args
from graph_embedding_dataset import GAEDataLoader
from preprocess_graph_features import *
from graph_autoencoder_model import GAE, VGAE
from clustering_metrics import NMI

# from torch.utils.tensorboard import SummaryWriter


def main(
    graph_path: str, corpus_info_path: str, feature_path: str, clusters: int
) -> None:
    print("start loading data...")
    data_loader = GAEDataLoader(graph_path, corpus_info_path, feature_path)
    adj = data_loader.A
    features = data_loader.X

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix(
        (adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape
    )
    adj_orig.eliminate_zeros()

    (
        adj_train,
        train_edges,
        val_edges,
        val_edges_false,
        test_edges,
        test_edges_false,
    ) = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Create Model
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    pos_weight = min(pos_weight, 10.0)  # 限制最大权重，避免极稀疏图梯度失控
    norm = (
        adj.shape[0]
        * adj.shape[0]
        / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    )

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    adj_norm = torch.sparse.FloatTensor(
        torch.LongTensor(adj_norm[0].T),
        torch.FloatTensor(adj_norm[1]),
        torch.Size(adj_norm[2]),
    )
    adj_label = torch.sparse.FloatTensor(
        torch.LongTensor(adj_label[0].T),
        torch.FloatTensor(adj_label[1]),
        torch.Size(adj_label[2]),
    )
    features = torch.sparse.FloatTensor(
        torch.LongTensor(features[0].T),
        torch.FloatTensor(features[1]),
        torch.Size(features[2]),
    )

    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight

    # init model and optimizer

    # model = getattr(model,args.model)(adj_norm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj_norm = adj_norm.to(device)
    model = None
    if args.graph_model == "GAE":
        model = GAE(adj_norm, device)
    elif args.graph_model == "VGAE":
        model = VGAE(adj_norm, device)
    else:
        raise NotImplementedError("model {} not implemented".format(args.model))
    model.to(device)

    features = features.to(device)
    adj_label = adj_label.to(device)
    weight_tensor = weight_tensor.to(device)
    # val_edges = val_edges.to(device)
    # val_edges_false = val_edges_false.to(device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # GAE evaluate
    def get_scores(edges_pos, edges_neg, adj_rec):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        preds = []
        pos = []
        for e in edges_pos:
            # print(e)
            # print(adj_rec[e[0], e[1]])
            preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
            pos.append(adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in edges_neg:

            preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
            neg.append(adj_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score

    def get_acc(adj_rec, adj_label):
        labels_all = adj_label.to(adj_rec.device).to_dense().view(-1).long()
        preds_all = (adj_rec > 0.5).view(-1).long()
        accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
        return accuracy

    # writer = SummaryWriter("./runs/GAE1_experiment_2d_100_epochs")
    # train model

    for epoch in range(args.num_epoch):
        t = time.time()

        A_pred = model(features)

        optimizer.zero_grad()
        loss = norm * F.binary_cross_entropy(
            A_pred.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor
        )
        
        if args.graph_model == "VGAE":    
            kl_divergence = (
                0.5
                / A_pred.size(0)
                * (
                    1
                    + 2 * model.logstd
                    - model.mean**2
                    - torch.exp(model.logstd) ** 2
                )
                .sum(1)
                .mean()
            )
            loss -= kl_divergence
        
        loss.backward()
        optimizer.step()
        
        train_acc = get_acc(A_pred.cpu(), adj_label)
        val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred.cpu())
        print(
            "Epoch:",
            "%04d" % (epoch + 1),
            "train_loss=",
            "{:.5f}".format(loss.item()),
            "train_acc=",
            "{:.5f}".format(train_acc),
            "val_roc=",
            "{:.5f}".format(val_roc),
            "val_ap=",
            "{:.5f}".format(val_ap),
            "time=",
            "{:.5f}".format(time.time() - t),
        )
        # writer.add_scalar("train_loss train", loss.item(), epoch)
        # writer.add_scalar("train_acc train", train_acc, epoch)
        # writer.add_scalar("val_roc train", val_roc, epoch)
        # writer.add_scalar("val_ap train", val_ap, epoch)
    A_pred = A_pred.cpu()
    test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
    print(
        "End of training!",
        "test_roc=",
        "{:.5f}".format(test_roc),
        "test_ap=",
        "{:.5f}".format(test_ap),
    )

    # remove derived node of Z
    z = model.mean
    indices = data_loader.base_nodes_id
    indices_z = torch.tensor(indices, dtype=torch.long)
    selected_z = z[indices_z]
    selected_z = selected_z.detach().cpu().numpy()

    # save GAE embedding
    np.savetxt(args.gae_feature_path, selected_z, delimiter=",")

    # Cluster
    C_model = KMeans(n_clusters=clusters, verbose=1, max_iter=100, tol=0.01, n_init=3)
    C_model.fit(selected_z)
    labels_pred = C_model.labels_
    labels_true = data_loader.Y

    def plot_embedding(data, label, title):
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)

        fig = plt.figure()
        ax = plt.subplot(111)
        for i in range(data.shape[0]):
            plt.text(
                data[i, 0],
                data[i, 1],
                str(label[i]),
                color=plt.cm.Set1(label[i] / 10.0),
                fontdict={"weight": "bold", "size": 9},
            )
        plt.xticks([])
        plt.yticks([])
        plt.title(title)
        return fig

    tsne = TSNE(n_components=2, init="pca", random_state=0)
    result = tsne.fit_transform(selected_z)
    fig = plot_embedding(result, labels_pred, "Cluster Pred")
    Path("./output/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig("./output/figures/savefig_pred_cluster.png")

    fig = plot_embedding(result, labels_true, "Cluster True")
    Path("./output/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig("./output/figures/savefig_true_cluster.png")

    ari = metrics.adjusted_rand_score(labels_true, labels_pred)
    nmi = NMI(labels_pred, labels_true)
    nmi = metrics.normalized_mutual_info_score(
        labels_true, labels_pred, average_method="min"
    )
    acc = metrics.accuracy_score(labels_true, labels_pred)
    precision_score = metrics.precision_score(
        labels_true, labels_pred, average="weighted", zero_division=1
    )
    recall_score = metrics.recall_score(
        labels_true, labels_pred, average="weighted", zero_division=1
    )
    f1_score = metrics.f1_score(
        labels_true, labels_pred, average="weighted", zero_division=1
    )
    print("ari(%): {:.2f}%".format((ari + 1) / 2 * 100))
    print("nmi(%): {:.2f}%".format(nmi * 100))
    print("acc(%): {:.2f}%".format(acc * 100))
    print("precision_score(%): {:.2f}%".format(precision_score * 100))
    print("recall_score(%): {:.2f}%".format(recall_score * 100))
    print("f1_score(%): {:.2f}%".format(f1_score * 100))


if __name__ == "__main__":
    main(
        graph_path=args.graph_path,
        corpus_info_path=args.corpus_info_path,
        feature_path=args.feature_path,
        clusters=30,
    )
