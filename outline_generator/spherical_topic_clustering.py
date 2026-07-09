import argparse
import os
import pickle as pk

# # # # import ipdb  # 注释掉，不是必需的
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


class AutoEncoder(nn.Module):
    def __init__(
        self, input_dim1, input_dim2, input_dim3, input_dim4, input_dim5, hidden_dims, agg, sep_decode
    ):
        super(AutoEncoder, self).__init__()

        self.agg = agg
        self.sep_decode = sep_decode

        print("hidden_dims:", hidden_dims)
        print(f"输入维度: input_dim1={input_dim1}, input_dim2={input_dim2}, input_dim3={input_dim3}, input_dim4={input_dim4}, input_dim5={input_dim5}")
        
        # 确保编码器和解码器的维度匹配
        self.encoder_layers = []
        self.encoder2_layers = []
        self.encoder3_layers = []
        self.encoder4_layers = []  # 简化后句子嵌入向量的编码器
        self.encoder5_layers = []  # 新增：事件嵌入向量的编码器
        dims = [[input_dim1, input_dim2, input_dim3, input_dim4, input_dim5]] + hidden_dims
        for i in range(len(dims) - 1):
            if i == 0:
                # 第一层，从输入维度到第一个隐藏层
                layer = nn.Sequential(nn.Linear(dims[i][0], dims[i + 1]), nn.ReLU())
                layer2 = nn.Sequential(nn.Linear(dims[i][1], dims[i + 1]), nn.ReLU())
                layer3 = nn.Sequential(nn.Linear(dims[i][2], dims[i + 1]), nn.ReLU())
                layer4 = nn.Sequential(nn.Linear(dims[i][3], dims[i + 1]), nn.ReLU())
                layer5 = nn.Sequential(nn.Linear(dims[i][4], dims[i + 1]), nn.ReLU())  # 新增：事件嵌入向量
            elif i != 0 and i < len(dims) - 2:
                # 中间层，隐藏层之间的转换
                layer = nn.Sequential(nn.Linear(dims[i], dims[i + 1]), nn.ReLU())
                layer2 = nn.Sequential(nn.Linear(dims[i], dims[i + 1]), nn.ReLU())
                layer3 = nn.Sequential(nn.Linear(dims[i], dims[i + 1]), nn.ReLU())
                layer4 = nn.Sequential(nn.Linear(dims[i], dims[i + 1]), nn.ReLU())
                layer5 = nn.Sequential(nn.Linear(dims[i], dims[i + 1]), nn.ReLU())  # 新增：事件嵌入向量
            else:
                # 最后一层，到最终隐藏层
                layer = nn.Linear(dims[i], dims[i + 1])
                layer2 = nn.Linear(dims[i], dims[i + 1])
                layer3 = nn.Linear(dims[i], dims[i + 1])
                layer4 = nn.Linear(dims[i], dims[i + 1])
                layer5 = nn.Linear(dims[i], dims[i + 1])  # 新增：事件嵌入向量
            self.encoder_layers.append(layer)
            self.encoder2_layers.append(layer2)
            self.encoder3_layers.append(layer3)
            self.encoder4_layers.append(layer4)
            self.encoder5_layers.append(layer5)  # 新增：事件嵌入向量
        self.encoder = nn.Sequential(*self.encoder_layers)
        self.encoder2 = nn.Sequential(*self.encoder2_layers)
        self.encoder3 = nn.Sequential(*self.encoder3_layers)
        self.encoder4 = nn.Sequential(*self.encoder4_layers)
        self.encoder5 = nn.Sequential(*self.encoder5_layers)  # 新增：事件嵌入向量

        # 解码器部分
        self.decoder_layers = []
        self.decoder2_layers = []
        self.decoder3_layers = []
        self.decoder4_layers = []  # 简化后句子嵌入向量的解码器
        self.decoder5_layers = []  # 新增：事件嵌入向量的解码器
        hidden_dims.reverse()
        dims = hidden_dims + [[input_dim1, input_dim2, input_dim3, input_dim4, input_dim5]]
        
        # 处理concat聚合方法
        if self.agg == "concat":
            # 对于concat方法，z的维度是5*最终隐藏层维度（增加了事件特征）
            first_decoder_dim = 5 * hidden_dims[0]
        else:
            first_decoder_dim = hidden_dims[0]
            
        for i in range(len(dims) - 1):
            if i == 0:
                # 第一层解码器，处理聚合后的z
                if not self.sep_decode:
                    # 如果不是分离解码，则所有解码器共享相同的输入维度
                    input_dim = first_decoder_dim
                else:
                    # 如果是分离解码，则每个解码器使用各自的z
                    input_dim = dims[i]
                    
                layer = nn.Sequential(nn.Linear(input_dim, dims[i + 1]), nn.ReLU())
                layer2 = nn.Sequential(nn.Linear(input_dim, dims[i + 1]), nn.ReLU())
                layer3 = nn.Sequential(nn.Linear(input_dim, dims[i + 1]), nn.ReLU())
                layer4 = nn.Sequential(nn.Linear(input_dim, dims[i + 1]), nn.ReLU())
                layer5 = nn.Sequential(nn.Linear(input_dim, dims[i + 1]), nn.ReLU())  # 新增：事件嵌入向量
            elif i < len(dims) - 2:
                # 中间层
                layer = nn.Sequential(nn.Linear(dims[i], dims[i + 1]), nn.ReLU())
                layer2 = nn.Sequential(nn.Linear(dims[i], dims[i + 1]), nn.ReLU())
                layer3 = nn.Sequential(nn.Linear(dims[i], dims[i + 1]), nn.ReLU())
                layer4 = nn.Sequential(nn.Linear(dims[i], dims[i + 1]), nn.ReLU())
                layer5 = nn.Sequential(nn.Linear(dims[i], dims[i + 1]), nn.ReLU())  # 新增：事件嵌入向量
            else:
                # 最后一层，输出到原始维度
                layer = nn.Linear(dims[i], dims[i + 1][0])
                layer2 = nn.Linear(dims[i], dims[i + 1][1])
                layer3 = nn.Linear(dims[i], dims[i + 1][2])
                layer4 = nn.Linear(dims[i], dims[i + 1][3])
                layer5 = nn.Linear(dims[i], dims[i + 1][4])  # 新增：事件嵌入向量
            self.decoder_layers.append(layer)
            self.decoder2_layers.append(layer2)
            self.decoder3_layers.append(layer3)
            self.decoder4_layers.append(layer4)
            self.decoder5_layers.append(layer5)  # 新增：事件嵌入向量
        self.decoder = nn.Sequential(*self.decoder_layers)
        self.decoder2 = nn.Sequential(*self.decoder2_layers)
        self.decoder3 = nn.Sequential(*self.decoder3_layers)
        self.decoder4 = nn.Sequential(*self.decoder4_layers)
        self.decoder5 = nn.Sequential(*self.decoder5_layers)  # 新增：事件嵌入向量
        
        print("编码器和解码器初始化完成")

    def forward(self, x1, x2, x3, x4, x5):  # 修改：添加x5参数（事件特征）
        # 编码阶段
        z1 = self.encoder(x1)
        z2 = self.encoder2(x2)
        z3 = self.encoder3(x3)
        z4 = self.encoder4(x4)  # 简化后句子嵌入向量的编码
        z5 = self.encoder5(x5)  # 新增：事件嵌入向量的编码

        # 聚合隐藏表示
        if self.agg == "max":
            z = torch.max(torch.stack([z1, z2, z3, z4, z5]), dim=0)[0]  # 修改：加入z5
        elif self.agg == "multi":
            z = z1 * z2 * z3 * z4 * z5  # 修改：加入z5
        elif self.agg == "sum":
            z = z1 + z2 + z3 + z4 + z5  # 修改：加入z5
        elif self.agg == "concat":
            z = torch.cat([z1, z2, z3, z4, z5], dim=1)  # 修改：加入z5
        
        # 解码阶段
        if self.sep_decode:
            # 分离解码：每个编码器的输出单独解码
            x_bar1 = self.decoder(z1)
            x_bar1 = F.normalize(x_bar1, dim=-1)
            x_bar2 = self.decoder2(z2)
            x_bar2 = F.normalize(x_bar2, dim=-1)
            x_bar3 = self.decoder3(z3)
            x_bar3 = F.normalize(x_bar3, dim=-1)
            x_bar4 = self.decoder4(z4)
            x_bar4 = F.normalize(x_bar4, dim=-1)
            x_bar5 = self.decoder5(z5)  # 新增：事件嵌入向量
            x_bar5 = F.normalize(x_bar5, dim=-1)  # 新增：事件嵌入向量
        else:
            # 联合解码：使用聚合后的z进行解码
            x_bar1 = self.decoder(z)
            x_bar1 = F.normalize(x_bar1, dim=-1)
            x_bar2 = self.decoder2(z)
            x_bar2 = F.normalize(x_bar2, dim=-1)
            x_bar3 = self.decoder3(z)
            x_bar3 = F.normalize(x_bar3, dim=-1)
            x_bar4 = self.decoder4(z)
            x_bar4 = F.normalize(x_bar4, dim=-1)
            x_bar5 = self.decoder5(z)  # 新增：事件嵌入向量
            x_bar5 = F.normalize(x_bar5, dim=-1)  # 新增：事件嵌入向量

        return x_bar1, x_bar2, x_bar3, x_bar4, x_bar5, z  # 修改：返回值增加x_bar5


class TopicCluster(nn.Module):
    def __init__(self, args):
        super(TopicCluster, self).__init__()
        self.alpha = 1.0
        self.dataset_path = "./dataset/{}".format(args.dataset)
        self.args = args
        self.device = args.device
        self.temperature = args.temperature
        self.distribution = args.distribution
        self.agg_method = args.agg_method
        self.sep_decode = args.sep_decode == 1

        input_dim1 = args.input_dim1
        input_dim2 = args.input_dim2
        input_dim3 = args.input_dim3
        input_dim4 = args.input_dim4  # 简化后句子嵌入向量的维度
        input_dim5 = args.input_dim5  # 新增：事件嵌入向量的维度
        # 将字符串转换为列表
        hidden_dims = eval(args.hidden_dims)
        print(f"解析后的hidden_dims: {hidden_dims}")
        # 保存最终输出维度，用于后续topic_emb初始化
        self.final_hidden_dim = hidden_dims[-1]
        self.model = AutoEncoder(
            input_dim1,
            input_dim2,
            input_dim3,
            input_dim4,
            input_dim5,  # 新增：事件嵌入向量
            hidden_dims,
            self.agg_method,
            self.sep_decode,
        )
        if self.agg_method == "concat":
            # 对于concat方法，z的维度是4*self.final_hidden_dim（增加了一个特征）
            self.topic_emb = Parameter(
                torch.Tensor(args.n_clusters, 4 * self.final_hidden_dim)
            )
        else:
            # 确保topic_emb与z的维度匹配，z的维度是self.final_hidden_dim
            self.topic_emb = Parameter(torch.Tensor(args.n_clusters, self.final_hidden_dim))
            print(f"最终隐藏层维度(final_dim): {self.final_hidden_dim}")
        # 使用xavier正态分布初始化权重
        torch.nn.init.xavier_normal_(self.topic_emb.data)
        print(f"初始化topic_emb维度: {self.topic_emb.shape}")

    def pretrain(self, input_data, pretrain_epoch=200):
        pretrained_path = os.path.join(
            self.dataset_path, f"pretrained_{self.args.suffix}.pt"
        )
        if os.path.exists(pretrained_path) and self.args.load_pretrain:
            # load pretrain weights
            print(f"loading pretrained model from {pretrained_path}")
            self.model.load_state_dict(torch.load(pretrained_path))
        else:
            train_loader = DataLoader(
                input_data, batch_size=self.args.batch_size, shuffle=True
            )
            optimizer = Adam(self.model.parameters(), lr=self.args.lr)
            for epoch in range(pretrain_epoch):
                total_loss = 0
                for batch_idx, (x1, x2, x3, x4, x5, _, weight) in enumerate(train_loader):  # 修改：增加x5参数
                    x1 = x1.to(self.device)
                    x2 = x2.to(self.device)
                    x3 = x3.to(self.device)
                    x4 = x4.to(self.device)
                    x5 = x5.to(self.device)  # 新增：事件嵌入向量
                    weight = weight.to(self.device)
                    optimizer.zero_grad()
                    x_bar1, x_bar2, x_bar3, x_bar4, x_bar5, z = self.model(x1, x2, x3, x4, x5)  # 修改：增加x5参数和x_bar5返回值
                    loss = (
                        cosine_dist(x_bar1, x1)
                        + cosine_dist(x_bar2, x2)
                        + cosine_dist(x_bar3, x3)
                        + cosine_dist(x_bar4, x4)
                        + cosine_dist(x_bar5, x5)  # 新增：事件嵌入向量的重构损失
                    )  # , weight)
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                print(f"epoch {epoch}: loss = {total_loss / (batch_idx+1):.4f}")
            torch.save(self.model.state_dict(), pretrained_path)
            print(f"model saved to {pretrained_path}")

    def cluster_assign(self, z):
        if self.distribution == "student":
            p = 1.0 / (
                1.0
                + torch.sum(torch.pow(z.unsqueeze(1) - self.topic_emb, 2), 2)
                / self.alpha
            )
            p = p.pow((self.alpha + 1.0) / 2.0)
            p = (p.t() / torch.sum(p, 1)).t()
        else:
            self.topic_emb.data = F.normalize(self.topic_emb.data, dim=-1)
            z = F.normalize(z, dim=-1)
            sim = torch.matmul(z, self.topic_emb.t()) / self.temperature
            p = F.softmax(sim, dim=-1)
        return p

    def forward(self, x1, x2, x3, x4, x5):  # 修改：增加x5参数（事件特征）
        x_bar1, x_bar2, x_bar3, x_bar4, x_bar5, z = self.model(x1, x2, x3, x4, x5)  # 修改：增加x5参数和x_bar5返回值
        p = self.cluster_assign(z)
        return x_bar1, x_bar2, x_bar3, x_bar4, x_bar5, z, p  # 修改：增加x_bar5返回值

    def target_distribution(self, x1, x2, x3, x4, x5, freq, method="all", top_num=0):  # 修改：增加x5参数（事件特征）
        _, _, _, _, _, z = self.model(x1, x2, x3, x4, x5)  # 修改：增加x5参数和x_bar5返回值
        p = self.cluster_assign(z).detach()
        if method == "all":
            q = p**2 / (p * freq.unsqueeze(-1)).sum(dim=0)
            q = (q.t() / q.sum(dim=1)).t()
        elif method == "top":
            assert top_num > 0
            q = p.clone()
            sim = torch.matmul(self.topic_emb, z.t())
            _, selected_idx = sim.topk(k=top_num, dim=-1)
            for i, topic_idx in enumerate(selected_idx):
                q[topic_idx] = 0
                q[topic_idx, i] = 1
        return p, q


def cosine_dist(x_bar, x, weight=None):
    if weight is None:
        weight = torch.ones(x.size(0), device=x.device)
    cos_sim = (x_bar * x).sum(-1)
    cos_dist = 1 - cos_sim
    cos_dist = (cos_dist * weight).sum() / weight.sum()
    return cos_dist


def train(args, emb_dict):
    # ipdb.set_trace()
    inv_vocab = {k: "/".join(v) for k, v in emb_dict["inv_vocab"].items()}
    for i, ent in enumerate(emb_dict["ents_vocab"]):
        inv_vocab[i] = inv_vocab[i] + "/" + "/".join(ent)
    for i in range(len(inv_vocab)):
        inv_vocab[i] = inv_vocab[i] + "/" + emb_dict["sents_id"][i]
    vocab = {" ".join(k): v for k, v in emb_dict["vocab"].items()}
    print(f"Vocab size: {len(vocab)}")
    
    # 打印各个嵌入向量的维度
    print(f"vs_emb 维度: {emb_dict['vs_emb'].shape}")
    print(f"oh_emb 维度: {emb_dict['oh_emb'].shape}")
    print(f"sents_graph_emb 维度: {emb_dict['sents_graph_emb'].shape}")
    print(f"summarized_emb 维度: {emb_dict['summarized_emb'].shape}")
    print(f"event_emb 维度: {emb_dict['event_emb'].shape}")  # 新增：事件嵌入向量
    
    embs = F.normalize(torch.tensor(emb_dict["vs_emb"]), dim=-1)
    embs2 = F.normalize(torch.tensor(emb_dict["oh_emb"]), dim=-1)
    embs3 = F.normalize(torch.tensor(emb_dict["sents_graph_emb"]), dim=-1)
    embs4 = F.normalize(torch.tensor(emb_dict["summarized_emb"]), dim=-1)  # 简化后句子嵌入向量
    embs5 = F.normalize(torch.tensor(emb_dict["event_emb"]), dim=-1)  # 新增：事件嵌入向量
    freq = np.array(emb_dict["tuple_freq"])
    if not args.use_freq:
        freq = np.ones_like(freq)

    input_data = TensorDataset(
        embs,
        embs2,
        embs3,
        embs4,
        embs5,  # 新增：事件嵌入向量
        torch.arange(embs.size(0)),
        torch.tensor(freq),
    )
    topic_cluster = TopicCluster(args).to(args.device)
    topic_cluster.pretrain(input_data, args.pretrain_epoch)
    train_loader = DataLoader(input_data, batch_size=args.batch_size, shuffle=False)
    optimizer = Adam(topic_cluster.parameters(), lr=args.lr)

    # topic embedding initialization
    embs = embs.to(args.device)
    embs2 = embs2.to(args.device)
    embs3 = embs3.to(args.device)
    embs4 = embs4.to(args.device)
    embs5 = embs5.to(args.device)  # 新增：事件嵌入向量
    # embs_ent = embs_ent.to(args.device)
    x_bar1, x_bar2, x_bar3, x_bar4, x_bar5, z = topic_cluster.model(embs, embs2, embs3, embs4, embs5)  # 修改：增加x5参数和x_bar5返回值
    z = F.normalize(z, dim=-1)

    n_samples = z.shape[0]
    effective_n_clusters = min(args.n_clusters, max(2, n_samples - 1))
    if effective_n_clusters != args.n_clusters:
        print(f"  WARNING: n_samples={n_samples} < n_clusters={args.n_clusters}, auto set to {effective_n_clusters}")

    print(f"Running K-Means for initialization (n_clusters={effective_n_clusters})")
    kmeans = KMeans(n_clusters=effective_n_clusters, n_init=5)
    if args.use_freq:
        y_pred = kmeans.fit_predict(z.data.cpu().numpy(), sample_weight=freq)
    else:
        y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    print(f"Finish K-Means")

    freq = torch.tensor(freq).to(args.device)

    y_pred_last = y_pred
    topic_cluster.topic_emb.data = torch.tensor(kmeans.cluster_centers_).to(args.device)

    topic_cluster.train()
    i = 0
    for epoch in range(50):
        if epoch % 5 == 0:
            _, _, _, _, _, z, p = topic_cluster(embs, embs2, embs3, embs4, embs5)  # 修改：增加x5参数和x_bar5返回值
            z = F.normalize(z, dim=-1)
            topic_cluster.topic_emb.data = F.normalize(
                topic_cluster.topic_emb.data, dim=-1
            )
            if not os.path.exists(
                os.path.join(topic_cluster.dataset_path, f"clusters_{args.suffix}")
            ):
                os.makedirs(
                    os.path.join(topic_cluster.dataset_path, f"clusters_{args.suffix}")
                )
            embed_save_path = os.path.join(
                topic_cluster.dataset_path, f"clusters_{args.suffix}/embed_{epoch}.pt"
            )
            torch.save(
                {
                    "inv_vocab": emb_dict["inv_vocab"],
                    "embed": z.detach().cpu().numpy(),
                    "topic_embed": topic_cluster.topic_emb.detach().cpu().numpy(),
                },
                embed_save_path,
            )
            cluster_path = os.path.join(
                topic_cluster.dataset_path, f"clusters_{args.suffix}/cluster.txt"
            )
            if os.path.exists(cluster_path):
                os.remove(cluster_path)
            f = open(
                # os.path.join(args.dataset_path, f"clusters_{args.suffix}/{epoch}.txt"),
                cluster_path,
                "w",
            )
            pred_cluster = p.argmax(-1).cpu()
            result_strings = []
            for j in range(args.n_clusters):
                if args.sort_method == "discriminative":
                    word_idx = torch.arange(embs.size(0))[pred_cluster == j]
                    sorted_idx = torch.argsort(
                        p[pred_cluster == j][:, j].cpu(), descending=True
                    )
                    word_idx = word_idx[sorted_idx]
                else:
                    sim = torch.matmul(topic_cluster.topic_emb[j], z.t())
                    _, word_idx = sim.topk(k=30, dim=-1)
                word_cluster = []

                freq_sum = 0
                for idx in word_idx:
                    freq_sum += freq[idx].item()
                    if inv_vocab[idx.item()] not in word_cluster:
                        word_cluster.append(str(idx.item()) + "/" + inv_vocab[idx.item()])
                        if len(word_cluster) >= 10:
                            break

                if len(word_cluster) == 0:
                    continue
                result_strings.append(
                    (
                        freq_sum,
                        f"Topic {j} ({freq_sum}): " + ", ".join(word_cluster) + "\n",
                    )
                )
            # result_strings = sorted(result_strings, key=lambda x: x[0], reverse=True)
            for result_string in result_strings:
                f.write(result_string[1])
            f.close()

        for x1, x2, x3, x4, x5, idx, weight in train_loader:  # 修改：增加x5参数（事件特征）

            if i % args.update_interval == 0:
                p, q = topic_cluster.target_distribution(
                    embs,
                    embs2,
                    embs3,
                    embs4,
                    embs5,  # 新增：事件嵌入向量
                    freq.clone().fill_(1),
                    method="all",
                    top_num=epoch + 1,
                )

                y_pred = p.cpu().numpy().argmax(1)
                delta_label = (
                    np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                )
                y_pred_last = y_pred

                if i > 0 and delta_label < args.tol:
                    print(f"delta_label {delta_label:.4f} < tol ({args.tol})")
                    print("Reached tolerance threshold. Stopping training.")
                    return z.detach().cpu().numpy()

            i += 1
            x1 = x1.to(args.device)
            x2 = x2.to(args.device)
            x3 = x3.to(args.device)
            x4 = x4.to(args.device)
            x5 = x5.to(args.device)  # 新增：事件嵌入向量
            idx = idx.to(args.device)
            weight = weight.to(args.device)

            x_bar1, x_bar2, x_bar3, x_bar4, x_bar5, _, p = topic_cluster(x1, x2, x3, x4, x5)  # 修改：增加x5参数和x_bar5返回值
            reconstr_loss = (
                cosine_dist(x_bar1, x1)
                + cosine_dist(x_bar2, x2)
                + cosine_dist(x_bar3, x3)
                + cosine_dist(x_bar4, x4)
                + cosine_dist(x_bar5, x5)  # 新增：事件嵌入向量的重构损失
            )  # , weight)
            kl_loss = F.kl_div(p.log(), q[idx], reduction="none").sum(-1)
            kl_loss = (kl_loss * weight).sum() / weight.sum()
            loss = args.gamma * kl_loss + reconstr_loss
            if i % args.update_interval == 0:
                print(f"KL loss: {kl_loss}; Reconstruction loss: {reconstr_loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return z.detach().cpu().numpy()


def filter_vocab(emb_dict):
    stop_words = set(stopwords.words("english"))
    new_inv_vocab = [
        w
        for w, _ in emb_dict["vocab"].items()
        if w not in stop_words and not w.startswith("##")
    ]
    new_vocab = {w: i for i, w in enumerate(new_inv_vocab)}
    new_avg_emb = emb_dict["avg_emb"][[emb_dict["vocab"][w] for w in new_inv_vocab]]
    return {"avg_emb": new_avg_emb, "vocab": new_vocab, "inv_vocab": new_inv_vocab}


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=0 python3 spherical_topic_clustering.py --dataset_path ./pandemic --input_emb_name po_tuple_features_all_svos.pk
    parser = argparse.ArgumentParser(
        description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dataset_path", default="./covid19", type=str)
    parser.add_argument(
        "--input_emb_name", default="po_tuple_features_all_svos.pk", type=str
    )
    # parser.add_argument('--dataset_path', type=str)
    # parser.add_argument('--input_emb_name', type=str)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--n_clusters", default=30, type=int)
    parser.add_argument(
        "--input_dim1", default=1000, type=int
    )  # default for covid19 dataset 736
    parser.add_argument(
        "--input_dim2", default=1000, type=int
    )  # default for covid19 dataset 242
    parser.add_argument("--input_dim3", default=1000, type=int)
    parser.add_argument("--input_dim4", default=1000, type=int) # 简化后句子嵌入向量维度
    parser.add_argument("--input_dim5", default=1000, type=int) # 新增：事件嵌入向量维度
    parser.add_argument(
        "--agg_method",
        default="multi",
        choices=["sum", "multi", "concat", "attend"],
        type=str,
    )
    parser.add_argument("--sep_decode", default=0, choices=[0, 1], type=int)
    parser.add_argument("--pretrain_epoch", default=300, type=int)
    parser.add_argument("--load_pretrain", default=False, action="store_true")
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument(
        "--sort_method", default="generative", choices=["generative", "discriminative"]
    )
    parser.add_argument(
        "--distribution", default="softmax", choices=["softmax", "student"]
    )
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--use_freq", default=False, action="store_true")
    parser.add_argument("--hidden_dims", default="[1000, 2000, 1000, 1000]", type=str)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument(
        "--gamma", default=5, type=float, help="weight of clustering loss"
    )
    parser.add_argument("--update_interval", default=100, type=int)
    parser.add_argument("--tol", default=0.001, type=float)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    args.device = torch.device("cuda" if args.cuda else "cpu")
    print(args)
    with open(os.path.join(args.dataset_path, args.input_emb_name), "rb") as fin:
        emb_dict = pk.load(fin)

    candidate_idx = train(args, emb_dict)
    print(candidate_idx)
