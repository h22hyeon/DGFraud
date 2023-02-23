"""
This code is attributed to Kay Liu (@kayzliu), Yingtong Dou (@YingtongDou)
and UIC BDSC Lab
DGFraud-TF2 (A Deep Graph-based Toolbox for Fraud Detection in TensorFlow 2.X)
https://github.com/safe-graph/DGFraud-TF2
"""
import os
import argparse
import numpy as np
from collections import namedtuple
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import tensorflow as tf

from algorithms.GraphConsis.GraphConsis import GraphConsis
from utils.data_loader import load_data_yelp
from utils.utils import preprocess_feature

# init the common args, expect the model specific args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=717, help='random seed')
parser.add_argument('--epochs', type=int, default=5,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--train_size', type=float, default=0.8,
                    help='training set percentage')
parser.add_argument('--lr', type=float, default=0.5, help='learning rate')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units')
parser.add_argument('--sample_sizes', type=list, default=[5, 5],
                    help='number of samples for each layer')
parser.add_argument('--identity_dim', type=int, default=0,
                    help='dimension of context embedding')
parser.add_argument('--eps', type=float, default=0.001,
                    help='consistency score threshold ε')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]="0"
# set seed
np.random.seed(args.seed)
tf.random.set_seed(args.seed)


def GraphConsis_main(neigh_dicts, features, labels, masks, num_classes, args):
    """
    neigh_dicts: 각 타입의 엣지에 대응하는 adjacency list (list of dictionary)
    features: 전체 그래프에 대응하는 노드의 feature matrix (CSC matrix)
    label: 전체 그래프에 대응하는 노드의 label (numpy array)
    [idx_train, idx_val, idx_test]: train/val/test 노드의 인덱스
    num_classes: Fraud의 종류 (2)
    """
    def generate_training_minibatch(nodes_for_training, all_labels,
                                batch_size, features):
        """
        nodes_for_training: train set에 대응하는 노드의 인덱스 (numpy array)
        all_labels: 전체 그래프에 대응하는 노드의 label (numpy array)
        features: features: 전체 그래프에 대응하는 노드의 feature matrix (CSC matrix)
        """
        nodes_for_epoch = np.copy(nodes_for_training) # 왜 copy 하는지...?
        ix = 0
        np.random.shuffle(nodes_for_epoch) # train set의 노드 인덱스를 셔플한다.
        """
        좀 이상하게 짜여있긴 하지만 1 epoch를 다 돌 때까지 카운팅하며,
        batch에 해당하는 노드와 label을 반환한다.
        """
        while len(nodes_for_epoch) > ix + batch_size:
            # 순차적으로 노드 할당
            mini_batch_nodes = nodes_for_epoch[ix:ix + batch_size]
            # 배치를 구성하는데
            batch = build_batch(mini_batch_nodes, neigh_dicts,
                                args.sample_sizes, features)
            labels = all_labels[mini_batch_nodes]
            ix += batch_size
            yield (batch, labels)
        mini_batch_nodes = nodes_for_epoch[ix:-1]
        batch = build_batch(mini_batch_nodes, neigh_dicts,
                            args.sample_sizes, features)
        labels = all_labels[mini_batch_nodes]
        yield (batch, labels)

    # train/val/test 노드의 인덱스를 정의한다.
    train_nodes = masks[0]
    val_nodes = masks[1]
    test_nodes = masks[2]

    # training
    model = GraphConsis(features.shape[-1], args.nhid,
                        len(args.sample_sizes), num_classes, len(neigh_dicts))
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    for epoch in range(args.epochs):
        print(f"Epoch {str(epoch).zfill(2)}: training...")
        minibatch_generator = generate_training_minibatch(train_nodes,
                                                          labels,
                                                          args.batch_size,
                                                          features)
        batchs = len(train_nodes) / args.batch_size
        for inputs, inputs_labels in tqdm(minibatch_generator, total=int(batchs)):

            with tf.GradientTape() as tape:
                predicted = model(inputs, features)
                loss = loss_fn(tf.convert_to_tensor(inputs_labels), predicted)
                acc = accuracy_score(inputs_labels,
                                     predicted.numpy().argmax(axis=1))
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            print(f" loss: {loss.numpy():.4f}, acc: {acc:.4f}")

        # validation
        print("Validating...")
        val_results = model(build_batch(val_nodes, neigh_dicts,
                                        args.sample_sizes, features), features)
        loss = loss_fn(tf.convert_to_tensor(labels[val_nodes]), val_results)
        val_acc = accuracy_score(labels[val_nodes],
                                 val_results.numpy().argmax(axis=1))
        print(f" Epoch: {epoch:d}, "
              f"loss: {loss.numpy():.4f}, "
              f"acc: {val_acc:.4f}")

    # testing
    print("Testing...")
    results = model(build_batch(test_nodes, neigh_dicts,
                                args.sample_sizes, features), features)
    test_acc = accuracy_score(labels[test_nodes],
                              results.numpy().argmax(axis=1))
    print(f"Test acc: {test_acc:.4f}")


def build_batch(nodes: list, neigh_dicts: dict, sample_sizes: list,
                features: np.array):
    """
    :param nodes: node ids
    :param neigh_dicts: BIDIRECTIONAL adjacency matrix in dict {node:[node]}
    :param sample_sizes: sample size for each layer
    :param features: 2d features of nodes
    :return a list of namedtuple minibatch
        "src_nodes": node ids to retrieve from raw feature and
        feed to the first layer
        "dstsrc2srcs": list of dstsrc2src matrices from last to first layer
        "dstsrc2dsts": list of dstsrc2dst matrices from last to first layer
        "dif_mats": list of dif_mat matrices from last to first layer
    """

    """
    nodes: 배치를 구성하는 노드의 인덱스
    neigh_dicts: 각 타입의 엣지에 대응하는 adjacency list (list of dictionary)
    sample_sizes: ??
    features: 전체 그래프에 대응하는 노드의 featrue matrix (CSC matrix)
    """

    output = []
    for neigh_dict in neigh_dicts: # 각 타입에 대응하는 엣지의 adjacency dictionary에 대하여
        dst_nodes = [nodes] # 배치를 구성하는 노드의 인덱스를 dst_nodes로 정의한다. 
        dstsrc2dsts = []
        dstsrc2srcs = []
        dif_mats = []

        # 전체 그래프에서 가장 큰 노드의 인덱스를 max_node_id로 정의한다. >??
        max_node_id = max(list(neigh_dict.keys()))

        for sample_size in reversed(sample_sizes):
            ds, d2s, d2d, dm = compute_diffusion_matrix(dst_nodes[-1],
                                                        neigh_dict,
                                                        sample_size,
                                                        max_node_id,
                                                        features
                                                        )
            dst_nodes.append(ds)
            dstsrc2srcs.append(d2s)
            dstsrc2dsts.append(d2d)
            dif_mats.append(dm)

        src_nodes = dst_nodes.pop()

        MiniBatchFields = ["src_nodes", "dstsrc2srcs",
                           "dstsrc2dsts", "dif_mats"]
        MiniBatch = namedtuple("MiniBatch", MiniBatchFields)
        output.append(MiniBatch(src_nodes, dstsrc2srcs, dstsrc2dsts, dif_mats))

    return output


def compute_diffusion_matrix(dst_nodes, neigh_dict, sample_size,
                             max_node_id, features):
    def calc_consistency_score(n, ns):
        """
        # Equation 3 in the paper
        Neighborhood sampling을 위해 consistancy score를 계산하는 함수이다.

        n: 타겟 노드의 인덱스 (Scalar)
        ns: 타겟 노드의 이웃 노드들의 인덱스 (numpy arrray)
        """
        # Neighborhood sampling을 위해 consistancy score를 계산한다!!
        consis = tf.exp(-tf.pow(tf.norm(tf.tile([features[n]], [len(ns), 1]) - features[ns], axis=1), 2))
        
        # Consistancy score가 eps보다 높은 값을 가지는 인덱스에 0을 할당한다.
        consis = tf.where(consis > args.eps, consis, 0)
        return consis

    def sample(n, ns):
        """
        n: Consistency score를 계산하기 위한 타겟 노드의 인덱스 (Scalar)
        ns: 타겟 노드의 이웃 노드들의 인덱스 (numpy arrray)
        """
        if len(ns) == 0:
            return [] # 타겟 노드가 singleton 노드일 경우 그냥 반환된다.
        
        # 타겟 노드와 이웃 노드 간의 consistancy score를 반환한다 (tensor).
        consis = calc_consistency_score(n, ns)

        """
        # Equation 4 in the paper
        Neighborhood sampling을 위해 consistancy score를 normalize 하는 코드이다.
        """
        prob = consis / tf.reduce_sum(consis)
        # normalized consistancy score를 확률로 이용하여 sample_size만큼 샘플링한다.
        # 샘플링된 타겟 노드의 이웃 노드들을 반환한다.
        return np.random.choice(ns, min(len(ns), sample_size),
                                replace=False, p=prob)

    def vectorize(ns):
        # 전체 그래프의 노드 수에 대응하는 차원의 0벡터를 생성하고,
        v = np.zeros(max_node_id + 1, dtype=np.float32)
        # 샘플링된 이웃 노드의 인덱스의 값을 1로 할당한다.
        v[ns] = 1
        return v # numpy array (shape: |V|,) 

    """ 2. Neighbor Sampling"""
    # 배치에 존재하는 노드들의 인덱스를 key로 하여 그들의 이웃 노드의 인덱스에 접근한다.
    # 이웃 노드와의 consistency score를 기준으로(normalize하여 probability로 사용함.) sample_size만금 샘플링한다.
    # adj_mat_full의 row는 배치를 구성하는 노드를 가리키며 column은 전체 그래프에서 이웃 노드들과의 연결 관계를 의미한다.
    adj_mat_full = np.stack([vectorize(sample(n, neigh_dict[n]))
                             for n in dst_nodes])
    # 배치를 구성하는 노드를 기준으로 전체 노드 중 샘플링되지 않는 이웃 노드들에 대한 마스크를 생성한다.
    # nonzero_cols_mask는 전체 노드 중 샘플링되는 이웃 노드만을 가리키는 마스크로 사용된다.
    nonzero_cols_mask = np.any(adj_mat_full.astype(bool), axis=0)

    # compute diffusion matrix
    # 배치 전체에서 참조되지 않는 이웃 노드들을 adj_mat_full에서 제거하여 adj_mat로 정의한다.
    adj_mat = adj_mat_full[:, nonzero_cols_mask]
    adj_mat_sum = np.sum(adj_mat, axis=1, keepdims=True)
    # 이웃 노드에 대한 mean aggregator를 의미한다.
    """self-loop을 더해주어야 하는지 코드를 통해 확인해야 함)."""
    _, zero_mask, _ = np.where([adj_mat_sum==0]) # 원래 코드에서 singleton node들은 row sum 값이 0이 되는데
    adj_mat_sum[zero_mask, 0] = 1 # 0으로 나누게 되어 에러 메시지와 런타임 경고가 발생하는데 이를 수정하였다.
    dif_mat = np.nan_to_num(adj_mat / adj_mat_sum) 

    # compute dstsrc mappings
    src_nodes = np.arange(nonzero_cols_mask.size)[nonzero_cols_mask]
    # np.union1d automatic sorts the return,
    # which is required for np.searchsorted
    dstsrc = np.union1d(dst_nodes, src_nodes)
    dstsrc2src = np.searchsorted(dstsrc, src_nodes)
    dstsrc2dst = np.searchsorted(dstsrc, dst_nodes)

    return dstsrc, dstsrc2src, dstsrc2dst, dif_mat


if __name__ == "__main__":
    # load the data
    # adj_list에는 각 타입의 adj가, split_ids에는 train_x,y/val_x,y/test_x,y가 리스트로 담겨있음.
    adj_list, features, split_ids, y = load_data_yelp(train_size=args.train_size)
    idx_train, _, idx_val, _, idx_test, _ = split_ids # train/val/test 노드의 인덱스를 정의한다.

    num_classes = len(set(y)) # Fraud의 클래스의 수를 계산한다.
    label = np.array([y]).T

    # Feature(CSC matrix format)를 row-normalize한다.
    features = preprocess_feature(features, to_tuple=False)
    # CSC matrix를 numpy 형태로 변환한다.
    features = np.array(features.todense())

    # Equation 2 in the paper
    """
    논문의 Context embedding에 해당하는 부분으로 feature vector에 trainable context embedding을 concatenate 하여
    노드의 feature가 구분될 수 있도록 한다. (numpy로 난수를 생성하는데, trainable 한지는 GraphConsis_main를 통해 확인해야 한다.)
    """
    features = np.concatenate((features,
                               np.random.rand(features.shape[0],
                                              args.identity_dim)), axis=1)

    neigh_dicts = []
    for net in adj_list: # adj_list에 존재하는 여러 타입의 엣지(CSC matrix format)를 차례로 불러와 이웃 노드들을 저장한다.
        neigh_dict = {} # 각 타입의 엣지로 연결된 노드의 이웃들을 딕셔너리 형태로 저장한다. 
        for i in range(len(y)): 
            neigh_dict[i] = [] # 노드의 수만큼 빈 리스트를 생성하고, key값을 대응하여 append하는 방식을 이용한다. 
        nodes1 = net.nonzero()[0] # CSC_matrix.nonzero()는 nonzero의 row_arr, colum_arr을 반환한다.
        nodes2 = net.nonzero()[1]
        
        # adj에서 값을 가지는 (row, col) 인덱스를 이용하여 딕셔너리를 채워넣는다.  
        for node1, node2 in zip(nodes1, nodes2):
            neigh_dict[node1].append(node2)
        # 노드의 이웃들의 리스트를 array로 형변환 하여 neigh_dicts에 append 한다.
        neigh_dicts.append({k: np.array(v, dtype=np.int64)
                            for k, v in neigh_dict.items()})

    """
    neigh_dicts: 각 타입의 엣지에 대응하는 adjacency list (list of dictionary)
    features: 노드의 feature matrix (CSC matrix)
    label: 노드의 label (numpy array)
    [idx_train, idx_val, idx_test]: train/val/test 노드의 인덱스
    num_classes: Fraud의 종류 (2)
    """
    GraphConsis_main(neigh_dicts, features, label,
                     [idx_train, idx_val, idx_test], num_classes, args)
