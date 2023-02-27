"""
This code is attributed to Kay Liu (@kayzliu), Yingtong Dou (@YingtongDou)
and UIC BDSC Lab
DGFraud-TF2 (A Deep Graph-based Toolbox for Fraud Detection in TensorFlow 2.X)
https://github.com/safe-graph/DGFraud-TF2
"""
import os
import time
import argparse
import numpy as np
import collections
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import tensorflow as tf

from algorithms.GraphSage.GraphSage import GraphSage
from utils.data_loader import load_data_yelp
from utils.utils import preprocess_feature
from utils.utils import log, print_config, test_gnn

# init the common args, expect the model specific args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=717, help='random seed')
parser.add_argument('--epochs', type=int, default=120,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--train_size', type=float, default=0.4,
                    help='training set percentage')
parser.add_argument('--lr', type=float, default=0.5, help='learning rate')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units')
parser.add_argument('--sample_sizes', type=list, default=[5, 5],
                    help='number of samples for each layer')
parser.add_argument('--valid_epochs', type=int, default=3, help='Number of valid epochs.')
parser.add_argument('--GPU_id', type=str, default="3", help='GPU index')

args = parser.parse_args()

# set seed
np.random.seed(args.seed)
tf.random.set_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"]= args.GPU_id

def GraphSage_main(neigh_dict, features, labels, masks, num_classes, args):
    """
    neigh_dict: neigh_dicts = 각 타입의 엣지에 대응하는 adjacency list (list of dictionary)
    features: 전체 그래프에 대응하는 노드의 feature matrix (CSC matrix)
    label: 전체 그래프에 대응하는 노드의 label (numpy array)
    masks: [idx_train, idx_val, idx_test] = train/val/test 노드의 인덱스
    num_classes: Fraud의 종류 (2)
    """
    def generate_minibatch(nodes_for_training,
                                all_labels, batch_size):
        """
        nodes_for_training: train set에 대응하는 노드의 인덱스 (numpy array)
        all_labels: 전체 그래프에 대응하는 노드의 label (numpy array)
        features: 전체 그래프에 대응하는 노드의 feature matrix (CSC matrix)
        """
        nodes_for_epoch = np.copy(nodes_for_training)
        ix = 0
        np.random.shuffle(nodes_for_epoch) # train set의 노드 인덱스를 셔플한다.
        """
        1 epoch를 다 돌때까지 카운팅하며 배치 노드와 label을 리턴한다.
        """
        while len(nodes_for_epoch) > ix + batch_size:
            # 인덱스에 따라 순차적으로 노드를 할당한다.
            mini_batch_nodes = nodes_for_epoch[ix:ix + batch_size]
            # 배치를 구성한다.
            # 이때, 배치는 네 개의 배열이 담긴 튜플로 반환된다.
            # src_nodes: 배치를 구성하는 노드들과 그 이웃 노드들의 인덱스가 담긴 배열
            # dstsrc2srcs: dstsrc에서 이웃 노드의 위치를 가리키는 인덱스 배열
            # dstsrc2dsts: dstsrc에서 배치 노드의 위치를 가리키는 인덱스 배열
            # dif_mats: mean aggragator 역할을 하는 행렬 (row-normalized)  
            batch = build_batch(mini_batch_nodes,
                                neigh_dict, args.sample_sizes)
            labels = all_labels[mini_batch_nodes]
            ix += batch_size
            yield (batch, labels)
        
        """남는 배치는 사용하지 않도록 한다."""
        # mini_batch_nodes = nodes_for_epoch[ix:-1]
        # batch = build_batch(mini_batch_nodes, neigh_dict, args.sample_sizes)
        # labels = all_labels[mini_batch_nodes]
        # yield (batch, labels)

    ckp = log()
    config_lines = print_config(vars(args))
    ckp.write_train_log(config_lines, print_line=False)
    ckp.write_valid_log(config_lines, print_line=False)
    ckp.write_test_log(config_lines, print_line=False)

    # train/val/test 노드의 인덱스를 정의한다.
    train_nodes = masks[0]
    val_nodes = masks[1]
    test_nodes = masks[2]

    # training
    """GraphSage 모델과 옵티마이저 생성 및 손실함수 정의"""
    model = GraphSage(features.shape[-1], args.nhid,
                      len(args.sample_sizes), num_classes)
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    for epoch in range(args.epochs):
        print(f"Epoch {str(epoch).zfill(2)}: training...")
        minibatch_generator = generate_minibatch(
            train_nodes, labels, args.batch_size)
        
        # iteration을 게산한다.
        total_loss = 0.0
        epoch_time = 0
        start_time = time.time()
        iters = int(len(train_nodes) / args.batch_size)
        for inputs, inputs_labels in tqdm(minibatch_generator, total=iters):
            # 모델에 대한 그래디언트를 계산하고 옵티마이저를 통해 가중치를 계산한다.
            with tf.GradientTape() as tape:
                # 최종적으로 생성된 배치 노드에 대한 fraud score를 predicted로 정의한다.
                predicted = model(inputs, features)
                # predicted를 통해 cross-entropy loss를 계산한다.
                loss = loss_fn(tf.convert_to_tensor(inputs_labels), predicted)
            # 역전파 과정을 통해 gradient를 계산하고 optimizer를 통해 가중치를 업데이트 한다.
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            total_loss += loss.numpy()
        end_time = time.time()
        epoch_time += end_time - start_time
        line = f'Epoch: {epoch}, loss: {total_loss / (iters * args.batch_size)}, time: {epoch_time}s'
        ckp.write_train_log(line)

        # validation!!
        # Valid the model for every $valid_epoch$ epoch
        if (epoch+1) % args.valid_epochs == 0:
            print("Valid at epoch {}".format(epoch))
            # 학습된 모델로부터 생성된 fraud score를 val_results로 정의한다.
            # 학습된 모델로부터 validation 과정을 수행한다.
            iters = int(len(val_nodes) / args.batch_size)
            minibatch_generator = generate_minibatch(val_nodes, labels, args.batch_size)
            acc_gnn_val, recall_gnn_val, f1_gnn_val = test_gnn(minibatch_generator, model, features, iters, ckp, flag="val")

    # testing!!
    print("Testing...")
    # 학습된 모델로부터 생성된 fraud score를 val_results로 정의한다.
    iters = int(len(val_nodes) / args.batch_size)
    minibatch_generator = generate_minibatch(test_nodes, labels, args.batch_size)
    acc_gnn_test, recall_gnn_test, f1_gnn_test = test_gnn(minibatch_generator, model, features, iters, ckp, flag="test") 


def build_batch(nodes, neigh_dict, sample_sizes):
    """
    :param [int] nodes: node ids
    :param {node:[node]} neigh_dict: BIDIRECTIONAL adjacency matrix in dict
    :param [sample_size]: sample sizes for each layer,
    lens is the number of layers
    :param tensor features: 2d features of nodes
    :return namedtuple minibatch
        "src_nodes": node ids to retrieve from raw feature
        and feed to the first layer
        "dstsrc2srcs": list of dstsrc2src matrices from last to first layer
        "dstsrc2dsts": list of dstsrc2dst matrices from last to first layer
        "dif_mats": list of dif_mat matrices from last to first layer
    """
    """
    nodes: 배치를 구성하는 노드의 인덱스
    neigh_dict: (homo type)각 타입의 엣지에 대응하는 adjacency list (list of dictionary)
    sample_sized: Neighbor sampling에서 샘플링할 노드의 수를 나타내는 배열 (numpy array)
    features: 전체 그래프에 대응하는 노드의 feture matrix (CSC matrix)
    """
    # GraphSage는 heterogeneous에서 작동하므로 모든 엣지의 adgacency list를 통합한다.
    # 배치를 구성하는 노드들과 그 이웃 노드들의 인덱스가 담긴 배열을 담을 리스트를 정의한다.
    # 추후 compute_diffusion_matrix 함수를 이용할 때 해당 리스트를 참조한다. (dst_nodes=[0-th layer, 1-th layer, ...], -1 인덱스로 참조해나감.)
    dst_nodes = [nodes]
    # 배치를 구성하는 노드의 인덱스를 dst_nodes로 정의한다.
    dstsrc2dsts = [] # 각 레이어에서 dst_nodes의 배치 노드의 위치를 가리키는 인덱스 배열을 담을 리스트를 선언한다.
    dstsrc2srcs = [] # 각 레이어에서 dst_nodes의 배치 노드의 이웃 노드의 위치를 가리키는 인덱스 배열을 담을 리스트를 선언한다. 
    dif_mats = [] # 특정 레이어에서 배치 노드의 mean aggregator의 역할을 하는 adjacency matrix를 담을 리스트를 정의한다.

    # 전체 그래프에서 가장 큰 노드의 인덱스를 max_node_id로 정의한다.
    # 이는 dif_mat(아래의 dm)을 생성하는 과정에서 일시적으로 필요하다.
    max_node_id = max(list(neigh_dict.keys()))

    for sample_size in reversed(sample_sizes):
        # dstsrc: 배치를 구성하는 노드들과 그 이웃 노드들의 인덱스가 담긴 배열
        # dstsrc2src: dstsrc에서 이웃 노드의 위치를 가리키는 인덱스 배열
        # dstsrc2dst: dstsrc에서 배치 노드의 위치를 가리키는 인덱스 배열
        # dif_mat: mean aggragator 역할을 하는 행렬 (row-normalized)
        ds, d2s, d2d, dm = compute_diffusion_matrix(dst_nodes[-1],
                                                    neigh_dict,
                                                    sample_size,
                                                    max_node_id,
                                                    )
        dst_nodes.append(ds)
        dstsrc2srcs.append(d2s)
        dstsrc2dsts.append(d2d)
        dif_mats.append(dm)

    # dst_nodes(0번 인덱스)는 기존의 노드 피처 하나가 더 담겨 잇었으므로 제거하여 수를 맞춘다.
    src_nodes = dst_nodes.pop()

    # 배치의 형태를 namedtuple로 변환하여 반환한다.
    MiniBatchFields = ["src_nodes", "dstsrc2srcs", "dstsrc2dsts", "dif_mats"]
    MiniBatch = collections.namedtuple("MiniBatch", MiniBatchFields)

    return MiniBatch(src_nodes, dstsrc2srcs, dstsrc2dsts, dif_mats)


def compute_diffusion_matrix(dst_nodes, neigh_dict, sample_size, max_node_id):
    """
    dst_nodes: 배치를 구성하는 노드들과 그 이웃 노드들의 인덱스가 담긴 배열
    neigh_dict: (homo type)각 타입의 엣지에 대응하는 adjacency list (list of dictionary)
    sample_size: 각 레이어에서 샘플링을 진행할 노드의 수 (list)
    """
    def sample(ns):
        """
        ns: 타겟 노드의 이웃 노드의 인덱스 리스트
        """
        # sample_size 혹은 이웃의 개수 중 적은 수로 샘플링 한다.
        return np.random.choice(ns, min(len(ns), sample_size), replace=False)

    def vectorize(ns):
        # 전체 그래프의 노드 수에 대응하는 차원의 0벡터를 생성하고,
        v = np.zeros(max_node_id + 1, dtype=np.float32)
        # 샘플링된 이웃 노드의 인덱스의 값만을 1로 할당한다.
        v[ns] = 1
        return v

    """sample neighbors"""
    # 각 노드의 이웃을 샘플링하여 벡터화 한 후 adj_mat_full로 정의한다.
    adj_mat_full = np.stack([vectorize(
        sample(neigh_dict[n])) for n in dst_nodes])
    
    # 배치를 구성하는 노드를 기준으로 전체 노드 중 샘플링되지 않는 이웃 노드들에 대한 마스크를 생성한다.
    # nonzero_cols_mask는 전체 노드 중 샘플링되는 이웃 노드만을 가리키는 마스크로 사용된다.
    nonzero_cols_mask = np.any(adj_mat_full.astype(bool), axis=0)

    # compute diffusion matrix
    # 배치에서 참조되지 않는 이웃 노드들을 adj_mat_full에서 제거하여 adj_mat로 정의한다.
    adj_mat = adj_mat_full[:, nonzero_cols_mask]
    # Row-normalize를 위해서 각 노드의 이웃 수를 구한다.
    adj_mat_sum = np.sum(adj_mat, axis=1, keepdims=True)
    """self-loop을 더해주어야 하는지 코드를 통해 확인해야 함)."""
    _, zero_mask, _ = np.where([adj_mat_sum==0]) # 원래 코드에서 singleton node들은 row sum 값이 0이 되는데
    adj_mat_sum[zero_mask, 0] = 1 # 0으로 나누게 되어 에러 메시지와 런타임 경고가 발생하는데 이를 수정하였다.
    # dif_mat은 mean aggragator 역할을 하는 행렬을 나타낸다. (B, # of neighborhood in batch)
    dif_mat = np.nan_to_num(adj_mat / adj_mat_sum)

    # compute dstsrc mappings
    # Aggregation 과정에서 message를 받아올 노드들의 인덱스가 담긴 배열을 src_nodes로 정의한다.
    src_nodes = np.arange(nonzero_cols_mask.size)[nonzero_cols_mask] # neighbor index array

    # np.union1d automatic sorts the return,
    # which is required for np.searchsorted
    # dst_nodes: 배치를 구성하는 노드의 인덱스가 담긴 배열
    # src_nodes: 모든 이웃 노드들의 인덱스가 담긴 배열
    dstsrc = np.union1d(dst_nodes, src_nodes) # 두 배열의 합집합을 정렬하여 array로 반환한다. ex) [480, 512, 782]
    dstsrc2src = np.searchsorted(dstsrc, src_nodes) # 정렬된 배열 dstsrc에서 이웃 노드의 인덱스를(mapping) array로 반환한다. ex) [1, 2]
    dstsrc2dst = np.searchsorted(dstsrc, dst_nodes) # 정렬된 배열 dstsrc에서 배치를 구성하는 노드의 인덱스를 array로 반환한다. ex) [0]

    # dstsrc: 배치를 구성하는 노드들과 그 이웃 노드들의 인덱스가 담긴 배열
    # dstsrc2src: dstsrc에서 이웃 노드의 위치를 가리키는 인덱스 배열
    # dstsrc2dst: dstsrc에서 배치 노드의 위치를 가리키는 인덱스 배열
    # dif_mat: mean aggragator 역할을 하는 행렬 (row-normalized) 
    return dstsrc, dstsrc2src, dstsrc2dst, dif_mat


if __name__ == "__main__":
    # load the data
    # adj_list: 각 타입에 해당하는 adjacency matrix
    # split_ids: train_x,y/val_x,y/test_x,y에 대응되는 노드 인덱스 리스트
    """
    GraphSage는 homogeneous graph를 다루기 때문에 relation을 통합한 homo 버전으로 불러와야 한다.
    """
    adj_list, features, split_ids, y = load_data_yelp(
        meta=False, train_size=args.train_size)
    # split_ids를 통해 train/val/test 노드의 인덱스를 정의한다.
    idx_train, _, idx_val, _, idx_test, _ = split_ids

    # Fraud의 클래스 수를 계산한다.
    num_classes = len(set(y))
    label = np.array([y]).T

    # Feature(CSC matrix format)을 row-normalize 한다.
    features = preprocess_feature(features, to_tuple=False)
    # CSC matrix를 numpy array로 형변환한다.
    features = np.array(features.todense())

    # 리스트로 구성된 딕셔너리를 neigh_dict로 정의한다.
    neigh_dict = collections.defaultdict(list)
    for i in range(len(y)): # 각 노드의 인덱스를 key로 하고 빈 리스트를 value로 할당한다.
        neigh_dict[i] = []

    # merge all relations into single graph
    for net in adj_list: # 각 타입의 엣지에 접근한다.
        nodes1 = net.nonzero()[0] # CSC_matrix.nonzero()는 nonzero value의 row_arr, colum_arr을 반환한다.
        nodes2 = net.nonzero()[1]
        # 노드의 수만큼 빈 리스트를 생성하고, key값을 대응하여 append하는 방식을 이용한다.
        for node1, node2 in zip(nodes1, nodes2):
            neigh_dict[node1].append(node2)

    # 각 노드의 이웃 노드를 가지고 있는 neigh_dict에 대하여 value를 리스트에서 numpy array로 형변환을 수행한다.
    neigh_dict = {k: np.array(v, dtype=np.int64)
                  for k, v in neigh_dict.items()}

    """
    neigh_dict: (homo type)각 타입의 엣지에 대응하는 adhacency list (list of dictionary)
    features: 노드의 feature matrix (row-normalized CSC matrix)
    label: 노드의 label (numpy array)
    [idx_train, idx_val, idx_test]: train/val/test 노드의 인덱스
    num_classes: Fraud의 종류 (2)
    """
    GraphSage_main(neigh_dict, features, label,
                   [idx_train, idx_val, idx_test], num_classes, args)
