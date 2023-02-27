"""
This code is attributed to Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection  in TensorFlow 2.X)
https://github.com/safe-graph/DGFraud-TF2
"""
import os
from tqdm import tqdm
from datetime import datetime
from typing import Tuple, Union
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score

class log:
	def __init__(self):
		self.log_dir_path = "./log"
		self.log_file_name = datetime.now().strftime("%Y-%m-%d %H:%M") + ".log"
		self.train_log_path = os.path.join(self.log_dir_path, "train", self.log_file_name)
		self.valid_log_path = os.path.join(self.log_dir_path, "valid", self.log_file_name)
		self.test_log_path = os.path.join(self.log_dir_path, "test", self.log_file_name)
		self.multi_run_log_path = os.path.join(self.log_dir_path, "multi-run(total)", self.log_file_name)
		os.makedirs(os.path.join(self.log_dir_path, "train"), exist_ok=True)
		os.makedirs(os.path.join(self.log_dir_path, "valid"), exist_ok=True)
		os.makedirs(os.path.join(self.log_dir_path, "test"), exist_ok=True)
		os.makedirs(os.path.join(self.log_dir_path, "multiple-run"), exist_ok=True)

	def write_train_log(self, line, print_line=True):
		if print_line:
			print(line)
		log_file = open(self.train_log_path, 'a')
		log_file.write(line + "\n")
		log_file.close()

	def write_valid_log(self, line, print_line=True):
		if print_line:
			print(line)
		log_file = open(self.valid_log_path, 'a')
		log_file.write(line + "\n")
		log_file.close()

	def write_test_log(self, line, print_line=True):
		if print_line:
			print(line)
		log_file = open(self.test_log_path, 'a')
		log_file.write(line + "\n")
		log_file.close()
	
	def multi_run_log(self, line, print_line=True):
		if print_line:
			print(line)
		log_file = open(self.multi_run_log_path, 'a')
		log_file.write(line + "\n")
		log_file.close()

def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    # Configuration 파일을 불러와 train setting을 출력한다.
    config_lines = ""
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        line = "{} -->   {}\n".format(keystr, val)
        config_lines += line
        print(line)
    print("**************** MODEL CONFIGURATION ****************")
    
    return config_lines

def test_gnn(minibatch_generator, model, features, iters, ckp, flag=None):
    
    f1_gnn = 0.0
    acc_gnn = 0.0
    recall_gnn = 0.0
    auc_gnn_list = []
    label_list = []
    
    for inputs, inputs_labels in tqdm(minibatch_generator, total=iters):
        predicted = model(inputs, features)
        f1_gnn += f1_score(inputs_labels, predicted.numpy().argmax(axis=1), average="macro")
        acc_gnn += accuracy_score(inputs_labels, predicted.numpy().argmax(axis=1))
        recall_gnn += recall_score(inputs_labels, predicted.numpy().argmax(axis=1), average="macro")
        auc_gnn_list.extend(predicted.numpy().argmax(axis=1))
        label_list.extend(inputs_labels)
    
    auc_gnn = roc_auc_score(label_list, np.array(auc_gnn_list))
    
    line1= f"GNN F1: {f1_gnn/iters:.4f}\tGNN AUC-ROC: {auc_gnn/iters:.4f}"+\
       f"\tGNN Recall: {recall_gnn/iters:.4f}\tGNN ACCuracy: {acc_gnn/iters:.4f}\n"
	
    if flag=="val":
        ckp.write_valid_log("Validation: "+ line1)
    elif flag=="test":
        ckp.write_test_log("Test: "+ line1)
    return acc_gnn, recall_gnn, f1_gnn

def sparse_to_tuple(sparse_mx: sp.coo_matrix) -> Tuple[np.array, np.array,
                                                       np.array]:
    """
    Convert sparse matrix to tuple representation.

    :param sparse_mx: the graph adjacency matrix in scipy sparse matrix format
    """

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    # lil matrix를 tuple 형태로 변환한다.
    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj: np.array) -> sp.coo_matrix:
    """
    Symmetrically normalize adjacency matrix
    Parts of this code file were originally forked from
    https://github.com/tkipf/gcn

    :param adj: the graph adjacency matrix
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj: np.array, to_tuple: bool = True) -> \
        Union[Tuple[np.array, np.array, np.array], sp.coo_matrix]:
    """
    Preprocessing of adjacency matrix for simple GCN model
    and conversion to tuple representation.
    Parts of this code file were originally forked from
    https://github.com/tkipf/gcn

    :param adj: the graph adjacency matrix
    """
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))

    if to_tuple:
        return sparse_to_tuple(adj_normalized)
    else:
        return adj_normalized


def preprocess_feature(features: np.array, to_tuple: bool = True) -> \
        Union[Tuple[np.array, np.array, np.array], sp.csr_matrix]:
    """
    Row-normalize feature matrix and convert to tuple representation
    Parts of this code file were originally forked from
    https://github.com/tkipf/gcn

    :param features: the node feature matrix
    :param to_tuple: whether cast the feature matrix to scipy sparse tuple
    """
    # Feature를 row-normalize 하는 함수이다.
    features = sp.lil_matrix(features)
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)

    # 이때 생성되는 features는 lil_matrix 형태이다.
    features = r_mat_inv.dot(features)
    if to_tuple:
        return sparse_to_tuple(features)
    else:
        return features


def sample_mask(idx: np.array, n_class: int) -> np.array:
    """
    Create mask for GCN.
    Parts of this code file were originally forked from
    https://github.com/tkipf/gcn

    :param idx: the train/val/test indices
    :param n_class: the number of classes for the data
    """
    mask = np.zeros(n_class)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def pad_adjlist(x_data):
    # Get lengths of each row of data
    lens = np.array([len(x_data[i]) for i in range(len(x_data))])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:, None]

    # Setup output array and put elements from data into masked positions
    padded = np.zeros(mask.shape)
    for i in range(mask.shape[0]):
        padded[i] = np.random.choice(x_data[i], mask.shape[1])
    padded[mask] = np.hstack((x_data[:]))
    return padded


def matrix_to_adjlist(M, pad=True): # CSC matrix를 adjacency list 형태로 변환하는 함수이다.
    adjlist = [] # Adjacency list
    for i in range(len(M)): # 리스트에서 각 노드의 인덱스에 이웃 노드들을 담을 리스트를 할당한다.
        adjline = [i] # self-loop이 추가된다.
        for j in range(len(M[i])):
            if M[i][j] == 1:
                adjline.append(j)
        adjlist.append(adjline) # 한 노드의 이웃 노드를 담고있는 리스트(adjline)를 adjlist에 추가한다.
    if pad:
        adjlist = pad_adjlist(adjlist)
    return adjlist


def pairs_to_matrix(pairs, nodes):
    # 노드 pair를 통해 adjaceny matrix를 구성하는 함수이다 (numpy dense matrix).
    M = np.zeros((nodes, nodes))
    for i, j in pairs:
        M[i][j] = 1
    return M


# Random walk on graph
def generate_random_walk(adjlist, start, walklength):
    t = 1
    # walk path를 노드 인덱스를 시작점으로 하여 정의한다.
    walk_path = np.array([start])
    while t <= walklength: # walklength만큼 walk을 수행한다.
        neighbors = adjlist[start] # 노드의 이웃들 중에서
        current = np.random.choice(neighbors) # 하나의 노드를 샘플링하여
        walk_path = np.append(walk_path, current) # walk path에 추가하고
        start = current # 해당 이웃 노드를 시작점으로 재지정한다.
        t += 1
    return walk_path # 노드에서 walklength만큼 walk하여 샘플링된 이웃 노드의 배열을 반환한다.


#  sample multiple times for each node
def random_walks(adjlist, numerate, walklength):
    """
    adjlist: 한 타입의 adjacency list
    numerate: 반복
    walklength: 길이
    """
    # 한 타입의 adjacency list를 이용해 random walk를 수행하고,
    # 샘플링된 walklength X numerate 개의 이웃 노드들과 pair를 생성하여 리스트로 구성하는 함수이다.
    nodes = range(0, len(adjlist))  # node index starts from zero
    walks = []

    # 각 노드별로 random walk를 수행하여 
    for n in range(numerate):
        for node in nodes:
            # 노드의 인덱스와 walklength를 고려하여 
            walks.append(generate_random_walk(adjlist, node, walklength))
    # walks는 리스트이다. (길이는 numerate X # of nodes)
    pairs = []
    for i in range(len(walks)):
        # 이웃들에 대한 pair를 순회하며 지정한다.
        for j in range(1, len(walks[i])):
            pair = [walks[i][0], walks[i][j]]
            pairs.append(pair) # 지정된 pair는 모두 pairs 리스트에 추가된다.
    # pairs는 리스트이다. (길이는 numerate X # of nodes X (walklength-1))
    return pairs # 한 타입의 adjacency list를 이용해 random walk를 수행하여 생성한 노드 pair 리스트를 반환한다.


def negative_sampling(adj_nodelist):
    # 노드의 차수을 계산하여 degree로 정의한다.
    degree = [len(neighbors) for neighbors in adj_nodelist]
    # 각 노드의 차수의 3/4 제곱을 수행하여 negative sampling 하도록 지정한다.
    node_negative_distribution = np.power(np.array(degree, dtype=np.float32),
                                          0.75)
    # node_negative_distribution를 normalize한다.
    node_negative_distribution /= np.sum(node_negative_distribution)
    node_sampling = AliasSampling(prob=node_negative_distribution)
    return node_negative_distribution, node_sampling


def get_negative_sampling(pairs, adj_nodelist, Q=3, node_sampling='numpy'):
    num_of_nodes = len(adj_nodelist)  # 전체 그래프의 노드의 수
    u_i = []
    u_j = []
    graph_label = []
    node_negative_distribution, nodesampling = negative_sampling(adj_nodelist)
    for index in range(0, num_of_nodes): # 전제 노드에 대해 순회하며
        u_i.append(pairs[index][0]) # 노드 i
        u_j.append(pairs[index][1]) # 노드 j
        graph_label.append(1) # graph_label에 1을 삽입한다.
        for i in range(Q): # Q만큼 반복하며
            while True:
                if node_sampling == 'numpy':
                    # node_negative_distribution로부터 num_of_nodes만큼 샘플링한다.
                    """p=node_negative_distribution으로 코드 수정"""
                    negative_node = np.random.choice(num_of_nodes, p=node_negative_distribution)
                    if negative_node not in adj_nodelist[pairs[index][0]]:
                        break
                elif node_sampling == 'atlas':
                    negative_node = nodesampling.sampling()
                    if negative_node not in adj_nodelist[pairs[index][0]]:
                        break
                elif node_sampling == 'uniform':
                    negative_node = np.random.randint(0, num_of_nodes)
                    if negative_node not in adj_nodelist[pairs[index][0]]:
                        break
            u_i.append(pairs[index][0]) # 타겟 노드와를 u_u에 삽입한다.
            u_j.append(negative_node) # 샘플링된 이웃 노드를 u_j에 삽입한다.
            graph_label.append(-1) # graph_label에 -1을 삽입한다.
    graph_label = np.array(graph_label, dtype=np.int32)
    graph_label = graph_label.reshape(graph_label.shape[0], 1)
    return u_i, u_j, graph_label


# Reference: https://en.wikipedia.org/wiki/Alias_method:  이산 확률 분포에서 샘플링하기 위한 알고리즘
class AliasSampling:

    def __init__(self, prob):
        """
        prob: normalized negative distribution을 나타내는 배열 (numpy array).
        """
        self.n = len(prob) # 배열의 길이
        self.U = np.array(prob) * self.n # 이 과정에서 1보다 큰 값과 작은 값으로 나뉘게 됨/
        self.K = [i for i in range(len(prob))] # 노드의 인덱스를 나타내는 배열을 K로 정의한다.
        
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U): # U의 값에 순차적으로 접근하여
            if U_i > 1: # 값이 1보다 크다면 overfull에
                overfull.append(i)
            elif U_i < 1: # 그렇지 않다면 underfull에 노드 인덱스를 할당한다.
                underfull.append(i)

        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop() # 둘 중에서 적은 수의 배열이 빌 때까지
            self.K[j] = i # 노드의 인덱스
            self.U[i] = self.U[i] - (1 - self.U[j]) 
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res
