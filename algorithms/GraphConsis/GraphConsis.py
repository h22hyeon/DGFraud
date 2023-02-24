"""
This code is attributed to Kay Liu (@kayzliu), Yingtong Dou (@YingtongDou)
and UIC BDSC Lab
DGFraud-TF2 (A Deep Graph-based Toolbox for Fraud Detection in TensorFlow 2.X)
https://github.com/safe-graph/DGFraud-TF2

Paper: 'Alleviating the Inconsistency Problem of
        Applying Graph Neural Network to Fraud Detection'
Link: https://arxiv.org/abs/2005.00625
"""

from collections import namedtuple

import tensorflow as tf
from tensorflow import keras

from layers.layers import ConsisMeanAggregator

init_fn = tf.keras.initializers.GlorotUniform


class GraphConsis(keras.Model):
    """
    The GraphConsis model
    """

    def __init__(self, features_dim: int, internal_dim: int, num_layers: int,
                 num_classes: int, num_relations: int) -> None:
        """
        :param int features_dim: input dimension
        :param int internal_dim: hidden layer dimension
        :param int num_layers: number of sample layer
        :param int num_classes: number of node classes
        :param int num_relations: number of relations
        """
        super().__init__()
        
        # 레이어를 담을 sequential 컨테이너를 정의한다.
        self.seq_layers = []
        
        # Relation attention을 수행하기 위해 concatenate할 relation vector를 정의한다.
        self.relation_vectors = tf.Variable(tf.random.uniform(
            [num_relations, internal_dim], dtype=tf.float32))
        
        # Relation attention을 수행하기 위한 attentiontion weight를 정의한다.
        self.attention_vec = tf.Variable(tf.random.uniform(
            [2 * internal_dim, 1], dtype=tf.float32))

        # 지정된 layer의 수만큼 aggregator_layer를 쌓는다.
        # 이때 layer의 수는 sample_size의 길이에 해당한다. (내부 값은 neighbor sampling에 사용할 이웃의 수이다.)
        for i in range(1, num_layers + 1):
            # Layer의 이름.
            layer_name = "agg_lv" + str(i)

            # 첫 layer의 dimension을 맞춰주기 위함.
            input_dim = internal_dim if i > 1 else features_dim
            # ConsisMeanAggregator를 통해 aggregator_layer를 생성한다. 
            aggregator_layer = ConsisMeanAggregator(input_dim, internal_dim,
                                                    name=layer_name)
            # sequential 컨테이너에 aggregator_lager를 쌓는다.
            self.seq_layers.append(aggregator_layer)

        # keras dense가 mlp와 동일한지 확인
        self.classifier = tf.keras.layers.Dense(num_classes,
                                                activation=tf.nn.softmax,
                                                use_bias=False,
                                                kernel_initializer=init_fn,
                                                name="classifier",
                                                )

    def call(self, minibatchs: namedtuple, features: tf.Tensor) -> tf.Tensor:
        """
        Forward propagation
        :param minibatchs: minibatch list of each relation
        :param features: 2d features of nodes
        """

        xs = []
        for i, minibatch in enumerate(minibatchs): # relation별로 반복을 돌린다.
            # 여기서 x는 배치를 구성하는 노드와 모든 이웃 노드의 피처로 구성된 행렬이다.
            x = tf.gather(tf.Variable(features, dtype=float),
                          tf.squeeze(minibatch.src_nodes))
            for aggregator_layer in self.seq_layers:
                
                x = aggregator_layer(x,
                                     minibatch.dstsrc2srcs.pop(),
                                     minibatch.dstsrc2dsts.pop(),
                                     minibatch.dif_mats.pop(),
                                     tf.nn.embedding_lookup(
                                         self.relation_vectors, i), # 해당 relation의 relation vector를 가져온다.
                                     self.attention_vec
                                     )
            xs.append(x)

        # 각 relation에 따른 노드 embedding을 열 방향으로 stack하고, 합을 구한다.
        # 이를 l2 normalize하고 MLP를 통해 최종 fraud score를 구한다.
        return self.classifier(tf.nn.l2_normalize(tf.reduce_sum(
            tf.stack(xs, 1), axis=1, keepdims=False), 1))
