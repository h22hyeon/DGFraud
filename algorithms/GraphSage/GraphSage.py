"""
This code is attributed to Kay Liu (@kayzliu), Yingtong Dou (@YingtongDou)
and UIC BDSC Lab
DGFraud-TF2 (A Deep Graph-based Toolbox for Fraud Detection in TensorFlow 2.X)
https://github.com/safe-graph/DGFraud-TF2

Paper: 'Inductive representation learning on large graphs'
Link: https://arxiv.org/abs/1706.02216
"""

import tensorflow as tf

from layers.layers import SageMeanAggregator

init_fn = tf.keras.initializers.GlorotUniform


class GraphSage(tf.keras.Model):
    """
    GraphSage model
    """

    def __init__(self, features_dim, internal_dim, num_layers, num_classes):
        """
        :param int features_dim: input dimension
        :param int internal_dim: hidden layer dimension
        :param int num_layers: number of sample layer
        :param int num_classes: number of node classes
        """
        super().__init__()
        # 레이어를 담을 sequential 컨테이너를 정의한다.
        self.seq_layers = []

        # 지정된 수만큼 aggregator_layer를 쌓는다.
        # 이때 layer의 수(num_layers)는 sample_size의 길이에 해당한다.
        for i in range(1, num_layers + 1):
            # Layer의 이름.
            layer_name = "agg_lv" + str(i)
            # 첫 layer의 dimension을 맞춰주기 위함.
            input_dim = internal_dim if i > 1 else features_dim
            # SageMeanAggragator를 통해 aggregation_lager를 생성한다.
            aggregator_layer = SageMeanAggregator(input_dim, internal_dim,
                                                  name=layer_name, activ=True)
            # sequential 컨테이너에 aggregator_layer를 쌓는다.
            self.seq_layers.append(aggregator_layer)

        # MLP로 fraud score 계산하도록 classifier를 생성한다.
        self.classifier = tf.keras.layers.Dense(num_classes,
                                                activation=tf.nn.softmax,
                                                use_bias=False,
                                                kernel_initializer=init_fn,
                                                name="classifier",
                                                )

    def call(self, minibatch, features):
        """
        :param namedtuple minibatch: minibatch of target nodes
        :param tensor features: 2d features of nodes
        """
        # 여기서 x는 배치를 구성하는 노드와 모든 이웃 노드의 피처로 구성된 행렬이다. (src_nodes는 전체!!)
        x = tf.gather(tf.constant(features, dtype=float),
                      tf.squeeze(minibatch.src_nodes))
        for aggregator_layer in self.seq_layers:
            # 배치 노드들의 embedding을 생성한다.
            x = aggregator_layer(x,
                                 minibatch.dstsrc2srcs.pop(),
                                 minibatch.dstsrc2dsts.pop(),
                                 minibatch.dif_mats.pop()
                                 )
        return self.classifier(x)
