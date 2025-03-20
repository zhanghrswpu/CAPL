from keras.layers import Layer
import tensorflow as tf

class CALayer(Layer):
    def __init__(self, lambda_acr=1.0, **kwargs):
        super(CALayer, self).__init__(**kwargs)
        self.lambda_acr = lambda_acr

    def build(self, input_shape):
        # 初始化中心向量，作为可学习的参数
        self.center = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True,
            name='center'
        )
        super(CALayer, self).build(input_shape)

    def call(self, inputs):
        # 正则化项：最小化样本与中心向量的距离
        acr_regularization = self.lambda_acr * tf.reduce_mean(tf.square(inputs - self.center))
        # 返回输入，便于后续网络处理
        self.add_loss(acr_regularization)
        return inputs-self.center
