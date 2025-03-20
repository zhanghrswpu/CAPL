from keras.layers import Layer
import tensorflow as tf

class CALayer(Layer):
    def __init__(self, lambda_acr=1.0, **kwargs):
        super(CALayer, self).__init__(**kwargs)
        self.lambda_acr = lambda_acr

    def build(self, input_shape):
        self.center = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True,
            name='center'
        )
        super(CALayer, self).build(input_shape)

    def call(self, inputs):
        acr_regularization = self.lambda_acr * tf.reduce_mean(tf.square(inputs - self.center))
        self.add_loss(acr_regularization)
        return inputs-self.center
