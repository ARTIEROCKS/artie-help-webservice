# python
# Archivo: 'lib/keras_custom_layers.py'
import tensorflow as tf
from keras import layers
from keras.saving import register_keras_serializable

@register_keras_serializable(package="Custom")
def compute_mask_layer(mask_value):
    def _fn(inp):
        return tf.cast(tf.reduce_any(inp != mask_value, axis=-1), tf.float32)
    return _fn

# Alias exacto que la Lambda guardada espera ('func').
# IMPORTANTE: ajusta el último dim (15) si tu modelo usa otro número de features.
@register_keras_serializable(package="Custom", name="func")
def func(inp):
    # Mismo comportamiento que el de entrenamiento: máscara por último eje != 0.0
    return tf.cast(tf.reduce_any(inp != 0.0, axis=-1), tf.float32)

@register_keras_serializable(package="Custom")
def squeeze_last_axis_func(t):
    return tf.squeeze(t, axis=-1)

@register_keras_serializable(package="Custom")
def mask_attention_scores_func(inputs):
    scores, mask = inputs
    minus_inf = -1e9
    return scores + (1.0 - mask) * minus_inf

@register_keras_serializable(package="Custom")
def apply_attention_func(inputs):
    x, attn = inputs
    return x * tf.expand_dims(attn, axis=-1)

# Funciones con especificación explícita de output_shape
@register_keras_serializable(package="Custom")
class SqueezeLastAxisLayer(layers.Layer):
    """Custom layer para squeeze que reemplaza la Lambda problemática"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, **kwargs):
        return tf.squeeze(inputs, axis=-1)

    def compute_output_shape(self, input_shape):
        if input_shape[-1] == 1:
            return input_shape[:-1]
        return input_shape

    def get_config(self):
        return super().get_config()

@register_keras_serializable(package="Custom")
class ComputeMaskLayer(layers.Layer):
    """Custom layer para compute mask que reemplaza la Lambda problemática"""
    def __init__(self, mask_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.mask_value = mask_value
        self.supports_masking = True

    def call(self, inputs, **kwargs):
        return tf.cast(tf.reduce_any(inputs != self.mask_value, axis=-1), tf.float32)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]  # Remove the last dimension

    def get_config(self):
        config = super().get_config()
        config.update({'mask_value': self.mask_value})
        return config

@register_keras_serializable(package="Custom")
class MaskAttentionScoresLayer(layers.Layer):
    """Custom layer para mask attention scores que reemplaza la Lambda problemática"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, **kwargs):
        scores, mask = inputs
        minus_inf = -1e9
        return scores + (1.0 - mask) * minus_inf

    def compute_output_shape(self, input_shapes):
        scores_shape, mask_shape = input_shapes
        return scores_shape

    def get_config(self):
        return super().get_config()

@register_keras_serializable(package="Custom")
class ApplyAttentionLayer(layers.Layer):
    """Custom layer para apply attention que reemplaza la Lambda problemática"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, **kwargs):
        x, attn = inputs
        return x * tf.expand_dims(attn, axis=-1)

    def compute_output_shape(self, input_shapes):
        x_shape, attn_shape = input_shapes
        return x_shape

    def get_config(self):
        return super().get_config()

@register_keras_serializable(package="Custom")
class MaskedRepeatVector(layers.Layer):
    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.n = int(n)
        self.supports_masking = True
        self._mask = None

    def call(self, inputs, mask=None):
        outputs = tf.repeat(tf.expand_dims(inputs, axis=1), repeats=self.n, axis=1)
        if mask is not None:
            mask = tf.cast(mask, tf.bool)
            self._mask = tf.repeat(tf.expand_dims(mask, axis=1), repeats=self.n, axis=1)
        return outputs

    def compute_mask(self, inputs, mask=None):
        return self._mask if mask is not None else None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n, input_shape[1])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n": self.n})
        return cfg

@register_keras_serializable(package="Custom")
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.W = None
        self.b = None
        self.u = None

    def build(self, input_shape):
        d = int(input_shape[-1])
        self.W = self.add_weight(
            name="attention_weight", shape=(d, 128),
            initializer="glorot_uniform", trainable=True
        )
        self.b = self.add_weight(
            name="attention_bias", shape=(128,),
            initializer="zeros", trainable=True
        )
        self.u = self.add_weight(
            name="context_vector", shape=(128, 1),
            initializer="glorot_uniform", trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)  # (b,t,128)
        ait = tf.tensordot(uit, self.u, axes=1)                        # (b,t,1)
        ait = tf.squeeze(ait, -1)                                      # (b,t)
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            ait += -10000.0 * (1.0 - mask)
        attn = tf.nn.softmax(ait)                                      # (b,t)
        attn = tf.expand_dims(attn, axis=-1)                           # (b,t,1)
        return tf.reduce_sum(inputs * attn, axis=1)                    # (b,f)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def get_config(self):
        return super().get_config()
