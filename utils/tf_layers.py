"""TensorFlow layers kept separate from utils/features.py so numpy-only clients don't import TF."""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="handsign")
class FeaturePreprocess(tf.keras.layers.Layer):
    """(B, T, 1662) raw holistic frames -> (B, T, num_features) selected, normalized features."""

    def __init__(self, columns, normalize, left_shoulder_cols, right_shoulder_cols,
                 default_center, default_scale, **kwargs):
        super().__init__(**kwargs)
        self.columns = list(columns)
        self.normalize = bool(normalize)
        self.left_shoulder_cols = list(left_shoulder_cols)
        self.right_shoulder_cols = list(right_shoulder_cols)
        self.default_center = list(default_center)
        self.default_scale = float(default_scale)
        self.num_points = len(self.columns) // 3

    def call(self, x):
        pts = tf.gather(x, self.columns, axis=-1)
        if not self.normalize:
            return pts
        shape = tf.shape(x)
        pts = tf.reshape(pts, [shape[0], shape[1], self.num_points, 3])

        ls = tf.gather(x, self.left_shoulder_cols, axis=-1)
        rs = tf.gather(x, self.right_shoulder_cols, axis=-1)
        both = tf.logical_and(tf.reduce_max(tf.abs(ls), axis=-1) > 0,
                              tf.reduce_max(tf.abs(rs), axis=-1) > 0)[..., None]   # (B, T, 1)

        center = tf.where(both, (ls + rs) / 2.0,
                          tf.broadcast_to(tf.constant(self.default_center, x.dtype), tf.shape(ls)))
        diff = ls[..., :2] - rs[..., :2]
        dist = tf.sqrt(tf.reduce_sum(diff * diff, axis=-1, keepdims=True))
        scale = tf.maximum(tf.where(both, dist, tf.ones_like(dist) * self.default_scale), 1e-3)

        present = tf.reduce_max(tf.abs(pts), axis=-1, keepdims=True) > 0
        out = (pts - center[:, :, None, :]) / scale[:, :, None, :]
        out = tf.where(present, out, tf.zeros_like(out))
        return tf.reshape(out, [shape[0], shape[1], self.num_points * 3])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], len(self.columns))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "columns": self.columns,
            "normalize": self.normalize,
            "left_shoulder_cols": self.left_shoulder_cols,
            "right_shoulder_cols": self.right_shoulder_cols,
            "default_center": self.default_center,
            "default_scale": self.default_scale,
        })
        return cfg
