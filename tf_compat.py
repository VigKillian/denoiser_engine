import tensorflow as _tf

try:
    tf = _tf.compat.v1
    tf.disable_v2_behavior()
except Exception:
    tf = _tf
