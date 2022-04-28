import tensorflow as tf
import tf2onnx

from tensorflow.keras.models import load_model

# Documentation: https://github.com/onnx/tensorflow-onnx

model = load_model("echoWeights.hdf5") #  change to model name

spec = (tf.TensorSpec((None, 30, 112, 112, 3), tf.float32, name="InputLayer"),) # parameters: input tensor shape, data type, name of input layer

output_path = "onnx_model.onnx" # directory and name for converted ONNX file

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path) # parameters: tf model, spec, opset (see documentation), output dir 

output_names = [n.name for n in model_proto.graph.output] # make a note of output layer name (list), used for inference

