#!/usr/bin/env python

import onnxruntime as ort
import onnx
import sys
import time

from onnx import numpy_helper
from google.protobuf import text_format

model_dir = "/cfs/klemming/home/p/pedroa/Private/Project/mnist-12"

# Load the model and create InferenceSession
providers=['CPUExecutionProvider']
session = ort.InferenceSession(f"{model_dir}/mnist-12.onnx", providers=providers)


# "Load and preprocess the input image inputTensor"
tensor = onnx.TensorProto()
input_file = f"{model_dir}/test_data_set_0/input_0.pb"
with open(input_file, 'rb') as f:
        tensor.ParseFromString(f.read())
        input_data = numpy_helper.to_array(tensor)
#print(tensor.name)
#print(tensor)

# Load expected output
output_tensor = onnx.TensorProto()
with open(f"{model_dir}/test_data_set_0/output_0.pb", 'rb') as f:
    output_tensor.ParseFromString(f.read())
    output_data = numpy_helper.to_array(output_tensor)
#print(output_tensor)
print(f"Expected output: {output_data}")

# get the name of the first input of the model
input_name = session.get_inputs()[0].name
print('Input Name:', input_name)

inference_time = 0
# Run inference
pre_inference_time = time.time()
outputs = session.run([], {input_name: input_data})[0]
pos_inference_time = time.time()
inference_time += pos_inference_time-pre_inference_time
print(outputs)
print(f"Inference Time: {inference_time}")
