#!/usr/bin/env python

import onnxruntime as ort
import onnx
import sys
import time

from onnx import numpy_helper
from google.protobuf import text_format

model_dir = "./mnist-12"

# Load the model and create InferenceSession
providers=['CPUExecutionProvider']
session = ort.InferenceSession(f"{model_dir}/mnist-12.onnx", providers=providers)

# "Load and preprocess the input image inputTensor"
tensor = onnx.TensorProto()
input_file = f"{model_dir}/test_data_set_0/input_0.pb"
with open(input_file, 'rb') as f:
    tensor.ParseFromString(f.read())
    input_data = numpy_helper.to_array(tensor)
    input_name = session.get_inputs()[0].name

# Load expected output
output_tensor = onnx.TensorProto()
with open(f"{model_dir}/test_data_set_0/output_0.pb", 'rb') as f:
    output_tensor.ParseFromString(f.read())
    expected_output = numpy_helper.to_array(output_tensor)

# Run inference
pre_inference_time = time.time()
output = session.run([], {input_name: input_data})[0]
pos_inference_time = time.time()
inference_time = pos_inference_time-pre_inference_time

result = ((output < expected_output*1.1) & (output > expected_output*0.9))
print(result.all())
if (not result.all()):
    print("Inference output is not what was expected.")
    print(f"Expected output: {expected_output}")
    print(f"Output is as expected: {output}")
else:
    print(f"Output is as expected: {output}")
    print(f"Inference Time: {inference_time}")
