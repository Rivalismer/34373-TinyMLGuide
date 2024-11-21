import ai_edge_torch
import numpy
import torch
import torchvision
import os
import cv2
import tensorflow as tf

# Ensure correct path
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Avoid CUDA conflicts
os.environ['PJRT_DEVICE'] = 'CPU'

# Instantiate the model with weights
# We use a pre-trained model here, but this could be exchanged for your own models
print("Fetching model")
resnet18 = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()

print("Model found, testing input")

# Test the model on ostrich picture
DATA_PATH = "goldfish.jpeg"
input_data = cv2.imread(DATA_PATH)
print(input_data.shape)

input_data = torch.tensor(input_data, dtype=torch.float32) / 255
input_data = input_data.reshape(3, input_data.shape[0], input_data.shape[1])

# Preprocess the image in the same way as the rest of the dataset for resnet18
preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)), # Same size as specified when converting, if you are unsure you can check this with input_details[0]['shape']
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

sample_inputs = (preprocess(input_data).unsqueeze(0), ) # Change channels to be second dimension
torch_output = resnet18(*sample_inputs)
print("Output received, converting")

# The convert function takes the model from PyTorch and converts it to LiteRT
# Note that it requires a sample input for tracing and shaping output during inference
resnet_model = ai_edge_torch.convert(resnet18, sample_inputs)
print("Model converting, testing LiteRT model")

resnet_output = resnet_model(*sample_inputs)
print("Model tested, verifying conversion")

# Model can then be validated for good measure
if (numpy.allclose(
    torch_output.detach().numpy(), 
    resnet_output,
    atol=1e-5,
    rtol=1e-5,
)):
    print("Inference result was within tolerance")
else:
    print("Something went wrong with the conversion")
    exit()

# Save model
resnet_model.export('model.tflite')

# Check probs
prob_res = tf.nn.softmax(resnet_output)
prob_torch = torch.nn.functional.softmax(torch_output, dim=0)

# Find top prob
max_torch = torch.argmax(prob_torch)
max_val = tf.reduce_max(prob_res, keepdims=True)
cond = tf.equal(prob_res, max_val)
res = tf.where(cond)
print(res, max_torch) # See that these match