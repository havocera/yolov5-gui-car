import torch
model = torch.load("bestv5s.onnx")
res = model("car.png")
print(res)