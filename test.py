from yolo import YOLOv5,Graph_YOLOv5,YOLOv5_for_Graph
import torch as pytorch
import oneflow as torch
import numpy as np
from PIL import Image
import yolo
import sys
model_name = []
model_in_hook = []


def hook(module, fea_in, fea_out):

    global module_name
    global model_in_hook
    model_in_hook.append(fea_in)
    model_name.append(module)
    return None
def register_hook(model):
    for i in model.children():
        register_hook(i)
    model.register_forward_hook(hook=hook)


yolo.setup_seed(1)
device = torch.device("cuda")
model = YOLOv5(80, (0.33, 0.5), img_sizes=[640, 640]).to(device)
parameters = pytorch.load("yolov5s.pth")
new_parameters = dict()
for key, value in parameters.items():
    if "num_batches_tracked" not in key:
        val = value.detach().cpu().numpy()
        val = torch.from_numpy(val).float()
        new_parameters[key] = val
model.load_state_dict(new_parameters)
model.to(device)
graph_model = Graph_YOLOv5(model)
target = [{'image_id': torch.tensor([2235]).to(device), 'boxes': torch.tensor([39.,489.,59.,590.]).view(1,4).to(device),
           'labels': torch.tensor([0]).to(device)}]
img = Image.open('images/r001.jpg')
k = torch.from_numpy(np.array(img).astype('float32'))[:640, :640].permute(2, 0, 1)
image = [k.to(device)]
register_hook(model)
t = model(image, target)
print(t)
with open('flow_net.txt','w') as f:
    for i in range(50):
        if len(model_in_hook[i]) >1:
            continue
        np.save('oneflow_np/'+str(i),model_in_hook[i][0].detach().numpy())
        f.write(str(i))
        f.write(':')
        f.write(model_name[i].__class__.__name__)
        f.write("\n")