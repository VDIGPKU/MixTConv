import torch
import matplotlib.pyplot as plt
import numpy as np
check_point = torch.load('pretrained/somethingv1_RGB_resnet50_ops_group1dk3outer_8_avg_segment8_e50/ckpt.best.pth.tar', map_location=torch.device('cpu'))
state_dict = (check_point['state_dict'])
parm = {}
for name, parameters in state_dict.items():
    print(name, ':', parameters.size())
    parm[name] = parameters.detach()
# layer = 'module.base_model.layer1.2.conv1.conv15d.weight'
layer = 'module.base_model.layer4.1.conv1.conv1d.weight'
size = parm[layer].size()

# print(size)
# print(parm[layer][: size[0] // 8, :, :, :, :])
# print(parm[layer][size[0] // 8: size[0] // 4, :, :, :, :])
print(parm[layer][size[0] // 4:, :, :, :, :])

dynamic1 = np.array(parm[layer][: size[0] // 8, :, :, :, :].squeeze())
dynamic2 = np.array(parm[layer][size[0] // 8: size[0] // 4, :, :, :, :].squeeze())
static = np.array(parm[layer][size[0] // 4:, :, :, :, :].squeeze())

from mpl_toolkits.mplot3d import Axes3D  # 用来给出三维坐标系。

# figure = plt.figure()

# 画出三维坐标系：

# axes = Axes3D(figure)

# axes.plot(np.array([0,1,2]), dynamic1[0], dynamic1[1])

plt.bar(np.array(range(static.shape[0])), static[:,2])
plt.show()
