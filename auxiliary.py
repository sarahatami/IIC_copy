# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import torch
#
#
# outmask = torch.tensor([[0, 1, 0, 0, 1],
#                         [1, 1, 1, 0, 0],
#                         [0, 0, 1, 1, 0],
#                         [1, 0, 0, 0, 1]])
# # outmask = trans(mask[0])
# # outmask.show()
#
# print("type")
# print(type(outmask))
# array = outmask.numpy()
# plt.imshow(array, cmap='gray', interpolation='nearest')
# plt.axis('off')
# plt.show()


import torch
from PIL import Image
#
# def show_tensor_as_image(tensor):
#     # Convert the tensor to a PIL image
#     # image = Image.fromarray((255 * tensor).byte().numpy(), mode='RGB')
#     tensor = tensor.byte().numpy()  # Convert to byte tensor
#     tensor = tensor.transpose(1, 2, 0)  # Transpose dimensions
#     image = Image.fromarray(tensor, mode='RGB')
#
#     # Display the image
#     image.show()
#
#
# import torch
#
# # Create a random 3D tensor of size (3, 4, 5)
# tensor = torch.randn(3, 128, 128)
#
# # Print the tensor
# print(tensor.size())
# print(tensor)
#
# # Display the tensor as an image
# show_tensor_as_image(tensor)
#


import numpy as np
import torch

x = np.arange(24).reshape((2, 3, 4))


res= np.argmax(x, axis=0)
print(x)
print(res)

# trans = torchvision.transforms.ToPILImage()
# outi = trans(imgs[0])
# # outi.show()
#
# out = trans(x_outs[0][0])
# # out.show()
#
# outtargets = trans(flat_targets[0])
# # outtargets.show()
#
# outmask = trans(mask[0])
# # outmask.show()

# def show_tensor_as_image(tensor):
#     # Convert the tensor to a PIL image
#     # image = Image.fromarray((255 * tensor).byte().numpy(), mode='RGB')
#     # tensor = tensor.byte().numpy()  # Convert to byte tensor
#     tensor = tensor  # Convert to byte tensor
#     tensor = tensor.transpose(1, 2, 0)  # Transpose dimensions
#     image = Image.fromarray(tensor, mode='RGB')
#     # Display the image
#     image.show()


# import matplotlib.pyplot as plt
# colormap = plt.cm.get_cmap('viridis')
# print(colormap)



######################### IN MAIN CODE ####################3
# 15
# colormap = plt.cm.get_cmap('Blues')
# output_normalized = out_image/ np.max(out_image)
# plt.imshow(output_normalized, cmap=colormap)
# plt.colorbar()
# plt.savefig("myimage.png")