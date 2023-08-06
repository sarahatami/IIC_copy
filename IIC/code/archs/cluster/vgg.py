import torch.nn as nn


class VGGTrunk(nn.Module):  # inherits from the nn.Module
  def __init__(self):
    super(VGGTrunk, self).__init__()  # about inheritance

  print("WE ARE IN VGG!")

  def _make_layers(self, batch_norm=True):
    layers = []
    in_channels = self.in_channels
    for tup in self.cfg:  # configuration of the VGGNet. a list of tuples
      assert (len(tup) == 2)
      print("tup: ", tup)
      # tup: (64, 1)
      # tup: (128, 1)
      # tup: ('M', None)
      # tup: (256, 1)
      # tup: (256, 1)
      # tup: (512, 2)

      out, dilation = tup  # number of output channels and dilation rate for a convolutional layer
      sz = self.conv_size
      stride = 1
      pad = self.pad  # to avoid shrinking

      if out == 'M':
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # max pooling layer is added
      elif out == 'A':
        layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
      else:
        conv2d = nn.Conv2d(in_channels, out, kernel_size=sz,
                           stride=stride, padding=pad,
                           dilation=dilation, bias=False)
        if batch_norm:
          layers += [conv2d, nn.BatchNorm2d(out,
                                            track_running_stats=self.batchnorm_track),
                     nn.ReLU(inplace=True)]
        else:
          layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = out
      # print("layers: ", layers)
    return nn.Sequential(*layers)


class VGGNet(nn.Module):
  def __init__(self):
    super(VGGNet, self).__init__()

  def _initialize_weights(self, mode='fan_in'):
    for m in self.modules():
      # print("m: ", m)
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity='relu')#Kaiming normal initialization method
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        assert (m.track_running_stats == self.batchnorm_track)
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()
