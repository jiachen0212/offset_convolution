import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append('./python')
import numpy as np
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import matplotlib.pyplot as plt

import pdb

proto_src = './examples/vgg9/lr1/s2/train_val.prototxt'
model_addr = './examples/vgg9/lr1/s2/model_bn/solver_iter_15000.caffemodel'

net_src = caffe.Net(proto_src, model_addr, caffe.TRAIN)
net_param_src = caffe_pb2.NetParameter()
text_format.Merge(open(proto_src).read(), net_param_src)
size = len(net_param_src.layer)

for i in xrange(size):
  layer = net_param_src.layer[i]
  layer_type = layer.type
  if layer_type == 'DeforConvolution':
    shift_float = net_src.params[layer.name][1].data[:].flatten()  # offset
    shift = np.round(shift_float)
    points = shift.shape[0] / 2 # inckhkw2 /2 offset points
    y_shift = []
    x_shift = []
    count = []
    ifPointExist = {}
    point_num = 0
    for n in xrange(points):
      if (shift[2 * n], shift[2 * n + 1]) in ifPointExist.keys():
        point_idx = ifPointExist[(shift[2 * n], shift[2 * n + 1])]
        count[point_idx] += 1
      else:
        x_shift.append(shift[2 * n])
        y_shift.append(shift[2 * n + 1])
        count.append(1)
        ifPointExist[(shift[2 * n], shift[2 * n + 1])] = point_num
        point_num += 1
    x_shift = np.array(x_shift, dtype=np.int32)
    y_shift = np.array(y_shift, dtype=np.int32)
    count = np.array(count, dtype=np.float32) ** 0.3
    count = count / np.max(count) * 15
    count = np.pi * (count ** 2)
    colors = np.random.rand(len(count))
    plt.figure(i)
    # count 越大, 点越大.
    plt.scatter(x_shift, y_shift, s=count, c=colors, alpha=0.5, marker='o')
    # plt.savefig('.examples/vgg9/lr1/s1/' + layer.name[:-7] + '.png')
    plt.savefig('examples/vgg9/lr1/s2/' + layer.name + '.png')
