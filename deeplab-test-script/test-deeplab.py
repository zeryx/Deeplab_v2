from __future__ import print_function
caffe_root = 'code/'
import sys
sys.path.insert(0, caffe_root + 'python')

import os
import numpy as np
from PIL import Image as PILImage
import caffe

def main():
  net_path = "deploy.prototxt"
  model_path = "trained.caffemodel"
  gpu_id = -1

  net = Segmenter(net_path, model_path, gpu_id)
  img = np.zeros((505,505,3))
  
  # print(img)
  
  segm = net.predict([img])
  
  # print("result is")
  # print(segm)

  assert np.all(segm==img.astype(np.float32).transpose(2, 0, 1)), "ERROR: Deeplab did not return expected result"
  print("SUCCESS")

class Segmenter(caffe.Net):
  def __init__(self, prototxt, model, gpu_id=-1):
    caffe.Net.__init__(self, prototxt, model, caffe.TEST)
   # self.set_phase_test()

    if gpu_id < 0:
      caffe.set_mode_cpu()
    else:
      caffe.set_mode_gpu()
      caffe.set_device(gpu_id)

  def predict(self, inputs):
    # uses MEMORY_DATA layer for loading images and postprocessing DENSE_CRF layer
    img = inputs[0].transpose((2, 0, 1))
    img = img[np.newaxis, :].astype(np.float32)
    label = np.zeros((1, 1, 1, 1), np.float32)
    data_dim = np.zeros((1, 1, 1, 2), np.float32)
    data_dim[0][0][0][0] = img.shape[2]
    data_dim[0][0][0][1] = img.shape[3]

    img      = np.ascontiguousarray(img, dtype=np.float32)
    label    = np.ascontiguousarray(label, dtype=np.float32)
    data_dim = np.ascontiguousarray(data_dim, dtype=np.float32)

    self.set_input_arrays(img, label)
    out = self.forward()

    predictions = out[self.outputs[0]] # the output layer should be called crf_inf
    segm_result = predictions[0].argmax(axis=0).astype(np.uint8)

    return segm_result 

if __name__ == '__main__':
  main()
