import numpy as np
import onnx
import onnxruntime as rt
import cv2
import os

def change_input_dim(model):
    # Use some symbolic name not used for any other dimension
    sym_batch_dim = "N"
    # or an actal value
    actual_batch_dim = 4
    # The following code changes the first dimension of every input to be batch-dim
    # Modify as appropriate ... note that this requires all inputs to
    # have the same batch_dim
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = sym_batch_dim
    #model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = sym_batch_dim

def apply(transform, infile, outfile):
    model = onnx.load(infile)
    transform(model)
    onnx.save(model, outfile)
    return model

class ReID_Inference():
    def __init__(self,  path='reid_pytorch/reid1.onnx'):
        if os.path.exists(path):
            self.model = onnx.load(path)
        else:
            print('model not exists!')
            self.model = apply(change_input_dim, "reid1.onnx", path)
        #create runtime session
        self.sess = rt.InferenceSession(path)
        # get output name
        self.input_name = self.sess.get_inputs()[0].name
        print("input name", self.input_name)
        self.output_name= self.sess.get_outputs()[0].name
        print("output name", self.output_name)
        self.output_shape = self.sess.get_outputs()[0].shape
        print("output shape", self.output_shape)


    def forward_batch(self, batch_imgs):
        ress = self.sess.run([self.output_name], {self.input_name: batch_imgs})
        return np.array(ress)
     
    def __call__(self, imgs):
        features = self.forward_batch(imgs)
        features = features.reshape(imgs.shape[0], -1)
        return features
