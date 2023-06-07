import numpy as np
import magicmind.python.runtime as mm

import argparse

class MagicmindModel(object):
    def __init__(self, model_path, device_id=0):
        self.model = mm.Model()
        self.model.deserialize_from_file(model_path)
        self.dev = mm.Device()
        self.dev.id = device_id
        assert self.dev.active().ok()
        self.engine = self.model.create_i_engine()
        self.context = self.engine.create_i_context()
        self.queue = self.dev.create_queue()
        self.inputs = self.context.create_inputs()
        self.outputs = []
    
    def run_infer(self, data):
        # data: detect face data, numpy array, nhwc
        self.inputs[0].from_numpy(data)
        assert self.context.enqueue(self.inputs, self.outputs, self.queue).ok()
        assert self.queue.sync().ok()
        res = self.outputs[0].asnumpy()
        return res

def load_data_npz(npz_path):
    d = np.load(npz_path)
    return d["image"], d["pose"]

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--magicmind_model_1', type=str, default='../model1_fp32_b1.mm', help=' .mm model name')
    parser.add_argument('--magicmind_model_2', type=str, default='../model2_fp32_b1.mm', help=' .mm model name')
    parser.add_argument('--magicmind_model_3', type=str, default='../model3_fp32_b1.mm', help=' .mm model name')
    parser.add_argument('--image_dir', type=str, default='../data/type1/AFLW2000.npz', help=' test data')
    parser.add_argument('--image_num', type=int, default='100', help=' test data nums, max 1969')
    parser.add_argument('--device_id', type=int, default=0, help='device id ')
    args = parser.parse_args()
    # init mm model
    model1 = MagicmindModel(model_path=args.magicmind_model_1, device_id=0)
    model2 = MagicmindModel(args.magicmind_model_2, 0)
    model3 = MagicmindModel(args.magicmind_model_3, 0)

    image,pose = load_data_npz(args.image_dir)
    x_data = []
    y_data = []

    for i in range(0,pose.shape[0]):
        temp_pose = pose[i,:]
        if np.max(temp_pose)<=99.0 and np.min(temp_pose)>=-99.0:
            x_data.append(image[i,:,:,:])
            y_data.append(pose[i,:])
    x_data = np.array(x_data)[:args.image_num]
    y_data = np.array(y_data)[:args.image_num]
    p_data = []
    for x in x_data:
        x = np.expand_dims(x, axis=0)
        res1 = model1.run_infer(x)
        res2 = model2.run_infer(x)
        res3 = model3.run_infer(x)
        res_x = np.concatenate([res1, res2, res3])
        p_result = [[np.mean(res_x[:,0]), np.mean(res_x[:,1]), np.mean(res_x[:,2])]]
        p_data.append(p_result)

    p_data = np.concatenate(p_data) 
    pose_matrix = np.mean(np.abs(p_data-y_data),axis=0)
    MAE = np.mean(pose_matrix)
    yaw = pose_matrix[0]
    pitch = pose_matrix[1]
    roll = pose_matrix[2]
    print("\n")
    print("test on AFLW200 samples: {}/1969, the MAE is:".format(args.image_num))
    print('--------------------------------------------------------------------------------')
    print(' MAE = %3.2f, [yaw,pitch,roll] = [%3.2f, %3.2f, %3.2f]'%(MAE, yaw, pitch, roll))
    print('--------------------------------------------------------------------------------')
        
if __name__ == '__main__':
    main()
