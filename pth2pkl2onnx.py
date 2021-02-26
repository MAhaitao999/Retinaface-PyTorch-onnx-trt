import os

import torch
import torch.nn as nn

from utils.config import cfg_mnet, cfg_re50
from nets.retinaface import RetinaFace
from utils.anchors import Anchors


class CompleteModel(nn.Module):

    def __init__(self, backbone):
        super(CompleteModel, self).__init__()
        self.backbone = backbone

    def forward(self, inputs):
        anchors = Anchors(cfg=cfg_mnet, image_size=(480, 640)).get_anchors()
        # print(anchors.shape)
        outputs = self.backbone(inputs)
        return anchors, torch.cat([outputs[0], outputs[1], outputs[2]], dim=-1)


if __name__ == '__main__':
    #----------------------------------------#
    #    加载网络结构图
    #----------------------------------------#
    net = RetinaFace(cfg=cfg_mnet, mode='eval').eval()
    print(net)

    #----------------------------------------#
    #    加载网络权重
    #----------------------------------------#
    print('Loading weights into state dict...')
    state_dict = torch.load('./model_data/Retinaface_mobilenet0.25.pth')
    print(state_dict)

    #----------------------------------------#
    #    将权重加载到图中并保存成pkl文件
    #----------------------------------------#
    net.load_state_dict(state_dict)
    torch.save(net, 'retinaface.pkl')
    print('Finished!')

    dummy_input = torch.randn(1, 3, 480, 640)
    input_names = ['input_image']
    output_names = ['bbox_regressions', 'classifications', 'ldm_regressions']

    torch.onnx.export(net,
                      dummy_input,
                      'retinaface.onnx',
                      dynamic_axes={'input_image': {0: 'batch', 2: 'H', 3: 'W'},
                                    'bbox_regressions': {0: 'batch', 1: 'S'},
                                    'classifications': {0: 'batch', 1: 'S'},
                                    'ldm_regressions': {0: 'batch', 1: 'S'}},
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=11)

    complete_model = CompleteModel(backbone=net)
    output_name = ['anchors_shape', 'complete_model_output']

    torch.onnx.export(complete_model,
                      dummy_input,
                      'complete_model.onnx',
                      dynamic_axes={'input_image': {0: 'batch', 2: 'H', 3: 'W'},
                                    'complete_model_output': {0: 'batch', 1: 'S'}},
                      input_names=input_names,
                      output_names=output_name,
                      opset_version=11
                      )
