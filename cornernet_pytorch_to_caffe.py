import sys
import os
import json

sys.path.insert(0, '.')
import torch
from torch.autograd import Variable
from torchvision.models import resnet
import pytorch_to_caffe

from config import system_configs
from nnet.py_factory import NetworkFactory

if __name__ == '__main__':
    # name = 'resnet18'
    # resnet18 = resnet.resnet18(pretrained=True)
    # checkpoint = torch.load("/home/shining/Downloads/resnet18-5c106cde.pth")
    name = 'cornernet'
    db = 1

    cfg_file = os.path.join(system_configs.config_dir, "CornerNet" + ".json")
    with open(cfg_file, "r") as f:
        configs = json.load(f)

    configs["system"]["snapshot_name"] = "CornerNet"
    system_configs.update_config(configs["system"])

    cornernet = NetworkFactory(db).model

    # resnet18.load_state_dict(checkpoint)
    cornernet.cuda()
    cornernet.eval()
    input = torch.ones([1, 3, 511, 511]).cuda()
    # input=torch.ones([1,3,224,224])
    pytorch_to_caffe.trans_net(cornernet, input, name)
    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))