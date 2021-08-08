from lib import *
from L2_norm import * 
from default_box import DefaultBox
def create_vgg():
    layers = []
    in_channels = 3
    cfgs = [64, 64, 'M', 128, 128, 'M',
            256, 256, 256, 'MC', 512, 512, 512, 'M',
            512, 512, 512]

    for cfg in cfgs:
        if cfg == 'M':
            layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
        elif cfg == "MC":
            layers += [nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode = True)]

        else:
            conv2d = nn.Conv2d(in_channels, cfg, kernel_size = 3, padding = 1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = cfg

    pool5 = nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1)
    conv6 = nn.Conv2d(512, 1024, kernel_size = 3, padding = 6, dilation = 6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size = 1)

    layers += [pool5, conv6,nn.ReLU(inplace = True), conv7,nn.ReLU(inplace = True)]

    return nn.ModuleList(layers)

def extras():
    layers = []
    in_channels = 1024

    cfgs = [256, 512, 128, 256,128,256,128,256]
    layers += [nn.Conv2d(in_channels, cfgs[0], kernel_size =1)]
    layers += [nn.Conv2d(cfgs[0], cfgs[1], kernel_size = 3, stride = 2, padding = 1)]
    layers += [nn.Conv2d(cfgs[1], cfgs[2], kernel_size = 1)]
    layers += [nn.Conv2d(cfgs[2], cfgs[3], kernel_size = 3, stride = 2, padding = 1)]
    layers += [nn.Conv2d(cfgs[3], cfgs[4], kernel_size = 1)]
    layers += [nn.Conv2d(cfgs[4], cfgs[5], kernel_size = 3)]
    layers += [nn.Conv2d(cfgs[5], cfgs[6], kernel_size = 1)]
    layers += [nn.Conv2d(cfgs[6], cfgs[7], kernel_size = 3)]

    return nn.ModuleList(layers)

def create_loc_conf(num_classes = 21, box_ratio_num = [4,6,6,6,4,4]):
    loc_layers = []
    conf_layers = []
    #source 1 
    #loc 
    loc_layers += [nn.Conv2d(512, box_ratio_num[0]*4, kernel_size = 3, padding = 1)]
    #conf 
    conf_layers += [nn.Conv2d(512, box_ratio_num[0]*num_classes, kernel_size = 3, padding = 1)]
    #source 2
    loc_layers += [nn.Conv2d(1024, box_ratio_num[1]*4, kernel_size = 3, padding = 1)]
    conf_layers += [nn.Conv2d(1024, box_ratio_num[1]*num_classes, kernel_size = 3, padding = 1)]
    #source 3 
    loc_layers += [nn.Conv2d(512, box_ratio_num[2]*4, kernel_size = 3, padding = 1)]
    conf_layers += [nn.Conv2d(512, box_ratio_num[2]*num_classes, kernel_size = 3, padding = 1)]
    #source 4 
    loc_layers += [nn.Conv2d(256, box_ratio_num[3]*4, kernel_size = 3, padding = 1)]
    conf_layers += [nn.Conv2d(256, box_ratio_num[3]*num_classes, kernel_size = 3, padding = 1)]

    #source 5 
    loc_layers += [nn.Conv2d(256, box_ratio_num[4]*4, kernel_size = 3, padding = 1)]
    conf_layers += [nn.Conv2d(256, box_ratio_num[4]*num_classes, kernel_size = 3, padding = 1)]

    #source 6 
    loc_layers += [nn.Conv2d(256, box_ratio_num[5]*4, kernel_size = 3, padding = 1)]
    conf_layers += [nn.Conv2d(256, box_ratio_num[5]*num_classes, kernel_size = 3, padding = 1)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)

cfg = {
    "num_classes":21,
    "input_size": 300, 
    "bbox_aspect_num": [4,6,6,6,4,4], # ti le khung hinh 
    "feature_maps": [38, 19, 10, 5, 3, 1],
    "steps": [8, 16, 32, 64, 100, 300], #size of default box 
    "min_size":[30, 50, 11, 162, 213, 264],
    "max_size":[50, 11, 162, 213, 264, 315],
    "aspect_ratios":[[2], [2, 3], [2,3], [2,3], [2], [2]]

}

class SSD(nn.Module):
    def __init__(self,phase, cfg):
        super(SSD, self).__init__()
        self.phase = phase
        #dinh nghia cac ham 
        self.vgg = create_vgg()
        self.L2Norm = L2Norm()
        self.extras = extras()
        self.loc_layers, self.conf_layers = create_loc_conf(cfg["num_classes"], cfg["bbox_aspect_num"])
        default = DefaultBox(cfg)
        self.default_box = default.create_defbox()
        if phase == "inference":
            self.detect = Detect()

def decode(loc, def_box):
    boxes = torch.cat((def_box[:, :2] + def_box[:,:2]*loc[:,:2]*0.1, def_box[:,2:]*torch.exp(0.2*loc[:, 2:])), dim = 1)

    boxes[:, :2] = boxes[:, :2] - boxes[:, 2:]
    boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
    return boxes
if __name__ == "__main__":
    # vgg = create_vgg()
    # print(vgg)
    # extras = extras()
    # print(extras)

    # loc, conf = create_loc_conf()
    # print(loc)
    # print(conf)
    ssd = SSD("train", cfg)
    print(ssd)