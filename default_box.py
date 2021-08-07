from lib import * 

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
 
class DefaultBox():
    def __init__(self, cfg):
        self.img_size = cfg["input_size"]
        self.feature_maps = cfg["feature_maps"]
        self.min_size = cfg["min_size"]
        self.max_size = cfg["max_size"]
        self.aspect_ratios = cfg["aspect_ratios"]
        self.steps = cfg['steps']

    def create_defbox(self):
        dbox_list = []

        for k, f in enumerate(self.feature_maps):
            for i, j in itertools.product(range(f), repeat=2):
                f_k = self.img_size/self.steps[k]

                cx = (i+0.5)/f_k
                cy = (j+0.5)/f_k
                #small box 
                s_k = self.min_size[k]/self.img_size 
                dbox_list += [cx, cy, s_k, s_k]

                #big box 
                s_k = sqrt(s_k* (self.max_size[k]/self.img_size))
                dbox_list += [cx, cy, s_k, s_k]

                for ar in self.aspect_ratios[k]:
                    dbox_list += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    dbox_list += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        output= torch.Tensor(dbox_list).view(-1, 4)
        output.clamp_(max = 1, min = 0)
        return output

# if __name__ == "__main__":
#     default = DefaultBox(cfg)
#     boxes = default.create_defbox()
#     print(boxes[0])