from extract_inform_annotation import Anno_xml
from utils.augumentation import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, \
    Expand, RandomSampleCrop, RandomMirror, \
    ToPercentCoords, Resize, SubtractMeans

from make_datapath import make_datapath_list
from lib import *

class DataTransform(object):
    def __init__(self, input_size, color_mean):
        self.data_trasform = {
            "train": Compose([              
                ConvertFromInts(),    #convert img from int to float32 
                ToAbsoluteCoords(), # back annotation to normal type
                PhotometricDistort(), #change color by random 
                Expand(color_mean),
                RandomSampleCrop(), #random crop image 
                RandomMirror(), # xoay anh theo kieu guong 
                ToPercentCoords(), # chuan hoa ve dang [0:1]
                Resize(input_size),
                SubtractMeans(color_mean)
            ]),
            "val": Compose([
                ConvertFromInts(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ])
        }

    def __call__(self, img, phase, boxes, labels):
        return self.data_trasform[phase](img, boxes, labels)

if __name__ == "__main__":
    root_path = "./data/VOC2012/"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path)
    img_file_path = val_img_list[1]
    img = cv2.imread(img_file_path) #BGR image (height, width, channel) 
    height, width, channel = img.shape
    classes =["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    #annotation 
    trans_anno = Anno_xml(classes)
    boxes = trans_anno(val_annotation_list[1],width, height)

    #plot origin image 
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    color_mean = [104, 117, 123]
    input_size = 300
    transform = DataTransform(input_size, color_mean)

    phase = "train"

    img_transformed, boxes, labels = transform(img,phase, boxes[:, :4], boxes[:, 4])
    
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
    plt.show()
