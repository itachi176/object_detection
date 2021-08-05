from lib import *

from make_datapath import  make_datapath_list

class Anno_xml(object):
    def __init__(self, classes):
        self.classes = classes


    def __call__(self, xml_path, width, height):
        ret = []
        
        #read file xml
        xml = ET.parse(xml_path).getroot()

        for obj in xml.iter('object'):
            difficult = int(obj.find("difficult").text)
            if (difficult == 1):
                    continue
            bb = []
            name = obj.find("name").text.lower().strip()
            bbox  = obj.find("bndbox")

            point = ["xmin", "ymin", "xmax", "ymax"]
            for pt in point:
                pixel = int(bbox.find(pt).text)-1

                if (pt == "xmin" or pt == "xmax"):
                    pixel /= width
                else:
                    pixel /= height
                bb.append(pixel)

            label_id = self.classes.index(name)
            bb.append(label_id)

            ret += [bb]

        return np.array(ret)

if __name__ == "__main__":
    classes =["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    
    anno_xml = Anno_xml(classes)   

    id = 1 
    root_path = "./data/VOC2012/"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path)
    img_file_path = val_img_list[id]
    img = cv2.imread(img_file_path)
    height, width, chanel  = img.shape
    bb = anno_xml(val_annotation_list[id],width, height)
    print(bb)