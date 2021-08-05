from extract_inform_annovation import Anno_xml
from transform import DataTransform
from lib import * 
from make_datapath import make_datapath_list

class MyDataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform, anno_xml):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.anno_xml = anno_xml

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, imdex):
        img, gt, height, width = self.pull_item(index)

        return img , gt

    def pull_item(self, index):
        img_file_path = self.img_list[index]
        img= cv2.imread(img_file_path)
        height, width, chanel = img.shape
        
        anno_file_path = self.anno_list[index]
        anno_infor = self.anno_xml(anno_file_path, width, height)

        img, boxes, label = self.transform(img, self.phase, anno_infor[:, :4], anno_infor[:, 4])

        img = torch.from_numpy(img[:, :, (2,1,0)]).permute(2,0,1)

        gt = np.hstack((boxes, np.expand_dims(label, axis = 1)))
        return img, gt, height, width 


if __name__ == "__main__":
    root_path = "./data/VOC2012/"
    classes =["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",\
                "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",\
                "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_datapath_list(root_path)
    color_mean = [104, 117, 123]
    input_size = 300

    train_dataset = MyDataset(train_img_list, train_annotation_list, phase="train", transform=DataTransform(input_size, color_mean), anno_xml=Anno_xml(classes))
    print(len(train_dataset))