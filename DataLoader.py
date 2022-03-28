from torch.utils.data import Dataset
import torch
from PIL import Image
import pandas as pd
import random
random.seed(10)

class AOIDataset(Dataset) : 
    def __init__(self, train : bool =None, val : bool =None, test : bool =None, transform=None) : 
    """"""""""""""""
    AOI 데이터를 불러오기 위한 데이터셋 클래스입니다. 
    작업 공간내에 인환님이 올려주신 train, validation,test csv 파일을 업로드해주시고, 아래의 read_csv의 파일 Path를 변경하여 사용해주시면 됩니다. 

    Args :
        train : bool, True로 설정 시 train용 csv 파일을 불러옵니다.
        val : bool, True로 설정 시 validation용 csv 파일을 불러옵니다. 
        test : bool, True로 설정 시 test용 csv 파일을 불러옵니다. 

    Return :
        img : 각 이미지를 PIL로 읽은 후 이를 반환하였습니다. 필히, transform에서 Tensor로 변환 후 학습하길 바랍니다. 
        label : defect 클래스에 따라 label encoding한 결과를 return해 주었습니다. 
    """"""""""""""""
        
        if train : 
            data = pd.read_csv('YOUR TRAIN DATA PATH')
        elif val : 
            data = pd.read_csv('YOUR VALIDATION DATA PATH')
        else : 
            data = pd.read_csv('YOUR TEST DATA PATH')
        
        self.imgs = data['path']
        self.imgs = list(self.imgs)
        self.labels = data['label']
        self.label_tags = {'normal' : 0,
                            'burr' : 1,
                            'substance' : 2,
                            'metalburr' : 3,
                            'crack' : 4,
                            'overflow' : 5,
                            'unfulfilled' : 6}
        
        self.labels = torch.tensor(self.labels.replace(self.label_tags), dtype=torch.long)

        #self.imgs.reset_index(drop=True, inplace=True)
        self.transform = transform

    def __len__(self) : 
        return len(self.imgs)

    def __getitem__(self, index) :
        img_path = self.imgs[index]
        img = Image.open(img_path)
        label = self.labels[index]

        if self.transform is not None : 
            img = self.transform(img)

        return img, label

