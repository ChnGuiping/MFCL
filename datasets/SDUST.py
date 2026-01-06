import os
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm
from torch.utils.data import DataLoader


signal_size = 2048
stride = 1024            # 步长
sample_num = 400        # 采样数


#2 Bearings with real damages caused by accelerated lifetime tests(10x)
RDBdata = ['NC','IF0.2','IF0.4','IF0.6','OF0.2','OF0.4','OF0.6','RF0.2','RF0.4','RF0.6']

label2 = [i for i in range(len(RDBdata))]

# working condition
WC = ["1800 20", "1800 40", "2000 20", "2000 40"]          # 轴承数据集



# generate Training Dataset and Testing Dataset
def get_files0(root, N, valid_labels):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab = []

    "读取源域数据，覆盖共享类别"

    return [data, lab]


# generate Training Dataset and Testing Dataset
def get_files1(root, N, valid_labels, unknown_labels=None):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab = []

    "读取目标域数据，覆盖全部类型，并且将源域类别之外的类别统一视为一个未知类别"

    return [data, lab]



def data_load(filename, name, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    fl = loadmat(filename)[name]
    fl = fl[0][0][1][0][0][0][:,0]  # Take out the data
    # print(fl)
    fl = fl.reshape(-1, 1)
    data = []
    lab = []

    for i in range(sample_num):
        start = i * stride
        end = signal_size + i * stride
        data.append(fl[start:end])
        lab.append(label)

    return data, lab



class SDUST(object):

    def __init__(self, data_dir, src_name, tgt_name, src_classes, tgt_classes, batch_size):
        self.root = data_dir
        self.src_name = src_name
        self.tgt_name = tgt_name
        self.src_classes = src_classes
        self.tgt_classes = tgt_classes
        self.batch_size = batch_size

    def create_loaders(self):
        num_classes = len(self.src_classes) + 1

        # 获取源域数据
        src_list_data = get_files0(self.root, self.src_name, self.src_classes)
        data_pd = pd.DataFrame({"data": src_list_data[0], "label": src_list_data[1]})
        src_train_pd, src_val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
        source_train = dataset(list_data=src_train_pd, transform=TwoStrongTransform())
        src_train_loader = DataLoader(source_train, batch_size=self.batch_size, shuffle=True, drop_last=False)
        print(src_val_pd.shape)

        # Dataset and DataLoader for source classes
        source_test = dataset(list_data=pd.DataFrame({"data": src_val_pd["data"], "label": src_val_pd["label"]}),
                                    transform=Compose([Reshape(), Normalize("0-1"), Retype()]))
        src_val_loader = DataLoader(source_test, batch_size=self.batch_size, shuffle=False, drop_last=False)

        # 获取目标域数据，包含unknown标签
        tgt_list_data = get_files1(self.root, self.tgt_name, self.tgt_classes)
        data_pd = pd.DataFrame({"data": tgt_list_data[0], "label": tgt_list_data[1]})
        tgt_train_pd, tgt_val_pd = train_test_split(data_pd, test_size=0.2, random_state=40, stratify=data_pd["label"])
        target_train = dataset(list_data=tgt_train_pd, transform=TwoStrongTransform())
        tgt_train_loader = DataLoader(target_train, batch_size=self.batch_size, shuffle=False, drop_last=False)
        print(tgt_val_pd.shape)

        # Dataset and DataLoader for target classes
        target_test = dataset(list_data=pd.DataFrame({"data": tgt_val_pd["data"], "label": tgt_val_pd["label"]}),
                                   transform=Compose([Reshape(), Normalize("0-1"), Retype()]))
        tgt_val_loader = DataLoader(target_test, batch_size=self.batch_size, shuffle=False, drop_last=False)

        return src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader, num_classes
