import torch
from models.DCViT import *
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import random
import logging
from pytz import timezone
from datetime import datetime
import sys
import torchvision.transforms as T

logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('Asia/Taipei')).timetuple()

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def NormalizeData_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

def get_data_loader(data_path="", data_path2="live", batch_size=5, shuffle=True, drop_last=True):
    # data path
    data = None
    live_spoof_label = None


    if data_path2 == "live":
        data = np.load(data_path)
        live_spoof_label = np.ones(len(data), dtype=np.int64)
    else:
        print_data = np.load(data_path)
        replay_data = np.load(data_path2)
        data = np.concatenate((print_data, replay_data), axis=0)
        live_spoof_label = np.zeros(len(data), dtype=np.int64)

    # dataset
    trainset = torch.utils.data.TensorDataset(torch.tensor(np.transpose(data, (0, 3, 1, 2))),
                                              torch.tensor(live_spoof_label))
    # free memory
    import gc
    del data
    gc.collect()
    # dataloader
    data_loader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              drop_last=drop_last)
    return data_loader


def get_inf_iterator(data_loader):
    # """Inf data iterator."""
    while True:
        for images,  live_spoof_labels in data_loader:
            yield (images, live_spoof_labels)


device_id = "cuda:0" 
results_filename = 'OMC(I)'
results_path = "E:/ICIP/" + results_filename
batch_size = 7

file_handler = logging.FileHandler(filename='C:/Users/Jxchong/Desktop/ICIp/logger/'+results_filename+'_train.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
date = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=date, handlers=handlers)

# replay casia Oulu MSU
dataset1 = "Oulu"
dataset2 = "MSU"
dataset3 = "casia"
logging.info(f"Train on {dataset1}, {dataset2}, {dataset3}")

live_path1 = 'C:/Users/Jxchong/Desktop/ICIp/domain-generalization/' + dataset1 + '_images_live.npy'
live_path2 = 'C:/Users/Jxchong/Desktop/ICIp/domain-generalization/' + dataset2 + '_images_live.npy'
live_path3 = 'C:/Users/Jxchong/Desktop/ICIp/domain-generalization/' + dataset3 + '_images_live.npy'

print_path1 = 'C:/Users/Jxchong/Desktop/ICIp/domain-generalization/' + dataset1 + '_print_images.npy'
print_path2 = 'C:/Users/Jxchong/Desktop/ICIp/domain-generalization/' + dataset2 + '_print_images.npy'
print_path3 = 'C:/Users/Jxchong/Desktop/ICIp/domain-generalization/' + dataset3 + '_print_images.npy'

replay_path1 = 'C:/Users/Jxchong/Desktop/ICIp/domain-generalization/' + dataset1 + '_replay_images.npy'
replay_path2 = 'C:/Users/Jxchong/Desktop/ICIp/domain-generalization/' + dataset2 + '_replay_images.npy'
replay_path3 = 'C:/Users/Jxchong/Desktop/ICIp/domain-generalization/' + dataset3 + '_replay_images.npy'


Fas_Net = vit_base_patch16_224(pretrained=True).to(device_id)

criterionCls = nn.CrossEntropyLoss().to(device_id)

optimizer_meta = optim.AdamW(Fas_Net.parameters(), lr=1e-5, betas=(0.5,0.8)) #
Fas_Net.train()

model_save_step = 50
model_save_epoch = 1

save_index = 0

data1_real = get_data_loader(data_path=live_path1, data_path2="live",
                             batch_size=batch_size, shuffle=True)
data2_real = get_data_loader(data_path=live_path2, data_path2="live",
                             batch_size=batch_size, shuffle=True)
data3_real = get_data_loader(data_path=live_path3, data_path2="live",
                             batch_size=batch_size, shuffle=True)
data1_fake = get_data_loader(data_path=print_path1, data_path2=replay_path1,
                             batch_size=batch_size, shuffle=True)
data2_fake = get_data_loader(data_path=print_path2, data_path2=replay_path2,
                             batch_size=batch_size, shuffle=True)
data3_fake = get_data_loader(data_path=print_path3, data_path2=replay_path3,
                             batch_size=batch_size, shuffle=True)

iternum = max(len(data1_real), len(data2_real),
              len(data3_real), len(data1_fake),
              len(data2_fake), len(data3_fake))
log_step = 20
logging.info(f"iternum={iternum}")
data1_real = get_inf_iterator(data1_real)
data2_real = get_inf_iterator(data2_real)
data3_real = get_inf_iterator(data3_real)
data1_fake = get_inf_iterator(data1_fake)
data2_fake = get_inf_iterator(data2_fake)
data3_fake = get_inf_iterator(data3_fake)
 
T_transform = torch.nn.Sequential(
        T.Pad(40, padding_mode="symmetric"),
        T.RandomRotation(30), 
        T.RandomHorizontalFlip(p=0.5),
        T.CenterCrop(224),
)



for epoch in range(100):

    for step in range(iternum):
        # ============ one batch extraction ============#
        img1_real, ls_lab1_real = next(data1_real)
        img1_fake, ls_lab1_fake= next(data1_fake)

        img2_real, ls_lab2_real = next(data2_real)
        img2_fake, ls_lab2_fake = next(data2_fake)

        img3_real, ls_lab3_real= next(data3_real)
        img3_fake, ls_lab3_fake = next(data3_fake)

        # ============ one batch collection ============#
        catimg = torch.cat([img1_real, img2_real, img3_real,
                            img1_fake, img2_fake, img3_fake], 0).to(device_id)
        ls_lab = torch.cat([ls_lab1_real, ls_lab2_real, ls_lab3_real,
                            ls_lab1_fake, ls_lab2_fake, ls_lab3_fake], 0).to(device_id)

        batchidx = list(range(len(catimg)))
        random.shuffle(batchidx)

        img_rand = catimg[batchidx, :]
        ls_lab_rand = ls_lab[batchidx]
        img_rand = T_transform(img_rand) 

        pred = Fas_Net(NormalizeData_torch(img_rand)) 
        Loss_cls = criterionCls(pred.squeeze(), ls_lab_rand)

        optimizer_meta.zero_grad()
        Loss_cls.backward()
        optimizer_meta.step()

        if (step + 1) % log_step == 0:
            logging.info('[epoch %d step %d]  Loss_cls %.8f'
                  % (epoch, step, Loss_cls.item()))
        if ((step + 1) % model_save_step == 0):
            mkdir(results_path)
            save_index += 1
            torch.save(Fas_Net.state_dict(), os.path.join(results_path,
                                                          "FASNet-{}.tar".format(save_index)))

    if ((epoch + 1) % model_save_epoch == 0):
        mkdir(results_path)
        save_index += 1
        torch.save(Fas_Net.state_dict(), os.path.join(results_path,
                                                      "FASNet-{}.tar".format(save_index)))
