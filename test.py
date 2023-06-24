import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
import torch 
from models.DCViT import * 
import logging
from pytz import timezone
from datetime import datetime
import torchvision.transforms as T
import sys
import torch.nn.functional as F

logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('Asia/Taipei')).timetuple()

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    # y = TPR - FPR
    y = TPR + (1 - FPR)
    # print(y)
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def NormalizeData_torch(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

# replay casia Oulu MSU
device_id = 'cuda:0'
batch_size = 20 
test_dataset = 'replay'
result_filename = 'OMC(I)'
result_path = "E:/ICIP/" + result_filename

file_handler = logging.FileHandler(filename='C:/Users/Jxchong/Desktop/ICIp/logger/'+ result_filename +'_test.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler] 
date = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=date, handlers=handlers)

T_transform = torch.nn.Sequential(
    T.CenterCrop(224),
) 

# replay casia Oulu MSU
live_path = 'C:/Users/Jxchong/Desktop/LDCNet-master/dataset/' + test_dataset + '_images_live.npy'
spoof_path = 'C:/Users/Jxchong/Desktop/LDCNet-master/dataset/' + test_dataset + '_images_spoof.npy'

live_data = np.load(live_path)
spoof_data = np.load(spoof_path)
live_label = np.ones(len(live_data), dtype=np.int64)
spoof_label = np.zeros(len(spoof_data), dtype=np.int64)

total_data = np.concatenate((live_data, spoof_data), axis=0)
total_label = np.concatenate((live_label, spoof_label), axis=0)

trainset = torch.utils.data.TensorDataset(torch.tensor(np.transpose(total_data, (0, 3, 1, 2))),
                                          torch.tensor(total_label))

data_loader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=False)

Fas_Net = vit_base_patch16_224(pretrained=True).to(device_id)


record = [1,100,100,100,100]
for epoch in range(1,61):

    Net_path = result_path + "/FASNet-" + str(epoch) + ".tar"
    Fas_Net.load_state_dict(torch.load(Net_path, map_location=device_id),strict=False) 
    Fas_Net.eval()

    score_list = [] 
    label_list = []
    TP = 0.0000001
    TN = 0.0000001
    FP = 0.0000001
    FN = 0.0000001
    for i, data in enumerate(data_loader, 0):
        images, labels = data
        images = images.to(device_id)
        images = T_transform(images)
        label_pred = Fas_Net(NormalizeData_torch(images))
        score = F.softmax(label_pred, dim=1).cpu().data.numpy()[:, 1]  # multi class

        for j in range(images.size(0)):
            score_list.append(score[j])
            label_list.append(labels[j])

    score_list = NormalizeData(score_list)

    fpr, tpr, thresholds = metrics.roc_curve(label_list, score_list)
    threshold, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)


    for i in range(len(score_list)):
        score = score_list[i]
        if (score >= threshold and label_list[i] == 1):
            TP += 1
        elif (score < threshold and label_list[i] == 0):
            TN += 1
        elif (score >= threshold and label_list[i] == 0):
            FP += 1
        elif (score < threshold and label_list[i] == 1):
            FN += 1

    APCER = FP / (TN + FP)
    NPCER = FN / (FN + TP)

    acer = '{:.5f}'.format(np.round((APCER + NPCER) / 2, 4))
    auc = '{:.5f}'.format(roc_auc_score(label_list, score_list))

    logging.info(f"Epoch {epoch} ACER {acer} AUC {auc} Test on {test_dataset}")
# conda rename -n hw4 -d Transformer
 
