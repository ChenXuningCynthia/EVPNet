import os
import scipy.io as scio
import numpy as np
import logging
import time
from termcolor import colored
import os.path as op
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE
import random
import torch
from sklearn.neighbors import NearestNeighbors

dataset_path = {'deap':'','hci':''}

'''
For loading data
'''

def load_data(dataset_name):
    if dataset_name in ['deap']:
        path = dataset_path[dataset_name]
        # data=np.load('processed_data_60s.npy')
        data= []
        data_bio = []
        label=[]
        for i in range(1,33):
            matdir=op.join(path,str(i)+'_de.mat')
            biodir=op.join(path,str(i)+'_bio.mat')
            de= scio.loadmat(matdir)['de']
            bio = scio.loadmat(biodir)['bio']
            if np.any(np.isnan(de)):
                print("NaN found in data, skipping file:", matdir)
                continue
            if np.any(np.isnan(bio)):
                print("NaN found in data, skipping file:", biodir)
                continue
            de_z = zscore(de, axis=0)
            bio_z = zscore(bio, axis=0)
            data.append(de_z)
            data_bio.append(bio_z)
            label_temp=np.round(scio.loadmat(matdir)['label'][:,0],1)
            for v in range(label_temp.shape[0]):
                if label_temp[v]>=5:
                    label_temp[v] = 1
                else:
                    label_temp[v]=0
            label.append(label_temp)
        return data,data_bio, label
    elif dataset_name in ['hci']:
        path = dataset_path[dataset_name]
        # data=np.load('processed_data_60s.npy')
        data= []
        data_bio = []
        label=[]
        notfound = [3, 9, 12, 15, 16, 26]
        for i in range(1,31):
            if i in notfound:
                continue
            matdir=op.join(path,str(i)+'_eeg_bio.mat')
            de= scio.loadmat(matdir)['eeg']
            bio = scio.loadmat(matdir)['bio']
            if np.any(np.isnan(de)):
                print("NaN found in data, skipping file:", matdir)
                continue
            if np.any(np.isnan(bio)):
                print("NaN found in data, skipping file:", matdir,'bio')
                continue
            de_z = zscore(de, axis=0)
            bio_z = zscore(bio, axis=0)
            data.append(de_z)
            data_bio.append(bio_z)
            label_temp=np.round(scio.loadmat(matdir)['a'],1)
            label_temp=label_temp.squeeze()
            for v in range(label_temp.shape[0]):
                if label_temp[v]>=5:
                    label_temp[v] = 1
                else:
                    label_temp[v]=0
            label.append(label_temp)
        return data,data_bio, label

def create_logger(args):
    os.makedirs(args.output_log_dir, exist_ok=True)
    time_str = time.strftime('%m-%d-%H-%M')
    log_file = 'CrossSubject_'+ args.dataset + '_lr_' + str(args.lr) + '_seed_' + str(args.seed) + '_{}.log'.format(time_str)
    final_log_file = os.path.join(args.output_log_dir, log_file)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = '[%(asctime)s] %(message)s'
    color_fmt = colored('[%(asctime)s]', 'green') + ' %(message)s'

    file = logging.FileHandler(filename=final_log_file, mode='a')
    file.setLevel(logging.INFO)
    file.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console)

    return logger

def apply_layered_smote(dataset, primary_modal='eeg', random_state=20):
    """
    SMOTE
    """
    eeg_data= torch.tensor(dataset['feature'], dtype=torch.float32)
    bio_data= torch.tensor(dataset['bio'], dtype=torch.float32)
    video_data= torch.tensor(dataset['video'], dtype=torch.float32)
    labels = torch.tensor(dataset['label'], dtype=torch.int64)

    # choose main modality
    if primary_modal == 'eeg':
        primary_data = eeg_data
        secondary_data = [video_data, bio_data]
    elif primary_modal == 'video':
        primary_data = video_data
        secondary_data = [eeg_data, bio_data]
    else:  # bio
        primary_data = bio_data
        secondary_data = [eeg_data, video_data]

    # SMOTE
    smote = SMOTE(random_state=random_state)
    primary_resampled, labels_resampled = smote.fit_resample(primary_data, labels)

    # get new index
    new_samples_mask = np.arange(len(primary_resampled)) >= len(primary_data)
    new_primary_samples = primary_resampled[new_samples_mask]
    new_labels = labels_resampled[new_samples_mask]

    # find kmeans
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(primary_data)
    _, indices = knn.kneighbors(new_primary_samples)
    neighbor_indices = indices.flatten()

    new_secondary_samples = []
    for secondary in secondary_data:
        neighbor_secondary = secondary[neighbor_indices]
        neighbor_secondary = torch.tensor(neighbor_secondary, dtype=torch.float32)
        noise = torch.randn_like(neighbor_secondary) * 0.05
        new_secondary = neighbor_secondary + noise
        new_secondary_samples.append(new_secondary)

    new_primary = torch.tensor(
        new_primary_samples.reshape(-1, *primary_data.shape[1:]),
        dtype=primary_data.dtype
    )

    if primary_modal == 'eeg':
        balanced_eeg = torch.cat([eeg_data, new_primary], dim=0)
        balanced_video = torch.cat([video_data, new_secondary_samples[0]], dim=0)
        balanced_bio = torch.cat([bio_data, new_secondary_samples[1]], dim=0)
    elif primary_modal == 'video':
        balanced_eeg = torch.cat([eeg_data, new_secondary_samples[0]], dim=0)
        balanced_video = torch.cat([video_data, new_primary], dim=0)
        balanced_bio = torch.cat([bio_data, new_secondary_samples[1]], dim=0)
    else:  # bio
        balanced_eeg = torch.cat([eeg_data, new_secondary_samples[0]], dim=0)
        balanced_video = torch.cat([video_data, new_secondary_samples[1]], dim=0)
        balanced_bio = torch.cat([bio_data, new_primary], dim=0)

    balanced_labels = torch.cat([labels, torch.tensor(new_labels, dtype=labels.dtype)], dim=0)
    souece_sample_num = balanced_eeg.shape[0]
    balanced_dataset = torch.utils.data.TensorDataset(balanced_eeg, torch.arange(souece_sample_num), balanced_labels,balanced_video, balanced_bio)

    return balanced_dataset,souece_sample_num

def set_seed(seed=20):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(2)


