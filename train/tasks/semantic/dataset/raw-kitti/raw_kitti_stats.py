#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt


# In[2]:


import numpy as np


# In[3]:


from pathlib import Path


# In[4]:


import yaml


# In[5]:


import pandas as pd


# In[6]:


from tqdm.notebook import tqdm


# In[7]:


from sklearn import preprocessing


# In[8]:


data_config_path = Path('/home/fusionresearch/AliThesis/lidar-bonnetal/train/tasks/semantic/config/labels/raw-kitti.yaml')


# In[9]:


labels_config = yaml.safe_load(open(data_config_path, 'r'))["labels"]


# In[10]:


dataset_root_dir = Path('/home/fusionresearch/Datasets/KITTI_raw_dataset/2011_09_26/')


# In[11]:


sequences = [i.name for i in sorted(dataset_root_dir.glob("*")) if i.is_dir()]


# In[12]:


sequences


# In[13]:


get_ipython().system(' ls /home/fusionresearch/Datasets/KITTI_raw_dataset/2011_09_26/2011_09_26_drive_0001_sync/BoundingBoxLabels/')


# In[14]:


def get_label_data(label_file_path):
    label_data = np.load(str(label_file_path))
    return label_data


# In[15]:


sequences_results = {}


# In[16]:


for seq in sequences:
    seq_result = {}
    seq_labels_path = dataset_root_dir / seq / "BoundingBoxLabels"
    seq_labels_files = [i.name for i in sorted(seq_labels_path.glob("*.npy"))]
    
    for label_file in tqdm(seq_labels_files, f"getting data for seq {seq}"):
        label_file_path =  dataset_root_dir / seq / "BoundingBoxLabels" / label_file
        label_data = get_label_data(label_file_path)
        classes_in_file, occurences = np.unique(label_data, return_counts=True)
        classes_numbers = list(zip(classes_in_file, occurences))
        for c, o in classes_numbers:
            class_name = labels_config[c]
            if c in seq_result:
                seq_result[class_name] += o
            else:
                seq_result[class_name] = o
        
    print(seq_result)    
    sequences_results[seq] = seq_result


# In[17]:


statistics_dataframe = pd.DataFrame(sequences_results)


# In[18]:


statistics_dataframe = statistics_dataframe.fillna(0)


# In[19]:


statistics_dataframe


# In[20]:


# statistics_dataframe.to_csv("raw_kitti_stats.csv")


# In[21]:


statistics_dataframe["class_sum"] = statistics_dataframe.sum(axis=1)


# In[22]:


statistics_dataframe["class_sum"]


# In[23]:


statistics_dataframe.loc['drive_sum'] = statistics_dataframe.sum(axis=0)


# In[24]:


statistics_dataframe


# In[25]:


statistics_dataframe['class_ratio'] = statistics_dataframe["class_sum"] / statistics_dataframe.loc['drive_sum', "class_sum"]


# In[26]:


statistics_dataframe


# In[36]:


sorted(statistics_dataframe.columns.values)


# In[43]:


statistics_dataframe.loc[:, sorted(statistics_dataframe.columns.values)[-15:]]


# In[40]:


type(sorted(statistics_dataframe.columns.values)[-13:])


# In[29]:


8056 * 0.2


# In[30]:


8056


# In[31]:


6360 /8056


# In[27]:


statistics_dataframe['class_ratio']


# statistics_dataframe = statistics_dataframe.drop('unlabeled')

# def normlize_col(col):
#     x = col.values.astype('float').reshape(1, -1)
# 
#     # Create a minimum and maximum processor object
#     min_max_scaler = preprocessing.Normalizer(norm='l1')
# 
#     # Create an object to transform the data to fit minmax processor
#     x_scaled = min_max_scaler.fit_transform(x).reshape(-1)
# 
#     return x_scaled

# statistics_dataframe["2011_09_26_drive_0001_sync"].plot.bar()

# statistics_dataframe_scaled = statistics_dataframe.apply(normlize_col, axis=0)

# statistics_dataframe_scaled

# statistics_dataframe_scaled["2011_09_26_drive_0001_sync"].plot.bar()
