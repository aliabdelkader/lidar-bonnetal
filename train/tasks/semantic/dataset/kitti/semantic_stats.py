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


# In[33]:


from tqdm.notebook import tqdm


# In[67]:


from sklearn import preprocessing


# In[6]:


data_config_path = Path('/home/fusionresearch/AliThesis/lidar-bonnetal/train/tasks/semantic/config/labels/semantic-kitti.yaml')


# In[18]:


labels_config = yaml.safe_load(open(data_config_path, 'r'))["labels"]


# In[7]:


dataset_root_dir = Path('/home/fusionresearch/SemanticKitti/dataset/sequences/')


# In[8]:


sequences = [i.name for i in sorted(dataset_root_dir.glob("*")) if i.is_dir()]


# In[9]:


sequences


# In[13]:


def get_label_data(label_file_path):
    label_data = np.fromfile(str(label_file_path), dtype=np.int32)
    label_data = label_data.reshape((-1))
    sem_label = label_data & 0xFFFF  
    return sem_label


# In[14]:


sequences_results = {}


# In[35]:


for seq in sequences:
    seq_result = {}
    seq_labels_path = dataset_root_dir / seq / "labels"
    seq_labels_files = [i.name for i in sorted(seq_labels_path.glob("*.label"))]
    
    for label_file in tqdm(seq_labels_files, f"getting data for seq {seq}"):
        label_file_path =  dataset_root_dir / seq / "labels" / label_file
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


# In[112]:


statistics_dataframe = pd.DataFrame(sequences_results)


# In[113]:


statistics_dataframe = statistics_dataframe.fillna(0)


# In[114]:


statistics_dataframe


# In[115]:


statistics_dataframe = statistics_dataframe.drop(statistics_dataframe.columns[-11:],axis=1)


# In[116]:


statistics_dataframe


# In[117]:


def normlize_col(col):
    x = col.values.astype('float').reshape(1, -1)
    print(x)
    # Create a minimum and maximum processor object
    min_max_scaler = preprocessing.Normalizer(norm='l1')

    # Create an object to transform the data to fit minmax processor
    x_scaled = min_max_scaler.fit_transform(x).reshape(-1)
    print(x_scaled)
    return x_scaled


# In[118]:


# statistics_dataframe.to_csv("dataset_stats.csv")


# In[119]:


statistics_dataframe


# In[120]:


statistics_dataframe["10"].plot.bar()


# In[121]:


statistics_dataframe_scaled = statistics_dataframe.apply(normlize_col, axis=0)


# In[122]:


statistics_dataframe_scaled


# In[123]:


statistics_dataframe_scaled["09"].plot.bar()


# In[124]:


statistics_dataframe_scaled["10"].plot.bar()

