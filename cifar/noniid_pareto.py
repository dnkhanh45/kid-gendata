import os
from pathlib import Path
import numpy as np
import math
from torchvision import datasets, transforms
import json

def pareto_noniid(dataset, total_client, total_sample=50000, total_label=10):
    labels = dataset.targets
    idxs = range(len(dataset))
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]
    tmp = 0
    list_tmp = []

    for i in range(len(dataset)):
        if(dataset[idxs[i]][1] == tmp):
            tmp +=1
            list_tmp.append(i)
    list_tmp.append(len(dataset))
      
    while(True):
        list_label ={}
        a = set()
        for i in range(total_client):
            list_label[i] = np.random.randint(0,total_label,3) 
            a.update(list_label[i])
        print(len(a))
        if len(a) >= total_label:
            break
    
    key  = True
    count = 0
    while(key):
        count += 1
        if count > 100:
            print("Infinite loop")
            exit(0)
            
        try:
            list_dict = [0] * total_label
            for i in range(total_label):
                list_dict[i] = idxs[list_tmp[i]:list_tmp[i+1]]

            dis = np.random.pareto(tuple([1] * total_client))
            dis = dis/np.sum(dis)
            percent = [0] * 100
            for i in range(total_client):
                for j in list_label[i]:
                    percent[j] += dis[i]

            maxx = max(percent)
            total = np.around(10000/maxx)
            sample_client = [math.ceil(total * dis[i]) for i in range(total_client)]
            for i in  range(len(sample_client)):
                if sample_client[i] == 0:
                    sample_client[i] = 1
                
            dict_client = {}
            for i in range(total_client):
                dict_client[i] = []
            for i in range(total_client):

                x = math.ceil(sample_client[i]/2)
                for j in list_label[i]:
                    a = np.random.choice(list_dict[j],x,replace=False)
                    list_dict[list_label[i][0]] = list(set(list_dict[list_label[i][0]]) - set(a))
                    dict_client[i] = dict_client[i]  + list(a)

                dict_client[i] = [int(j) for j in dict_client[i]]
            key = False
        except:
            key = True
        
    return dict_client


def cifar100_noniid_pareto(dataset, total_client, total_label=100):
    total_label = total_label
    labels = dataset.targets
    idxs = range(len(dataset))
    
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]
    tmp = 0
    list_tmp = []

    for i in range(len(dataset)):
        if(dataset[idxs[i]][1] == tmp):
            tmp +=1
            list_tmp.append(i)
    list_tmp.append(len(dataset))
    
    while(True):
        list_label ={}
        a = set()
        for i in range(total_client):
            list_label[i] = np.random.randint(0,total_label,3) 
            a.update(list_label[i])
        if len(a) >= total_label:
            break
    
    key  = True
    count = 0
    while(key):
        count += 1
        if count > 200:
            print("Infinite loop")
            return None
            
        try:
            list_dict = [0] * total_label
            for i in range(total_label):
                list_dict[i] = idxs[list_tmp[i]:list_tmp[i+1]]

            dis = np.random.pareto(tuple([1] * total_client))
            dis = dis/np.sum(dis)
            percent = [0] * total_label
            for i in range(total_client):
                for j in list_label[i]:
                    percent[j] += dis[i]/3
            print(dis)
            print(percent)
            print(sum(percent))
            maxx = max(percent)
            # total = np.around(100/maxx)
            total = np.around(1000/maxx)
            # total = 49900
            sample_client = [math.ceil(total * dis[i]) for i in range(total_client)]
            print(sample_client)
            for i in range(len(sample_client)):
                if sample_client[i] == 0:
                    sample_client[i] = 1
                
            dict_client = {}
            for i in range(total_client):
                dict_client[i] = []
            for i in range(total_client):

                x = math.ceil(sample_client[i]/3)
                for j in list_label[i]:
                    a = np.random.choice(list_dict[j],x,replace=False)
                    list_dict[list_label[i][0]] = list(set(list_dict[list_label[i][0]]) - set(a))
                    dict_client[i] = dict_client[i]  + list(a)

                dict_client[i] = [int(j) for j in dict_client[i]]
            key = False
        except:
            key = True
        
    return dict_client

import pandas as pd 
def sta(client_dict, train_dataset, num_client=20, num_label=10):
    rs = []
    for client in range(num_client):
        tmp = []
        for i in range(num_label):
            tmp.append(sum(train_dataset[j][1] == i for j in client_dict[client]))
        rs.append(tmp)
    df = pd.DataFrame(rs,columns=[f"Label_{i}" for i in range(num_label)])
    return df

if __name__ == "__main__":
    train_dataset = datasets.CIFAR10(
        root="../data/cifar10/", 
        # split="letters",
        train=True, 
        download=True
    )
    
    dict_client = None
    while dict_client is None:
        dict_client = pareto_noniid(train_dataset, total_client=20, total_label=10)
    df = sta(dict_client,train_dataset,20,10)
    store_path = "cifar10/pareto/"
    if not Path(store_path).exists():
        os.makedirs(store_path)
    with open(f"{store_path}/CIFAR10_pareto_10clients_10class.json","w") as f:
        json.dump(dict_client, f)
    df.to_csv(f"{store_path}/CIFAR10_pareto_10clients_10class_stat.csv")