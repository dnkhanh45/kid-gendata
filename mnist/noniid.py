from xml.dom.expatbuilder import parseString
import numpy as np
import torch


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    print(num_users)
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # idx label da sort
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        print(i)
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]), axis=0
            )

    for i in range(num_users):
        dict_users[i] = [int(j) for j in dict_users[i]]
    return dict_users


import math


import math
def poreto_noniid(dataset, num_users):
    labels = dataset.targets
    idxs = range(60000)
    # pair = np.vstack((range(60000),np.array(list_label)))
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]
    tmp = 0
    list_tmp = []

    for i in range(60000):
        if(dataset[idxs[i]][1] == tmp):
            tmp +=1
            list_tmp.append(i)
    list_tmp.append(60000)
    

    #list label cho tung client
    while(True):  
        list_label ={}
        a = set()
        for i in range(num_users):
            list_label[i] = np.random.randint(0,10,3) 
            a.update(list_label[i])
        print(len(a))
        if len(a) == 10:
            break
    
    key  = True
    while(key):
        try:
            list_dict = [0] * 10
            for i in range(10):
                list_dict[i] = idxs[list_tmp[i]:list_tmp[i+1]]
            # print(list_label)
            #ty le du lieu tung client
            dis = np.random.pareto(tuple([1] * num_users))
            dis = dis/np.sum(dis)
            percent = [0] * 100
            for i in range(num_users):
                for j in list_label[i]:
                    percent[j] += dis[i]
            # print(percent)
            maxx = max(percent)
            # print(maxx)
            total = np.around(10000/maxx)
            print(total)
            sample_client = [math.ceil(total * dis[i]) for i in range(num_users)]
            # sample_client = [1 for i in sample_client if i == 0]
            for i in  range(len(sample_client)):
                if sample_client[i] == 0:
                    sample_client[i] = 1
                
            # print(sample_client)
            dict_client = {}
            for i in range(num_users):
                dict_client[i] = []
            for i in range(num_users):
                # breakpoint()
                x = math.ceil(sample_client[i]/2)
                for j in list_label[i]:
                    # breakpoint()
                    a = np.random.choice(list_dict[j],x,replace=False)
                    list_dict[list_label[i][0]] = list(set(list_dict[list_label[i][0]]) - set(a))
                    dict_client[i] = dict_client[i]  + list(a)
                    # else :
                        
                    #     dict_client[i] = dict_client[i] +  list(a)
                    # dict_client[i]
                dict_client[i] = [int(j) for j in dict_client[i]]
            key = False
        except:
            key = True
        
    return dict_client


def mnist_noniid_featured(dataset, num_users, ratio):
    group_mem = math.ceil(num_users * ratio)
    print(f"Group_men : {group_mem}")
    labels = dataset.targets
    idxs = range(60000)
    # pair = np.vstack((range(60000),np.array(list_label)))
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]
    tmp = 0
    list_tmp = []
    for i in range(60000):
        if dataset[idxs[i]][1] == tmp:
            tmp += 1
            list_tmp.append(i)
    list_tmp.append(60000)

    list_dict = [0] * 10
    for i in range(10):
        list_dict[i] = idxs[list_tmp[i] : list_tmp[i + 1]]
    # print(list_dict)
    dict_client = {}
    n_samples = int(5800 / group_mem)
    for i in range(num_users):
        dict_client[i] = []
    arr = [1, 4, 3, 2, 0]
    # client_dis = [1.4,1,0.4,1.2,0.7,1.3,1.5,0.6,2,2.5]
    client_dis = [1] * 100
    # client_dis = np.random.normal(1, 0.1, 100)
    # client_dis[:group_mem] = client_dis[:group_mem] / sum(client_dis[:group_mem]) * group_mem
    
    key = True
    while(key):
        try:
            for i in range(group_mem):
                label_client = [8,9]
                # label_client = range(arr[0]*2,arr[0]*2+2)
                print(label_client)
                for j in label_client:
                    a = np.random.choice(
                        list_dict[j], int(n_samples * client_dis[i]), replace=False
                    )
                    list_dict[j] = list(set(list_dict[j]) - set(a))
                    dict_client[i] = dict_client[i] + list(a)
                dict_client[i] = [int(k) for k in dict_client[i]]

            for i in range(group_mem, num_users, 1):
                print(label_client)
                label_client = np.random.choice(range(8), 2, replace=False)
                for j in label_client:
                    print(n_samples * client_dis[i])
                    a = np.random.choice(
                        list_dict[j], int(n_samples * client_dis[i]), replace=False
                    )
                    list_dict[j] = list(set(list_dict[j]) - set(a))
                    dict_client[i] = dict_client[i] + list(a)
                dict_client[i] = [int(k) for k in dict_client[i]]
            key = False
        except:
            key = True
    # print(dict_client)
    return dict_client
# def mnist_noniid_featured(dataset, num_users, ratio):
#     group_mem = math.ceil(num_users * ratio)
#     print(f"Group_men : {group_mem}")
#     labels = dataset.targets
#     idxs = range(60000)
#     # pair = np.vstack((range(60000),np.array(list_label)))
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]
#     labels = idxs_labels[1, :]
#     tmp = 0
#     list_tmp = []
#     for i in range(60000):
#         if dataset[idxs[i]][1] == tmp:
#             tmp += 1
#             list_tmp.append(i)
#     list_tmp.append(60000)

#     list_dict = [0] * 10
#     for i in range(10):
#         list_dict[i] = idxs[list_tmp[i] : list_tmp[i + 1]]
#     # print(list_dict)
#     dict_client = {}
#     n_samples = int(6000 / group_mem)
#     for i in range(num_users):
#         dict_client[i] = []
#     arr = [1, 4, 3, 2, 0]
#     # client_dis = [1.4,1,0.4,1.2,0.7,1.3,1.5,0.6,2,2.5]
#     client_dis = [1]* 100

#     key = True
#     while key:
#         try:
#             # client_dis = np.random.normal(1, 0.1, 100)
#             # client_dis[:group_mem] = (
#             #     client_dis[:group_mem] / sum(client_dis[:group_mem]) * group_mem
#             # )
#             # print(client_dis[:group_mem].sum())
#             for i in range(group_mem):
#                 label_client = [0, 1]
#                 # label_client = range(arr[0]*2,arr[0]*2+2)
#                 print(label_client)
#                 for j in label_client:
#                     a = np.random.choice(
#                         list_dict[j], int(n_samples * client_dis[i]), replace=False
#                     )
#                     list_dict[j] = list(set(list_dict[j]) - set(a))
#                     dict_client[i] = dict_client[i] + list(a)
#                 dict_client[i] = [int(k) for k in dict_client[i]]

#             while True:
#                 list_label = {}
#                 a = set()
#                 for i in range(group_mem, num_users, 1):
#                     list_label[i] = np.random.randint(2, 10, 2)
#                     a.update(list_label[i])
#                 print(len(a))
#                 if len(a) == 8:
#                     break
#             for i in range(group_mem, num_users, 1):
#                 # print(label_client)
#                 # label_client = np.random.choice(range(2,10),2,replace=False)
#                 for j in list_label[i]:
#                     print(n_samples * client_dis[i])
#                     a = np.random.choice(
#                         list_dict[j], int(n_samples * client_dis[i]), replace=False
#                     )
#                     list_dict[j] = list(set(list_dict[j]) - set(a))
#                     dict_client[i] = dict_client[i] + list(a)
#                 dict_client[i] = [int(k) for k in dict_client[i]]
#             key = False
#         except:
#             key = True
#         # print(dict_client)
#     return dict_client

def mnist_noniid_quantitative(dataset, num_users):
    labels = dataset.targets
    idxs = range(60000)
    # pair = np.vstack((range(60000),np.array(list_label)))
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]
    tmp = 0
    list_tmp = []
    for i in range(60000):
        if dataset[idxs[i]][1] == tmp:
            tmp += 1
            list_tmp.append(i)
    list_tmp.append(60000)

    list_dict = [0] * 10
    for i in range(10):
        list_dict[i] = idxs[list_tmp[i] : list_tmp[i + 1]]
    # print(list_dict)
    dict_client = {}
    n_samples = 10
    for i in range(num_users):
        dict_client[i] = []
    print("WHILE")
    try:
        key = True
        while(key):
            # client_dis = np.random.normal(1,0.3,100)
            client_dis = [1]* 150
            print(client_dis)
            for i in range(45):
                label_client = range(0,2)
                for j in label_client:
                    a = np.random.choice(
                        list_dict[j], int(n_samples * client_dis[i]), replace=False
                    )
                    list_dict[j] = list(set(list_dict[j]) - set(a))
                    dict_client[i] = dict_client[i] + list(a)
                dict_client[i] = [int(k) for k in dict_client[i]]
                print(dict_client)
            for i in range(45,90,1):
                label_client = range(2,4)
                for j in label_client:
                    a = np.random.choice(
                        list_dict[j], int(n_samples * client_dis[i]), replace=False
                    )
                    list_dict[j] = list(set(list_dict[j]) - set(a))
                    dict_client[i] = dict_client[i] + list(a)
                dict_client[i] = [int(k) for k in dict_client[i]]
            for i in range(90,120):
                label_client = range(4,6)
                for j in label_client:
                    a = np.random.choice(
                        list_dict[j], int(n_samples * client_dis[i]), replace=False
                    )
                    list_dict[j] = list(set(list_dict[j]) - set(a))
                    dict_client[i] = dict_client[i] + list(a)
                dict_client[i] = [int(k) for k in dict_client[i]]
            for i in range(120,135):
                label_client = range(6,8)
                for j in label_client:
                    a = np.random.choice(
                        list_dict[j], int(n_samples * client_dis[i]), replace=False
                    )
                    list_dict[j] = list(set(list_dict[j]) - set(a))
                    dict_client[i] = dict_client[i] + list(a)
                dict_client[i] = [int(k) for k in dict_client[i]]
            for i in range(135,150):
                label_client = range(8,10)
                for j in label_client:
                    a = np.random.choice(
                        list_dict[j], int(n_samples * client_dis[i]), replace=False
                    )
                    list_dict[j] = list(set(list_dict[j]) - set(a))
                    dict_client[i] = dict_client[i] + list(a)
                dict_client[i] = [int(k) for k in dict_client[i]]
            key = False
    except:
        key= True 
    # print(dict_client)
    return dict_client


from torchvision import datasets, transforms
import json


def save_dataset_idx(list_idx_sample, path="dataset_idx.json"):
    with open(path, "w+") as outfile:
        json.dump(list_idx_sample, outfile)


import pandas as pd


def gen_full(dataset, num_users):
    client_dict = {}
    for i in range(1):
        if i == 0:
            client_dict[i] = range(60000)
        else:
            client_dict[i] = []
        client_dict[i] = [int(k) for k in client_dict[i]]
    return client_dict


def sta(client_dict,train_dataset):
    rs = []
    for client in range(50):
        tmp = []
        for i in range(10):
            tmp.append(sum(train_dataset[j][1] == i for j in client_dict[client]))
        rs.append(tmp)
    df = pd.DataFrame(rs,columns=[f"Label_{i}" for i in range(10)])
    return df


if __name__ == "__main__":
    # poreto_noniid(0,10)
    apply_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        "./data/mnist/", train=True, download=True, transform=apply_transform
    )
    # for i in train_dataset:
    #     print(i[0].shape)
    client_dict = poreto_noniid(train_dataset, 50)
    save_dataset_idx(client_dict, f"mnist/mnist_pareto_50clients.json")
    print(">>>>>>STA")
    df = sta(client_dict, train_dataset)
    df.to_csv(f"mnist/mnist_pareto_50clients.csv")
