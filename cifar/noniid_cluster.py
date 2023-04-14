import numpy as np
import math

def cifar_noniid_quantitative_cluster(dataset, total_client, total_label=10):
    labels = dataset.targets
    total_sample = len(dataset)
    idxs = range(total_sample)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]
    tmp = 0
    list_tmp = []
    for i in range(total_sample):
        if(dataset[idxs[i]][1] == tmp):
            tmp +=1
            list_tmp.append(i)
    list_tmp.append(total_sample)

    list_dict = [0] * total_label
    for i in range(total_label):
        list_dict[i] = idxs[list_tmp[i]:list_tmp[i+1]]

    dict_client = {}
    n_samples= 30
    key = True
    count = 0
    while(key):
        count += 1
        try:
            client_dis = np.random.normal(1, 0.3, total_label)
            for i in range(total_client):
                dict_client[i] = []
                
            for i in range(int(total_client * 0.3)):
                label_client = range(0,int(total_label * 0.2))
                for j in label_client:
                    a = np.random.choice(list_dict[j],int(n_samples *client_dis[i]),replace=False)
                    list_dict[j] = list(set(list_dict[j]) - set(a))
                    dict_client[i] = dict_client[i]  + list(a)
                dict_client[i] = [int(k) for k in dict_client[i]]
                
            for i in range(int(total_client * 0.3), int(total_client * 0.6), 1):
                label_client = range(int(total_label * 0.2), int(total_label * 0.2) + int(total_label * 0.2))
                for j in label_client:
                    a = np.random.choice(list_dict[j],int(n_samples *client_dis[i]),replace=False)
                    list_dict[j] = list(set(list_dict[j]) - set(a))
                    dict_client[i] = dict_client[i]  + list(a)
                dict_client[i] = [int(k) for k in dict_client[i]]
                
            for i in range(int(total_client * 0.6), int(total_client * 0.8), 1):
                label_client = range(int(total_label * 0.4), int(total_label * 0.6))
                for j in label_client:
                    a = np.random.choice(list_dict[j],int(n_samples *client_dis[i]),replace=False)
                    list_dict[j] = list(set(list_dict[j]) - set(a))
                    dict_client[i] = dict_client[i]  + list(a)
                dict_client[i] = [int(k) for k in dict_client[i]]
            
            for i in range(int(total_client * 0.8), int(total_client * 0.9), 1):
                label_client = range(int(total_label * 0.6), int(total_label * 0.8))
                for j in label_client:
                    a = np.random.choice(list_dict[j],int(n_samples *client_dis[i]),replace=False)
                    list_dict[j] = list(set(list_dict[j]) - set(a))
                    dict_client[i] = dict_client[i]  + list(a)
                dict_client[i] = [int(k) for k in dict_client[i]]

            for i in range(int(total_client * 0.9), int(total_client * 1.0),1):
                label_client = range(int(total_label * 0.8), int(total_label * 1.0))
                for j in label_client:
                    a = np.random.choice(list_dict[j],int(n_samples *client_dis[i]),replace=False)
                    list_dict[j] = list(set(list_dict[j]) - set(a))
                    dict_client[i] = dict_client[i]  + list(a)
                dict_client[i] = [int(k) for k in dict_client[i]]
            
            key = False
            if count > 100:
                break
        except:
            if count > 100:
                break
            
    print("Count", count)
    return dict_client


def cifar100_noniid_cluster(dataset, total_client, total_label=100):
    total_label = total_label
    label_list = [i for i in range(total_label)]
    label_per_client = 2
    total_sample = len(dataset)
    
    labels = dataset.targets
    idxs = range(total_sample)
    idxs_labels = np.vstack((idxs, labels)).T
    
    dict_client = {}
    client_labels = []
    
    for _ in range(math.ceil(total_label/label_per_client)):
        this_set = np.random.choice(label_list, label_per_client, replace=False)
        client_labels.append(list(this_set))
        label_list = list(set(label_list) - set(this_set))
    num_added_total = (total_client - len(client_labels))
    num_added = np.random.default_rng().pareto(1.5, len(client_labels))
    num_added = np.round(num_added/np.sum(num_added) * num_added_total)
    num_added = np.sort(num_added)
    num_added[-1] = num_added_total - np.sum(num_added) + num_added[-1]
    
    adds = []
    print(num_added)
    print(len(client_labels), len(num_added))
    for i in range(len(client_labels)):
        for j in range(int(num_added[i])):
            adds += [client_labels[i]]
        
    client_labels += adds
    sample_per_client = int(np.floor(total_sample/total_label * 1/(num_added[-1] + 1)))
    
    for client_idx, client_label in zip(range(total_client), client_labels):
        label_1, label_2 = client_label
        
        idxes_1 = idxs_labels[idxs_labels[:,1] == label_1][:,0]
        idxes_2 = idxs_labels[idxs_labels[:,1] == label_2][:,0]

        # print(idxes_1.shape, idxes_2.shape)
        
        label_1_idxes = np.random.choice(idxes_1, sample_per_client, replace=False)
        label_2_idxes = np.random.choice(idxes_2, sample_per_client, replace=False)
        
        dict_client[client_idx] = label_1_idxes.tolist()
        dict_client[client_idx] += label_2_idxes.tolist()

        idxs_labels[label_1_idxes] -= 100
        idxs_labels[label_2_idxes] -= 100
        
    return dict_client



from torchvision import datasets
if __name__ == "__main__":
    # a = np.random.default_rng().pareto(0.75, 50)
    # a = a/np.sum(a)
    # a = np.round(a * 50)
    # a = np.sort(a)
    # a[-1] = 50 - np.sum(a) + a[-1]
    # print(np.sum(a))
    # print([5] * 0)
    train_dataset = datasets.CIFAR100( "./data/cifar/", train=True, download=False, transform=None)
    dict_return = cifar100_noniid_cluster(train_dataset, 100)