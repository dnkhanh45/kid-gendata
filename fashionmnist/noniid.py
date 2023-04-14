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
        if dataset[idxs[i]][1] == tmp:
            tmp += 1
            list_tmp.append(i)
    list_tmp.append(60000)

    # list label cho tung client
    while True:
        list_label = {}
        a = set()
        for i in range(num_users):
            list_label[i] = np.random.randint(0, 10, 2)
            a.update(list_label[i])
        print(len(a))
        if len(a) == 10:
            break

    key = True
    while key:
        try:
            list_dict = [0] * 10
            for i in range(10):
                list_dict[i] = idxs[list_tmp[i] : list_tmp[i + 1]]
            # print(list_label)
            # ty le du lieu tung client
            dis = np.random.pareto(tuple([1] * num_users))
            dis = dis / np.sum(dis)
            percent = [0] * 100
            for i in range(num_users):
                for j in list_label[i]:
                    percent[j] += dis[i]
            # print(percent)
            maxx = max(percent)
            # print(maxx)
            total = np.around(15000 / maxx)
            print(total)
            sample_client = [math.ceil(total * dis[i]) for i in range(num_users)]
            # sample_client = [1 for i in sample_client if i == 0]
            for i in range(len(sample_client)):
                if sample_client[i] == 0:
                    sample_client[i] = 1

            # print(sample_client)
            dict_client = {}
            for i in range(num_users):
                dict_client[i] = []
            for i in range(num_users):
                # breakpoint()
                x = math.ceil(sample_client[i] / 2)
                for j in list_label[i]:
                    # breakpoint()
                    a = np.random.choice(list_dict[j], x, replace=False)
                    list_dict[list_label[i][0]] = list(
                        set(list_dict[list_label[i][0]]) - set(a)
                    )
                    dict_client[i] = dict_client[i] + list(a)
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
    n_samples = int(6000/ group_mem)
    for i in range(num_users):
        dict_client[i] = []
    # client_dis = [1.4,1,0.4,1.2,0.7,1.3,1.5,0.6,2,2.5]
    client_dis = [1]* 100
    print("check_")
    key = True
    while key:
        # print("check_")
        try:
            # client_dis = np.random.normal(1, 0.1, 100)
            # client_dis[:group_mem] = (
            #     client_dis[:group_mem] / sum(client_dis[:group_mem]) * group_mem
            # )
            # print(client_dis[:group_mem].sum())
            for i in range(group_mem):
                label_client = [0, 1]
                # label_client = range(arr[0]*2,arr[0]*2+2)
                print(label_client)
                for j in label_client:
                    a = np.random.choice(
                        list_dict[j], int(n_samples * client_dis[i]), replace=False
                    )
                    list_dict[j] = list(set(list_dict[j]) - set(a))
                    dict_client[i] = dict_client[i] + list(a)
                dict_client[i] = [int(k) for k in dict_client[i]]

            while True:
                print("check")
                list_label = {}
                a = set()
                for i in range(group_mem, num_users, 1):
                    list_label[i] = np.random.randint(2, 10, 2)
                    a.update(list_label[i])
                print(len(a))
                if len(a) == 8:
                    break
            for i in range(group_mem, num_users, 1):
                # print(label_client)
                # label_client = np.random.choice(range(2,10),2,replace=False)
                for j in list_label[i]:
                    print(n_samples * client_dis[i])
                    a = np.random.choice(
                        list_dict[j], int(n_samples * client_dis[i]/2), replace=False
                    )
                    list_dict[j] = list(set(list_dict[j]) - set(a))
                    dict_client[i] = dict_client[i] + list(a)
                dict_client[i] = [int(k) for k in dict_client[i]]
            key = False
        except:
            key = True
        # print(dict_client)
    return dict_client


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
    n_samples = 980
    for i in range(num_users):
        dict_client[i] = []
    arr = [0, 3, 2, 1, 4]
    # client_dis = [1.3,0.7,.4,1.2,1,1.4,1.7,0.8,3,2.5]
    for i in range(6):
        label_client = range(arr[0] * 2, arr[0] * 2 + 2)
        for j in label_client:
            a = np.random.choice(
                list_dict[j], int(n_samples * client_dis[i]), replace=False
            )
            list_dict[j] = list(set(list_dict[j]) - set(a))
            dict_client[i] = dict_client[i] + list(a)
        dict_client[i] = [int(k) for k in dict_client[i]]

    for i in range(6, 10, 1):
        label_client = range((arr[i - 5]) * 2, 2 + (arr[i - 5]) * 2)
        for j in label_client:
            # breakpoint
            print(n_samples * client_dis[i])
            a = np.random.choice(
                list_dict[j], int(n_samples * client_dis[i]), replace=False
            )
            list_dict[j] = list(set(list_dict[j]) - set(a))
            dict_client[i] = dict_client[i] + list(a)
        dict_client[i] = [int(k) for k in dict_client[i]]
    # print(dict_client)
    return dict_client


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 600, 100
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 4
    max_shard = 8

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard + 1, size=num_users)
    random_shard_size = np.around(
        random_shard_size / sum(random_shard_size) * num_shards
    )
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )

        random_shard_size = random_shard_size - 1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )

    for i in range(num_users):
        dict_users[i] = [int(j) for j in dict_users[i]]

    return dict_users


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


# def mnist_noniid_unequal(dataset, num_users):
#     """
#     Sample non-I.I.D client data from MNIST dataset s.t clients
#     have unequal amount of data
#     :param dataset:
#     :param num_users:
#     :returns a dict of clients with each clients assigned certain
#     number of training imgs
#     """
#     # 60,000 training imgs --> 50 imgs/shard X 1200 shards
#     num_shards, num_imgs = 50, 1200
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([]) for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.targets.numpy()

#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]

#     # Minimum and maximum shards assigned per client:
#     min_shard = 3
#     max_shard = 7

#     # Divide the shards into random chunks for every client
#     # s.t the sum of these chunks = num_shards
#     random_shard_size = np.random.randint(min_shard, max_shard+1,
#                                           size=num_users)
#     random_shard_size = np.around(random_shard_size /
#                                   sum(random_shard_size) * num_shards)
#     random_shard_size = random_shard_size.astype(int)

#     # Assign the shards randomly to each client
#     if sum(random_shard_size) > num_shards:

#         for i in range(num_users):
#             # First assign each client 1 shard to ensure every client has
#             # atleast one shard of data
#             rand_set = set(np.random.choice(idx_shard, 1, replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[i] = np.concatenate(
#                     (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
#                     axis=0)

#         random_shard_size = random_shard_size-1

#         # Next, randomly assign the remaining shards
#         for i in range(num_users):
#             if len(idx_shard) == 0:
#                 continue
#             shard_size = random_shard_size[i]
#             if shard_size > len(idx_shard):
#                 shard_size = len(idx_shard)
#             rand_set = set(np.random.choice(idx_shard, shard_size,
#                                             replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[i] = np.concatenate(
#                     (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
#                     axis=0)
#     else:

#         for i in range(num_users):
#             shard_size = random_shard_size[i]
#             rand_set = set(np.random.choice(idx_shard, shard_size,
#                                             replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[i] = np.concatenate(
#                     (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
#                     axis=0)

#         if len(idx_shard) > 0:
#             # Add the leftover shards to the client with minimum images:
#             shard_size = len(idx_shard)
#             # Add the remaining shard to the client with lowest data
#             k = min(dict_users, key=lambda x: len(dict_users.get(x)))
#             rand_set = set(np.random.choice(idx_shard, shard_size,
#                                             replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[k] = np.concatenate(
#                     (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
#                     axis=0)

#     for i in range(num_users):
#         dict_users[i] = [int(j) for j in dict_users[i]]

#     return dict_users
def sta(client_dict, train_dataset, num_users):
    print("<<<< STA")
    rs = []
    for client in range(num_users):
        tmp = []
        for i in range(10):
            tmp.append(sum(train_dataset[j][1] == i for j in client_dict[client]))
        rs.append(tmp)
    df = pd.DataFrame(rs, columns=[f"Label_{i}" for i in range(10)])
    return df


def fashionmnist_noniid_featured(dataset, num_users):
    # group_mem = math.ceil(num_users*ratio)
    # print(f"Group_men : {group_mem}")
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
    # n_samples= int(5800/group_mem)
    n_samples = 1000
    for i in range(num_users):
        dict_client[i] = []
    arr = [3, 1, 4, 0, 2]
    # client_dis = [1.4,1,0.4,1.2,0.7,1.3,1.5,0.6,2,2.5]
    client_dis = [1] * 100
    # client_dis = np.random.normal(1,0.1,10)
    # client_dis[:6] = client_dis[:6]/sum(client_dis[:6]) * 6
    # print(client_dis[:40].sum())
    for i in range():
        # label_client = arr
        label_client = range(arr[0] * 2, arr[0] * 2 + 2)
        print(label_client)
        for j in label_client:
            a = np.random.choice(
                list_dict[j], int(n_samples * client_dis[i]), replace=False
            )
            list_dict[j] = list(set(list_dict[j]) - set(a))
            dict_client[i] = dict_client[i] + list(a)
        dict_client[i] = [int(k) for k in dict_client[i]]

    for i in range(6, 10, 1):
        # print(label_client)
        label_client = range(arr[i - 5] * 2, arr[i - 5] * 2 + 2)
        # label_client = np.random.choice(range(2,10),2,replace=False)
        for j in label_client:
            # print(n_samples*client_dis[i])
            a = np.random.choice(
                list_dict[j], int(n_samples * client_dis[i]), replace=False
            )
            list_dict[j] = list(set(list_dict[j]) - set(a))
            dict_client[i] = dict_client[i] + list(a)
        dict_client[i] = [int(k) for k in dict_client[i]]
    # print(dict_client)
    return dict_client


if __name__ == "__main__":
    # poreto_noniid(0,10)
    apply_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.FashionMNIST(
        "./data/fashionmnist/", train=True, download=True, transform=apply_transform
    )
    # for i in train_dataset:
    #     print(i)
    #     breakpoint()
    # breakpoint()
    num_users = 10
    client_dict = poreto_noniid(train_dataset, num_users)
    index = 1
    save_dataset_idx(
        client_dict, f"10client/FashionMNIST-noniid-featured_bias_sample_{index}.json"
    )
    df = sta(client_dict, train_dataset, num_users)
    df.to_csv(f"10client/FashionMNIST-noniid-featured_bias_sample_{index}.csv")
