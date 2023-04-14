from noniid_cluster import cifar100_noniid_cluster
from noniid_featured import cifar100_noniid_featured
from noniid_pareto import cifar100_noniid_pareto
from noniid_quantittative import cifar100_noniid_quantitative
from torchvision import datasets
from torchvision import datasets, transforms
import json
import os
from pathlib import Path

def save_dataset_idx(list_idx_sample, path, filename="dataset_idx.json"):
    if not Path(path).exists():
        os.system(f"mkdir -p {path}")
    
    with open(path + filename, "w+") as outfile:
        json.dump(list_idx_sample, outfile)

import pandas as pd

def sta(client_dict, train_dataset, num_client=100):
    rs = []
    for client in range(num_client):
        tmp = []
        for i in range(num_client):
            tmp.append(sum(train_dataset[j][1] == i for j in client_dict[client]))
        rs.append(tmp)
    df = pd.DataFrame(rs,columns=[f"Label_{i}" for i in range(num_client)])
    return df
  
def gen_full(dataset,total_client, num_sample=50000):
    client_dict = {}
    for i in range(1):
        if i ==0:
            client_dict[i] = range(num_sample)
        else:
            client_dict[i] = []
        client_dict[i] = [int(k) for k in client_dict[i]]
    return client_dict


if __name__ == '__main__':
    train_dataset = datasets.CIFAR100( "./data/cifar/", train=True, download=True, transform=None)
    print("total sample of dataset", len(train_dataset))
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=['featured', 'quantitative', 'pareto', 'cluster'])
    parser.add_argument("--total_client", type=int, default=100, required=False)
    args = parser.parse_args()
    
    if args.method == 'featured':
        client_dict = cifar100_noniid_featured(train_dataset, total_client=args.total_client)
    elif args.method == 'quantitative':
        client_dict = cifar100_noniid_quantitative(train_dataset, total_client=args.total_client)
    elif args.method == 'pareto':
        client_dict = cifar100_noniid_pareto(train_dataset, total_client=args.total_client)
    elif args.method == 'cluster':
        client_dict = cifar100_noniid_cluster(train_dataset, total_client=args.total_client)
    else:
        raise Exception("Not recognise data distributing method")
    
    print("Gen done!")
    save_dataset_idx(client_dict, f"cifar100/{args.total_client}client/", f"CIFAR100_{args.total_client}client_{args.method}.json")
    df = sta(client_dict, train_dataset, num_client=args.total_client)
    print("Total sample in modified dataset", df.values.sum())
    df.to_csv(f"cifar100/{args.total_client}client/CIFAR100_{args.total_client}client_{args.method}.csv", index=False, header=False)
  
    