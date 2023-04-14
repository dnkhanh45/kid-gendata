import json
def save_dataset_idx(list_idx_sample,path="dataset_idx.json"):
    with open(path, "w+") as outfile:
        json.dump(list_idx_sample, outfile)