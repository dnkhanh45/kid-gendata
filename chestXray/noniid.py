def chest_noniid(dataset, num_users):
    """
    Generates a non-iid dataset of size num_users.

    Args:
        dataset: A list of tuples of the form (image, label).
        num_users: The number of users to generate.

    Returns:
        A list of tuples of the form (image, label).
    """
    print(num_users)
    num_shards, num_imgs = 500, 200
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets.numpy()
