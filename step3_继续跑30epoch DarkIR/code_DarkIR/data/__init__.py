from .dataset_reader.dataset_LOLBlur import main_dataset_lolblur

def create_dataloaders(rank, world_size, datasets_opt, mode, return_path=False, samples_per_scene=15):
    """
    datasets_opt: dict from YAML key 'datasets'
    mode: 'train' or 'val'
    return_path: if True, return image paths along with images (for visualization)
    """
    if mode == 'train':
        opt = datasets_opt['train']
    elif mode == 'val':
        opt = datasets_opt['val']
    else:
        raise ValueError(f"Invalid mode {mode}")

    name = datasets_opt['name'] if 'name' in datasets_opt else 'LOLBlur'
    data_path = opt['dataroot']
    batch_size = opt.get('batch_size', 1)
    verbose = datasets_opt.get('verbose', True)
    num_workers = opt.get('n_workers', 4)

    if rank != 0:
        verbose = False
    samplers = None

    if name == 'LOLBlur':
        loader, samplers = main_dataset_lolblur(
            rank=rank,
            test_path=data_path,
            batch_size_test=batch_size,
            verbose=verbose,
            num_workers=num_workers,
            world_size=world_size,
            return_path=return_path,
            samples_per_scene=samples_per_scene
        )
    else:
        raise NotImplementedError(f"{name} is not implemented")

    if rank == 0:
        print(f"Using {name} Dataset for {mode}")

    return loader, samplers
