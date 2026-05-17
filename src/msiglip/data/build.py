# import logging

import torch
import torchvision.transforms as T
# from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler

# from datasets.sampler import RandomIdentitySampler
# from datasets.sampler_ddp import RandomIdentitySampler_DDP
# from msiglip.utils.comm import get_world_size
# from msiglip.utils.tokenizer_utils import get_tokenizer

# from .bases import ImageDataset, ImageTextDataset, ImageTextMLMDataset, TextDataset
# from .cuhkpedes import CUHKPEDES
# from .icfgpedes import ICFGPEDES
# from .rstpreid import RSTPReid
# from .vn3k import VN3K

# __factory = {
#     "CUHK-PEDES": CUHKPEDES,
#     "ICFG-PEDES": ICFGPEDES,
#     "RSTPReid": RSTPReid,
#     "VN3K": VN3K,
# }


def build_transforms(img_size=(384, 128), aug=False, is_train=True) -> T.Compose:
    """
    Build a transform function for the dataset.

    Args:
        img_size: tuple, (height, width)
        aug: bool, whether to use data augmentation
        is_train: bool, whether to use training transform
    Returns:
        transform: torchvision.transforms.Compose
    """
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose(
            [
                T.Resize((height, width)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )
        return transform

    # transform for training
    if aug:
        transform = T.Compose(
            [
                T.Resize((height, width)),
                T.RandomHorizontalFlip(0.5),
                T.Pad(10),
                T.RandomCrop((height, width)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
                T.RandomErasing(scale=(0.02, 0.4), value=mean),
            ]
        )
    else:
        transform = T.Compose(
            [
                T.Resize((height, width)),
                T.RandomHorizontalFlip(0.5),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )
    return transform


def collate(batch):
    keys = set([key for b in batch for key in b.keys()])
    # turn list of dicts data structure to dict of lists data structure
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], int):
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):
            batch_tensor_dict.update({k: torch.stack(v)})
        else:
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")

    return batch_tensor_dict


# def collate(batch):
#     # Initialize dictionary to hold batched data
#     batched_data = {}

#     # Get all keys from the first item in the batch to identify field names
#     keys = batch[0].keys()

#     for key in keys:
#         if isinstance(batch[0][key], dict):
#             # If the field is a nested dictionary, process each sub-field
#             sub_keys = batch[0][key].keys()
#             batched_data[key] = {}
#             for sub_key in sub_keys:
#                 sub_values = [item[key][sub_key] for item in batch]
#                 if isinstance(sub_values[0], torch.Tensor):
#                     # Concatenate tensors
#                     batched_data[key][sub_key] = torch.cat(sub_values, dim=0)
#                 else:
#                     # Assume list of primitive types and convert to tensor
#                     batched_data[key][sub_key] = torch.tensor(
#                         sub_values, dtype=torch.int64
#                     )
#         else:
#             # If the field is not a nested dictionary, process directly
#             values = [item[key] for item in batch]
#             if isinstance(values[0], torch.Tensor):
#                 # Stack tensors if they are of the same size
#                 batched_data[key] = torch.stack(values)
#             else:
#                 # Assume list of primitive types and convert to tensor
#                 batched_data[key] = torch.tensor(values, dtype=torch.int64)


#     return batched_data

# def build_dataloader(args, tranforms=None):
#     logger = logging.getLogger("IRRA.dataset")

#     num_workers = args.num_workers
#     dataset = __factory[args.dataset_name](root=args.root_dir)
#     num_classes = len(dataset.train_id_container)

#     tokenizer = get_tokenizer(args)

#     if args.training:
#         train_transforms = build_transforms(
#             img_size=args.img_size, aug=args.img_aug, is_train=True
#         )
#         val_transforms = build_transforms(img_size=args.img_size, is_train=False)

#         if args.MLM:
#             train_set = ImageTextMLMDataset(
#                 tokenizer, dataset.train, train_transforms, text_length=args.text_length
#             )
#         else:
#             train_set = ImageTextDataset(
#                 tokenizer, dataset.train, train_transforms, text_length=args.text_length
#             )

#         if args.sampler == "identity":
#             if args.distributed:
#                 logger.info("using ddp random identity sampler")
#                 logger.info("DISTRIBUTED TRAIN START")
#                 mini_batch_size = args.batch_size // get_world_size()
#                 # TODO wait to fix bugs
#                 data_sampler = RandomIdentitySampler_DDP(
#                     dataset.train, args.batch_size, args.num_instance
#                 )
#                 batch_sampler = torch.utils.data.sampler.BatchSampler(
#                     data_sampler, mini_batch_size, True
#                 )

#             else:
#                 logger.info(
#                     f"using random identity sampler: batch_size: {args.batch_size}, id: {args.batch_size // args.num_instance}, instance: {args.num_instance}"
#                 )
#                 train_loader = DataLoader(
#                     train_set,
#                     batch_size=args.batch_size,
#                     sampler=RandomIdentitySampler(
#                         dataset.train, args.batch_size, args.num_instance
#                     ),
#                     num_workers=num_workers,
#                     collate_fn=collate,
#                 )
#         elif args.sampler == "random":
#             # TODO add distributed condition
#             logger.info("using random sampler")
#             train_loader = DataLoader(
#                 train_set,
#                 batch_size=args.batch_size,
#                 shuffle=True,
#                 num_workers=num_workers,
#                 collate_fn=collate,
#             )
#         else:
#             logger.error(
#                 "unsupported sampler! expected softmax or triplet but got {}".format(
#                     args.sampler
#                 )
#             )

#         # use test set as validate set
#         ds = dataset.val if args.val_dataset == "val" else dataset.test
#         val_img_set = ImageDataset(ds["image_pids"], ds["img_paths"], val_transforms)
#         val_txt_set = TextDataset(
#             tokenizer, ds["caption_pids"], ds["captions"], text_length=args.text_length
#         )

#         val_img_loader = DataLoader(
#             val_img_set,
#             batch_size=args.batch_size,
#             shuffle=False,
#             num_workers=num_workers,
#             collate_fn=collate,
#         )
#         val_txt_loader = DataLoader(
#             val_txt_set,
#             batch_size=args.batch_size,
#             shuffle=False,
#             num_workers=num_workers,
#             collate_fn=collate,
#         )

#         return train_loader, val_img_loader, val_txt_loader, num_classes

#     else:
#         # Build dataloader for testing
#         if tranforms:
#             test_transforms = tranforms
#         else:
#             test_transforms = build_transforms(img_size=args.img_size, is_train=False)

#         ds = dataset.test
#         test_img_set = ImageDataset(ds["image_pids"], ds["img_paths"], test_transforms)
#         test_txt_set = TextDataset(
#             tokenizer, ds["caption_pids"], ds["captions"], text_length=args.text_length
#         )

#         test_img_loader = DataLoader(
#             test_img_set,
#             batch_size=args.test_batch_size,
#             shuffle=False,
#             num_workers=num_workers,
#             collate_fn=collate,
#         )
#         test_txt_loader = DataLoader(
#             test_txt_set,
#             batch_size=args.test_batch_size,
#             shuffle=False,
#             num_workers=num_workers,
#             collate_fn=collate,
#         )
#         return test_img_loader, test_txt_loader, num_classes
