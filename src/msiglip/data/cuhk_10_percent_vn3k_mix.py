import os.path as op
from typing import List
from random import Random

from msiglip.utils.iotools import read_json
from .bases import BaseDataset


class TenPercentCUHK_VN3KMIX(BaseDataset):
    """
    annotation format:
    [{'split', str,
      'captions', list,
      'file_path', str,
      'processed_tokens', list,
      'id', int}...]
    """

    dataset_dir_CUHK = "CUHK-PEDES"
    dataset_dir_VN3K = "VN3K"

    def __init__(self, root="", verbose=True, seed=42):
        super(TenPercentCUHK_VN3KMIX, self).__init__()
        self.dataset_dir_CUHK = op.join(root, self.dataset_dir_CUHK)
        self.img_dir_CUHK = op.join(self.dataset_dir_CUHK, "imgs/")
        self.anno_path_CUHK = op.join(self.dataset_dir_CUHK, "reid_raw.json")
        # self._check_before_run()

        self.random_generator = Random(seed)

        self.dataset_dir_VN3K = op.join(root, self.dataset_dir_VN3K)
        self.img_dir_VN3K = op.join(self.dataset_dir_VN3K, "imgs/")
        self.anno_path_VN3K = op.join(self.dataset_dir_VN3K, "data_captions.json")
        # self._check_before_run()

        # Use 0.1 of the CUHK
        train_annos_CUHK, test_annos_CUHK, val_annos_CUHK = self._split_anno(
            self.anno_path_CUHK, self.img_dir_CUHK, proportion=0.1
        )
        # Use full of VN3K
        train_annos_VN3K, test_annos_VN3K, val_annos_VN3K = self._split_anno(
            self.anno_path_VN3K, self.img_dir_VN3K
        )

        # Shift PID of VN3K to the end of CUHK
        max_cuhk_pid = max(
            int(anno["id"])
            for anno in train_annos_CUHK + test_annos_CUHK + val_annos_CUHK
        )
        for anno in train_annos_VN3K + test_annos_VN3K + val_annos_VN3K:
            anno["id"] = int(anno["id"]) + max_cuhk_pid + 1  # +1 for safety margin

        self.train_annos = train_annos_CUHK + train_annos_VN3K
        self.test_annos = test_annos_CUHK + test_annos_VN3K
        self.val_annos = val_annos_CUHK + val_annos_VN3K

        self.train, self.train_id_container = self._process_anno(
            self.train_annos, training=True
        )
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            self.logger.info("=> CUHK-PEDES Images and Captions are loaded")
            self.show_dataset_info()

    def _split_anno(self, anno_path: str, dir_path: str, proportion=None):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        source = None
        if self.dataset_dir_CUHK in anno_path:
            source = "CUHK"
        elif self.dataset_dir_VN3K in anno_path:
            source = "VN3K"

        for anno in annos:
            # Make sure the file path is full path
            anno["file_path"] = op.join(dir_path, anno["file_path"])
            if anno["split"] == "train":
                train_annos.append(anno)
            elif anno["split"] == "test":
                test_annos.append(anno)
            else:
                val_annos.append(anno)

        if proportion is not None:
            if len(train_annos) > 0:
                number_of_samples = int(len(train_annos) * proportion)
                train_annos = self.random_generator.sample(
                    train_annos, number_of_samples
                )
                self.logger.info(
                    f"Using {number_of_samples} = {proportion} of the training set for {source}"
                )
        return train_annos, test_annos, val_annos

    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        if training:
            dataset = []
            image_id = 0
            for anno in annos:
                pid = int(anno["id"]) - 1
                pid_container.add(pid)
                img_path = op.join(anno["file_path"])
                captions = anno["captions"]  # caption list
                for caption in captions:
                    dataset.append((pid, image_id, img_path, caption))
                image_id += 1
            # for idx, pid in enumerate(pid_container):
            #     # check pid begin from 0 and no break
            #     assert idx == pid, f"idx: {idx} and pid: {pid} are not match in {pid_container}"
            # Shuffle the dataset
            return dataset, pid_container
        else:
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            for anno in annos:
                pid = int(anno["id"])
                pid_container.add(pid)
                img_path = op.join(anno["file_path"])
                img_paths.append(img_path)
                image_pids.append(pid)
                caption_list = anno["captions"]  # caption list
                for caption in caption_list:
                    captions.append(caption)
                    caption_pids.append(pid)
            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions,
            }
            return dataset, pid_container

    # def _check_before_run(self):
    #     """Check if all files are available before going deeper"""
    #     if not op.exists(self.dataset_dir):
    #         raise RuntimeError("'{}' is not available".format(self.dataset_dir))
    #     if not op.exists(self.img_dir):
    #         raise RuntimeError("'{}' is not available".format(self.img_dir))
    #     if not op.exists(self.anno_path):
    #         raise RuntimeError("'{}' is not available".format(self.anno_path))
