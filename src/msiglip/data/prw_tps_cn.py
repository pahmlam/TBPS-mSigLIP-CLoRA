import os.path as op
from typing import List

from msiglip.utils.iotools import read_json
from .bases import BaseDataset


class PRW_TPS_CN(BaseDataset):
    """
    PRW-TPS-CN

    Chinese text-based person search on PRW dataset.

    annotation format:
    [{'split', str,
      'captions', list,
      'file_path', str,
      'id', int}...]
    """

    dataset_dir = "PRW-TPS-CN"

    def __init__(self, root="", verbose=True, seed=42):
        super(PRW_TPS_CN, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.img_dir = op.join(self.dataset_dir, "imgs/")

        self.anno_path = op.join(self.dataset_dir, "prw_cn_caption_crops.json")
        self._check_before_run()

        self.train_annos, self.test_annos, self.val_annos = self._split_anno(
            self.anno_path
        )

        self.train, self.train_id_container = self._process_anno(
            self.train_annos, training=True
        )
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            self.logger.info("=> PRW-TPS-CN Images and Captions are loaded")
            self.show_dataset_info()

    def _split_anno(self, anno_path: str):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        for anno in annos:
            if anno["split"] == "train":
                train_annos.append(anno)
            elif anno["split"] == "test":
                test_annos.append(anno)
            else:
                val_annos.append(anno)
        return train_annos, test_annos, val_annos

    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        skipped = 0
        if training:
            dataset = []
            image_id = 0
            for anno in annos:
                img_path = op.join(self.img_dir, anno["file_path"])
                if not op.exists(img_path):
                    skipped += 1
                    continue
                pid = int(anno["id"]) - 1  # make pid begin from 0
                pid_container.add(pid)
                captions = anno["captions"]  # caption list
                for caption in captions:
                    dataset.append((pid, image_id, img_path, caption))
                image_id += 1
            # Remap pids to be contiguous from 0
            old_pids = sorted(pid_container)
            pid_map = {old: new for new, old in enumerate(old_pids)}
            dataset = [(pid_map[pid], img_id, path, cap) for pid, img_id, path, cap in dataset]
            pid_container = set(pid_map.values())
            if skipped > 0:
                self.logger.info(f"Skipped {skipped} train annotations with missing images")
            return dataset, pid_container
        else:
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            for anno in annos:
                img_path = op.join(self.img_dir, anno["file_path"])
                if not op.exists(img_path):
                    skipped += 1
                    continue
                pid = int(anno["id"])
                pid_container.add(pid)
                img_paths.append(img_path)
                image_pids.append(pid)
                caption_list = anno["captions"]  # caption list
                for caption in caption_list:
                    captions.append(caption)
                    caption_pids.append(pid)
            if skipped > 0:
                self.logger.info(f"Skipped {skipped} eval annotations with missing images")
            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions,
            }
            return dataset, pid_container

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not op.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))
