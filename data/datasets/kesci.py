# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset


class Kesci(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'kesci'

    def __init__(self, root='/media/yj_data/DataSet/reid', verbose=True, **kwargs):
        super(Kesci, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_file = osp.join(self.dataset_dir, 'train_list.txt')
        self.query_file = osp.join(self.dataset_dir, 'query_list.txt')
        self.gallery_file = osp.join(self.dataset_dir, 'gallery_list.txt')

        self._check_before_run()

        train = self._process_dir(self.train_file, camid = 0, relabel=True)
        query = self._process_dir(self.query_file, camid = 1, relabel=False)
        gallery = self._process_dir(self.gallery_file, camid = 2, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_file):
            raise RuntimeError("'{}' is not available".format(self.train_file))
        if not osp.exists(self.query_file):
            raise RuntimeError("'{}' is not available".format(self.query_file))
        if not osp.exists(self.gallery_file):
            raise RuntimeError("'{}' is not available".format(self.gallery_file))

    def _process_dir(self, file_path, camid, relabel=False):
        # train/427696570.png 4258
        lines = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        pid_container = set()
        for line in lines:
            img_path, pid = line.strip().split()
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
            img_path = osp.join(self.dataset_dir, img_path)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for line in lines:
            img_path, pid = line.strip().split()
            if pid == -1: continue  # junk images are just ignored
            if relabel: pid = pid2label[pid]
            img_path = osp.join(self.dataset_dir, img_path)
            dataset.append((img_path, pid, camid))

        return dataset

class KesciA(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'kesci'

    def __init__(self, root='/media/yj_data/DataSet/reid', verbose=True, **kwargs):
        super(KesciA, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_file = osp.join(self.dataset_dir, 'train_list.txt')
        self.query_file = osp.join(self.dataset_dir, 'query_a_list.txt')
        self.gallery_file = osp.join(self.dataset_dir, 'gallery_a_list.txt')

        self._check_before_run()

        train = self._process_dir(self.train_file, camid = 0, relabel=True)
        query = self._process_dir(self.query_file, camid = 1, relabel=False)
        gallery = self._process_dir(self.gallery_file, camid = 2, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_file):
            raise RuntimeError("'{}' is not available".format(self.train_file))
        if not osp.exists(self.query_file):
            raise RuntimeError("'{}' is not available".format(self.query_file))
        if not osp.exists(self.gallery_file):
            raise RuntimeError("'{}' is not available".format(self.gallery_file))

    def _process_dir(self, file_path, camid, relabel=False):
        # train/427696570.png 4258
        lines = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        pid_container = set()
        for line in lines:
            img_path, pid = line.strip().split()
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
            img_path = osp.join(self.dataset_dir, img_path)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for line in lines:
            img_path, pid = line.strip().split()
            if pid == -1: continue  # junk images are just ignored
            if relabel: pid = pid2label[pid]
            img_path = osp.join(self.dataset_dir, img_path)
            dataset.append((img_path, pid, camid))

        return dataset
