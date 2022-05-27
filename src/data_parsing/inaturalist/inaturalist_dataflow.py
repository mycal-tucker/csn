#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: iNaturalist.py
from tensorflow.keras.utils import to_categorical
import json
import os
import os.path as osp
import pickle
import cv2
import numpy as np

__all__ = ['iNaturalistMeta', 'iNaturalist', 'iNaturalistFiles']
ground_truth_dir = '/mnt/mount_sda2/src/inaturalist/'


class DataFlow(object):
    """ Base class for all DataFlow """

    def get_data(self):
        """
        The method to generate datapoints.
        Yields:
            list: The datapoint, i.e. list of components.
        """

    def size(self):
        """
        Returns:
            int: size of this data flow.
        Raises:
            :class:`NotImplementedError` if this DataFlow doesn't have a size.
        """
        raise NotImplementedError()

    def reset_state(self):
        """
        Reset state of the dataflow.
        It **has to** be called once and only once before producing datapoints.
        Note:
            1. If the dataflow is forked, each process will call this method
               before producing datapoints.
            2. The caller thread of this method must remain alive to keep this dataflow alive.
        For example, RNG **has to** be reset if used in the DataFlow,
        otherwise it won't work well with prefetching, because different
        processes will have the same RNG state.
        """
        pass


class RNGDataFlow(DataFlow):
    """ A DataFlow with RNG"""

    def reset_state(self):
        """ Reset the RNG """
        print("Resetting random seed for numpy.")
        # np.random.seed(int(np.random.random() * 100))
        self.rng = np.random.shuffle

class iNaturalistMeta(object):
    """
    Provide methods to access metadata for ILSVRC dataset.
    """

    def __init__(self, dir=None):
        pass

    def get_image_list(self, name):
        """
        Args:
            name (str): 'train' or 'val' or 'test'
            dir_structure (str): same as in :meth:`iNaturalist.__init__()`.
        Returns:
            list: list of (image filename, label)
        """
        assert name in ['train', 'val', 'test']
        ret = []

        if name == 'train':
            fname = osp.join(ground_truth_dir, 'train2019.json')
            assert os.path.isfile(fname), fname

            with open(fname, 'r') as f:
                train2019 = json.load(f)
            for i in range(len(train2019['images'])):
                name = train2019['images'][i]['file_name'] #train_va/2018/Aves/2820/285hy8uryu8w989.jpg
                cls = train2019['annotations'][i]['category_id']
                ret.append((name.strip(), cls))

        if name == 'val':
            fname = osp.join(ground_truth_dir, 'val2019.json')
            assert os.path.isfile(fname), fname

            with open(fname, 'r') as f:
                val2019 = json.load(f)
            for i in range(len(val2019['images'])):
                name = val2019['images'][i]['file_name'] #train_va/2018/Aves/2820/285hy8uryu8w989.jpg
                cls = val2019['annotations'][i]['category_id']
                ret.append((name.strip(), cls))

        if name == 'test':
            fname = osp.join(ground_truth_dir, 'test2019.json')
            assert os.path.isfile(fname), fname

            with open(fname, 'r') as f:
                test2019 = json.load(f)
            for i in range(len(test2019['images'])):
                name = test2019['images'][i]['file_name'] #train_va/2018/Aves/2820/285hy8uryu8w989.jpg
                id = test2019['images'][i]['id']
                ret.append((name.strip(), id))

        assert len(ret)
        return ret

class iNaturalistFiles(RNGDataFlow):
    """
    Same as :class:`iNaturalist`, but produces filenames of the images instead of nparrays.
    This could be useful when ``cv2.imread`` is a bottleneck and you want to
    decode it in smarter ways (e.g. in parallel).
    """
    def __init__(self, dir, name, meta_dir=None,
                 shuffle=True):
        """
        Same as in :class:`iNaturalist`.
        """
        assert name in ['train', 'test', 'val'], name
        assert os.path.isdir(dir), dir
        self.full_dir = dir
        self.name = name
        assert os.path.isdir(self.full_dir), self.full_dir
        assert meta_dir is None or os.path.isdir(meta_dir), meta_dir
        if shuffle is None:
            shuffle = name == 'train'
        self.shuffle = shuffle

        meta = iNaturalistMeta(meta_dir)
        self.imglist = meta.get_image_list(name)

        for fname, _ in self.imglist[:10]:
            fname = os.path.join(self.full_dir, fname)
            assert os.path.isfile(fname), fname

    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            # assert False, "Didn't implement suffling"
            self.rng(idxs)
            pass
        for k in idxs:
            fname, label = self.imglist[k]
            fname = os.path.join(self.full_dir, fname)
            yield [fname, label]

class iNaturalist(iNaturalistFiles):
    """
    Produces uint8 iNaturalist images of shape [h, w, 3(BGR)], and a label between [0, 8141]. num_classes=8142
    """
    def __init__(self, dir, name, meta_dir=None,
                 shuffle=None):
        """
        Args:
            dir (str): A directory containing a subdir named ``name``,
                containing the images in a structure described below.
            name (str): One of 'train' or 'val' or 'test'.
            shuffle (bool): shuffle the dataset.
                Defaults to True if name=='train'.
            dir_structure (str): One of 'original' or 'train'.
        Examples:
        When `dir_structure=='original'`, `dir` should have the following structure:
            dir/
              train/
                n02134418/
                  n02134418_198.JPEG
                  ...
                ...
              val(test)/
                ILSVRC2012_val_00000001.JPEG
        When `dir_structure=='train'`, `dir` should have the following structure:
            dir/
              train/
                n02134418/
                  n02134418_198.JPEG
                  ...
              val/
                n01440764/
                  ILSVRC2012_val_00000293.JPEG
                  ...
              test/
                ILSVRC2012_test_00000001.JPEG
        """
        super(iNaturalist, self).__init__(
            dir, name, meta_dir, shuffle)

        self.tree = pickle.load(open('data_parsing/inaturalist/inaturalist19_tree.pkl', 'rb'))
        print("Tree", self.tree)
        all_positions = self.tree.treepositions()
        level_sizes = [3, 4, 9, 34, 57, 72, 1010]
        levels = [[] for _ in level_sizes]
        global_tree_positions = []
        for idx, trace in enumerate(all_positions[1:]):  # Skip the root node because doesn't matter
            if idx < 10:
                print(global_tree_positions)
            level = len(trace) - 1
            print("Level", level)
            matching_elts = len(levels[level])
            levels[level].append(matching_elts)
            if level == 6:
                print("Bottom!")
                global_tree_positions.append([stored_level[-1] for stored_level in levels])
                # print(global_tree_positions)
        print("Global", global_tree_positions)
        self.tree_positions = global_tree_positions

    def __iter__(self):
        for fname, label in super(iNaturalist, self).get_data():
            # print("Label", label)
            im = cv2.imread(fname, cv2.IMREAD_COLOR) / 255
            # im = cv2.imread(fname, cv2.IMREAD_COLOR)
            # cv2.imshow('temp', im)
            # cv2.multiply(im, 255, im)
            # cv2.imshow('new temp', im)
            # cv2.waitKey()
            assert im is not None, fname
            # yield [im, label]
            ancestry = self.tree_positions[label]
            # relevant_labels = ancestry[-1]  # Just the label
            # relevant_labels = ancestry[-2:]  # One level of hierarchy.
            # arrayed = [to_categorical(relevant_labels[0], num_classes=72),
            #            to_categorical(relevant_labels[1], num_classes=1010)]
            relevant_labels = ancestry
            arrayed = [to_categorical(relevant_labels[0], num_classes=3),
                       to_categorical(relevant_labels[1], num_classes=4),
                       to_categorical(relevant_labels[2], num_classes=9),
                       to_categorical(relevant_labels[3], num_classes=34),
                       to_categorical(relevant_labels[4], num_classes=57),
                       to_categorical(relevant_labels[5], num_classes=72),
                       to_categorical(relevant_labels[6], num_classes=1010)]
            yield [im] + arrayed

    def __len__(self):
        return 265213

    def reset_state(self):
        super(iNaturalist, self).reset_state()
