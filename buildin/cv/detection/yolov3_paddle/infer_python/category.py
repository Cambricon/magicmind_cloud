# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['get_categories']


def get_categories(metric_type, anno_file=None, arch=None):
    """
    Get class id to category id map and category id
    to category name map from annotation file.

    Args:
        metric_type (str): metric type, currently support 'coco', 'voc', 'oid'
            and 'widerface'.
        anno_file (str): annotation file path
    """
    if arch == 'keypoint_arch':
        return (None, {'id': 'keypoint'})

    if anno_file == None or (not os.path.isfile(anno_file)):
        logger.warning(
            "anno_file '{}' is None or not set or not exist, "
            "please recheck TrainDataset/EvalDataset/TestDataset.anno_path, "
            "otherwise the default categories will be used by metric_type.".
            format(anno_file))

    if metric_type.lower() == 'coco' or metric_type.lower(
    ) == 'rbox' or metric_type.lower() == 'snipercoco':
        if anno_file and os.path.isfile(anno_file):
            if anno_file.endswith('json'):
                # lazy import pycocotools here
                from pycocotools.coco import COCO
                coco = COCO(anno_file)
                cats = coco.loadCats(coco.getCatIds())

                clsid2catid = {i: cat['id'] for i, cat in enumerate(cats)}
                catid2name = {cat['id']: cat['name'] for cat in cats}

            elif anno_file.endswith('txt'):
                cats = []
                with open(anno_file) as f:
                    for line in f.readlines():
                        cats.append(line.strip())
                if cats[0] == 'background': cats = cats[1:]

                clsid2catid = {i: i for i in range(len(cats))}
                catid2name = {i: name for i, name in enumerate(cats)}

            else:
                raise ValueError("anno_file {} should be json or txt.".format(
                    anno_file))
            return clsid2catid, catid2name

        # anno file not exist, load default categories of COCO17
        else:
            if metric_type.lower() == 'rbox':
                logger.warning(
                    "metric_type: {}, load default categories of DOTA.".format(
                        metric_type))
                return _dota_category()
            logger.warning("metric_type: {}, load default categories of COCO.".
                           format(metric_type))
            return _coco17_category()

    elif metric_type.lower() == 'voc':
        if anno_file and os.path.isfile(anno_file):
            cats = []
            with open(anno_file) as f:
                for line in f.readlines():
                    cats.append(line.strip())

            if cats[0] == 'background':
                cats = cats[1:]

            clsid2catid = {i: i for i in range(len(cats))}
            catid2name = {i: name for i, name in enumerate(cats)}

            return clsid2catid, catid2name

        # anno file not exist, load default categories of
        # VOC all 20 categories
        else:
            logger.warning("metric_type: {}, load default categories of VOC.".
                           format(metric_type))
            return _vocall_category()

    elif metric_type.lower() == 'oid':
        if anno_file and os.path.isfile(anno_file):
            logger.warning("only default categories support for OID19")
        return _oid19_category()

    elif metric_type.lower() == 'widerface':
        return _widerface_category()

    elif metric_type.lower() == 'keypointtopdowncocoeval' or metric_type.lower(
    ) == 'keypointtopdownmpiieval':
        return (None, {'id': 'keypoint'})

    elif metric_type.lower() in ['mot', 'motdet', 'reid']:
        if anno_file and os.path.isfile(anno_file):
            cats = []
            with open(anno_file) as f:
                for line in f.readlines():
                    cats.append(line.strip())
            if cats[0] == 'background':
                cats = cats[1:]
            clsid2catid = {i: i for i in range(len(cats))}
            catid2name = {i: name for i, name in enumerate(cats)}
            return clsid2catid, catid2name
        # anno file not exist, load default category 'pedestrian'.
        else:
            logger.warning(
                "metric_type: {}, load default categories of pedestrian MOT.".
                format(metric_type))
            return _mot_category(category='pedestrian')

    elif metric_type.lower() in ['kitti', 'bdd100kmot']:
        return _mot_category(category='vehicle')

    elif metric_type.lower() in ['mcmot']:
        if anno_file and os.path.isfile(anno_file):
            cats = []
            with open(anno_file) as f:
                for line in f.readlines():
                    cats.append(line.strip())
            if cats[0] == 'background':
                cats = cats[1:]
            clsid2catid = {i: i for i in range(len(cats))}
            catid2name = {i: name for i, name in enumerate(cats)}
            return clsid2catid, catid2name
        # anno file not exist, load default categories of visdrone all 10 categories
        else:
            logger.warning(
                "metric_type: {}, load default categories of VisDrone.".format(
                    metric_type))
            return _visdrone_category()

    else:
        raise ValueError("unknown metric type {}".format(metric_type))
