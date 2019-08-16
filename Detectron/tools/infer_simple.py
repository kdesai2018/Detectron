#!/usr/bin/python2

# Copyright (c) 2017-present, Facebook, Inc.
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
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

#ROS imports
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from PIL import Image as PL


c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

# os.chdir("/home/dirac/catkin_ws/src/detectron/Detectron")

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        # default='configs/12_2017_baselines/retinanet_R-101-FPN_2x.yaml',
        default='configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml',
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default='https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold for visualizing detections',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--kp-thresh',
        dest='kp_thresh',
        help='Threshold for visualizing keypoints',
        default=2.0,
        type=float
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default='.'
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

class get_image:
    def __init__(self):
        self.bridge = CvBridge()

        # edit this topic based on where the camera topic is located (eg: "/usb_cam/image_raw")
        self.image_subscriber = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback)

    def callback(self,data):
        cv_image = None
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        args.im_or_folder = cv_image

def talker(i):
    image_publisher = rospy.Publisher('detectron_output', Image, queue_size=10)
    brdg = CvBridge()
    image_publisher.publish(brdg.cv2_to_imgmsg(i, "bgr8"))

def main(args):
    #ros initialization
    rospy.init_node('get_image', anonymous=True)

    args.im_or_folder = get_image()

    while not rospy.is_shutdown():
        merge_cfg_from_file(args.cfg)
        cfg.NUM_GPUS = 1
        args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
        assert_and_infer_cfg(cache_urls=False)

        assert not cfg.MODEL.RPN_ONLY, \
            'RPN models are not supported'
        assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
            'Models that require precomputed proposals are not supported'

        model = infer_engine.initialize_model_from_cfg(args.weights)
        dummy_coco_dataset = dummy_datasets.get_coco_dataset()
        # set im list to be the ros image
        im = args.im_or_folder

        timers = defaultdict(Timer)
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(model, im, None, timers=timers
            )
        # calls the method that performs actual detection and inference
        fig = vis_utils.vis_one_image_opencv(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dummy_coco_dataset,
            show_class=True,
            thresh=args.thresh,
            kp_thresh=args.kp_thresh,
        )

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(fig, cv2.COLOR_RGB2BGR)


        image_publisher = rospy.Publisher('detectron_output', Image, queue_size=10)
        brdg = CvBridge()
        image_publisher.publish(brdg.cv2_to_imgmsg(img, "bgr8"))
        try:
            start_time = time.time()
            print(time.time()-start_time)
            main(args)
        except KeyboardInterrupt:
            print("shutting down")
            cv2.destroyAllWindows()




if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
