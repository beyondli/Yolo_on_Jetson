"""eval_yolo.py

This script is for evaluating engine of YOLO models.
"""
import os
import sys
import json
import argparse

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import cv2
import torch

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from Processor import Processor
from utils.general import coco80_to_coco91_class

# converts 80-index (val2014) to 91-index (paper)
coco91class = coco80_to_coco91_class()

VAL_IMGS_DIR = "./images_test"
SAVE_IMGS_DIR = "./images_test/ret"

def parse_args():
    """Parse input arguments."""
    desc = 'test image detection of YOLO TRT model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '--imgs-dir', type=str, default=VAL_IMGS_DIR,
        help='directory of validation images [%s]' % VAL_IMGS_DIR)
    parser.add_argument(
        '-c', '--category-num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '--img-size', nargs='+', type=int, default=[640, 640], help='image size')
    parser.add_argument(
        '-m', '--model', type=str, default='./weights/yolov5s_fp32.engine',
        help=('trt model path'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    parser.add_argument(
        '--conf-thres', type=float, default=0.6,
        help='object confidence threshold')
    parser.add_argument(
        '--iou-thres', type=float, default=0.6,
        help='IOU threshold for NMS')
    args = parser.parse_args()
    return args


def check_args(args):
    """Check and make sure command-line arguments are valid."""
    if not os.path.isdir(args.imgs_dir):
        sys.exit('%s is not a valid directory' % args.imgs_dir)


def generate_results(processor, imgs_dir, jpgs, conf_thres, iou_thres, non_coco):
    """Run detection on each jpg and write results to file."""
    results = []

    with open("trt/labels.txt", "r") as tf:
        category = tf.read().split('\n')

    
    i = 0
    for jpg in jpgs:
        i+=1
        if(i%20 == 0):
            print('Processing {} images'.format(i))
        img = cv2.imread(os.path.join(imgs_dir, jpg))
        output = processor.detect(img)

        img_save = img
        
        pred = processor.post_process(output, img.shape, conf_thres=conf_thres, iou_thres=iou_thres)
        for p in pred.tolist():
            
            x = float(p[0])
            y = float(p[1])
            w = float(p[2] - p[0])
            h = float(p[3] - p[1])

            category_id = coco91class[int(p[5])] if not non_coco else int(p[5])

            img_save = cv2.putText(img_save, category[category_id - 1], (int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2);

            cv2.rectangle(img_save,(int(x),int(y)),(int(p[2]),int(p[3])),(0,255,0),2)
        
        cv2.imwrite(os.path.join(SAVE_IMGS_DIR,'img_test_{}.jpg'.format(i)),img_save)
        

def main():
    
    args = parse_args()
    check_args(args)

    # setup processor

    processor = Processor(model=args.model, letter_box=True)

    jpgs = [j for j in os.listdir(args.imgs_dir) if j.endswith('.jpg')]

    if (len(jpgs) == 0):
        assert(0), 'no images in {}'.format(args.imgs_dir)

    generate_results(processor, args.imgs_dir, jpgs, args.conf_thres, args.iou_thres,
                     non_coco=False)


if __name__ == '__main__':
    main()
