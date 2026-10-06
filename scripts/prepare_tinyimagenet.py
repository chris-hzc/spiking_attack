"""
Reorganize the Tiny-ImageNet validation split into one sub-folder per class so
that it can be loaded with torchvision.datasets.ImageFolder.

Usage:
    python scripts/prepare_tinyimagenet.py --root ./data/tiny-imagenet-200
"""

import argparse
import os
import shutil

parser = argparse.ArgumentParser(description='Prepare Tiny-ImageNet validation split')
parser.add_argument('--root', default='./data/tiny-imagenet-200', type=str,
                    help='path to the extracted tiny-imagenet-200 folder')
args = parser.parse_args()

val_dir = os.path.join(args.root, 'val')
img_dir = os.path.join(val_dir, 'images')
annotations = os.path.join(val_dir, 'val_annotations.txt')

if not os.path.isdir(img_dir):
    print(f'{img_dir} not found; validation split already reorganized?')
    raise SystemExit(0)

with open(annotations) as f:
    for line in f:
        fname, wnid = line.split('\t')[:2]
        os.makedirs(os.path.join(val_dir, wnid), exist_ok=True)
        shutil.move(os.path.join(img_dir, fname), os.path.join(val_dir, wnid, fname))

os.rmdir(img_dir)
print(f'Done: {val_dir} now has one sub-folder per class.')
