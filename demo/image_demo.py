#!/home/jichao/anaconda3/envs/open-mmlab/bin/python
import os

print('Current working path',
      os.getcwd())  # please enter "/home/jichao/python_ws/Swin-Transformer-Semantic-Segmentation-main/demo" to run the python file

import sys

print('当前 Python 解释器路径：', sys.executable)
parent_path = os.path.dirname(sys.path[0])
print('Import libraries from', parent_path)
if parent_path not in sys.path:
    sys.path.append(parent_path)

import cv2
from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

image_file = './demo2.png'
config_file = '../configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py'
checkpoint_file = '../upernet_swin_tiny_patch4_window7_512x512.pth'


def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default=image_file, help='Image file')
    parser.add_argument('--config', default=config_file, help='Config file')
    parser.add_argument('--checkpoint', default=checkpoint_file, help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_segmentor(model, args.img)
    print('result', type(result[0]), result[0].shape, result)
    # show the results
    segment_image = show_result_pyplot(model, args.img, result, get_palette(args.palette), display=False)
    cv2.imwrite('./demo2_segmented.png', segment_image)
    return result[0], segment_image


if __name__ == '__main__':
    main()
