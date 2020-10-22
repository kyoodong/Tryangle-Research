
import argparse
import cv2

import torch
from background_classification.model.model import BGClassification
from background_classification.data.data import FastBaseTransform
import background_classification.data.config as config

# ================================================
# How to test model
# python train.py --dataset=path/to/dataset
#
# ================================================

def main():
    parse_args()
    evaluate()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    global args
    parser = argparse.ArgumentParser(
        description='BGClassification Eval Script')
    parser.add_argument('--trained_model',
                        default='weights/model_base_44_8100.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--image',
                        default=None, type=str,
                        help='A path to an image to use')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')

    parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True, autoscale=True)
    args = parser.parse_args()

    def replace(name, value):
        if getattr(args, name) == None: setattr(args, name, value)

    replace("image", "images/test7.jpg")


def evaluate():
    import time

    model = BGClassification()
    model.load_weights(args.trained_model)
    model.eval()

    if args.cuda:
        model = model.cuda()

    img = cv2.imread(args.image)
    frame = torch.from_numpy(img).float()
    if args.cuda:
        frame = frame.cuda()
    batch = FastBaseTransform()(frame.unsqueeze(0))

    with torch.no_grad():
        result = model(batch)

        st = time.time()
        outputs = model(batch)
        print(f"{time.time() - st} sec .....")

        _, pred = torch.max(outputs, 1)

        print(config.CLASSES[pred])


if __name__ == "__main__":
    main()