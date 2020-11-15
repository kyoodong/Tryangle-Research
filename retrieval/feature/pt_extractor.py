import struct
import glob
import numpy as np
import os

import torch.nn as nn
import torchvision
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2
import PIL.Image as Image

'''
이미지에서 특징을 뽑아서 binary 형태로 저장
'''
class SimpleModel(nn.Module):

    def __init__(self):
        super(SimpleModel, self).__init__()
        # resnet = torchvision.models.resnet50(pretrained=True)
        #
        # self._backbone = _backbone = nn.Sequential(
        #     resnet.conv1,
        #     resnet.bn1,
        #     resnet.relu,
        #     resnet.maxpool,
        #     resnet.layer1,
        #     resnet.layer2,
        #     resnet.layer3,
        #     resnet.avgpool
        # )
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        backbone.out_channel = 1280
        self._backbone = _backbone = backbone

    def forward(self, x):
        x = self._backbone(x)
        x = x.flatten(start_dim=1)
        return x

class TransformCVResize(object):

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        assert isinstance(size, int) or (isinstance(size, tuple) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        img = np.array(img)
        img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(np.uint8(img))
        return img

def extract(image_dataset, store_dir="features", store_file="fvecs"):
    if not os.path.exists(store_dir):
        os.mkdir(store_dir)
    binary_file = f"{store_dir}/{store_file}.bin"
    name_file = f"{store_dir}/{store_file}_names.txt"

    batch_size = 100
    preprocess = transforms.Compose([
        TransformCVResize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = SimpleModel()
    model.eval()

    fnames = glob.glob(os.path.join(image_dataset, "**", "*.jpg"), recursive=True)
    dataset = torchvision.datasets.ImageFolder(root=image_dataset, transform=preprocess)
    dataloaders = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    with torch.no_grad():
        with open(binary_file, 'wb') as f:
            for i, (image, labels) in enumerate(dataloaders):
                fvecs = model(image)
                fvecs = fvecs.numpy()
                # fvecs의 길이 만큼 fmt설정
                fmt = f'{np.prod(fvecs.shape)}f'

                # fmt = 포맷, fvecs.flatten() = 벡터를 한줄로 변환
                # struct.pack을 사용하여 패킹한다.
                f.write(struct.pack(fmt, *(fvecs.flatten())))

                print(f"[INFO] Process {i * batch_size}/{len(fnames)} images.....")

    with open(name_file, 'w') as f:
        f.write('\n'.join(fnames))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help='Dataset path format, ex) ./image/**/*.jpg')
    parser.add_argument("--directory", required=False,
                        default="output",
                        help='Features and Image Path store File')
    parser.add_argument("--store", required=False,
                        default="fvecs",
                        help='Features and Image Path store File')
    args = parser.parse_args()

    extract(image_dataset=args.dataset,
            store_dir=args.directory,
            store_file=args.store)
