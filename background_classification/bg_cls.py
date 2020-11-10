import torch
import os
import sys

from background_classification.model.model import BGClassification
from background_classification.data.data import FastBaseTransform

# Root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)  # To find local version of the library

CUDA = False

# 모델 생성하는 부분
MODEL = BGClassification()
MODEL.load_weights("{}/weights/model_base_44_8100.pth".format(ROOT_DIR), CUDA)
MODEL.eval()

if CUDA:
    MODEL = MODEL.cuda()


def get_bg(img):
    # 모델에 넣을 이미지 불러오기 -> YOLACT랑 똑같이 만듬
    frame = torch.from_numpy(img).float()
    if CUDA:
        frame = frame.cuda()
    batch = FastBaseTransform(CUDA)(frame.unsqueeze(0))

    # 실제로 모델 테스트
    with torch.no_grad():
        outputs = MODEL(batch)
        # 나온 확률 중 가장 큰 값
        _, pred = torch.max(outputs, 1)

    return int(pred)


if __name__ == "__main__":
    import cv2
    import os
    import background_classification.data.config as config

    image_dir = "../images"
    image_names = [f"test{i}.jpg"for i in range(1, 27)]

    for img_name in image_names:
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)

        b = get_bg(img)
        print(f"Success {img_name} is {b}, {config.CLASSES[b]}")
