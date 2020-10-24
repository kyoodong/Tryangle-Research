from pycocotools.coco import COCO

import shutil
import os
import json
"""

SRC_COCO_IMAGE_DIR, SRC_COCO_ANNOTATION_PATH -> 기본 COCO 데이터셋의 위치
pick_category -> 뽑아낼 카테고리

--------
output
--------
output_dir 위치에 해당하는 디렉토리에 annotations와 images폴더가 생기며, 
해당 폴더들 안에 뽑아낸 이미지와 annotation이 들어가 있음

"""

# =================Default Setting==================

output_dir = "coco_data_selected"

SRC_COCO_IMAGE_DIR = "coco/images"
SRC_COCO_ANNOTATION_PATH = "coco/annotations/instances_train2017.json"
annotation_file = SRC_COCO_ANNOTATION_PATH.split("/")[-1]
pick_category = {
        1: "person", 44: "bottle", 46: "wine glass", 47: "cup"
}

# =================== Process =======================
def main():
    process()


def process(copy_images:bool=True, copy_annotations:bool=True):
    annotation_dir = os.path.join(output_dir, "annotations")
    images_dir = os.path.join(output_dir, "images")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        os.mkdir(annotation_dir)
        os.mkdir(images_dir)
    else:
        if not os.path.exists(annotation_dir):
            os.mkdir(annotation_dir)
        if not os.path.exists(images_dir):
            os.mkdir(images_dir)

    coco_annotations = COCO(SRC_COCO_ANNOTATION_PATH)
    print(f"[INFO] Origin dataset annotations size is {len(coco_annotations.dataset['annotations'])}")
    print(f"[INFO] Origin dataset images size is {len(coco_annotations.dataset['images'])}")

    if copy_images:
        extract_images(coco_annotations, images_dir)
    if copy_annotations:
        extract_annotation(coco_annotations, annotation_dir)

    print(f"[INFO] finished all processes..")


def extract_images(coco_annotations:COCO, images_dir:str):

    print("\n====================================")
    print_bool = True
    print(f"[INFO] Image copy process...")
    for key in pick_category.keys():
        catToImgs = coco_annotations.catToImgs[key]
        imgs_len = len(catToImgs)
        print(f"[INFO] category key is {key}, total size is {imgs_len}...")
        loadImgs = coco_annotations.loadImgs(ids=catToImgs)

        iteration = 0
        for img_info in loadImgs:
            file_name = img_info['file_name']
            src_file_path = os.path.join(SRC_COCO_IMAGE_DIR, file_name)
            dst_file_path = os.path.join(images_dir, file_name)

            shutil.copyfile(src_file_path, dst_file_path)
            iteration += 1
            p = (iteration / imgs_len) * 100
            if int(p) % 10 == 0 and int(p) != 0:
                if print_bool:
                    print_bool = False
                    print(f"[INFO] {iteration}/{imgs_len} ({round(p, 4)}%)image processed...")
            else:
                print_bool = True

    print(f"[INFO] finished images copy processes..")


def extract_annotation(coco_annotations:COCO, annotation_dir:str):

    dataset = dict()
    dataset['info'] = coco_annotations.dataset['info']
    dataset['licenses'] = coco_annotations.dataset['licenses']
    dataset['categories'] = coco_annotations.loadCats(ids=pick_category.keys())

    print("\n====================================")
    print(f"[INFO] Annotation copy process...")
    img_ids = []
    pick_category_key = pick_category.keys()
    for key in pick_category_key:
        catToImgs = coco_annotations.catToImgs[key]
        img_ids.extend(catToImgs)
        print(f"[INFO] category key is {key}, total size is {len(catToImgs)}...")

    img_ids = list(set(img_ids))
    loadImgs = coco_annotations.loadImgs(ids=img_ids)
    loadAnns = []
    for id in img_ids:
        for annotation in coco_annotations.imgToAnns[id]:
            if annotation['category_id'] in pick_category_key:
                loadAnns.append(annotation)

    print(f"[INFO] annotations size is {len(loadAnns)}...")
    print(f"[INFO] images size is {len(loadImgs)}...")

    dataset['images'] = loadImgs
    dataset['annotations'] = loadAnns

    print(f"[INFO] save annotation file {annotation_dir}/{annotation_file}...")
    with open(os.path.join(annotation_dir, annotation_file), "w") as f:
        json.dump(dataset, f)

    print(f"[INFO] finished annotations copy processes..")


if __name__ == "__main__":
    main()



