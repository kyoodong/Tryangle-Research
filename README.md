# 곰손 카메라 이미지 처리 모듈
## DataFormatConverter
skyfinder, ade20k 데이터 셋을 coco 데이터 셋으로 변환하기 위한 프로젝트  
### 필요 데이터
- datasets (dir)
- datasets/ADE20K_2016_07_26 (dir)
- datasets/ADE20K_2016_07_26/images (ade20k 데이터셋)
- datasets/coco_annotations (dir)
- datasets/coco_annotations/instances_train2017.json
- datasets/coco_annotations/instances_val2017.json
- datasets/skyfinder (dir)
- datasets/skyfinder/images (dir)
- datasets/skyfinder/images/*.jpg (all of skyfinder images)
- datasets/skyfinder/masks (dir)
- datasets/skyfinder/images/*.png (all of skyfinder masks)

coco_annotations 은 기존의 coco annotation으로 skyfinder 혹은 ade20k 데이터셋이 이 기존 어노테이션과 합쳐진 어노테이션이 결과로 생겨남  
