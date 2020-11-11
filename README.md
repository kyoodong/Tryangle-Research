# 곰손 카메라 이미지 처리 모듈
#Requirement
- Python 3.7

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

## background_classification
배경을 인식하여 분류하는 문제를 해결하기 위함  
추천 이미지를 제공할 때 배경이 비슷한 사진을 우선적으로 제공하는 기능에 사용됨  

## human_pose
사람 관절의 위치를 파악하여 가이드를 제공하기 위함  

## ptyolact
빠른 이미지 세그멘테이션을 위해 Mask RCNN을 대신하여 차용된 알고리즘

## retrieval
장소 기반 가이드 이미지를 제공할 때 비슷하게 생긴 이미지를 우선적으로 제공하기 위함  

## retrieval_mobile
모바일에서 빠른 구동을 위해 캐시 이미지 내에서 검색하는 모듈  

## Opencv
Opencv 모듈을 테스트하고자 만든 프로젝트  

