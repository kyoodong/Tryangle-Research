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

## InstagramCrawling
인스타그램 이미지를 크롤링하기 위한 프로젝트.
크롤링하고자 하는 해시태그를 배열로서 적어두면 크롬 브라우저를 통해 스크롤하며 자동으로 이미지를 크롤링함

## MaskRCNN
MaskRCNN을 직접구 현하고자 만든 프로젝트 (현재는 시간 상 구현 불가로 판단)  
  
## Opencv
Opencv 모듈을 테스트하고자 만든 프로젝트  

## aktwelve_mask_rcnn
https://github.com/akTwelve/Mask_RCNN
해당 공개 소스를 인용하여 작업중인 프로젝트  
현재 TriAngle 의 핵심 소스가 구현된 프로젝트이다.  

main.py : 이미지를 선택하면 해당 이미지에 대한 가이드를 제공하는 파일  
coco.py : MaskRCNN을 Finetuning 하는 파일. 현재 하늘, 땅, 바다를 추가 학습 중임  
