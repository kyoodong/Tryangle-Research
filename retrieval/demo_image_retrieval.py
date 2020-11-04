import time
# import psutil
import os

import cv2
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model


from retrieval.feature_extractor import preprocess
from retrieval.image_retrieval import ImageRetrieval


def retrieval_res(image_path, binary_file, name_file, index_type, k):
    # query Image feature 뽑는 과정
    # 현재는 간단하게 Resnet152로 이미지의 feature를 뽑아내서 비교
    dim = 1280
    input_shape = (224, 224, 3)
    base = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                             include_top=False,
                                             weights='imagenet')
    base.trainable = False
    model = Model(inputs=base.input, outputs=layers.GlobalAveragePooling2D()(base.output))

    # 이미지 로드
    st = time.time()
    img = preprocess(image_path, input_shape)
    img = tf.reshape(img, (1,) + input_shape)
    print(f"[INFO] image load time {time.time() - st}")

    # 이미지에서 feature 뽑아내기
    st = time.time()
    fvec = model.predict(img)
    print(f"[INFO] feature extract time {time.time() - st}")

    # 이미지 검색 클래스 생성
    imageRetrieval = ImageRetrieval(fvec_file=binary_file,
                                    fvec_img_file_name=name_file,
                                    fvec_dim=dim,
                                    index_type=index_type)

    st = time.time()
    results = imageRetrieval.search(fvec, k=k)
    print(f"[INFO] image retrieval time {time.time() - st}")

    # 찾은 이미지들의 경로 출력 및 저장
    with open('retrieval_result.txt', 'w') as f:
        # index type save
        f.writelines(f"{index_type}\n")
        # query image path save
        f.writelines(f"{image_path}\n")

        # result image path save
        for path, dst in results:
            print(f"Image path: {path}, dst: {dst}")
            f.writelines(f"{path}\n")

    # 메모리 사용량 측정
    pid = os.getpid()
    # current_process = psutil.Process(pid)
    # current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2. ** 20
    # print(f"BEFORE CODE: Current memory KB   : {current_process_memory_usage_as_KB: 9.3f} KB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True,
                        help="Query Image")
    parser.add_argument("--features", required=True,
                        help='Features Binary File, .bin')
    parser.add_argument("--names", required=True,
                        help='Features image name File, .txt')
    parser.add_argument("--index", required=False,
                        default='hnsw',
                        help='Faiss Index type, hnsw or l2 or IVFFlat or IVFPQ')
    parser.add_argument("--k", required=False,
                        default=10,
                        help='k')
    args = parser.parse_args()

    retrieval_res(args.image, args.features, args.names, args.index, int(args.k))

