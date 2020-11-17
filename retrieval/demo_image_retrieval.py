import time
import tensorflow as tf

from retrieval.image_retrieval import ImageRetrieval
from retrieval.feature.tf_extractor import (
    preprocess,
    MODEL,
    DIMENTION,
    INPUT_SIZE
)

def retrieval(image_path, binary_file, name_file, index_type, k):
    # query Image feature 뽑는 과정
    # 현재는 간단하게 Resnet152로 이미지의 feature를 뽑아내서 비교

    # 이미지 로드
    st = time.time()

    img = preprocess(image_path)
    img = tf.reshape(img, (1,) + INPUT_SIZE)
    print(f"[INFO] image load time {time.time() - st}")

    # 이미지에서 feature 뽑아내기
    st = time.time()
    fvec = MODEL.predict(img)
    print(f"[INFO] feature extract time {time.time() - st}")

    # 이미지 검색 클래스 생성
    imageRetrieval = ImageRetrieval(fvec_file=binary_file,
                                    fvec_img_file_name=name_file,
                                    fvec_dim=DIMENTION,
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
    # pid = os.getpid()
    # current_process = psutil.Process(pid)
    # current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2. ** 20
    # print(f"BEFORE CODE: Current memory KB   : {current_process_memory_usage_as_KB: 9.3f} KB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True,
                        help="Query Image")
    parser.add_argument("--store_dir", required=False,
                        default="feature/output",
                        help='Features and Image Path store directory')
    parser.add_argument("--store", required=False,
                        default="fvecs",
                        help='Features and Image Path store File')
    parser.add_argument("--index", required=False,
                        default='l2',
                        help='Faiss Index type, hnsw or l2 or IVFFlat or IVFPQ')
    parser.add_argument("--k", required=False,
                        default=10,
                        help='k')
    args = parser.parse_args()

    binary_file = f"{args.store_dir}/{args.store}.bin"
    name_file = f"{args.store_dir}/{args.store}_names.txt"

    retrieval(args.image, binary_file, name_file, args.index, int(args.k))

