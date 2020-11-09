import os
import time
import math
import numpy as np
from sklearn.preprocessing import normalize
import faiss

from retrieval.feature.tf_extractor import (
    preprocess,
    MODEL as fmodel,
    DIMENTION
)

def certain_retrieval(query_image,
                      image_names,
                      feature_dir="feature/output"):
    query_image = preprocess(query_image)
    query_fvec = fmodel.predict(query_image)

    fvecs = []
    for image_name in image_names:
        fvec_file = os.path.join(feature_dir, f"{image_name}.feature")
        fvec = np.memmap(fvec_file, dtype='float32', mode='r').view('float32').reshape(DIMENTION)
        fvecs.append(fvec)
    fvecs = np.array(fvecs)

    index = faiss.IndexFlatL2(DIMENTION)
    index.add(normalize(fvecs))
    dsts, idxs = index.search(normalize(query_fvec), len(image_names))

    return_names = []
    for idx in idxs[0]:
        return_names.append(image_names[idx])

    return return_names



class ImageRetrieval:
    def __init__(self,
                 fvec_file: str,
                 fvec_img_file_name: str,
                 fvec_dim: int,
                 index_type='hnsw'):
        '''
        :param fvec_file: binary형태의 image_feature 파일
        :param fvec_img_file_name: 각 feature에 맞는 이미지 이름 정보
        :param fvec_dim: feature vector의 dimention 크기
        :param index_type: indexing하는 방법, ( hnsw, l2, IVFFlat, IVFPQ)

        ----------------------------------------------
        약 10만장 기준으로 해본 결과
        hnsw    | sec: 0.0054초 정도 | mem: 975KB 정도
        l2      | sec: 0.0398초 정도 | mem: 945KB 정도
        IVFFlat | sec: 0.0036초 정도 | mem: 943KB 정도
        IVFPQ   | sec: 0.0029초 정도 | mem: 455KB 정도
        ----------------------------------------------

        '''

        with open(fvec_img_file_name) as f:
            self.fname = f.readlines()

        # index file name
        index_file = f'{fvec_file}.{index_type}.index'

        # 메모리에 모든 정보를 올리면 메모리가 꽉 차니까 np.memmap으로 정보 불러오기
        fvecs = np.memmap(fvec_file, dtype='float32', mode='r').view('float32').reshape(-1, fvec_dim)

        self.index = None
        # 인덱스 파일이 있으면 불러오고 없으면 생성
        if os.path.exists(index_file):
            self.index = faiss.read_index(index_file)
            if index_type == 'hnsw':
                self.index.hnsw.efSearch = 256
        else:
            self.index = self.__get_index(index_type, fvec_dim)
            if index_type == 'IVFFlat' or index_type == 'IVFPQ':
                self.index = self.__train(self.index, fvecs)
            self.index = self.__populate(self.index, fvecs)
            faiss.write_index(self.index, index_file)
        print(f"[INFO]index n total {self.index.ntotal}")
        print(f"[INFO]index type: {index_type}")

    def search(self, query_fvecs, k=10):
        '''
        :param fvecs:
        :param k:
        :return: k개의 비슷한 이미지의 이미지이름, 거리를 반환해 줌 -> (k, (img_path, dst))
        '''
        dsts, idxs = self.index.search(normalize(query_fvecs), k)

        result = []
        for (dst, idx) in zip(dsts[0], idxs[0]):
            result.append((self.fname[idx].strip(), dst))

        return result

    def __get_index(self, index_type, dim):
        if index_type == 'hnsw':
            m = 48
            index = faiss.IndexHNSWFlat(dim, m)
            index.hnsw.efConstruction = 128
            return index
        elif index_type == 'l2':
            return faiss.IndexFlatL2(dim)
        elif index_type == 'IVFFlat':
            nlist = 100
            quantizer = faiss.IndexFlatL2(dim)  # the other index
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            return index
        elif index_type == 'IVFPQ':
            nlist = 100
            m = 8
            quantizer = faiss.IndexFlatL2(dim)  # the other index
            index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
            return index
        raise

    def __populate(self, index, fvecs, batch_size=1000):
        index.add(normalize(fvecs))
        # nloop = math.ceil(fvecs.shape[0] / batch_size)
        # for n in range(nloop):
        #     s = time.time()
        #     index.add(normalize(fvecs[n * batch_size: min((n + 1) * batch_size, fvecs.shape[0])]))
        #     print(n * batch_size, time.time() - s)

        return index

    def __train(self, index, fvecs):
        assert not index.is_trained
        index.train(normalize(fvecs))
        assert index.is_trained
        return index



if __name__ == "__main__":
    from PIL import Image
    image_dir = "../images"
    image_names = [f"test{i}.jpg"for i in range(1, 27)]
    query_img_path = os.path.join(image_dir, "asd.jpg")

    query_img = Image.open(query_img_path)
    query_img = np.array(query_img)

    certain_retrieval(query_img, image_names)
