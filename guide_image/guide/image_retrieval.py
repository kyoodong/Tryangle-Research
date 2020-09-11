import os
import time
import math
import numpy as np
from sklearn.preprocessing import normalize
import faiss


class ImageRetrieval:
    def __init__(self,
                 fvec_file,
                 fvec_img_file_name):
        '''
        :param feature_file: binary형태의 image_feature 파일
        :param feature_file_name: 각 feature에 맞는 이미지 이름 정보
        '''
        
        with open(fvec_img_file_name) as f:
            self.fname = f.readlines()

        #feature dimention
        dim = 1280
        # indexing 하는 방법
        index_type = 'hnsw'
        # index_type = 'l2'

        # index file name
        index_file = f'{fvec_file}.{index_type}.index'

        # 메모리에 모든 정보를 올리면 메모리가 꽉 차니까 np.memmap으로 정보 불러오기
        fvecs = np.memmap(fvec_file, dtype='float32', mode='r').view('float32').reshape(-1, dim)

        self.index = None
        # 인덱스 파일이 있으면 불러오고 없으면 생성
        if os.path.exists(index_file):
            self.index = faiss.read_index(index_file)
            if index_type == 'hnsw':
                self.index.hnsw.efSearch = 256
        else:
            self.index = self.__get_index(index_type, dim)
            self.index = self.__populate(self.index, fvecs)
            faiss.write_index(self.index, index_file)
        print("index n total", self.index.ntotal)

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
        raise

    def __populate(self, index, fvecs, batch_size=1000):
        nloop = math.ceil(fvecs.shape[0] / batch_size)
        for n in range(nloop):
            s = time.time()
            index.add(normalize(fvecs[n * batch_size: min((n + 1) * batch_size, fvecs.shape[0])]))
            print(n * batch_size, time.time() - s)

        return index
