// DemoPytorch.cpp : 애플리케이션의 진입점을 정의합니다.
//

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <fstream>
#include <string>
#include <cassert>

namespace utils {
	template <typename T_, typename TI_>
	struct CMax;

	// traits of minheaps = heaps where the minimum value is stored on top
	// useful to find the *max* values of an array
	template <typename T_, typename TI_>
	struct CMin {
		typedef T_ T;
		typedef TI_ TI;
		typedef CMax<T_, TI_> Crev;
		inline static bool cmp(T a, T b) {
			return a < b;
		}
		inline static T neutral() {
			return std::numeric_limits<T>::lowest();
		}
	};

	template <typename T_, typename TI_>
	struct CMax {
		typedef T_ T;
		typedef TI_ TI;
		typedef CMin<T_, TI_> Crev;
		inline static bool cmp(T a, T b) {
			return a > b;
		}
		inline static T neutral() {
			return std::numeric_limits<T>::max();
		}
	};

	template <typename C>
	struct HeapArray {
		typedef typename C::TI TI;
		typedef typename C::T T;

		size_t nh;    ///< number of heaps
		size_t k;     ///< allocated size per heap
		TI* ids;     ///< identifiers (size nh * k)
		T* val;      ///< values (distances or similarities), size nh * k

		/// Return the list of values for a heap
		T* get_val(size_t key) { return val + key * k; }

		/// Correspponding identifiers
		TI* get_ids(size_t key) { return ids + key * k; }

		/// prepare all the heaps before adding
		void heapify();

		/** add nj elements to heaps i0:i0+ni, with sequential ids
		 *
		 * @param nj    nb of elements to add to each heap
		 * @param vin   elements to add, size ni * nj
		 * @param j0    add this to the ids that are added
		 * @param i0    first heap to update
		 * @param ni    nb of elements to update (-1 = use nh)
		 */
		void addn(size_t nj, const T* vin, TI j0 = 0,
			size_t i0 = 0, int64_t ni = -1);

		/** same as addn
		 *
		 * @param id_in     ids of the elements to add, size ni * nj
		 * @param id_stride stride for id_in
		 */
		void addn_with_ids(
			size_t nj, const T* vin, const TI* id_in = nullptr,
			int64_t id_stride = 0, size_t i0 = 0, int64_t ni = -1);

		/// reorder all the heaps
		void reorder();

		/** this is not really a heap function. It just finds the per-line
		 *   extrema of each line of array D
		 * @param vals_out    extreme value of each line (size nh, or NULL)
		 * @param idx_out     index of extreme value (size nh or NULL)
		 */
		void per_line_extrema(T* vals_out, TI* idx_out) const;

	};


	typedef HeapArray<CMax<float, int64_t> > float_maxheap_array_t;

	/*******************************************************************
	*Basic heap ops : push and pop
	******************************************************************* /

	/** Pops the top element from the heap defined by bh_val[0..k-1] and
	 * bh_ids[0..k-1].  on output the element at k-1 is undefined.
	 */
	template <class C> inline
		void heap_pop(size_t k, typename C::T* bh_val, typename C::TI* bh_ids)
	{
		bh_val--; /* Use 1-based indexing for easier node->child translation */
		bh_ids--;
		typename C::T val = bh_val[k];
		size_t i = 1, i1, i2;
		while (1) {
			i1 = i << 1;
			i2 = i1 + 1;
			if (i1 > k)
				break;
			if (i2 == k + 1 || C::cmp(bh_val[i1], bh_val[i2])) {
				if (C::cmp(val, bh_val[i1]))
					break;
				bh_val[i] = bh_val[i1];
				bh_ids[i] = bh_ids[i1];
				i = i1;
			}
			else {
				if (C::cmp(val, bh_val[i2]))
					break;
				bh_val[i] = bh_val[i2];
				bh_ids[i] = bh_ids[i2];
				i = i2;
			}
		}
		bh_val[i] = bh_val[k];
		bh_ids[i] = bh_ids[k];
	}



	/** Pushes the element (val, ids) into the heap bh_val[0..k-2] and
	 * bh_ids[0..k-2].  on output the element at k-1 is defined.
	 */
	template <class C> inline
		void heap_push(size_t k,
			typename C::T* bh_val, typename C::TI* bh_ids,
			typename C::T val, typename C::TI ids)
	{
		bh_val--; /* Use 1-based indexing for easier node->child translation */
		bh_ids--;
		size_t i = k, i_father;
		while (i > 1) {
			i_father = i >> 1;
			if (!C::cmp(val, bh_val[i_father]))  /* the heap structure is ok */
				break;
			bh_val[i] = bh_val[i_father];
			bh_ids[i] = bh_ids[i_father];
			i = i_father;
		}
		bh_val[i] = val;
		bh_ids[i] = ids;
	}

	template <typename T> inline
		void maxheap_pop(size_t k, T* bh_val, int64_t* bh_ids)
	{
		heap_pop<CMax<T, int64_t> >(k, bh_val, bh_ids);
	}


	template <typename T> inline
		void maxheap_push(size_t k, T* bh_val, int64_t* bh_ids, T val, int64_t ids)
	{
		heap_push<CMax<T, int64_t> >(k, bh_val, bh_ids, val, ids);
	}



	/*******************************************************************
	 * Heap initialization
	 *******************************************************************/

	 /* Initialization phase for the heap (with unconditionnal pushes).
	  * Store k0 elements in a heap containing up to k values. Note that
	  * (bh_val, bh_ids) can be the same as (x, ids) */
	template <class C> inline
		void heap_heapify(
			size_t k,
			typename C::T* bh_val,
			typename C::TI* bh_ids,
			const typename C::T* x = nullptr,
			const typename C::TI* ids = nullptr,
			size_t k0 = 0)
	{
		if (k0 > 0) assert(x);

		if (ids) {
			for (size_t i = 0; i < k0; i++)
				heap_push<C>(i + 1, bh_val, bh_ids, x[i], ids[i]);
		}
		else {
			for (size_t i = 0; i < k0; i++)
				heap_push<C>(i + 1, bh_val, bh_ids, x[i], i);
		}

		for (size_t i = k0; i < k; i++) {
			bh_val[i] = C::neutral();
			bh_ids[i] = -1;
		}

	}
	template <typename T> inline
		void maxheap_heapify(
			size_t k,
			T* bh_val,
			int64_t* bh_ids,
			const T* x = nullptr,
			const int64_t* ids = nullptr,
			size_t k0 = 0)
	{
		heap_heapify< CMax<T, int64_t> >(k, bh_val, bh_ids, x, ids, k0);
	}

	/*******************************************************************
	 * Heap finalization (reorder elements)
	 *******************************************************************/


	 /* This function maps a binary heap into an sorted structure.
		It returns the number  */
	template <typename C> inline
		size_t heap_reorder(size_t k, typename C::T* bh_val, typename C::TI* bh_ids)
	{
		size_t i, ii;

		for (i = 0, ii = 0; i < k; i++) {
			/* top element should be put at the end of the list */
			typename C::T val = bh_val[0];
			typename C::TI id = bh_ids[0];

			/* boundary case: we will over-ride this value if not a true element */
			heap_pop<C>(k - i, bh_val, bh_ids);
			bh_val[k - ii - 1] = val;
			bh_ids[k - ii - 1] = id;
			if (id != -1) ii++;
		}
		/* Count the number of elements which are effectively returned */
		size_t nel = ii;

		memmove(bh_val, bh_val + k - ii, ii * sizeof(*bh_val));
		memmove(bh_ids, bh_ids + k - ii, ii * sizeof(*bh_ids));

		for (; ii < k; ii++) {
			bh_val[ii] = C::neutral();
			bh_ids[ii] = -1;
		}
		return nel;
	}

	template <typename T> inline
		size_t maxheap_reorder(size_t k, T* bh_val, int64_t* bh_ids)
	{
		return heap_reorder< CMax<T, int64_t> >(k, bh_val, bh_ids);
	}
}


struct Index {

	int d;                 ///< vector dimension
	int64_t ntotal;          ///< total nb of indexed vectors

	std::vector<float> xb;
	std::vector<std::string> fnames;

	explicit Index(int d = 0) :
		d(d),
		ntotal(0) {}

	/**************************************************
	param n			: query 벡터의 개수
	param x			: query 벡터
	param k			: 찾을 이미지의 개수
	param distances	: query와 찾은 이미지 벡터의 거리
	param labels	: 찾은 이미지의 인덱스
	****************************************************/
	void search(int n, const float* x, int k, float* distances, int64_t* labels)
	{
		utils::float_maxheap_array_t res = {
			size_t(n), size_t(k), labels, distances };
		knn_L2sqr(x, xb.data(), d, n, ntotal, &res);
	}

	/**************************************************
	param fvecs : feature map 파일의 이름
	param name	: 이미지 파일의 이름을 저장한 파일
	param n		: 이미지의 개수
	****************************************************/
	void load(const char* fvecs, const char* name, int64_t n)
	{
		std::ifstream data_file;      // pay attention here! ofstream
		data_file.open(fvecs, std::ios::in | std::ios::binary);
		xb.resize(n * d);
		data_file.read(reinterpret_cast<char*>(&xb[0]), n * d * sizeof(float));
		data_file.close();
		ntotal = n;

		std::string line;
		std::fstream fs;
		fs.open(name, std::fstream::in);
		while (getline(fs, line)) {
			fnames.push_back(line);
		}
		fs.close();
	}

	/**************************************************
	param fvecs : feature map 파일의 이름
	param name	: 이미지 파일의 이름을 저장한 파일
	param n		: 이미지의 개수
	****************************************************/
	std::vector<std::string> get_filename(int64_t* I, int k)
	{
		std::vector<std::string> names;

		for (int i = 0; i < k; i++)
			names.push_back(fnames[I[i]]);

		return names;
	}

	/**************************************************
	param x		: query feature 벡터
	param y		: 찾을 이미지들의 feature 벡터
	param d		: vector의 dimension
	param nx	: query벡터의 개수
	param ny	: 찾을 이미지들의 feature 벡터 개수
	param res	: 결과를 담는 heap array
	****************************************************/
	void knn_L2sqr(
		const float* x,
		const float* y,
		size_t d, size_t nx, size_t ny,
		utils::float_maxheap_array_t* res) {
		size_t k = res->k;
		for (int64_t i = 0; i < nx; i++) {
			const float* x_i = x + i * d;
			const float* y_j = y;
			size_t j;
			float* simi = res->get_val(i);
			int64_t* idxi = res->get_ids(i);

			utils::maxheap_heapify(k, simi, idxi);
			for (j = 0; j < ny; j++) {
				float disij = fvec_L2sqr(x_i, y_j, d);

				if (disij < simi[0]) {
					utils::maxheap_pop(k, simi, idxi);
					utils::maxheap_push(k, simi, idxi, disij, j);
				}
				y_j += d;
			}
			utils::maxheap_reorder(k, simi, idxi);
		}
	}
	/**************************************************
	param x : 벡터
	param y	: 벡터
	param d	: vector의 dimension
	****************************************************/
	float fvec_L2sqr(const float* x, const float* y, size_t d)
	{
		float sum = 0;
		for (int64_t i = 0; i < d; i++) {
			float dst = (x[i] - y[i]);
			sum += dst * dst;
		}
		return sqrt(sum);
	}
};


std::vector<std::string> image_retrieval(
	cv::Mat img);
at::Tensor to_tensor(cv::Mat src);

int main()
{
	cv::Mat img;
	img = cv::imread("test1.jpg");
	std::vector<std::string> image_names = image_retrieval(img);

	for (int i = 0; i < image_names.size(); i++) {
		std::cout << image_names[i] << "\n";
	}

	return 0;
}

std::vector<std::string> image_retrieval(
	cv::Mat img) {

	torch::jit::script::Module model;
	try {
		// torch::jit::load()을 사용해 ScriptModule을 파일로부터 역직렬화
		model = torch::jit::load("resnet50_model.pt");
		model.eval();
	}
	catch (const c10::Error& e) {
		std::cerr << e.msg() << "\n";
		std::cerr << "error loading the model\n";
		return std::vector<std::string>();
	}

	int k = 4;
	std::vector<std::string> image_names;

	Index index(1024);
	index.load("vecs.bin", "vecs_names.txt", 1791);

	// Create Input
	at::Tensor input_tensor = to_tensor(img);
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(input_tensor);

	// Run model
	auto output = model.forward(inputs).toTensor();
	float* feature = output.data<float>();

	// find similar image name
	int64_t* I = new int64_t[k];
	float* D = new float[k];
	index.search(1, feature, k, D, I);
	image_names = index.get_filename(I, k);

	return image_names;
}

at::Tensor to_tensor(cv::Mat _src)
{
	cv::Mat src = _src.clone();
	cv::cvtColor(src, src, cv::COLOR_BGR2RGB);

	cv::Mat img;
	src.convertTo(img, CV_32FC3, (1.0 / 255.0));
	cv::resize(img, img, cv::Size(224, 224));

	cv::Mat rgb[3];
	cv::split(img, rgb);

	float* data = new float[3 * 224 * 224];
	memcpy(data, rgb[0].data, 224 * 224 * sizeof(float));
	memcpy(data + 224 * 224, rgb[1].data, 224 * 224 * sizeof(float));
	memcpy(data + 2 * 224 * 224, rgb[2].data, 224 * 224 * sizeof(float));

	float mean[] = { 0.485, 0.456, 0.406 };
	float std[] = { 0.229, 0.224, 0.225 };
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 224 * 224; j++) {
			data[224 * 224 * i + j] = (data[224 * 224 * i + j] - mean[i]) / std[i];
		}
	}

	auto tensor = torch::from_blob(data, { 1, 3, 224, 224 }, torch::kFloat32);

	return tensor;
}