
#include <cublas_v2.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <string>
#include <sstream>
#include <vector>
#include <memory>

#include <curand_kernel.h>
#include <assert.h>

#include <thrust/device_vector.h>
#include <thrust/memory.h>


#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define IDX3C(i,j,k,rows,cols) (k*(rows)*(cols)+IDX2C(i,j,rows))

#define SIGMOID_0 0
#define LINEAR 1

#define LEARNING_RATE 0.029
#define SPARSITY .05f
#define SPARSITY_WEIGHT .05f


using namespace std;

cublasHandle_t handle;


class Autoencoder;
class Layer;
class Cube;
class Matrix;
class MatrixBatched;
class Vector;
class VectorBatched;
typedef MatrixBatched* matrixb_ptr;
typedef VectorBatched* vectorb_ptr;
typedef Layer* layer_ptr;
typedef Cube* cube_ptr;
typedef Matrix* matrix_ptr;
typedef Vector* vector_ptr;
typedef Autoencoder* autoencoder_ptr;

inline void check_cuda_errors(const char *filename, const int line_number)
{
      cudaDeviceSynchronize();
      cudaError_t error = cudaGetLastError();
      if(error != cudaSuccess) {
          printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
          exit(-1);
      }
}

class Cube {
public:
	int rows;
	int cols;
	int levels;

	thrust::device_vector<float> d_data;

	Cube(int _rows, int _cols, int _levels) : rows(_rows), cols(_cols), levels(_levels), d_data(rows*cols*levels) {
		thrust::fill(d_data.begin(), d_data.end(), 0.f);
	}

	void reset() {
		thrust::fill(d_data.begin(), d_data.end(), 0.f);
	}

	__host__ __device__ int numel() const { return rows*cols*levels; }

	virtual ~Cube() {
		d_data.clear();
		d_data.shrink_to_fit();
	}

	void copy(cube_ptr m) {
		thrust::copy(m->d_data.begin(),m->d_data.end(), d_data.begin());
	}

	void randomize() {
		for(int i=0;i<numel();i++) {
			d_data[i]=(((float)rand())/RAND_MAX)/10.;
		}
	}
};

class MatrixBatched : public Cube {
public:
	MatrixBatched(int _rows, int _cols, int _batchSize) : Cube(_rows, _cols, _batchSize) {
	}

	void print(int l) {
		thrust::host_vector<float> hv(d_data.begin(), d_data.end());
		for(int i=0;i<rows;i++) {
			for(int j=0;j<cols;j++) {
				cout << hv[IDX3C(i,j,l,rows,cols)] << ",";
			}
//			printf("\n");
		}
	}
	void _printf(int l) {
		thrust::host_vector<float> hv(d_data.begin(), d_data.end());
		for(int i=0;i<rows;i++) {
			for(int j=0;j<cols;j++) {
				printf("%02.3f,",hv[IDX3C(i,j,l,rows,cols)] );
			}
//			printf("\n");
		}
	}

	void println(int l) {
		thrust::host_vector<float> hv(d_data.begin(), d_data.end());
		for(int i=0;i<rows;i++) {
			for(int j=0;j<cols;j++) {
				printf("%f,",hv[IDX3C(i,j,l,rows,cols)] );
			}
			printf("\n");
		}
	}
};

class VectorBatched : public MatrixBatched {
public:
	VectorBatched(int _rows, int _cols, int _batchSize) : MatrixBatched(_rows, _cols, _batchSize) {
		assert(_rows==1 || _cols==1);
	}
};

class RowVectorBatched : public VectorBatched {
public:
	RowVectorBatched(int size, int batchSize) : VectorBatched(1, size, batchSize) {
	}
};

class ColVectorBatched : public VectorBatched {
public:
	ColVectorBatched(int size, int batchSize) : VectorBatched(size, 1, batchSize) {
	}
};

class Matrix : public MatrixBatched {
public:
	Matrix(int _rows, int _cols) : MatrixBatched(_rows,_cols,1) {
	}
	Matrix(int _size) : MatrixBatched(_size, _size, 1){
		for(int i=0;i<rows;i++)
			d_data[IDX2C(i,i,rows)]=1.f;
	}
	virtual ~Matrix() {
	}

	void print() {
		thrust::host_vector<float> hv(d_data.begin(), d_data.end());
		for(int i=0;i<rows;i++) {
			for(int j=0;j<cols;j++) {
				printf("%f,",hv[IDX2C(i,j,rows)]);
			}
//			printf("\n");
		}
	}
	void printd() {
		thrust::host_vector<float> hv(d_data.begin(), d_data.end());
		for(int i=0;i<rows;i++) {
			for(int j=0;j<cols;j++) {
				printf("%.0f,",hv[IDX2C(i,j,rows)]);
			}
//			printf("\n");
		}
	}


	
};

class Vector : public Matrix {
public:
	Vector(int rows, int cols): Matrix(rows, cols) {
	}	
	virtual ~Vector() {
	}
	virtual float operator[](const int idx) = 0;
	virtual void set(int idx,float a) = 0;
	void set(Vector &v) {
		for(int i=0;i<numel() && i<v.numel(); i++) {
			set(i, v[i]);
		}
	}
};

class ColVector : public Vector {
public:
	ColVector(int _size) : Vector(_size, 1) {
	}
	virtual ~ColVector() {
	}

	virtual float operator[](const int idx) {
			return d_data[IDX2C(idx, 0, rows)];
	}
	virtual void set(int idx, float a) {
		d_data[IDX2C(idx,0,rows)]=a;
	}
};

class RowVector : public Vector {
public:
	RowVector(int _size) : Vector(1,_size) {
	}
	virtual ~RowVector() {
	}
	virtual float operator[](const int idx) {
			return d_data[IDX2C(0,idx, rows)];
	}
	virtual void set(int idx, float a) {
		d_data[IDX2C(0,idx,rows)]=a;
	}
};

// ----------------------------------------------------------------------------
//                IO part
// ----------------------------------------------------------------------------


struct IO {
	int inputs, outputs, samples;
	const char *infile,*outfile;

    vectorb_ptr input;
    vectorb_ptr output;

		void read_i(const char* infile) {
			ifstream ifile(infile);

			for(int i=0;i<samples;i++) {
				string line;
				getline(ifile,line);
				istringstream is(line);
				char c;
				for(int p=0;p<inputs;p++) {
					float z=0;
					is>>z>>c;
					input->d_data[IDX3C(0,p,i,1,inputs)]=z;
				}
			}
		}

		void read_o(const char *outfile) {
			ifstream ofile(outfile);
				
			for(int i=0;i<samples;i++) {
				string line;
				getline(ofile,line);
				istringstream is(line);
				char c;
				for(int p=0;p<outputs;p++) {
					float z=0;
					is>>z>>c;
					output->d_data[IDX3C(0,p,i,1,outputs)]=z;
				}
			}
		}

		void read_io(const char* infile, const char* outfile) {
			read_i(infile);
			read_o(outfile);
		}


	IO(int _inputs, int _outputs, int _samples, const char* _infile, const char* _outfile) : inputs(_inputs), outputs(_outputs), samples(_samples),
		infile(_infile), outfile(_outfile) {
		input=new RowVectorBatched(inputs, samples);
		output=new RowVectorBatched(outputs, samples);
		read_io(infile,outfile);
	}
};



__device__ inline float sigmoid(float signal) {
	return 1./(1+exp(-1.*signal));
}

__device__ inline float sigmoid_derived(float signal) {
	float s=sigmoid(signal);
	return s*(1-s);
}

__global__ void cuSigmoid(int rows, int cols,int levels,float* v, float* output) {
	int l=blockIdx.x;

	while(l<levels) {
		int vIdx=threadIdx.x;
					while(vIdx<cols) {
						output[IDX3C(0,vIdx,l,rows,cols)]=sigmoid(v[IDX3C(0,vIdx,l,rows,cols)]);
						//assert(output[IDX3C(0,vIdx,l,rows,cols)]==output[IDX3C(0,vIdx,l,rows,cols)]);
						vIdx+=blockDim.x;
					}
			l+=gridDim.x;
	}
}

__global__ void cuSigmoidDerived(int rows, int cols, int levels, float* v, float* outputDerived) {

	int l=blockIdx.x;

	while(l<levels) {

		int vIdx=threadIdx.x;
					while(vIdx<cols) {
						outputDerived[IDX3C(vIdx,vIdx, l, cols, cols)]=sigmoid_derived(v[IDX3C(0,vIdx,l, rows,cols)]);
						vIdx+=blockDim.x;
					}
		l+=gridDim.x;
	}
}

void sigmoid(vectorb_ptr v, vectorb_ptr output, matrixb_ptr outputDerived, bool last) {
	cuSigmoid<<<16,16>>>(v->rows, v->cols, v->levels, thrust::raw_pointer_cast(v->d_data.data()), thrust::raw_pointer_cast(output->d_data.data()));
	cuSigmoidDerived<<<16,16>>>(v->rows, v->cols,v->levels, thrust::raw_pointer_cast(v->d_data.data()), thrust::raw_pointer_cast(outputDerived->d_data.data()));
}

__global__ void cuLinear(int rows, int cols, int levels, float* v, float* output) {

	int l=blockIdx.x;

	while(l<levels) {
		int vIdx=threadIdx.x;
					while(vIdx<cols) {
						output[IDX3C(0,vIdx,l,rows,cols)]=v[IDX3C(0,vIdx, l,rows,cols)];
						vIdx+=blockDim.x;
					}
		l+=gridDim.x;
	}
}

__global__ void cuLinearDerived(int rows, int cols, int levels, float* v, float* outputDerived) {
	
	int l=blockIdx.x;

	while(l<levels) {
	int vIdx=threadIdx.x;
					while(vIdx<cols) {
						outputDerived[IDX3C(vIdx,vIdx,l,cols,cols)]=1;
						vIdx+=blockDim.x;
					}
			l+=gridDim.x;
	}
}

void linear(vectorb_ptr v, vectorb_ptr output, matrixb_ptr outputDerived, bool last) {
/*
	for(int i=0;i<v->numel();i++) {
		outputDerived->d_data[IDX2C(i,i,outputDerived->rows)]=1;
		output->d_data[i]=v->d_data[i];
	}
*/
	cuLinear<<<16,16>>>(v->rows, v->cols, v->levels, thrust::raw_pointer_cast(v->d_data.data()),thrust::raw_pointer_cast(output->d_data.data()));
	cuLinearDerived<<<16,16>>>(v->rows, v->cols, v->levels, thrust::raw_pointer_cast(v->d_data.data()),thrust::raw_pointer_cast(outputDerived->d_data.data()));
/*
	if(!last) {
		output->d_data[v->numel()-1]=1.f;
		outputDerived->d_data[IDX2C(v->numel()-1,v->numel()-1,outputDerived->rows)]=0.f;
	}
*/
}

typedef void (*neuron_func_t)(vectorb_ptr, vectorb_ptr,matrixb_ptr,bool last);

neuron_func_t neuron_func[]={sigmoid,linear};


void cublasMul(float alpha,float beta, matrix_ptr m1,cublasOperation_t transa, matrix_ptr m2, cublasOperation_t transb, matrix_ptr p) {
	int check1=(transa==CUBLAS_OP_N?m1->cols:m1->rows);
	int check2=(transb==CUBLAS_OP_N?m2->rows:m2->cols);

	int lda=m1->rows;
	int ldb=m2->rows;

	//printf("%d %d\n", check1,check2);
	assert(check1==check2);

	p->rows=(transa==CUBLAS_OP_N?m1->rows:m1->cols);
	p->cols=(transb==CUBLAS_OP_N?m2->cols:m2->rows);


	cublasStatus_t status = cublasSgemm (handle, transa, transb, p->rows, p->cols, check1, &alpha, thrust::raw_pointer_cast (m1->d_data.data()), lda, thrust::raw_pointer_cast (m2->d_data.data()), ldb, &beta, thrust::raw_pointer_cast (p->d_data.data()), p->rows);
	if (status != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "!!!! kernel execution error.\n";
    }
}

inline void cublasMul(matrix_ptr m1, cublasOperation_t transa, matrix_ptr m2, cublasOperation_t transb, matrix_ptr p) {
	cublasMul(1.0f,0.f,m1, transa, m2, transb, p);
}
inline void cublasMul(matrix_ptr m1, matrix_ptr m2, matrix_ptr p) {
	cublasMul(1.0f,0.f,m1, CUBLAS_OP_N, m2, CUBLAS_OP_N, p);
}
inline void cublasMul(float alpha,float beta,matrix_ptr m1, matrix_ptr m2, matrix_ptr p) {
	cublasMul(alpha,beta,m1, CUBLAS_OP_N, m2, CUBLAS_OP_N, p);
}

inline void cublasMul(float alpha, float beta, matrixb_ptr m1, cublasOperation_t transa, matrix_ptr m2, cublasOperation_t transb, matrixb_ptr p) {
	int check1=(transa==CUBLAS_OP_N?m1->cols:m1->rows);
	int check2=(transb==CUBLAS_OP_N?m2->rows:m2->cols);

	int lda=m1->rows;
	int ldb=m2->rows;

	//printf("%d %d\n", check1,check2);
	assert(check1==check2);
	assert(m1->levels==p->levels);

	p->rows=(transa==CUBLAS_OP_N?m1->rows:m1->cols);
	p->cols=(transb==CUBLAS_OP_N?m2->cols:m2->rows);

	for(int l=0; l < m1->levels; l++ ) {
			cublasStatus_t status = cublasSgemm (handle, transa, transb, p->rows, p->cols, check1, &alpha,
				 thrust::raw_pointer_cast (&m1->d_data[IDX3C(0,0,l,m1->rows,m1->cols)]), lda, 
				 thrust::raw_pointer_cast (m2->d_data.data()), ldb, &beta,
				 thrust::raw_pointer_cast (&p->d_data[IDX3C(0,0,l,p->rows, p->cols)]), p->rows);
			if (status != CUBLAS_STATUS_SUCCESS) {
			  std::cerr << "!!!! kernel execution error.\n";
			}
	}

}

__global__ void cuMulMatrixBatched(float alpha, float beta, int transa, int transb, int m, int n, int k, int levels, float *a, float *b, float *c) {
	int level=blockIdx.x;

	while(level<levels) {
		int tx=threadIdx.x;
		int ty=threadIdx.y;
		float Pvalue=0;
		for(int i=0;i<n;i++) {
			float Mdelement=a[IDX3C(tx,i,level, m,n)];
			float Ndelement=b[IDX3C(i,ty,level, n,k)];
			Pvalue+=Mdelement*Ndelement;
		}
		c[IDX3C(tx,ty,level, m,k)]=Pvalue;

		level+=gridDim.x;
	}
}

inline void cublasMul(float alpha, float beta, matrixb_ptr m1, cublasOperation_t transa, matrixb_ptr m2, cublasOperation_t transb, matrixb_ptr p) {
	int check1=(transa==CUBLAS_OP_N?m1->cols:m1->rows);
	int check2=(transb==CUBLAS_OP_N?m2->rows:m2->cols);

	int lda=m1->rows;
	int ldb=m2->rows;

	//printf("%d %d\n", check1,check2);
	assert(check1==check2);
	assert(m1->levels==p->levels);
	assert(m1->levels==m2->levels);

	p->rows=(transa==CUBLAS_OP_N?m1->rows:m1->cols);
	p->cols=(transb==CUBLAS_OP_N?m2->cols:m2->rows);

	for(int l=0; l < m1->levels; l++ ) {
			cublasStatus_t status = cublasSgemm (handle, transa, transb, p->rows, p->cols, check1, &alpha,
				 thrust::raw_pointer_cast (&m1->d_data[IDX3C(0,0,l,m1->rows,m1->cols)]), lda, 
				 thrust::raw_pointer_cast (&m2->d_data[IDX3C(0,0,l,m2->rows,m2->cols)]), ldb, &beta,
				 thrust::raw_pointer_cast (&p->d_data[IDX3C(0,0,l,p->rows, p->cols)]), p->rows);
			if (status != CUBLAS_STATUS_SUCCESS) {
			  std::cerr << "!!!! kernel execution error.\n";
			}
	}
	//cuMulMatrixBatched<<<m1->levels,dim3(m1->rows,m2->cols)>>>(alpha,beta, transa, transb, m1->rows, m1->cols, m2->cols,m1->levels, thrust::raw_pointer_cast(m1->d_data.data()), thrust::raw_pointer_cast(m2->d_data.data()), thrust::raw_pointer_cast(p->d_data.data()));

}

inline void cublasMul(matrixb_ptr m1, cublasOperation_t transa, matrix_ptr m2, cublasOperation_t transb, matrixb_ptr p) {
	cublasMul(1.0f,0.f,m1, transa, m2, transb, p);
}
inline void cublasMul(matrixb_ptr m1, matrix_ptr m2, matrixb_ptr p) {
	cublasMul(1.0f,0.f,m1, CUBLAS_OP_N, m2, CUBLAS_OP_N, p);
}
inline void cublasMul(float alpha,float beta,matrixb_ptr m1, matrix_ptr m2, matrixb_ptr p) {
	cublasMul(alpha,beta,m1, CUBLAS_OP_N, m2, CUBLAS_OP_N, p);
}

inline void cublasMul(matrixb_ptr m1, cublasOperation_t transa, matrixb_ptr m2, cublasOperation_t transb, matrixb_ptr p) {
	cublasMul(1.0f,0.f,m1, transa, m2, transb, p);
}
inline void cublasMul(matrixb_ptr m1, matrixb_ptr m2, matrixb_ptr p) {
	cublasMul(1.0f,0.f,m1, CUBLAS_OP_N, m2, CUBLAS_OP_N, p);
}
inline void cublasMul(float alpha,float beta,matrixb_ptr m1, matrixb_ptr m2, matrixb_ptr p) {
	cublasMul(alpha,beta,m1, CUBLAS_OP_N, m2, CUBLAS_OP_N, p);
}

inline void cublasMul(float alpha, float beta, matrixb_ptr m1, cublasOperation_t transa, matrix_ptr m2, cublasOperation_t transb, matrix_ptr p) {
	int check1=(transa==CUBLAS_OP_N?m1->cols:m1->rows);
	int check2=(transb==CUBLAS_OP_N?m2->rows:m2->cols);

	int lda=m1->rows;
	int ldb=m2->rows;

	//printf("%d %d\n", check1,check2);
	assert(check1==check2);

	p->rows=(transa==CUBLAS_OP_N?m1->rows:m1->cols);
	p->cols=(transb==CUBLAS_OP_N?m2->cols:m2->rows);

	cublasStatus_t status = cublasSgemm (handle, transa, transb, p->rows, p->cols, check1, &alpha,
		 thrust::raw_pointer_cast (&m1->d_data[IDX3C(0,0,0,m1->rows,m1->cols)]), lda, 
		 thrust::raw_pointer_cast (m2->d_data.data()), ldb, &beta,
		 thrust::raw_pointer_cast (p->d_data.data()), p->rows);
	if (status != CUBLAS_STATUS_SUCCESS) {
	  std::cerr << "!!!! kernel execution error.\n";
	}

}

inline void cublasMul(matrixb_ptr m1, cublasOperation_t transa, matrix_ptr m2, cublasOperation_t transb, matrix_ptr p) {
	cublasMul(1.0f,0.f,m1, transa, m2, transb, p);
}
inline void cublasMul(matrixb_ptr m1, matrix_ptr m2, matrix_ptr p) {
	cublasMul(1.0f,0.f,m1, CUBLAS_OP_N, m2, CUBLAS_OP_N, p);
}
inline void cublasMul(float alpha,float beta,matrixb_ptr m1, matrix_ptr m2, matrix_ptr p) {
	cublasMul(alpha,beta,m1, CUBLAS_OP_N, m2, CUBLAS_OP_N, p);
}

inline void cublasAddSub(float alpha, float beta, matrixb_ptr m1, matrixb_ptr m2, matrixb_ptr p) {
	assert(m1->rows==m2->rows);
//	printf("%d %d\n",m1->cols, m2->cols);
	assert(m1->cols==m2->cols);

	int lda=m1->rows;
	int ldb=m2->rows;
	assert(m1->levels==m2->levels);
	assert(m1->levels==p->levels);

	for(int l=0; l < m1->levels; l++ ) {
		cublasStatus_t status=cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m1->rows, m2->cols, &alpha,
			 thrust::raw_pointer_cast(&m1->d_data[IDX3C(0,0,l, m1->rows, m1->cols)]), lda, &beta,
			 thrust::raw_pointer_cast(&m2->d_data[IDX3C(0,0,l, m2->rows, m2->cols)]), ldb, 
			 thrust::raw_pointer_cast(&p->d_data[IDX3C(0,0,l, p->rows,p->cols)]), p->rows);
		if (status != CUBLAS_STATUS_SUCCESS) {
			  std::cerr << "!!!! kernel execution error.\n";
		}
	}
}
inline void cublasSub(matrixb_ptr m1, matrixb_ptr m2, matrixb_ptr p) {

	float alpha=1.f;
	float beta=-1.f;

	cublasAddSub(alpha, beta, m1, m2, p);
}

inline void cublasAdd(matrixb_ptr m1, matrixb_ptr m2, matrixb_ptr p) {
	float alpha=1.f;
	float beta=1.f;
	cublasAddSub(alpha, beta, m1, m2, p);
}

inline void cublasAddSub(float alpha, float beta, matrix_ptr m1, matrix_ptr m2, matrix_ptr p) {
	assert(m1->rows==m2->rows);
//	printf("%d %d\n",m1->cols, m2->cols);
	assert(m1->cols==m2->cols);

	int lda=m1->rows;
	int ldb=m2->rows;
	cublasStatus_t status=cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m1->rows, m2->cols, &alpha, thrust::raw_pointer_cast(m1->d_data.data()), lda, &beta, thrust::raw_pointer_cast(m2->d_data.data()), ldb, thrust::raw_pointer_cast(p->d_data.data()), p->rows);
}

inline void cublasSub(matrix_ptr m1, matrix_ptr m2, matrix_ptr p) {

	

/*
	assert(m1->rows==m2->rows);
	assert(m1->cols==m2->cols);

	int lda=m1->rows;
	int ldb=m2->rows;
*/
	float alpha=1.f;
	float beta=-1.f;

	cublasAddSub(alpha, beta, m1, m2, p);
//	cublasStatus_t status=cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m1->rows, m2->cols, &alpha, thrust::raw_pointer_cast(m1->d_data.data()), lda, &beta, thrust::raw_pointer_cast(m2->d_data.data()), ldb, thrust::raw_pointer_cast(p->d_data.data()), p->rows);
}

inline void cublasAdd(matrix_ptr m1, matrix_ptr m2, matrix_ptr p) {

/*
	assert(m1->rows==m2->rows);
//	printf("%d %d\n",m1->cols, m2->cols);
	assert(m1->cols==m2->cols);

	int lda=m1->rows;
	int ldb=m2->rows;
*/
	float alpha=1.f;
	float beta=1.f;
	cublasAddSub(alpha, beta, m1, m2, p);

//	cublasStatus_t status=cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m1->rows, m2->cols, &alpha, thrust::raw_pointer_cast(m1->d_data.data()), lda, &beta, thrust::raw_pointer_cast(m2->d_data.data()), ldb, thrust::raw_pointer_cast(p->d_data.data()), p->rows);
}

inline void cublasMul(float alpha, vector_ptr v) {
	cublasStatus_t status=cublasSscal(handle, v->numel(), &alpha, thrust::raw_pointer_cast(v->d_data.data()), 1);
	if (status != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "!!!! kernel execution error.\n";
    }
}

__global__ void cuMulElementwiseT(int rows, int cols, float *a, float *b, float *c) {
	int row=threadIdx.x;
	int col=threadIdx.y;

	while(col<cols) {
		while(row<rows) {
			c[IDX2C(col,row,rows)]=a[IDX2C(row,col,rows)]*b[IDX2C(row,col,rows)];
			row+=blockDim.x;
		}
		col+=blockDim.y;
	}
}

void mulElementwiseT(matrix_ptr m1, matrix_ptr m2, matrix_ptr p) {
	assert(m1->rows==m2->rows);
	assert(m1->cols==m2->cols);
	assert(m1->rows==p->cols);
	assert(m1->cols==p->rows);

	cuMulElementwiseT<<<1,dim3(16,16)>>>(m1->rows, m1->cols, thrust::raw_pointer_cast(m1->d_data.data()), thrust::raw_pointer_cast(m2->d_data.data()), thrust::raw_pointer_cast(p->d_data.data()));
}

__global__ void cuSum(int rows, int cols, int levels, float *a) {
	int row=threadIdx.x;

	while(row<rows) {
		int col=threadIdx.y; 

		while(col<cols) {
			for(int k=1;k<levels;k++) {
				a[IDX3C(row,col,0,rows,cols)]+= a[IDX3C(row,col,k,rows,cols)];
			}
/*
			printf("levels %d\n", levels);
			for(int stride=levels>>1;stride>0;stride>>=1) {
				__syncthreads();
//				printf("stride: %d\n", stride);
				if(threadIdx.z<stride) {
					a[IDX3C(row,col,threadIdx.z,rows,cols)]+=a[IDX3C(row,col,threadIdx.z+stride,rows,cols)];
				}
			}
*/
			col+=blockDim.y;
		}
		row+=blockDim.x;
	}
}

void mySum(matrixb_ptr m) {

//	cout << m->rows << " " << m->cols << " " << m->levels << endl;
	cuSum<<<1,dim3( m->rows, m->cols)>>>(m->rows, m->cols, m->levels, thrust::raw_pointer_cast(m->d_data.data()));
	check_cuda_errors(__FILE__, __LINE__);


/*
	for(int i=0;i<m->rows;i++)
		for(int j=0;j<m->cols;j++) {
			for(int k=1;k<m->levels;k++) {
				m->d_data[IDX3C(i,j,0,m->rows,m->cols)]+= m->d_data[IDX3C(i,j,k,m->rows,m->cols)];
			}
		}
*/
}



class Layer {
public:
	int neuronType;
	int neuronNum;
	vector_ptr bias;
	vectorb_ptr activation;
	float activationNum;
	int batchSize;

	layer_ptr nextLayer;
	layer_ptr prevLayer;
	matrix_ptr nextMatrix;
	matrix_ptr prevMatrix;
	
	vectorb_ptr output;
	matrixb_ptr outputDerived;
	vectorb_ptr delta;
	vector_ptr sparsity;
	vectorb_ptr error_derived;
	vectorb_ptr mse;
	matrix_ptr unit;
	vectorb_ptr input;
	matrixb_ptr weightAdj;
	matrixb_ptr mTmp;


	Layer(int _neuronNum, int _neuronType, int _batchSize, bool _bias=true) : neuronNum(_neuronNum), neuronType(_neuronType),  batchSize(_batchSize), nextLayer(0), prevLayer(0), weightAdj(0), mTmp(0) {
		cout << __PRETTY_FUNCTION__ << _neuronNum << endl;
		activationNum=0;

		output=new RowVectorBatched(neuronNum, batchSize);
		input=new RowVectorBatched(neuronNum, batchSize);

		bias=new RowVector(neuronNum);
		outputDerived=new MatrixBatched(neuronNum, neuronNum, batchSize);
			
		activation=new RowVectorBatched(neuronNum,batchSize);
		delta=new ColVectorBatched(neuronNum, batchSize);
		sparsity=new ColVector(neuronNum);
		error_derived=new RowVectorBatched(neuronNum,batchSize);
		mse=new RowVectorBatched(1,batchSize);
		unit=new Matrix(neuronNum);

	}	

	virtual ~Layer() {
		delete  output;
		cout << __PRETTY_FUNCTION__ << neuronNum << endl;
	}

	void excite() {
		if(prevLayer!=0) {
			cublasMul(prevLayer->output, prevMatrix, input);
			//cublasAdd(input, bias, input);
			neuron_func[neuronType](input,output, outputDerived,nextLayer==0);
/*
cout << "output////////////////////////" << endl;
			outputDerived->println(1);
cout << "////////////////////////" << endl;
*/
			activation->copy(output);
			mySum(activation);
			activationNum+=batchSize;
		}
		if(nextLayer!=0)
			nextLayer->excite();
	}
	layer_ptr error(vectorb_ptr desiredOutput) {
		if(prevLayer!=0) {
			cublasSub(output,desiredOutput,error_derived);

			cublasMul(error_derived, CUBLAS_OP_N, error_derived, CUBLAS_OP_T, mse);
			mySum(mse);
/*
cout << "*-------------------------" << endl;
			error_derived->print(3);
cout << "*-------------------------" << endl;
*/
			cublasMul(outputDerived, CUBLAS_OP_N, error_derived, CUBLAS_OP_T, delta);
//			mulElementwiseT(outputDerived, error_derived, delta);
			
		}
		return this;
	}

	void backpropagation() {
		if(nextLayer!=0 && prevLayer!=0) {
			MatrixBatched m(outputDerived->rows, nextMatrix->cols,batchSize);
			cublasMul(outputDerived, CUBLAS_OP_N, nextMatrix, CUBLAS_OP_N, &m);
			cublasMul(&m,nextLayer->delta, delta);
		}
		if(prevLayer!=0)
			prevLayer->backpropagation();
	}

	void backpropagationSparse() {
		if(nextLayer!=0 && prevLayer!=0) {
			if(mTmp==0) {
				mTmp=new MatrixBatched(outputDerived->rows, nextMatrix->cols, batchSize);
			}
			cublasMul(outputDerived, CUBLAS_OP_N, nextMatrix, CUBLAS_OP_N, mTmp);
			cublasMul(mTmp,nextLayer->delta, delta);

//			cublasMul(-1./SPARSITY, activation);

			for(int i=0;i<sparsity->numel();i++) {
				float a=activation->d_data[IDX3C(0,i,0,activation->rows, activation->cols)]/activationNum;
				sparsity->d_data[i]=-(SPARSITY/a) + (1-SPARSITY)/(1-a);
//				cout << a << ",";
			}
//			cout << endl;
/*
			cout << "sparsity ";
			sparsity->print();
			cout << endl;
*/

			cublasMul(SPARSITY_WEIGHT, 1.f, outputDerived, CUBLAS_OP_N, sparsity, CUBLAS_OP_N, delta);
		}
		if(prevLayer!=0)
			prevLayer->backpropagationSparse();
	}
	
	void calculateSparseActivation() {
#if 1
			if(activationNum!=0) {
//				cublasMul(1./activationNum, activation);
//				activation->print();
				activationNum=0.f;
//				cout << endl;
			}
			if(prevLayer!=0)
				prevLayer->calculateSparseActivation();
#endif
	}
	void nextRound() {
			activation->reset();
			if(nextLayer!=0)
				nextLayer->nextRound();
	}

	void adjust() {
		if(prevLayer!=0) {
			if(weightAdj==0)
				weightAdj=new MatrixBatched(prevMatrix->cols, prevMatrix->rows, batchSize);
			//printf("delta %d %d\n", delta->rows, delta->cols);
			cublasMul(1./*LEARNING_RATE*/, 0.0f, delta, CUBLAS_OP_N, prevLayer->output, CUBLAS_OP_N, weightAdj);
//			cublasMul(-1.f, 1.f, weightAdj, CUBLAS_OP_T, unit, CUBLAS_OP_N, prevMatrix);
//			cublasMul(-LEARNING_RATE, 1.f, delta, CUBLAS_OP_T, unit, CUBLAS_OP_N, bias);
		}
		if(nextLayer!=0)
			nextLayer->adjust();
	}

	void adjustAdd() {
		if(prevLayer!=0) {
			mySum(weightAdj);
			cublasMul(-LEARNING_RATE, 1.f, weightAdj, CUBLAS_OP_T, unit, CUBLAS_OP_N, prevMatrix);
			weightAdj->reset();
//			cublasMul(0.f,0.f, delta, CUBLAS_OP_N, prevLayer->output, CUBLAS_OP_N, weightAdj);
		}
		if(nextLayer!=0)
			nextLayer->adjustAdd();
	}

	void addTail(layer_ptr layer) {
		layer_ptr last=this;
		while(last->nextLayer!=0) {
			last=last->nextLayer;
		}
		last->nextLayer=layer;
		layer->prevLayer=last;
		
	}
};

struct Autoencoder {
	layer_ptr inputLayer;
	layer_ptr hiddenLayer;
	layer_ptr outputLayer;

	autoencoder_ptr next;
	autoencoder_ptr prev;
	
	Autoencoder(int _inputNum, int _hiddenNum,int batchSize) : next(0), prev(0) {
		inputLayer=new Layer(_inputNum, SIGMOID_0,batchSize);
		hiddenLayer=new Layer(_hiddenNum, SIGMOID_0,batchSize);
		outputLayer=new Layer(_inputNum, LINEAR,batchSize,false);

		inputLayer->addTail(hiddenLayer);
		inputLayer->addTail(outputLayer);

		for(layer_ptr i=inputLayer;i!=outputLayer;i=i->nextLayer) {
			i->nextMatrix=new Matrix(i->neuronNum, i->nextLayer->neuronNum);
			i->nextLayer->prevMatrix=i->nextMatrix;
			i->nextMatrix->randomize();
		}	
	}

	
	void addTail(autoencoder_ptr autoencoder) {
		autoencoder_ptr last=this;
		while(last->next!=0) {
			last=last->next;
		}
		last->next=autoencoder;
		autoencoder->prev=last;
	}

	void excite(vectorb_ptr p) {
		inputLayer->output->copy(p);
		inputLayer->excite();
	}
	
	void train(vectorb_ptr p) {
		for(int i=0;i<1;i++) {
				excite(p);
				outputLayer->error(p)->backpropagationSparse();
				inputLayer->adjust();
		}
//		inputLayer->adjustAdd();
/*
		if(next!=0) {
			next->train(hiddenLayer->output);
		}
*/
	}

};


struct NeuralNet {
	layer_ptr inputLayer;
	layer_ptr outputLayer;
	vector_ptr error;
	autoencoder_ptr autoencoder;
	autoencoder_ptr autoencoder_pretrain;
	int bias;

	NeuralNet() {};
public:
	NeuralNet(int batchSize, int _inputNum,int _outputNum, int _hiddenLayerNum, std::vector<int> layerNeuronNum) : autoencoder(0), bias(1){
		cout << __PRETTY_FUNCTION__ << endl;
		error=new ColVector(_outputNum);
		printf("_inputNum %d\n", _inputNum);
		
		inputLayer=new Layer(_inputNum, SIGMOID_0,batchSize);
		for(int i=0; i<_hiddenLayerNum;i++) {

			if(autoencoder==0) {
				autoencoder=new Autoencoder(_inputNum, layerNeuronNum[i],batchSize);
			} else {
				autoencoder->addTail(new Autoencoder(layerNeuronNum[i-1], layerNeuronNum[i],batchSize));
			}

			inputLayer->addTail(new Layer(layerNeuronNum[i], SIGMOID_0,batchSize));
		}

		outputLayer=new Layer(_outputNum, LINEAR,batchSize,false);
		inputLayer->addTail(outputLayer);
		autoencoder->addTail(new Autoencoder(layerNeuronNum[_hiddenLayerNum-1],_outputNum,batchSize));

		autoencoder_ptr a=autoencoder;

		for(layer_ptr i=inputLayer;i!=outputLayer;i=i->nextLayer) {
//			i->nextMatrix=new Matrix(i->neuronNum, i->nextLayer->neuronNum);
			i->nextMatrix=a->inputLayer->nextMatrix;
			i->nextLayer->prevMatrix=i->nextMatrix;
			i->nextMatrix->randomize();
			if(a!=0)
				a=a->next;
		}
		autoencoder_pretrain=autoencoder;
	}

	void pretrain(vectorb_ptr p) {
		autoencoder_ptr a=autoencoder;
		if(a==autoencoder_pretrain) {
			autoencoder_pretrain->train(p);
		} else {
			/*
				a->excite(p);
				a=a->next;
				while(a!=autoencoder_pretrain) {
					a->excite(a->prev->hiddenLayer->output);
					a=a->next;
				}
*/
				autoencoder_pretrain->train(autoencoder_pretrain->prev->hiddenLayer->output);
		}
	}

	void nextRound() {
		autoencoder_pretrain->hiddenLayer->calculateSparseActivation();
		autoencoder_pretrain->inputLayer->nextRound();
		autoencoder_pretrain->inputLayer->adjustAdd();
	}

	void pretrainNext() {
		autoencoder_pretrain=autoencoder_pretrain->next;
	}

	void pretrainAdjust() {
		autoencoder_pretrain->inputLayer->adjustAdd();
	}

	
	vectorb_ptr excite(vectorb_ptr p) {
		inputLayer->output->copy(p);
		inputLayer->excite();
/*
		layer_ptr layer=inputLayer->nextLayer;
		layer->excite();
*/
/*
		while(layer!=0) {
			layer->excite();
			layer=layer->nextLayer;
		}
*/
		return outputLayer->output;
	}

	void backpropagation(vectorb_ptr desiredOutput) {
		layer_ptr layer=outputLayer;
		layer->error(desiredOutput)->backpropagation();
/*
		layer=layer->prevLayer;
		layer->backpropagation();
*/

/*
		while(layer!=0) {
			layer->backpropagation();
			layer=layer->prevLayer;
		}
*/
		inputLayer->adjust();
	}

	void adjust() {
		inputLayer->adjustAdd();
	}

	virtual ~NeuralNet() {
/*
		for(layer_ptr i=outputLayer; i!=0UL;i=i->prevLayer) {
			delete i->nextMatrix;
			delete i->nextLayer;
		}
		delete inputLayer;
		delete error;
*/
		cout << __PRETTY_FUNCTION__ << endl;
	}
} nn_t;


// ----------------------------------------------------------------------------------------------------
//                       main function
// ----------------------------------------------------------------------------------------------------

void usage(char** argv) {
	fprintf(stderr,"Usage %s [-g genome] [-i indata] [-o outdata] [-s samples] [-n input_size] [-u output_size]\n", argv[0]);
	exit(-1);

}

int main(int argc, char **argv) {
	srand(time(0));
	int opt;
	if(argc!=13) {
		usage(argv);
	}
	const char *filename, *infile, *outfile;
	int samples, input_size, output_size;

	while((opt=getopt(argc, argv, "i:o:g:s:n:u:"))!=-1) {
		switch(opt) {
			case 'g':
				filename=optarg;
				break;
			case 'i':
				infile=optarg;
				break;
			case 'o':
				outfile=optarg;
				break;
			case 's':
				samples=atoi(optarg);
				break;
			case 'n':
				input_size=atoi(optarg);
				break;
			case 'u':
				output_size=atoi(optarg);
				break;
			default:
				usage(argv);
		}
	}
	IO io(input_size, output_size, samples,infile, outfile);

	cublasStatus_t status=cublasCreate(&handle);
	if(status!=CUBLAS_STATUS_SUCCESS) {
		cerr << "cublas init failed" << endl;
	}
/*
	Matrix m1(3,2);
	m1.d_data[IDX2C(0,0,3)]=1;
	m1.d_data[IDX2C(1,0,3)]=2;
	m1.d_data[IDX2C(2,0,3)]=3;

	m1.d_data[IDX2C(0,1,3)]=2;
	m1.d_data[IDX2C(1,1,3)]=3;
	m1.d_data[IDX2C(2,1,3)]=4;

	Matrix m2(2,3);
	m2.d_data[IDX2C(0,0,2)]=2;
	m2.d_data[IDX2C(1,0,2)]=3;

	m2.d_data[IDX2C(0,1,2)]=3;
	m2.d_data[IDX2C(1,1,2)]=4;

	m2.d_data[IDX2C(0,2,2)]=4;
	m2.d_data[IDX2C(1,2,2)]=5;

	Matrix p(3,2);
	Matrix m3(2);

	cublasMul(&m1,&m3, &p);
	p.print();
*/
//	exit(0);

/*
	for(int i=0;i<p.rows;i++) {
		for(int j=0;j<p.cols;j++)
			cout << p.d_data[IDX2C(i,j,p.rows)] << ",";
		cout << endl;
	}
*/

	int d[]={16};
	int l=sizeof(d)/sizeof(int);
	NeuralNet nn(samples,input_size,output_size,l,std::vector<int>(d, d+l));


	
#if 1
	cout << "pretraining ..." << endl;
	for(int r=0;r<l;r++) {
			cout << "autoencoder " << r << endl;
			for(int k=0;k<20000;k++) {
					nn.pretrain(io.input);
					nn.nextRound();
//					nn.pretrainAdjust();
			}
			nn.pretrainNext();
	}
#endif
	cout << "training ..." << endl;
#if 1
	ofstream mseFile("./mse.dat");
for(int k=0;k<100000;k++) {
if(k%100==0)
		cout << k << endl;

		vectorb_ptr o=nn.excite(io.input);
		nn.backpropagation(io.output);
		nn.adjust();
		mseFile << k << "\t" << nn.outputLayer->mse->d_data[0] << endl;
		if(k%100==0) {
			for(int i=0;i<samples;i++) {
				cout << "input: ";
				io.input->print(i);
				cout << " output: ";
				io.output->print(i);
				cout << "fit: ";
				o->_printf(i);
				cout << " ";
//				cout << "err: "; nn.outputLayer->error_derived->print(i); 
				int ham=0;
				for(int r=0;r<output_size;r++) {
					int a=o->d_data[IDX3C(0,r,i,o->rows, o->cols)]>.5?1:0;
					int b=io.output->d_data[IDX3C(0,r,i,io.output->rows, io.output->cols)]>.5?1:0;
					ham+=a^b;
					printf("%d,", a);
				}
				cout << " distance: " << ham;
				cout << endl;
			}
		}

	if(k%100==0)
			cout << "-------------------------------------------------------------------------" << endl;
}

	mseFile.close();

#endif
	status=cublasDestroy(handle);
	if(status!=CUBLAS_STATUS_SUCCESS) {
		cerr << "cublas shutdown failed" << endl;
	}
	return 0;
}
