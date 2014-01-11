
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

#define SIGMOID_0 0
#define LINEAR 1

#define LEARNING_RATE 0.17


using namespace std;

cublasHandle_t handle;

// ----------------------------------------------------------------------------
//                IO part
// ----------------------------------------------------------------------------


struct IO {
	int inputs, outputs, samples;
	const char *infile,*outfile;

    float *input;
    float *output;

		void read_i(const char* infile) {
			ifstream ifile(infile);

			for(int i=0;i<samples;i++) {
				string line;
				getline(ifile,line);
				istringstream is(line);
				char c;
				for(int p=0;p<inputs;p++) {
					is>>input[IDX2C(p,i,inputs)]>>c;
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
					is>>output[IDX2C(p,i,outputs)]>>c;
				}
			}
		}

		void read_io(const char* infile, const char* outfile) {
			read_i(infile);
			read_o(outfile);
		}


	IO(int _inputs, int _outputs, int _samples, const char* _infile, const char* _outfile) : inputs(_inputs), outputs(_outputs), samples(_samples),
		infile(_infile), outfile(_outfile) {
		input=new float[inputs*samples];
		output=new float[outputs*samples];
		read_io(infile,outfile);
	}
};

class Layer;
class Matrix;
class Vector;
typedef Layer* layer_ptr;
typedef Matrix* matrix_ptr;
typedef Vector* vector_ptr;

class Matrix {
public:
	int rows;
	int cols;
	thrust::device_vector<float> d_data;
	Matrix(int _rows, int _cols) : rows(_rows), cols(_cols), d_data(rows*cols) {
		for(int i=0;i<numel();i++) {
			d_data[i]=0.f;
		}
	}
	Matrix(int _size) : rows(_size), cols(_size), d_data(_size*_size) {
		for(int i=0;i<numel();i++) {
			d_data[i]=0.f;
		}
		for(int i=0;i<rows;i++)
			d_data[IDX2C(i,i,rows)]=1.f;
	}
	int numel() {return rows*cols;}
	virtual ~Matrix() {
		d_data.clear();
		d_data.shrink_to_fit();
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

	void randomize() {
		for(int i=0;i<numel();i++) {
			d_data[i]=(((float)rand())/RAND_MAX)/5.;
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
		for(int i=0;i<numel(); i++) {
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

inline float sigmoid(float signal) {
	return 1./(1+exp(-1.*signal));
}

inline float sigmoid_derived(float signal) {
	float s=sigmoid(signal);
	return s*(1-s);
}

void sigmoid(vector_ptr v, vector_ptr output, matrix_ptr outputDerived) {
	for(int i=0;i<v->numel();i++) {
		outputDerived->d_data[IDX2C(i,i, outputDerived->rows)]=sigmoid_derived(v->d_data[i]);
		output->d_data[i]=sigmoid(v->d_data[i]);
	}
}

void linear(vector_ptr v, vector_ptr output, matrix_ptr outputDerived) {
	for(int i=0;i<v->numel();i++) {
		outputDerived->d_data[IDX2C(i,i,outputDerived->rows)]=1;
		output->d_data[i]=v->d_data[i];
	}
}

typedef void (*neuron_func_t)(vector_ptr, vector_ptr,matrix_ptr);

neuron_func_t neuron_func[]={sigmoid,linear};


void cublasMul(float alpha,float beta, matrix_ptr m1,cublasOperation_t transa, matrix_ptr m2, cublasOperation_t transb, matrix_ptr p) {
	int check1=(transa==CUBLAS_OP_N?m1->cols:m1->rows);
	int check2=(transb==CUBLAS_OP_N?m2->rows:m2->cols);

	int lda=m1->rows;
	int ldb=m2->rows;

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

inline void cublasSub(matrix_ptr m1, matrix_ptr m2, matrix_ptr p) {

	assert(m1->rows==m2->rows);
	assert(m1->cols==m2->cols);

	int lda=m1->rows;
	int ldb=m2->rows;
	float alpha=1.f;
	float beta=-1.f;
	cublasStatus_t status=cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m1->rows, m2->cols, &alpha, thrust::raw_pointer_cast(m1->d_data.data()), lda, &beta, thrust::raw_pointer_cast(m2->d_data.data()), ldb, thrust::raw_pointer_cast(p->d_data.data()), p->rows);
}



class Layer {
public:
	int neuronType;
	int neuronNum;
	bool bias;
	layer_ptr nextLayer;
	layer_ptr prevLayer;
	matrix_ptr nextMatrix;
	matrix_ptr prevMatrix;
	
	vector_ptr output;
	matrix_ptr outputDerived;
	vector_ptr delta;
	vector_ptr error_derived;
	matrix_ptr unit;
	vector_ptr input;
	matrix_ptr weightAdj;

	Layer(int _neuronNum, int _neuronType, bool _bias=false) : neuronNum(_neuronNum), neuronType(_neuronType), bias(_bias), nextLayer(0), prevLayer(0), weightAdj(0) {
		cout << __PRETTY_FUNCTION__ << _neuronNum << endl;
		output=new RowVector(neuronNum);
		input=new RowVector(neuronNum);
		outputDerived=new Matrix(neuronNum,neuronNum);
		delta=new ColVector(neuronNum);
		error_derived=new RowVector(neuronNum);
		unit=new Matrix(neuronNum);

	}	

	virtual ~Layer() {
		delete  output;
		cout << __PRETTY_FUNCTION__ << neuronNum << endl;
	}

	void excite() {
		if(prevLayer!=0) {
			cublasMul(prevLayer->output, prevMatrix, input);
			neuron_func[neuronType](input,output, outputDerived);
		}
/*
		if(nextLayer!=0)
			nextLayer->excite();
*/
	}
	void error(vector_ptr desiredOutput) {
		if(prevLayer!=0) {
			cublasSub(output,desiredOutput,error_derived);
			cublasMul(outputDerived, CUBLAS_OP_N, error_derived, CUBLAS_OP_T, delta);
			
		}
/*
		if(nextLayer!=0)
			nextLayer->excite();
*/
	}

	void backpropagation() {
		if(nextLayer!=0 && prevLayer!=0) {
			Matrix m(outputDerived->rows, nextMatrix->cols);
			cublasMul(outputDerived, CUBLAS_OP_N, nextMatrix, CUBLAS_OP_N, &m);
			cublasMul(&m,nextLayer->delta, delta);
		}
/*
		if(prevLayer!=0)
			prevLayer->backpropagation();
*/
	}

	void adjust() {
		if(prevLayer!=0) {
			if(weightAdj==0)
				weightAdj=new Matrix(prevMatrix->cols, prevMatrix->rows);
			cublasMul(LEARNING_RATE, 0.0f, delta, CUBLAS_OP_N, prevLayer->output, CUBLAS_OP_N, weightAdj);
			cublasMul(-1.f, 1.f, weightAdj, CUBLAS_OP_T, unit, CUBLAS_OP_N, prevMatrix);
		}
		if(nextLayer!=0)
			nextLayer->adjust();
	}

	void adjustAdd() {
		if(prevLayer!=0) {
			cublasMul(-1.f, 1.f, weightAdj, CUBLAS_OP_T, unit, CUBLAS_OP_N, prevMatrix);
			cublasMul(0.f,0.f, delta, CUBLAS_OP_N, prevLayer->output, CUBLAS_OP_N, weightAdj);
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


struct NeuralNet {
	layer_ptr inputLayer;
	layer_ptr outputLayer;
	vector_ptr error;

	NeuralNet() {};
public:
	NeuralNet(int _inputNum,int _outputNum, int _hiddenLayerNum, std::vector<int> layerNeuronNum) {
		cout << __PRETTY_FUNCTION__ << endl;
		error=new ColVector(_outputNum);
		
		inputLayer=new Layer(_inputNum, SIGMOID_0);
		for(int i=0; i<_hiddenLayerNum;i++) {
			inputLayer->addTail(new Layer(layerNeuronNum[i], SIGMOID_0));
		}

		outputLayer=new Layer(_outputNum, LINEAR);
		inputLayer->addTail(outputLayer);

		for(layer_ptr i=inputLayer;i!=outputLayer;i=i->nextLayer) {
			i->nextMatrix=new Matrix(i->neuronNum, i->nextLayer->neuronNum);
			i->nextLayer->prevMatrix=i->nextMatrix;
			i->nextMatrix->randomize();
		}
	}
	vector_ptr excite(vector_ptr p) {
		inputLayer->output->set(*p);
		layer_ptr layer=inputLayer->nextLayer;
		while(layer!=0) {
			layer->excite();
			layer=layer->nextLayer;
		}
		return outputLayer->output;
	}

	void backpropagation(vector_ptr desiredOutput) {
		layer_ptr layer=outputLayer;
		layer->error(desiredOutput);
		layer=layer->prevLayer;
		while(layer!=0) {
			layer->backpropagation();
			layer=layer->prevLayer;
		}
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
//	exit(0);

/*
	for(int i=0;i<p.rows;i++) {
		for(int j=0;j<p.cols;j++)
			cout << p.d_data[IDX2C(i,j,p.rows)] << ",";
		cout << endl;
	}
*/

	int d[]={4,4,4,4,4,4,4,4};
	int l=sizeof(d)/sizeof(int);
	NeuralNet nn(input_size,output_size,l,std::vector<int>(d, d+l));

	RowVector v(input_size);
	RowVector v1(output_size);
for(int k=0;k<5000;k++) {
if(k%50==0)
		cout << k << endl;
	for(int i=0;i<samples;i++) {
		for(int j=0;j<input_size;j++) {
			v.set(j,io.input[IDX2C(j,i,input_size)]);
		}
		for(int j=0;j<output_size; j++) {
			v1.set(j, io.output[IDX2C(j,i,output_size)]);
		}
		vector_ptr o=nn.excite(&v);
if(k%50==0) {
		cout << "input: ";
		v.printd();
		cout << " output: ";
		v1.printd();
//		cout << endl;
		cout << "fit: ";
		o->print();
		cout << " ";
		int ham=0;
		for(int r=0;r<output_size;r++) {
			int a=o->d_data[r]>.5?1:0;
			int b=v1.d_data[r]>.5?1:0;
			ham+=a^b;
			printf("%d,", a);
		}
		cout << " distance: " << ham;
		cout << endl;
}

		nn.backpropagation(&v1);
	}
//	nn.adjust();
if(k%50==0)
		cout << "-------------------------------------------------------------------------" << endl;
}

	status=cublasDestroy(handle);
	if(status!=CUBLAS_STATUS_SUCCESS) {
		cerr << "cublas shutdown failed" << endl;
	}
	return 0;
}
