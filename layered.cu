#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <thrust/sort.h>
#include <thrust/functional.h>

#include <curand_kernel.h>
#include <assert.h>
#include <helper_cuda.h>

using namespace std;

#define NEURON_NUM 16
#define HIDDEN_LAYER_NUM 2
#define POPULATION_SIZE 256
#define MUTATION_PROB 0.01
#define CROSSOVER_PROB 0.8
#define BEST_INDIVIDUALS 2

#define LEARNING_RATE 0.2

#define NR_INPUTS 5
#define NR_OUTPUTS 15
#define SAMPLES 16

#define BP_NUM 400

#define _THREAD_COUNT (blockDim.x*gridDim.x)
#define THREAD_ID (blockIdx.x*blockDim.x + threadIdx.x)
#define CHAR(x) ((char*)x)

#define NN_CONNECT(connect,x,y) (connect[(y*NEURON_NUM+x)/8]&(1<<((y*NEURON_NUM+x)%8)))
#define NN_CONNECT_INPUT(connect, iIdx, nIdx) (connect[(iIdx*NEURON_NUM+nIdx)/8]&(1<<((iIdx*NEURON_NUM+nIdx)%8)))
#define NN_CONNECT_OUTPUT(connect, oIdx, nIdx) (connect[(oIdx*NEURON_NUM+nIdx)/8]&(1<<((oIdx*NEURON_NUM+nIdx)%8)))

#define NEURON_IDX(output_neuron_idx) ((output_neuron_idx&0xf)%NEURON_NUM)
#define CORTEX_IDX(output_neuron_idx) (((output_neuron_idx>>4)&0xf)%CORTEX_NUM)

#define FITNESS(population, index) ((population)->fitness[index])
//#define DEBUG

typedef struct {
	char connect[NEURON_NUM*NEURON_NUM/8];
} ga_hidden_layer_t;

typedef struct {
	char connect[NR_INPUTS*NEURON_NUM/8];
} ga_input_layer_t;

typedef struct {
	char connect[NEURON_NUM*NR_OUTPUTS/8];
} ga_output_layer_t;

typedef struct {
	ga_input_layer_t ga_input;
	ga_output_layer_t ga_output;
	ga_hidden_layer_t ga_hidden[HIDDEN_LAYER_NUM];
} ga_genome_t;

typedef struct {
	float weight[NR_INPUTS];
	float weight_adjust[NR_INPUTS];
} bp_input_layer_t;

typedef struct {
	float weight[NEURON_NUM][NEURON_NUM];
	float weight_adjust[NEURON_NUM][NEURON_NUM];
} bp_hidden_layer_t;

typedef struct {
	float weight[NR_OUTPUTS][NEURON_NUM];
	float weight_adjust[NR_OUTPUTS][NEURON_NUM];
} bp_output_layer_t;

typedef struct {
	bp_input_layer_t bp_input;
	bp_output_layer_t bp_output;
	bp_hidden_layer_t bp_hidden[HIDDEN_LAYER_NUM];
} bp_genome_t;

typedef struct {
	ga_genome_t ga_genome[POPULATION_SIZE];
	bp_genome_t bp_genome[POPULATION_SIZE];


	float hidden_layer_neuron_input[POPULATION_SIZE][HIDDEN_LAYER_NUM][NEURON_NUM];
	float hidden_layer_neuron_output[POPULATION_SIZE][HIDDEN_LAYER_NUM][NEURON_NUM];
	float hidden_layer_neuron_output_derived[POPULATION_SIZE][HIDDEN_LAYER_NUM][NEURON_NUM];


	float input_layer_neuron_input[POPULATION_SIZE][NR_INPUTS];
	float input_layer_neuron_output[POPULATION_SIZE][NR_INPUTS];
	float input_layer_neuron_output_derived[POPULATION_SIZE][NR_INPUTS];


	float output_layer_neuron_input[POPULATION_SIZE][NR_OUTPUTS];
	float output_layer_neuron_output[POPULATION_SIZE][NR_OUTPUTS];
	float output_layer_neuron_output_derived[POPULATION_SIZE][NR_OUTPUTS];

	float error[POPULATION_SIZE][NR_OUTPUTS];
	float error_derived[POPULATION_SIZE][NR_OUTPUTS];
	float delta_output_layer[POPULATION_SIZE][NR_OUTPUTS];
	float delta_hidden_layer[POPULATION_SIZE][HIDDEN_LAYER_NUM][NEURON_NUM];
	float delta_input_layer[POPULATION_SIZE][NR_INPUTS];
	float mse[POPULATION_SIZE];

	
	float fitness[POPULATION_SIZE];
	int map[POPULATION_SIZE];
	curandState_t curandState[POPULATION_SIZE];

} population_t;

typedef struct {
    float inputs[NR_INPUTS][SAMPLES];
    float outputs[NR_OUTPUTS][SAMPLES];
    float errors[NR_OUTPUTS];
} IO_t;

struct Counter {
	int count;
	__device__ Counter() : count(0) {};
    __device__ int getPopulationIndex() const { int r=_THREAD_COUNT*count+THREAD_ID; return r;};
    __device__ int getPopulationIndexInc() { int r=getPopulationIndex(); count++; return r; };
    __device__ int getPopulationIndexEven() { int r=_THREAD_COUNT*count+THREAD_ID; return r; };
    __device__ int getPopulationIndexOdd() { int r=_THREAD_COUNT*(count+1)+THREAD_ID; return r; };
    __device__ void reset() { count=0; };
};

float *deviceBestIndividualFitness;
float *hostBestIndividualFitness;


// ------------------------------------------------------------------------------------------------
inline void check_cuda_errors(const char *filename, const int line_number)
{
/*
      cudaDeviceSynchronize();
      cudaError_t error = cudaGetLastError();
      if(error != cudaSuccess) {
          printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
          exit(-1);
      }
*/
}

__device__ inline float sigmoid(float signal) {
	return 1./(1+exp(-1.*signal));
}

__device__ inline float sigmoid_derived(float signal) {
	float s=sigmoid(signal);
	return s*(1-s);
}

__device__ 
void
swap (int array[], int i, int j)
{
  int tmp = array[i];
  array[i] = array[j];
  array[j] = tmp;
}

__device__ void
qs (population_t *population, int array[], int left, int right)
{
  if (left < right)
    {
      double p = FITNESS(population, array[left + (right - left) / 2]);
      int i = left;
      int j = right;

      while (i < j)
	{

	  while (FITNESS(population,array[i]) > p && i < right)
	    i++;
	  while (FITNESS(population,array[j]) < p && j > left)
	    j--;
	  if (i <= j)
	    {
	      swap (array, i, j);
	      i++;
	      j--;
	    }
	}

      qs (population,array, i, right);
      qs (population,array, left, j);
    }

}

__device__ void qsort(population_t *population) {
    qs(population, population->map, 0, POPULATION_SIZE-1);
}


__global__ void MatrixMul(float *Md, float *Nd, float *Pd, int width) {
	int tx=threadIdx.x;
	int ty=threadIdx.y;

	float Pval=0.f;
	for(int k=0;k<width;k++) {
		float Mdelement=Md[ty*width+k];
		float Ndelement=Nd[k*width+tx];
		Pval+=Mdelement*Ndelement;
	}

	Pd[ty*width+tx]=Pval;
}


//<<<POPULATION_SIZE, dim3(NR_INPUTS)>>>
__global__ void cuResetWeightsInputLayer(population_t *p) {
	int g=blockIdx.x;

	int inputIdx=threadIdx.x%NR_INPUTS;

	while(g<POPULATION_SIZE) {
		p->bp_genome[g].bp_input.weight[inputIdx]=1.f;
		g+=gridDim.x;
	}
}

//<<<POPULATION_SIZE,dim3(NEURON_NUM,NEURON_NUM>>>
__global__ void cuResetWeightsHiddenLayer(population_t *p) {
	int g=blockIdx.x;

	int neuronIdx=threadIdx.x%NEURON_NUM;
	int neuronIdx1=threadIdx.y%NEURON_NUM;

	while(g<POPULATION_SIZE) {
		for(int i=0;i<HIDDEN_LAYER_NUM;i++)
			p->bp_genome[g].bp_hidden[i].weight[neuronIdx][neuronIdx1]=1.f;
		g+=gridDim.x;
	}
}

//<<<POPULATION_SIZE,dim3(NEURON_NUM,NR_OUTPUTS)>>>
__global__ void cuResetWeightsOutputLayer(population_t *p) {
	int g=blockIdx.x;

	int outputIdx=threadIdx.x%NR_OUTPUTS;
	int neuronIdx=threadIdx.y%NEURON_NUM;

	while(g<POPULATION_SIZE) {
		p->bp_genome[g].bp_output.weight[outputIdx][neuronIdx]=1.f;
		g+=gridDim.x;
	}
}

//<<<POPULATION_SIZE, NR_INPUTS>>>
__global__ void cuResetNeuronInputInputLayer(population_t *p) {
	int g=blockIdx.x;
	
	int inputIdx=threadIdx.x%NR_INPUTS;
	while(g<POPULATION_SIZE) {
		p->input_layer_neuron_input[g][inputIdx]=0.f;
		g+=gridDim.x;
	}
}

//<<<POPULATION_SIZE, dim3(HIDDEN_LAYER_NUM,NEURON_NUM)>>>
__global__ void cuResetNeuronInputHiddenLayer(population_t *p) {
	int g=blockIdx.x;
	
	int layerIdx=threadIdx.x%HIDDEN_LAYER_NUM;
	int neuronIdx=threadIdx.y%NEURON_NUM;

	while(g<POPULATION_SIZE) {
		p->hidden_layer_neuron_input[g][layerIdx][neuronIdx]=0.f;	
		g+=gridDim.x;
	}
}

//<<<POPULATION_SIZE, dim3(NR_OUTPUTS)>>>
__global__ void cuResetNeuronInputOutputLayer(population_t *p) {
	int g=blockIdx.x;

	int outputIdx=threadIdx.x%NR_OUTPUTS;
	while(g<POPULATION_SIZE) {
		p->output_layer_neuron_input[g][outputIdx]=0.f;
		g+=gridDim.x;
	}
}

//<<<POPULATION_SIZE, dim3(NR_INPUTS)>>>
__global__ void cuResetNeuronOutputInputLayer(population_t *p) {
	int g=blockIdx.x;

	int inputIdx=threadIdx.x%NR_INPUTS;
	while(g<POPULATION_SIZE) {
		p->input_layer_neuron_output[g][inputIdx]=0.f;
		p->input_layer_neuron_output_derived[g][inputIdx]=0.f;
		g+=gridDim.x;
	}
}

//<<<POPULATION_SIZE, dim3(HIDDEN_LAYER_NUM, NEURON_NUM)>>>
__global__ void cuResetNeuronOutputHiddenLayer(population_t *p) {
	int g=blockIdx.x;
	int layer=threadIdx.x%HIDDEN_LAYER_NUM;
	int neuron=threadIdx.y%NEURON_NUM;

	while(g < POPULATION_SIZE) {
		p->hidden_layer_neuron_output[g][layer][neuron]=0.f;
		p->hidden_layer_neuron_output_derived[g][layer][neuron]=0.f;
		g+=gridDim.x;
	}
}


//<<<POPULATION_SIZE, dim3(NR_OUTPUTS)>>>
__global__ void cuResetNeuronOutputOutputLayer(population_t *p) {
	int g=blockIdx.x;
	int outputIdx=threadIdx.x%NR_OUTPUTS;

	while(g<POPULATION_SIZE) {
		p->output_layer_neuron_output[g][outputIdx]=0.f;
		p->output_layer_neuron_output_derived[g][outputIdx]=0.f;
		g+=gridDim.x;
	}
}

//<<<POPULATION_SIZE, NR_INPUTS>>>
__global__ void cuResetDeltaInputLayer(population_t *p) {
	int g=blockIdx.x;
	int inputIdx=threadIdx.x%NR_INPUTS;

	while(g<POPULATION_SIZE) {
		p->delta_input_layer[g][inputIdx]=0.f;
		g+=gridDim.x;
	}
}

//<<<POPULATION_SIZE, dim3(HIDDEN_LAYER_NUM, NEURON_NUM)>>>
__global__ void cuResetDeltaHiddenLayer(population_t *p) {
	int g=blockIdx.x;
	int layerIdx=threadIdx.x%HIDDEN_LAYER_NUM;
	int neuronIdx=threadIdx.y%NEURON_NUM;

	while(g<POPULATION_SIZE) {
		p->delta_hidden_layer[g][layerIdx][neuronIdx]=0.f;
		g+=gridDim.x;
	}
}

//<<<POPULATION_SIZE, NR_OUTPUTS>>>
__global__ void cuResetDeltaOutputLayer(population_t *p) {
	int g=blockIdx.x;
	int outputIdx=threadIdx.x%NR_OUTPUTS;
	while(g<POPULATION_SIZE) {
		p->delta_output_layer[g][outputIdx]=0.f;
		g+=gridDim.x;
	}
}

//<<<POPULATION_SIZE, NR_INPUTS>>>
__global__ void cuResetWeightAdjustInputLayer(population_t *p) {
	int g=blockIdx.x;
	
	int inputIdx=threadIdx.x%NR_INPUTS;
	while(g<POPULATION_SIZE) {
		p->bp_genome[g].bp_input.weight_adjust[inputIdx]=0.f;
		g+=gridDim.x;
	}
}

///<<<POPULATION_SIZE, dim3(NEURON_NUM,NEURON_NUM)>>>
__global__ void cuResetWeightAdjustHiddenLayer(population_t *p) {
	int g=blockIdx.x;
	int neuronIdx1=threadIdx.x%NEURON_NUM;
	int neuronIdx2=threadIdx.y%NEURON_NUM;

	while(g<POPULATION_SIZE) {
		for(int i=0;i<HIDDEN_LAYER_NUM;i++)
			p->bp_genome[g].bp_hidden[i].weight_adjust[neuronIdx1][neuronIdx2]=0.f;
		g+=gridDim.x;
	}
}

//<<<POPULATION_SIZE, dim3(NR_OUTPUTS, NEURON_NUM)>>>
__global__ void cuResetWeightAdjustOutputLayer(population_t *p) {
	int g=blockIdx.x;

	int outputIdx=threadIdx.x%NR_OUTPUTS;
	int neuronIdx=threadIdx.y%NEURON_NUM;

	while(g<POPULATION_SIZE) {
		p->bp_genome[g].bp_output.weight_adjust[outputIdx][neuronIdx]=0.f;
		g+=gridDim.x;
	}
}

__global__ void cuResetMse(population_t *p) {
	int g=blockIdx.x;
	while(g<POPULATION_SIZE){
		p->mse[g]=0.f;
		g+=gridDim.x;	
	}
}


// ---------------------------------------------------------------------------------------
//                    Reset part
// ---------------------------------------------------------------------------------------


__host__ void hoResetWeights(population_t *p) {
	cuResetWeightsInputLayer<<<POPULATION_SIZE, dim3(NR_INPUTS)>>>(p);
	check_cuda_errors(__FILE__,__LINE__);
	cuResetWeightsHiddenLayer<<<POPULATION_SIZE, dim3(NEURON_NUM, NEURON_NUM)>>>(p);
	check_cuda_errors(__FILE__,__LINE__);
	cuResetWeightsOutputLayer<<<POPULATION_SIZE, dim3(NR_OUTPUTS, NEURON_NUM)>>>(p);
	check_cuda_errors(__FILE__,__LINE__);
}

__host__ void hoResetInput(population_t *p) {
	cuResetNeuronInputInputLayer<<<POPULATION_SIZE, dim3(NR_INPUTS)>>>(p);
	cuResetNeuronInputHiddenLayer<<<POPULATION_SIZE, dim3(HIDDEN_LAYER_NUM, NEURON_NUM)>>>(p);
	cuResetNeuronInputOutputLayer<<<POPULATION_SIZE, dim3(NR_OUTPUTS)>>>(p);
}

__host__ void hoResetOutput(population_t *p) {
	cuResetNeuronOutputInputLayer<<<POPULATION_SIZE, dim3(NR_INPUTS)>>>(p);
	cuResetNeuronOutputHiddenLayer<<<POPULATION_SIZE, dim3(HIDDEN_LAYER_NUM, NEURON_NUM)>>>(p);	
	cuResetNeuronOutputOutputLayer<<<POPULATION_SIZE, dim3(NR_OUTPUTS)>>>(p);
}

__host__ void hoResetDelta(population_t *p) {
	cuResetDeltaInputLayer<<<POPULATION_SIZE, NR_INPUTS>>>(p);
	check_cuda_errors(__FILE__,__LINE__);
	cuResetDeltaHiddenLayer<<<POPULATION_SIZE, dim3(HIDDEN_LAYER_NUM, NEURON_NUM)>>>(p);
	check_cuda_errors(__FILE__,__LINE__);
	cuResetDeltaOutputLayer<<<POPULATION_SIZE, NR_OUTPUTS>>>(p);
	check_cuda_errors(__FILE__,__LINE__);
}

__host__ void hoResetWeightAdjust(population_t *p) {
	cuResetWeightAdjustInputLayer<<<POPULATION_SIZE, NR_INPUTS>>>(p);
	check_cuda_errors(__FILE__,__LINE__);
	cuResetWeightAdjustHiddenLayer<<<POPULATION_SIZE, dim3(NEURON_NUM,NEURON_NUM)>>>(p);
	check_cuda_errors(__FILE__,__LINE__);
	cuResetWeightAdjustOutputLayer<<<POPULATION_SIZE, dim3(NR_OUTPUTS, NEURON_NUM)>>>(p);
	check_cuda_errors(__FILE__,__LINE__);
}

__host__ void hoResetMse(population_t *p) {
	cuResetMse<<<POPULATION_SIZE,1>>>(p);
}


__host__ void hoResetExceptWeights(population_t *p) {
	check_cuda_errors(__FILE__,__LINE__);
	hoResetInput(p);
	check_cuda_errors(__FILE__,__LINE__);
	hoResetOutput(p);
	check_cuda_errors(__FILE__,__LINE__);
	hoResetDelta(p);
	check_cuda_errors(__FILE__,__LINE__);
	hoResetWeightAdjust(p);
	check_cuda_errors(__FILE__,__LINE__);
	hoResetMse(p);
	check_cuda_errors(__FILE__,__LINE__);
}
__host__ void hoReset(population_t *p) {
	hoResetExceptWeights(p);
	hoResetWeights(p);
}

//---------------------------------------------------------------------------------------
//                   Excite part
//---------------------------------------------------------------------------------------

//<<<POPULATION_SIZE,NR_INPUTS>>>
__global__ void cuInputsOutputsInputLayer(population_t *p, IO_t *io, int sample) {
	int g=blockIdx.x;

	int neuronIdx=threadIdx.x%NR_INPUTS;
	while(g<POPULATION_SIZE) {
		p->input_layer_neuron_input[g][neuronIdx]=p->bp_genome[g].bp_input.weight[neuronIdx]*io->inputs[neuronIdx][sample];
#ifdef DEBUG
		printf("p->bp_genome[%d].bp_input.weight[%d]=%f\n", g, neuronIdx, p->bp_genome[g].bp_input.weight[neuronIdx]);
#endif

		p->input_layer_neuron_output[g][neuronIdx]=sigmoid(p->input_layer_neuron_input[g][neuronIdx]);
		p->input_layer_neuron_output_derived[g][neuronIdx]=sigmoid_derived(p->input_layer_neuron_input[g][neuronIdx]);

		assert(p->bp_genome[g].bp_input.weight[neuronIdx]==p->bp_genome[g].bp_input.weight[neuronIdx]);
		assert(p->input_layer_neuron_output[g][neuronIdx]==p->input_layer_neuron_output[g][neuronIdx]);
		assert(p->input_layer_neuron_output_derived[g][neuronIdx]==p->input_layer_neuron_output_derived[g][neuronIdx]);
		g+=gridDim.x;
	}
}


//<<<POPULATION_SIZE, dim3(NEURON_NUM)>>>
__global__ void cuInputsHiddenLayer(population_t *p) {
	int g=blockIdx.x;
	
	int neuronIdx=threadIdx.x%NEURON_NUM;

	while(g<POPULATION_SIZE) {
		for(int i=0;i<NR_INPUTS;i++) 
			if(NN_CONNECT_INPUT(p->ga_genome[g].ga_input.connect, /*neuronIdx,i*/ i, neuronIdx)) {
				p->hidden_layer_neuron_input[g][0][neuronIdx]+=p->input_layer_neuron_output[g][i]*
/*!!!!!!!!!!!!!*/			p->bp_genome[g].bp_hidden[0].weight[neuronIdx][i];
				assert(p->hidden_layer_neuron_input[g][0][neuronIdx]==p->hidden_layer_neuron_input[g][0][neuronIdx]);
			}
		g+=gridDim.x;
	}
}

//<<<POPULATION_SIZE,NEURON_NUM)>>>
__global__ void cuExciteHidden(population_t *p, int layerIdx) {
	int g=blockIdx.x;
	
	int neuronIdx=threadIdx.x%NEURON_NUM;

	while(g<POPULATION_SIZE) {
		p->hidden_layer_neuron_output[g][layerIdx][neuronIdx]=sigmoid(p->hidden_layer_neuron_input[g][layerIdx][neuronIdx]);
		p->hidden_layer_neuron_output_derived[g][layerIdx][neuronIdx]=sigmoid_derived(p->hidden_layer_neuron_input[g][layerIdx][neuronIdx]);	

		assert(p->hidden_layer_neuron_output[g][layerIdx][neuronIdx]==p->hidden_layer_neuron_output[g][layerIdx][neuronIdx]);
		assert(p->hidden_layer_neuron_output_derived[g][layerIdx][neuronIdx]==p->hidden_layer_neuron_output_derived[g][layerIdx][neuronIdx]);
		g+=gridDim.x;
	}
}

//<<<POPULATION_SIZE, NEURON_NUM>>>
__global__ void cuInputsHiddenLayer(population_t *p, int layerIdx) {
	int g=blockIdx.x;
	int neuronIdx=threadIdx.x%NEURON_NUM;

	while(g<POPULATION_SIZE) {
		for(int i=0;i<NEURON_NUM;i++) {
			if(NN_CONNECT(p->ga_genome[g].ga_hidden[layerIdx].connect, neuronIdx,i)) {
				p->hidden_layer_neuron_input[g][layerIdx][neuronIdx]+=p->hidden_layer_neuron_output[g][layerIdx-1][i]*p->bp_genome[g].bp_hidden[layerIdx].weight[neuronIdx][i];
				assert(p->hidden_layer_neuron_input[g][layerIdx][neuronIdx]==p->hidden_layer_neuron_input[g][layerIdx][neuronIdx]);
			}
		}
		g+=gridDim.x;
	}
}

//<<<POPULATION_SIZE,NR_OUTPUTS>>>
__global__ void cuInputsOutputLayer(population_t *p) {
	int g=blockIdx.x;

	int neuronIdx=threadIdx.x%NR_OUTPUTS;
	while(g<POPULATION_SIZE) {
		for(int i=0;i<NEURON_NUM;i++) {
			if(NN_CONNECT_OUTPUT(p->ga_genome[g].ga_output.connect, neuronIdx,i)) {
				p->output_layer_neuron_input[g][neuronIdx]+=p->hidden_layer_neuron_output[g][HIDDEN_LAYER_NUM-1][i]*p->bp_genome[g].bp_output.weight[neuronIdx][i];
				assert(p->output_layer_neuron_input[g][neuronIdx]==p->output_layer_neuron_input[g][neuronIdx]);
			}
		}
		g+=gridDim.x;
	}
}


//<<<POPULATION_SIZE, NR_OUTPUTS>>>
__global__ void cuExciteOutputLayer(population_t *p) {
	int g=blockIdx.x;

	int outputIdx=threadIdx.x%NR_OUTPUTS;
	while(g<POPULATION_SIZE) {
		p->output_layer_neuron_output[g][outputIdx]=sigmoid(p->output_layer_neuron_input[g][outputIdx]);
		p->output_layer_neuron_output_derived[g][outputIdx]=sigmoid_derived(p->output_layer_neuron_input[g][outputIdx]);

		assert(p->output_layer_neuron_output[g][outputIdx]==p->output_layer_neuron_output[g][outputIdx]);
		assert(p->output_layer_neuron_output_derived[g][outputIdx]==p->output_layer_neuron_output_derived[g][outputIdx]);
		g+=gridDim.x;
	}
}


__host__ void hoExcite(population_t *p, IO_t *io, int sample) {
	cuInputsOutputsInputLayer<<<POPULATION_SIZE, NR_INPUTS>>>(p, io, sample);
	cuInputsHiddenLayer<<<POPULATION_SIZE, NEURON_NUM>>>(p);
	cuExciteHidden<<<POPULATION_SIZE, NEURON_NUM>>>(p, 0);

	for(int i=1;i<HIDDEN_LAYER_NUM;i++) {
		cuInputsHiddenLayer<<<POPULATION_SIZE, NEURON_NUM>>>(p, i);
		cuExciteHidden<<<POPULATION_SIZE, NEURON_NUM>>>(p,i);
	}

	cuInputsOutputLayer<<<POPULATION_SIZE,NR_OUTPUTS>>>(p);
	cuExciteOutputLayer<<<POPULATION_SIZE,NR_OUTPUTS>>>(p);
}


// ----------------------------------------------------------------------------------------
//                   Backpropagation part
// ----------------------------------------------------------------------------------------


//<<<POPULATION_SIZE, NR_OUTPUTS>>>
__global__ void cuDeltaOutputLayer(population_t *p) {
	int g=blockIdx.x;

	int neuronIdx=threadIdx.x%NR_OUTPUTS;

	while(g<POPULATION_SIZE) {
		p->delta_output_layer[g][neuronIdx]=p->output_layer_neuron_output_derived[g][neuronIdx]*
			p->error_derived[g][neuronIdx];
#ifdef DEBUG
		printf("p->delta_output_layer[%d][%d]=%f error_derived[%d][%d]=%f p->output_layer_neuron_output_derived[%d][%d]=%f\n", g,neuronIdx, p->delta_output_layer[g][neuronIdx], g, neuronIdx, p->error_derived[g][neuronIdx], g, neuronIdx, p->output_layer_neuron_output_derived[g][neuronIdx]);
#endif
		assert(p->delta_output_layer[g][neuronIdx]==p->delta_output_layer[g][neuronIdx]);
		g+=gridDim.x;
	}
}


//<<<POPULATION_SIZE, NEURON_NUM>>>
__global__ void cuDeltaHiddenLayerFromOutputLayer(population_t *p) {
	int g=blockIdx.x;

	int neuronIdx=threadIdx.x%NEURON_NUM;
	while(g<POPULATION_SIZE) {
		for(int i=0;i<NR_OUTPUTS;i++) 
			if(NN_CONNECT_OUTPUT(p->ga_genome[g].ga_output.connect, i, neuronIdx))  {
				p->delta_hidden_layer[g][HIDDEN_LAYER_NUM-1][neuronIdx]+=p->delta_output_layer[g][i]
						*p->bp_genome[g].bp_output.weight[i][neuronIdx];
#ifdef DEBUG
				printf("delta_output_layer[%d][%i]=%f output_weight[%d][%d]=%f p->delta_hidden_layer[%d][%d][%d]=%f\n", g,i,p->delta_output_layer[g][i], i, neuronIdx, p->bp_genome[g].bp_output.weight[i][neuronIdx], g, HIDDEN_LAYER_NUM-1, neuronIdx, p->delta_hidden_layer[g][HIDDEN_LAYER_NUM-1][neuronIdx]);
#endif
				assert(p->delta_hidden_layer[g][HIDDEN_LAYER_NUM-1][neuronIdx]==p->delta_hidden_layer[g][HIDDEN_LAYER_NUM-1][neuronIdx]);
			}
		g+=gridDim.x;
	}
}

//<<<POPULATION_SIZE, NEURON_NUM>>>
__global__ void cuDeltaHiddenLayer(population_t *p, int layerIdx) {
	int g=blockIdx.x;

	int neuronIdx=threadIdx.x%NEURON_NUM;
	while(g<POPULATION_SIZE) {
		for(int i=0;i<NEURON_NUM;i++) {
			if(NN_CONNECT(p->ga_genome[g].ga_hidden[layerIdx].connect, i, neuronIdx)) {
				p->delta_hidden_layer[g][layerIdx-1][neuronIdx]+=p->delta_hidden_layer[g][layerIdx][i]
					*p->bp_genome[g].bp_hidden[layerIdx].weight[i][neuronIdx];
#ifdef DEBUG
				printf("delta_hidden_layer[%d][%d][%d]=%f delta_hidden_layer[%d][%d][%d]=%f p->bp_genome[g].bp_hidden[%d].weight[%d][%d]=%f\n", g , layerIdx-1, neuronIdx, p->delta_hidden_layer[g][layerIdx-1][neuronIdx], g, layerIdx, i, p->delta_hidden_layer[g][layerIdx][i], g,layerIdx, neuronIdx, p->bp_genome[g].bp_hidden[layerIdx].weight[i][neuronIdx]);
#endif
				assert(p->delta_hidden_layer[g][layerIdx-1][neuronIdx]==p->delta_hidden_layer[g][layerIdx-1][neuronIdx]);
			}
		}
		g+=gridDim.x;
	}
}

//<<<POPULATION_SIZE, NEURON_NUM>>>
__global__ void cuMulDerivedHiddenLayer(population_t *p, int layerIdx) {
	int g=blockIdx.x;

	int neuronIdx=threadIdx.x%NEURON_NUM;
	
	while(g<POPULATION_SIZE) {
		p->delta_hidden_layer[g][layerIdx][neuronIdx]*=p->hidden_layer_neuron_output_derived[g][layerIdx][neuronIdx];
#ifdef DEBUG
		printf("p->hidden_layer_neuron_output_derived[%d][%d][%d]=%f  p->delta_hidden_layer[%d][%d][%d]=%f\n",g, layerIdx, neuronIdx, p->hidden_layer_neuron_output_derived[g][layerIdx][neuronIdx],  g, layerIdx, neuronIdx, p->delta_hidden_layer[g][layerIdx][neuronIdx]);
#endif
		g+=gridDim.x;
	}
}


//<<<POPULATION_SIZE, NR_INPUTS>>>
__global__ void cuDeltaInputLayerFromHiddenLayer(population_t *p) {
	int g=blockIdx.x;

	int inputIdx=threadIdx.x%NR_INPUTS;
	while(g<POPULATION_SIZE) {
		for(int i=0;i<NEURON_NUM;i++) {
			if(NN_CONNECT_INPUT(p->ga_genome[g].ga_input.connect, /*i, inputIdx*/ inputIdx, i))  {
				p->delta_input_layer[g][inputIdx]+=p->delta_hidden_layer[g][0][i]*p->bp_genome[g].bp_hidden[0].weight[i][inputIdx];
//				printf("%f %f\n", p->delta_hidden_layer[g][0][i], p->bp_genome[g].bp_hidden[0].weight[i][inputIdx]);
				assert(p->bp_genome[g].bp_hidden[0].weight[i][inputIdx]==p->bp_genome[g].bp_hidden[0].weight[i][inputIdx]);
				assert(p->delta_hidden_layer[g][0][i]==p->delta_hidden_layer[g][0][i]);
				assert(p->delta_input_layer[g][inputIdx]==p->delta_input_layer[g][inputIdx]);
			}
		}
		g+=gridDim.x;
	}
}

//<<<POPULATION_SIZE, NR_INPUTS>>>
__global__ void cuMulDerivedInputLayer(population_t *p) {
	int g=blockIdx.x;
	
	int inputIdx=threadIdx.x%NR_INPUTS;

	while(g<POPULATION_SIZE) {
		p->delta_input_layer[g][inputIdx]*=p->input_layer_neuron_output_derived[g][inputIdx];
		g+=gridDim.x;
	}
}

__host__ void hoBackpropagation(population_t *p) {
	cuDeltaOutputLayer<<<POPULATION_SIZE, NR_OUTPUTS>>>(p);
	cuDeltaHiddenLayerFromOutputLayer<<<POPULATION_SIZE, NEURON_NUM>>>(p);
	cuMulDerivedHiddenLayer<<<POPULATION_SIZE, NEURON_NUM>>>(p, HIDDEN_LAYER_NUM-1);

	for(int i=HIDDEN_LAYER_NUM-1;i>0;i--) {
		cuDeltaHiddenLayer<<<POPULATION_SIZE, NEURON_NUM>>>(p, i);
		cuMulDerivedHiddenLayer<<<POPULATION_SIZE, NEURON_NUM>>>(p, i-1);
	}
	cuDeltaInputLayerFromHiddenLayer<<<POPULATION_SIZE, NR_INPUTS>>>(p);
	cuMulDerivedInputLayer<<<POPULATION_SIZE, NR_INPUTS>>>(p);
}

// ----------------------------------------------------------------------------------
//                     Add weights for adjustenment
// ----------------------------------------------------------------------------------

//<<<POPULATION_SIZE, NR_INPUTS>>>
__global__ void cuAddAdjustInputLayer(population_t *p, IO_t *io, int sample) {
	int g=blockIdx.x;
	int inputIdx=threadIdx.x%NR_INPUTS;

	while(g<POPULATION_SIZE) {
		p->bp_genome[g].bp_input.weight_adjust[inputIdx]+=p->delta_input_layer[g][inputIdx]*
			io->inputs[inputIdx][sample];
#ifdef DEBUG
		printf("p->delta_input_layer[%d][%d]=%f\n", g, inputIdx, p->delta_input_layer[g][inputIdx]);
#endif
		g+=gridDim.x;
	}
}

//<<<POPULATION_SIZE, dim3(NEURON_NUM,NR_INPUTS>>>
__global__ void cuAddAdjustHiddenLayerFromInputLayer(population_t *p) {
	int g=blockIdx.x;
	int neuronIdx=threadIdx.x%NEURON_NUM;
	int inputIdx=threadIdx.y%NR_INPUTS;

	while(g<POPULATION_SIZE) {
		if(NN_CONNECT_INPUT(p->ga_genome[g].ga_input.connect, /*neuronIdx, inputIdx*/ inputIdx, neuronIdx)) 
			p->bp_genome[g].bp_hidden[0].weight_adjust[neuronIdx][inputIdx]+=
				p->input_layer_neuron_output[g][inputIdx]*p->delta_hidden_layer[g][0][neuronIdx];
		g+=gridDim.x;
	}
}

//<<<POPULATION_SIZE, dim3(NEURON_NUM, NEURON_NUM)>>>
__global__ void cuAddAjustHiddenLayer(population_t *p) {
	int g=blockIdx.x;
	int neuronIdx1=threadIdx.x%NEURON_NUM;
	int neuronIdx2=threadIdx.y%NEURON_NUM;

	while(g<POPULATION_SIZE) {
		for(int i=1;i<HIDDEN_LAYER_NUM;i++) {
			if(NN_CONNECT(p->ga_genome[g].ga_hidden[i].connect, neuronIdx2,neuronIdx1))
				p->bp_genome[g].bp_hidden[i].weight_adjust[neuronIdx2][neuronIdx1]+=
					p->hidden_layer_neuron_output[g][i-1][neuronIdx1]*p->delta_hidden_layer[g][i][neuronIdx2];
		}
		g+=gridDim.x;
	}
}

//<<<POPULATION_SIZE, NEURON_NUM>>>
__global__ void cuAddAdjustOutputLayer(population_t *p) {
	int g=blockIdx.x;
	int neuronIdx=threadIdx.x%NEURON_NUM;
	
	while(g<POPULATION_SIZE) {
		for(int i=0;i<NR_OUTPUTS;i++) {
			if(NN_CONNECT_OUTPUT(p->ga_genome[g].ga_output.connect, i, neuronIdx)) 
				p->bp_genome[g].bp_output.weight_adjust[i][neuronIdx]+=
					p->hidden_layer_neuron_output[g][HIDDEN_LAYER_NUM-1][neuronIdx]*p->delta_output_layer[g][i];
		}
		
		g+=gridDim.x;
	}
}

__host__ void hoAddAdjustWeights(population_t *p, IO_t *io, int sample) {
	cuAddAdjustInputLayer<<<POPULATION_SIZE, NR_INPUTS>>>(p, io, sample);
	cuAddAdjustHiddenLayerFromInputLayer<<<POPULATION_SIZE, dim3(NEURON_NUM,NR_INPUTS)>>>(p);
	cuAddAjustHiddenLayer<<<POPULATION_SIZE, dim3(NEURON_NUM, NEURON_NUM)>>>(p);
	cuAddAdjustOutputLayer<<<POPULATION_SIZE, NEURON_NUM>>>(p);
	
}

// ---------------------------------------------------------------------------------------
//                           Weight adjustenment part
// ---------------------------------------------------------------------------------------


//<<<POPULATION_SIZE, NR_INPUTS>>>
__global__ void cuAdjustInputLayer(population_t *p) {
	int g=blockIdx.x;
	int inputIdx=threadIdx.x%NR_INPUTS;

	while(g<POPULATION_SIZE) {
		p->bp_genome[g].bp_input.weight[inputIdx]-=LEARNING_RATE*p->bp_genome[g].bp_input.weight_adjust[inputIdx];
#ifdef DEBUG
		printf("p->bp_genome[%d].bp_input.weight_adjust[%d]=%f\n", g, inputIdx, p->bp_genome[g].bp_input.weight_adjust[inputIdx]);
#endif
		g+=gridDim.x;
	}
}

//<<<POPULATION_SIZE, dim3(NEURON_NUM,NR_INPUTS>>>
__global__ void cuAdjustHiddenLayerFromInputLayer(population_t *p) {
	int g=blockIdx.x;
	int neuronIdx=threadIdx.x%NEURON_NUM;
	int inputIdx=threadIdx.y%NR_INPUTS;

	while(g<POPULATION_SIZE) {
		if(NN_CONNECT_INPUT(p->ga_genome[g].ga_input.connect, /*neuronIdx, inputIdx*/ inputIdx, neuronIdx))  {
			p->bp_genome[g].bp_hidden[0].weight[neuronIdx][inputIdx]-=LEARNING_RATE*p->bp_genome[g].bp_hidden[0].weight_adjust[neuronIdx][inputIdx];
#ifdef DEBUG
			printf("p->bp_genome[%d].bp_hidden[0].weight_adjust[%d][%d]=%f\n", g, neuronIdx,inputIdx,p->bp_genome[g].bp_hidden[0].weight_adjust[neuronIdx][inputIdx]);
#endif
		}
		g+=gridDim.x;
	}
}

//<<<POPULATION_SIZE, dim3(NEURON_NUM, NEURON_NUM)>>>
__global__ void cuAjustHiddenLayer(population_t *p) {
	int g=blockIdx.x;
	int neuronIdx1=threadIdx.x%NEURON_NUM;
	int neuronIdx2=threadIdx.y%NEURON_NUM;

	while(g<POPULATION_SIZE) {
		for(int i=1;i<HIDDEN_LAYER_NUM;i++) {
			if(NN_CONNECT(p->ga_genome[g].ga_hidden[i].connect, neuronIdx2,neuronIdx1)) {
				p->bp_genome[g].bp_hidden[i].weight[neuronIdx2][neuronIdx1]-=LEARNING_RATE*p->bp_genome[g].bp_hidden[i].weight_adjust[neuronIdx2][neuronIdx1];
#ifdef DEBUG
				printf("p->bp_genome[%d].bp_hidden[%d].weight_adjust[%d][%d]=%f\n", g,i, neuronIdx2, neuronIdx1, p->bp_genome[g].bp_hidden[i].weight_adjust[neuronIdx2][neuronIdx1]);
#endif
			}
		}
		g+=gridDim.x;
	}
}

//<<<POPULATION_SIZE, NEURON_NUM>>>
__global__ void cuAdjustOutputLayer(population_t *p) {
	int g=blockIdx.x;
	int neuronIdx=threadIdx.x%NEURON_NUM;
	
	while(g<POPULATION_SIZE) {
		for(int i=0;i<NR_OUTPUTS;i++) {
			if(NN_CONNECT_OUTPUT(p->ga_genome[g].ga_output.connect, i, neuronIdx)) 
				p->bp_genome[g].bp_output.weight[i][neuronIdx]-=LEARNING_RATE*p->bp_genome[g].bp_output.weight_adjust[i][neuronIdx];
		
		}
		g+=gridDim.x;
	}
}

__host__ void hoAdjustWeights(population_t *p) {
	cuAdjustInputLayer<<<POPULATION_SIZE, NR_INPUTS>>>(p);
	cuAdjustHiddenLayerFromInputLayer<<<POPULATION_SIZE, dim3(NEURON_NUM,NR_INPUTS)>>>(p);
	cuAjustHiddenLayer<<<POPULATION_SIZE, dim3(NEURON_NUM, NEURON_NUM)>>>(p);
	cuAdjustOutputLayer<<<POPULATION_SIZE, NEURON_NUM>>>(p);
	
}




// ----------------------------------------------------------------------------------------
//                  GA part
// ----------------------------------------------------------------------------------------

__device__ inline float my_random (population_t *p) {
  return ((float) (curand (&p->curandState[THREAD_ID]) / ((float)(0x0FFFFFFFFUL))));
}

__device__ void mutate (population_t *p, double prob, ga_genome_t * genome) {
  double r = my_random (p);
  if (r < prob) {
	int s = curand(&p->curandState[THREAD_ID]) % (sizeof (ga_genome_t) * 8);
    ((char *) genome)[s / 8] ^= (1 << (s % 8));
    }
}

__device__ void cross(void* ng1, void* g1,void* ng2, void* g2, int s, int size) {

      memcpy (ng1, g1, s / 8);
      memcpy (CHAR (ng1) + s / 8, CHAR (g2) + s / 8, size - s / 8);
      char mask = (1 << (s % 8)) - 1;
      CHAR (ng1)[s / 8] = (CHAR (g1)[s / 8] & ~mask) | (CHAR (g2)[s / 8] & mask);

      memcpy (ng2, g2, s / 8);
      memcpy (CHAR (ng2) + s / 8 , CHAR (g1) + s / 8 , size - s / 8);
      mask = (1 << (s % 8)) - 1;
      CHAR (ng1)[s / 8] = (CHAR (g2)[s / 8] & ~mask) | (CHAR (g1)[s / 8] & mask); 
}

__device__ void _crossover (Counter *c, double prob, population_t * population, population_t * new_population) {


  float r = my_random (population);
  float total = 0;
  assert(c->getPopulationIndex() < POPULATION_SIZE);
  ga_genome_t *ng1 = &new_population->ga_genome[c->getPopulationIndexInc()];
  assert(c->getPopulationIndex() < POPULATION_SIZE);
  ga_genome_t *ng2 = &new_population->ga_genome[c->getPopulationIndexInc()];


  for (int i = 0; i < POPULATION_SIZE; i++)
    {
      total += population->fitness[i];
    }
  ga_genome_t *g1 = 0;
  ga_genome_t *g2 = 0;
  while (g1 == g2) {
      float r1 = my_random (population) * total;
      float r2 = my_random (population) * total;
      float sum = 0;
      for (int i = 0; i < POPULATION_SIZE; i++)
	{
	  sum += population->fitness[i];
	  if (sum >= r1)
	    {
	      g1 = &population->ga_genome[i];
	      break;
	    }
	}
      sum = 0;
      for (int i = 0; i < POPULATION_SIZE; i++)
	{
	  sum += population->fitness[i];
	  if (sum >= r2)
	    {
	      g2 = &population->ga_genome[i];
	      break;
	    }
	}
  }

  if (r < prob)
    {

       int s= curand(&population->curandState[THREAD_ID]) % (sizeof(ga_genome_t)*8);

       cross(ng1,g1,ng2,g2,s,sizeof(ga_genome_t));
    }
  else
    {
      memcpy (ng1, g1, sizeof (ga_genome_t));
      memcpy (ng2, g2, sizeof (ga_genome_t));
    }

  mutate (population, MUTATION_PROB, ng1);
  mutate (population, MUTATION_PROB, ng2);
}

__global__ void crossover (double prob, population_t * population, population_t * new_population) {


	Counter c;
/*
    if(THREAD_ID==0) {
		for(int i=0; i<BEST_INDIVIDUALS && i<POPULATION_SIZE;i++) {
	        c.getPopulationIndexInc();
		}
    }
	__syncthreads();
*/

    while(c.getPopulationIndexEven() < POPULATION_SIZE && c.getPopulationIndexOdd() < POPULATION_SIZE) {
	     _crossover(&c, prob, population, new_population);
    }

}

__global__ void init_random_population (population_t* currentPopulation)
{
  int g=blockIdx.x;
  while(g < POPULATION_SIZE) {
      for (int j = 0; j < sizeof (ga_genome_t); j++) {
    	  ((char *) &(currentPopulation->ga_genome[g]))[j] = curand (&currentPopulation->curandState[g]) % ~(0U);
	}
	g+=gridDim.x;
 }
}

__global__ void find_best_individual(population_t* population, float *deviceBestIndividualFitness) {
  if(THREAD_ID==0) {
//	qsort(population);
    *deviceBestIndividualFitness=population->fitness[population->map[0]];
/*
	for(int i=0;i<POPULATION_SIZE;i++) 
		printf("%f,",population->fitness[population->map[i]]);
	printf("\n\n");
*/
  }
}

__host__ void host_find_best_individual(population_t *p, float *deviceBestIndividualFitness) {

	find_best_individual<<<1,1>>>(p, deviceBestIndividualFitness);	
	check_cuda_errors(__FILE__, __LINE__);
}

__global__ void copy_best_individuals (population_t * p1, population_t * p2)
{
    if(THREAD_ID==0) {
		Counter c;
        for(int i=0;i<BEST_INDIVIDUALS && i<POPULATION_SIZE && c.getPopulationIndex() < POPULATION_SIZE;i++) {
		  int pi=c.getPopulationIndexInc();
          memcpy (&p2->ga_genome[pi], &p1->ga_genome[p1->map[i]], sizeof (ga_genome_t));
          memcpy (&p2->bp_genome[pi], &p1->bp_genome[p1->map[i]], sizeof (bp_genome_t));
		  
        }
    }
}

__global__ void cuError(population_t *p, IO_t *io, int sample) {
	int g=blockIdx.x;
	int oIdx=threadIdx.x%NR_OUTPUTS;

	while(g < POPULATION_SIZE) {
		__shared__ float error[NR_OUTPUTS+1];

		if(threadIdx.x==0)
			error[NR_OUTPUTS]=0.;
		__syncthreads();

		//ga_genome_t *ga_genome=&p->ga_genome[g];

		float output=p->output_layer_neuron_output[g][oIdx];
		float ed=output-io->outputs[oIdx][sample];
		p->error_derived[g][oIdx]=ed;
		error[oIdx]=.5*pow(ed,2);
		p->error[g][oIdx]=error[oIdx];
		
		for(int stride=16>>1;stride>0;stride>>=1) {
			__syncthreads();
			if(threadIdx.x<stride) {
				error[threadIdx.x]+=error[threadIdx.x+stride];
			}
		}

		__syncthreads();
		if(threadIdx.x==0) {
			p->mse[g]+=error[0];
		}
		__syncthreads();

		g+=gridDim.x;
	}
}

__global__ void cuFitness(population_t *p) {
	int g=blockIdx.x;

	while(g < POPULATION_SIZE) {
		p->fitness[g]=1./(p->mse[g]+0.00001);
		p->map[g]=g;
		g+=gridDim.x;
	}
}

__global__ void cuMaxFitness(population_t *p) {
	for(int stride=blockDim.x>>1;stride>0;stride>>=1) {
		__syncthreads();
		if(threadIdx.x<stride) {
			if(p->fitness[p->map[threadIdx.x]]<p->fitness[p->map[threadIdx.x+stride]]) {
				int v=p->map[threadIdx.x];
				p->map[threadIdx.x]=p->map[threadIdx.x+stride];
				p->map[threadIdx.x+stride]=v;
			}
		}
	}
	
}

void resetFitness(population_t *p) {
		hoResetInput(p);
		hoResetOutput(p);
		hoResetMse(p);	
}

void fitness(population_t *population1,IO_t *io, int sample) {
		hoExcite(population1, io, sample);

		(cuError<<<POPULATION_SIZE,NR_OUTPUTS>>>(population1, io, sample));
		check_cuda_errors(__FILE__,__LINE__);

		(cuFitness<<<POPULATION_SIZE,1>>>(population1));
		check_cuda_errors(__FILE__,__LINE__);
		(cuMaxFitness<<<1,POPULATION_SIZE>>>(population1));
		check_cuda_errors(__FILE__,__LINE__);
}

void train(population_t *p, IO_t *io, int start_sample, int stop_sample) {
	for(int j=0;j<BP_NUM;j++) {
		for(int sample=start_sample;sample<stop_sample; sample++) {
		hoResetWeightAdjust(p);
//		check_cuda_errors(__FILE__,__LINE__);
			hoResetDelta(p);
			hoResetOutput(p);
			hoResetInput(p);
			check_cuda_errors(__FILE__,__LINE__);

			hoExcite(p, io, sample);
			check_cuda_errors(__FILE__,__LINE__);
			cuError<<<POPULATION_SIZE,NR_OUTPUTS>>>(p, io, sample);
			check_cuda_errors(__FILE__,__LINE__);
			hoBackpropagation(p);
			check_cuda_errors(__FILE__,__LINE__);
			hoAddAdjustWeights(p, io, sample);
			check_cuda_errors(__FILE__,__LINE__);
			hoAdjustWeights(p);
		}
	}
}

__global__ void print_outputs(population_t *p, IO_t *io,int sample) {
		int b=p->map[0];
		printf("input: ");
		for(int j=0;j<NR_INPUTS;j++) {
			printf("%d,", io->inputs[j][sample]>0.5?1:0);
		}
		printf(" output: ");
		for(int j=0;j<NR_OUTPUTS;j++) {
			printf("%d,",io->outputs[j][sample]>.5?1:0);
		}
		int ham=0;
		printf("\noutput: [");
		for(int j=0;j<NR_OUTPUTS;j++) {
			int out=p->output_layer_neuron_output[b][j]>0.5?1:0;
			ham+=out^(io->outputs[j][sample]>0.5?1:0);
			printf("%d,", out);
		}
		printf("]  ");
		for(int j=0;j<NR_OUTPUTS;j++) {
			printf("%f,", p->output_layer_neuron_output[b][j]);
		}
		printf(" err: %f ", p->mse[b]);
		printf("fitness: %f", p->fitness[b]);
		printf(" distance: %d\n", ham);
}

void print_best(population_t *population, IO_t *io, float *deviceBestIndividualFitness, int start_sample, int stop_sample) {
	for(int i=start_sample;i<stop_sample;i++) {
		resetFitness(population);
		fitness(population, io, i);
		host_find_best_individual(population, deviceBestIndividualFitness);
		print_outputs<<<1,1>>>(population,io, i);
	}
}

__global__ void cuRandInit(population_t *p) {
		int g=blockIdx.x;
		while(g < POPULATION_SIZE) {
	    	curand_init(2345,THREAD_ID, 0, &p->curandState[g]);
			g+=blockDim.x;
		}
}

void cuInit(population_t* p1, population_t *p2) {
		cuRandInit<<<POPULATION_SIZE,1>>>(p1);
		hoReset(p1);
		cuRandInit<<<POPULATION_SIZE,1>>>(p2);
		hoReset(p2);
}

// ----------------------------------------------------------------------------------------------------
//                       MAIN PART
// ----------------------------------------------------------------------------------------------------
void genetic (IO_t *io,population_t* population1, population_t* population2, float *deviceBestIndividualFitness) {


      int start_sample=0;
      int stop_sample=1;
      int stride=SAMPLES;

	  cuInit(population1, population2);

  	  init_random_population<<<128,1>>> (population1);
  	  init_random_population<<<128,1>>> (population2);

	 float tmpFitness=-10000;

      while(stride <= SAMPLES) {
          int g=0;
          stop_sample=stride;
          start_sample=stride-stop_sample;

        while(stop_sample<=SAMPLES) {  
              printf("training %d-%d ...\n", start_sample, stop_sample-1);

          int it=0;

          do {
			
//			hoReset(population1);
			  hoResetExceptWeights(population1);
			  train(population1, io,start_sample, stop_sample);
		hoResetMse(population1);
			  for(int i=start_sample; i<stop_sample;i++) {
		hoResetInput(population1);
		hoResetOutput(population1);
//				  resetFitness(population1);
				  fitness(population1, io, i);
			  }
/*
		  	  host_find_best_individual(population1,deviceBestIndividualFitness);
			  copy_best_individuals<<<POPULATION_SIZE/2,1>>> (population1, population2);
			  check_cuda_errors(__FILE__,__LINE__);
*/

              if(g%10==0) {
			  	  printf("generation %d\n", it);
				  host_find_best_individual(population1,deviceBestIndividualFitness);
                  checkCudaErrors(cudaMemcpy(hostBestIndividualFitness, deviceBestIndividualFitness, sizeof(float), cudaMemcpyDeviceToHost));
				  printf("%f %f\n", tmpFitness, *hostBestIndividualFitness);
//				  assert(tmpFitness <= *hostBestIndividualFitness);
			  	  tmpFitness=*hostBestIndividualFitness;
				  printf("best fitness: %f %d-%d\n", *hostBestIndividualFitness,start_sample, stop_sample-1);
				  print_best(population1, io, deviceBestIndividualFitness, start_sample, stop_sample);
				  check_cuda_errors(__FILE__, __LINE__);
				  printf("---------------------------------------\n");
              }

//			  if(it>0) { // kvoli pocitaniu fitness, pri 0 este nie je vypocitany
			 	  crossover<<<POPULATION_SIZE/2,1>>> (CROSSOVER_PROB, population1, population2);
				  check_cuda_errors(__FILE__,__LINE__);
//			  }

//		  	  host_find_best_individual(population1,deviceBestIndividualFitness);
			  copy_best_individuals<<<POPULATION_SIZE/2,1>>> (population1, population2);
			  check_cuda_errors(__FILE__,__LINE__);


              {
                  population_t *p = population1;
                  population1 = population2;
                  population2 = p;
              }



            g++;
            it++;
          } while (*hostBestIndividualFitness < 5);
	  		*hostBestIndividualFitness=0.;

			tmpFitness=-10000;
          stop_sample+=stride;
          start_sample=stop_sample-stride;
        }
        stride<<=1;
    }
    start_sample=0;

}

void read_io(IO_t &io) {
#if 0
  const char* in[]={
#include "switch_in.dat.1"
      };
  const char* out[]={
#include "switch_out.dat.1"
      };
  for(int k=0;k<SAMPLES;k++) {
      sscanf(in[k],"%f,%f,%f", 
        &io.inputs[0][k],
        &io.inputs[1][k],
        &io.inputs[2][k]
        );
  //    cout << io.inputs[0][k] <<"," << io.inputs[1][k] << "," << io.inputs[2][k] << endl;
  }
  for(int k=0;k<SAMPLES;k++) {
      sscanf(out[k],"%f,%f", 
        &io.outputs[0][k],
        &io.outputs[1][k]
        );
#else
      const char* in[]={
#include "bch_input.dat.3"
      };
      const char* out[]={
#include "bch_output.dat.3"
      };
  for(int k=0;k<SAMPLES;k++) {
      sscanf(in[k],"%f,%f,%f,%f,%f", 
        &io.inputs[0][k],
        &io.inputs[1][k],
        &io.inputs[2][k],
        &io.inputs[3][k],
        &io.inputs[4][k]
        );
  //    cout << io.inputs[0][k] <<"," << io.inputs[1][k] << "," << io.inputs[2][k] << endl;
  }
  for(int k=0;k<SAMPLES;k++) {
      sscanf(out[k],"%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f", 
        &io.outputs[0][k],
        &io.outputs[1][k],
        &io.outputs[2][k],
        &io.outputs[3][k],
        &io.outputs[4][k],
        &io.outputs[5][k],
        &io.outputs[6][k],
        &io.outputs[7][k],
        &io.outputs[8][k],
        &io.outputs[9][k],
        &io.outputs[10][k],
        &io.outputs[11][k],
        &io.outputs[12][k],
        &io.outputs[13][k],
        &io.outputs[14][k]
        );
#endif
   //   cout << io.outputs[0][k] <<"," << io.outputs[1][k] << endl;
  }
      /*
  memset(io.inputs,0, sizeof(io.inputs));
  for(int k=0;k<=SAMPLES;k++) {
      for(int l=0;l<20;l++) {
          io.inputs[keno_input2[k][l]-1][k]=1;
      }
  }
  memset(io.outputs,0, sizeof(io.outputs));
  for(int k=0;k<SAMPLES;k++) {
      for(int l=0;l<20;l++) {
          io.outputs[keno_output2[k][l]-1][k]=1;
      }
  }
  cout << "sizeof " << sizeof(io.inputs) << "/" << sizeof(double)  << endl;
  */

}
int main() {
	assert(NR_INPUTS<=NEURON_NUM);
	printf("population: %ld\n", sizeof(population_t)/1024/1024);
	printf("ga_genome_t %ld\n", sizeof(ga_genome_t));
	printf("bp_genome_t %ld\n", sizeof(bp_genome_t));

	population_t *p1,*p2;
	IO_t tmp;
	IO_t *io;

	read_io(tmp);
	
	cudaMalloc((void**)&p1, sizeof(population_t));
	cudaMalloc((void**)&p2, sizeof(population_t));
  	cudaMalloc((void**)&deviceBestIndividualFitness, sizeof(double));
	cudaHostAlloc((void**)&hostBestIndividualFitness,sizeof(double), cudaHostAllocWriteCombined);
	cudaMalloc((void**)&io,sizeof(IO_t));

	cudaMemcpy(io,&tmp,sizeof(IO_t),cudaMemcpyHostToDevice);

	hoReset(p1);
	hoReset(p2);
	genetic(io, p1,p2, deviceBestIndividualFitness);

	cudaFree(p1);
	cudaFree(p2);
	cudaFree(io);
	cudaFree(hostBestIndividualFitness);
	return 0;
}
