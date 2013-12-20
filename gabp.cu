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

using namespace std;

#define NEURON_NUM 8
#define CORTEX_NUM 1
#define THREAD_PER_NEURON 1
#define POPULATION_SIZE 256
#define MUTATION_PROB 0.15
#define CROSSOVER_PROB 0.8
#define BEST_INDIVIDUALS 2

#define NR_INPUTS 5
#define NR_OUTPUTS 15
#define SAMPLES 16

#define _THREAD_COUNT (blockDim.x*gridDim.x)
#define THREAD_ID (blockIdx.x*blockDim.x + threadIdx.x)
#define CHAR(x) ((char*)x)

#define INTERCONNECT_VALUE(i,c,n) (i[c][n])
#define INTERCONNECT_CORTEX(i,c,n) (INTERCONNECT_VALUE(i,c,n)&0xf)
#define INTERCONNECT_NEURON(i,c,n) ((INTERCONNECT_VALUE(i,c,n)>>4)&0x7)
#define NN_CONNECT(connect,x,y) (connect[(y*NEURON_NUM+x)/8]&(1<<((y*NEURON_NUM+x)%8)))
#define CC_CONNECT(connect,x,y) ((INTERCONNECT_VALUE(connect,x,y)>>4)&0x8)

#define NEURON_IDX(output_neuron_idx) ((output_neuron_idx&0xf)%NEURON_NUM)
#define CORTEX_IDX(output_neuron_idx) (((output_neuron_idx>>4)&0xf)%CORTEX_NUM)

typedef struct {
	char connect[NEURON_NUM*NEURON_NUM/8];
} ga_cortex_t;

typedef struct {
	ga_cortex_t ga_cortex[CORTEX_NUM];
	char connect[CORTEX_NUM][NEURON_NUM]; //8 bit
	unsigned char output_neuron_idx[NR_OUTPUTS];
} ga_genome_t;

typedef struct {
	float weight[NEURON_NUM][NEURON_NUM];
} bp_cortex_t;

typedef struct {
	bp_cortex_t bp_cortex[CORTEX_NUM];
	float weight[CORTEX_NUM][NEURON_NUM][CORTEX_NUM][NEURON_NUM];
	float input_weight[NR_INPUTS][CORTEX_NUM][NEURON_NUM];
} bp_genome_t;

typedef struct {
	ga_genome_t ga_genome[POPULATION_SIZE];
	bp_genome_t bp_genome[POPULATION_SIZE];
	float fitness[POPULATION_SIZE];

	float neuron_output[POPULATION_SIZE][CORTEX_NUM][NEURON_NUM];
	float neuron_output_derived[POPULATION_SIZE][CORTEX_NUM][NEURON_NUM];
	float input[NR_INPUTS];
	float output[NR_OUTPUTS];
	float error[POPULATION_SIZE];
	float error_t[POPULATION_SIZE][NR_OUTPUTS];
	float error_t_derived[POPULATION_SIZE][NR_OUTPUTS];
	float delta[POPULATION_SIZE][CORTEX_NUM][NEURON_NUM];
	float outputs[POPULATION_SIZE][NR_OUTPUTS];
	
	curandState_t curandState[POPULATION_SIZE];
	int map[POPULATION_SIZE];
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


// --------------------------------------------------------------------------------------------------------

inline void check_cuda_errors(const char *filename, const int line_number)
{
      cudaDeviceSynchronize();
      cudaError_t error = cudaGetLastError();
      if(error != cudaSuccess) {
          printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
          exit(-1);
      }
}

__device__ inline float sigmoid(float signal) {
	return 1./(1+exp(-2.*signal));
}

__device__ inline float sigmoid_derived(float signal) {
	float s=sigmoid(signal);
	return s*(1-s);
}


__global__ void cuReset(population_t *p) {
	if(THREAD_ID==0) {
		memset(p->fitness, 0, sizeof(p->fitness));
		memset(p->neuron_output, 0, sizeof(p->neuron_output));
		memset(p->input, 0, sizeof(p->input));
		memset(p->output, 0, sizeof(p->output));
		memset(p->error, 0, sizeof(p->error));
		memset(p->outputs, 0, sizeof(p->outputs));
		for(int i=0;i<POPULATION_SIZE;i++) {
			p->map[i]=i;
		}
	}
		
}

__global__ void cuResetError(population_t *p) {
	int g=blockIdx.x;
	while(g < POPULATION_SIZE) {
		p->error[g]=0.f;
		g+=gridDim.x;
	}
}

__global__ void cuResetNeurons(population_t *p) {
	int g=blockIdx.x;

	while(g < POPULATION_SIZE) {
		memset(p->neuron_output[g], 0, sizeof(float)*CORTEX_NUM*NEURON_NUM);
		g+=gridDim.x;
	}
}


__global__ void cuResetDelta(population_t *p) {
	int g=blockIdx.x;

	while(g<POPULATION_SIZE) {
		p->delta[g][threadIdx.x%CORTEX_NUM][threadIdx.z%NEURON_NUM]=0.f;
		g+=gridDim.x;
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
		cuReset<<<1,1>>>(p1);
		cuRandInit<<<POPULATION_SIZE,1>>>(p2);
		cuReset<<<1,1>>>(p2);
}

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
    if(THREAD_ID==0) {
		for(int i=0; i<BEST_INDIVIDUALS && i<POPULATION_SIZE;i++) {
	        c.getPopulationIndexInc();
		}
    }

/*
    while(c.getPopulationIndexEven() < POPULATION_SIZE && c.getPopulationIndexOdd() < POPULATION_SIZE) {
//		if(c.getPopulationIndexEven()==0 ||
//			c.getPopulationIndexEven()==16)
//		printf("cross na %d %d\n", c.getPopulationIndexEven(), c.getPopulationIndexOdd());
	     _crossover(&c, prob, population, new_population);
    }
*/

}

__global__ void cuInputs(population_t *p, IO_t *io,int sample) {
	p->input[threadIdx.x]=io->inputs[threadIdx.x][sample];
//	printf("in %d %f,", threadIdx.x, p->input[threadIdx.x]);
}

__global__ void cuOutputs(population_t *p, IO_t *io, int sample) {
	p->output[threadIdx.x]=io->outputs[threadIdx.x][sample];
//	printf("out: %d %f,", threadIdx.x, p->output[threadIdx.x]);
}

__global__ void cuExcite(population_t* p) {
	int g=blockIdx.x;

	int cIdx=threadIdx.x;
	int nIdx=threadIdx.y;



	while(g < POPULATION_SIZE) {
		__shared__ float neuron_output[CORTEX_NUM][NEURON_NUM];
//		__shared__ float neuron_output_derived[CORTEX_NUM][NEURON_NUM];
		__shared__ float _signal[CORTEX_NUM][NEURON_NUM][THREAD_PER_NEURON];
		ga_genome_t *ga_genome=&p->ga_genome[g];
		bp_genome_t *bp_genome=&p->bp_genome[g];

			

		neuron_output[cIdx][nIdx]=p->neuron_output[g][cIdx][nIdx];
		_signal[cIdx][nIdx][threadIdx.z]=0.f;

		__syncthreads();

		float signal=0.;
		// toto je vypocet jedneho cortexu
		for(int n=threadIdx.z;n<NEURON_NUM;n+=blockDim.z) {
			if(NN_CONNECT(ga_genome->ga_cortex[cIdx].connect,nIdx,n)) {
				float factor=bp_genome->bp_cortex[cIdx].weight[nIdx][n];
				_signal[cIdx][nIdx][threadIdx.z]+=factor*neuron_output[cIdx][n];
			}
		}
		__syncthreads();
		// toto su signaly z ostatnych cortexov
		for(int n=0;n<CORTEX_NUM;n++) {
			for(int m=threadIdx.z;m<NEURON_NUM;m+=blockDim.z) {
				if(CC_CONNECT(ga_genome->connect,cIdx,n)) {
						if(INTERCONNECT_CORTEX(ga_genome->connect,n,m)==cIdx)
							if( INTERCONNECT_NEURON(ga_genome->connect,n,m)==nIdx) {
							float factor=bp_genome->weight[cIdx][nIdx][n][m];
							_signal[cIdx][nIdx][threadIdx.z]+=factor*neuron_output[n][m];
						}
				}
			}
		}
		__syncthreads();
		// toto su signaly z zo vstupov
		for(int n=threadIdx.z;n<NR_INPUTS;n+=blockDim.z) {
			float factor=bp_genome->input_weight[n][cIdx][nIdx];
			_signal[cIdx][nIdx][threadIdx.z]+=factor*p->input[n];
		}

		for(int stride=blockDim.z>>1;stride>0;stride>>=1) {
			__syncthreads();
			if(threadIdx.z<stride)
				_signal[cIdx][nIdx][threadIdx.z]+=_signal[cIdx][nIdx][threadIdx.z+stride];
		}
		__syncthreads();
		if(threadIdx.z==0) {
			signal=_signal[cIdx][nIdx][0];
		}

		__syncthreads();
		if(threadIdx.z==0)
			neuron_output[cIdx][nIdx]=sigmoid(signal);
//		printf("neuron_output: %d %d %d %f\n", g, cIdx,nIdx,neuron_output[cIdx][nIdx]);

		__syncthreads();	
		if(threadIdx.z==0) {
			p->neuron_output[g][cIdx][nIdx]=neuron_output[cIdx][nIdx];
			p->neuron_output_derived[g][cIdx][nIdx]=sigmoid_derived(neuron_output[cIdx][nIdx]);
		}
		__syncthreads();
		g+=gridDim.x;
	}
}

__global__ void cuError(population_t *p) {
	int g=blockIdx.x;
	int oIdx=threadIdx.x%NR_OUTPUTS;

	while(g < POPULATION_SIZE) {
		__shared__ float error[NR_OUTPUTS+1];
		if(threadIdx.x==0)
			error[NR_OUTPUTS]=0.;
		__syncthreads();
		ga_genome_t *ga_genome=&p->ga_genome[g];
		int nIdx=NEURON_IDX(ga_genome->output_neuron_idx[oIdx]);
		int cIdx=CORTEX_IDX(ga_genome->output_neuron_idx[oIdx]);
		float output=p->neuron_output[g][cIdx][nIdx];
//		printf("neuron: %d %d %f\n", g, oIdx, output);
		p->outputs[g][oIdx]=output;
//        int b1=p->output[oIdx]>0.?1:0;
//        int b2=output>0.?1:0;
//		printf("e: %d %d %d %d %d %f %f\n",g, cIdx, nIdx, b1, b2, output, p->output[oIdx]); 
		p->error_t_derived[g][oIdx]=powf(p->output[oIdx] - output,2);
		error[oIdx]=.5*p->error_t_derived[g][oIdx];//(SAMPLES*NR_OUTPUTS);
		p->error_t[g][oIdx]=error[oIdx];
		
		for(int stride=16>>1;stride>0;stride>>=1) {
			__syncthreads();
			if(threadIdx.x<stride) {
				error[threadIdx.x]+=error[threadIdx.x+stride];
			}
		}

		__syncthreads();
		if(threadIdx.x==0) {
			p->error[g]+=error[0];
//				printf("error: %d %f\n", g, p->error[g]);
		}
		__syncthreads();

		g+=gridDim.x;
	}
}

__global__ void cuBackpropagation(population_t *p) {
	int g=blockDim.x;

	int cIdx=threadIdx.x%CORTEX_NUM;
/*
	int nIdx=threadIdx.y%NEURON_NUM;
*/

	while(g<POPULATION_SIZE) {
		ga_genome_t *ga_genome=&p->ga_genome[g];
		bp_genome_t *bp_genome=&p->bp_genome[g];
		for(int i=0;i<NR_OUTPUTS;i++) {
			int nIdx=NEURON_IDX(ga_genome->output_neuron_idx[i]);
			int cIdx=CORTEX_IDX(ga_genome->output_neuron_idx[i]);
			p->delta[g][cIdx][nIdx]+=p->error_t_derived[g][i]*p->neuron_output_derived[g][cIdx][nIdx];
		}

		__syncthreads();
		for(int i=0;i<NEURON_NUM;i++) {
			for(int j=0;j<NEURON_NUM;j++) {
				if(NN_CONNECT(ga_genome->ga_cortex[cIdx].connect, j,i)) {
					p->delta[g][cIdx][i]+=bp_genome->bp_cortex[cIdx].weight[j][i]*p->delta[g][cIdx][j];
				}
			}	
		}

			
		__syncthreads();
		g+=gridDim.x;
	}
}

__global__ void cuAdjustWeights(population_t *p,IO_t *io, int sample) {
	int g=blockDim.x;

	int cIdx=threadIdx.x%CORTEX_NUM;

	while(g < POPULATION_SIZE) {
		ga_genome_t *ga_genome=&p->ga_genome[g];
		bp_genome_t *bp_genome=&p->bp_genome[g];

		for(int i=0;i<NEURON_NUM;i++) {
			for(int j=0;j<NEURON_NUM;j++) {
				if(NN_CONNECT(ga_genome->ga_cortex[cIdx].connect, j,i)) {
					bp_genome->bp_cortex[cIdx].weight[j][i]-=.2*p->neuron_output[g][cIdx][i]*p->delta[g][cIdx][j];
				}
			}
		}

		__syncthreads();
		for(int i=0;i<NEURON_NUM;i++) {
			for(int j=0;j<NR_INPUTS;j++) {
				bp_genome->input_weight[j][cIdx][i]-=.2*io->inputs[j][sample]*p->delta[g][cIdx][i];
			}
		}
		
		__syncthreads();
		g+=gridDim.x;
	}
}

__global__ void cuFitness(population_t *p) {
	int g=blockIdx.x;

	while(g < POPULATION_SIZE) {
		p->fitness[g]=1./(p->error[g]+0.00001);
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
//    qsort(population);
    *deviceBestIndividualFitness=population->fitness[population->map[0]];
/*
	for(int i=0;i<POPULATION_SIZE;i++) 
		printf("%f,",population->fitness[population->map[i]]);
	printf("\n\n");
*/
  }
}

__host__ void host_find_best_individual(population_t *p, float *deviceBestIndividualFitness) {

//	qsort<<<1,1>>>(p);
	find_best_individual<<<1,1>>>(p, deviceBestIndividualFitness);	
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

void fitness(population_t *population1,IO_t *io, int sample) {
		cuInputs<<<1,NR_INPUTS>>>(population1, io,sample);
		check_cuda_errors(__FILE__, __LINE__);
		cuOutputs<<<1,NR_OUTPUTS>>>(population1, io,sample);
		check_cuda_errors(__FILE__, __LINE__);
		for(int i=0;i<4;i++) {
			cuExcite<<<POPULATION_SIZE,dim3(CORTEX_NUM,NEURON_NUM,THREAD_PER_NEURON)>>>(population1);
			check_cuda_errors(__FILE__, __LINE__);
		}
		cuError<<<256,NR_OUTPUTS>>>(population1);
		check_cuda_errors(__FILE__, __LINE__);
		cuResetDelta<<<POPULATION_SIZE, dim3(CORTEX_NUM,NEURON_NUM)>>>(population1);
		check_cuda_errors(__FILE__, __LINE__);

		for(int i=0;i<4;i++) {
			cuBackpropagation<<<POPULATION_SIZE,1>>>(population1);
			check_cuda_errors(__FILE__, __LINE__);
		}
		cuAdjustWeights<<<POPULATION_SIZE, 1>>>(population1, io, sample);
		check_cuda_errors(__FILE__, __LINE__);

		cuFitness<<<256,1>>>(population1);
		check_cuda_errors(__FILE__, __LINE__);
		cuMaxFitness<<<1,POPULATION_SIZE>>>(population1);
		check_cuda_errors(__FILE__, __LINE__);
//		cuTotal<<<1,1>>>(population1);
//		check_cuda_errors(__FILE__, __LINE__);
}

__global__ void print_outputs(population_t *p, IO_t *io,int sample) {
		int b=p->map[0];
		printf("input: ");
		for(int j=0;j<NR_INPUTS;j++) {
			printf("%d,", p->input[j]>0?1:0);
		}
		int ham=0;
		printf("\noutput: [");
		for(int j=0;j<NR_OUTPUTS;j++) {
			int out=p->outputs[b][j]>0?1:0;
			ham+=out^(io->outputs[j][sample]>0?1:0);
			printf("%d,", out);
		}
		printf("]  ");
		for(int j=0;j<NR_OUTPUTS;j++) {
			printf("%f,", p->outputs[b][j]);
		}
		printf(" err: %f ", p->error[b]);
		printf("fitness: %f", p->fitness[b]);
		printf(" distance: %d\n", ham);
}

void print_best(population_t *population, IO_t *io, float *deviceBestIndividualFitness, int start_sample, int stop_sample) {
	cuResetNeurons<<<128,1>>>(population);
	cuResetError<<<128,1>>>(population);
	for(int i=start_sample;i<stop_sample;i++) {
        cuResetNeurons<<<128,1>>> (population); 
        check_cuda_errors(__FILE__, __LINE__);
		fitness(population, io, i);
		host_find_best_individual(population, deviceBestIndividualFitness);
		print_outputs<<<1,1>>>(population,io, i);
	}
}

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
			  host_find_best_individual(population1,deviceBestIndividualFitness);
              copy_best_individuals<<<POPULATION_SIZE/2,1>>> (population1, population2);
              check_cuda_errors(__FILE__, __LINE__);
              cuResetNeurons<<<128,1>>> (population2); 
              cuResetError<<<128,1>>> (population2); 
              check_cuda_errors(__FILE__, __LINE__);

			  if(it>0) // kvoli pocitaniu fitness, pri 0 este nie je vypocitany
				crossover<<<POPULATION_SIZE/2,1>>> (CROSSOVER_PROB, population1, population2);
              check_cuda_errors(__FILE__, __LINE__);


              {
                  population_t *p = population1;
                  population1 = population2;
                  population2 = p;
              }

              cuResetError<<<128,1>>> (population1); 
			  for(int i=start_sample; i<stop_sample;i++) {
              	  cuResetNeurons<<<128,1>>> (population1); 
              	  check_cuda_errors(__FILE__, __LINE__);
				  fitness(population1, io, i);
			  }

              if(g%100==0) {
			  	  printf("generation %d\n", it);
//				printf("deviceBest\n");
				  host_find_best_individual(population1,deviceBestIndividualFitness);
                  check_cuda_errors(__FILE__, __LINE__);
                  cudaMemcpy(hostBestIndividualFitness, deviceBestIndividualFitness, sizeof(float), cudaMemcpyDeviceToHost);
                  check_cuda_errors(__FILE__, __LINE__);
//					printf("old %f new %f\n", tmpFitness, *hostBestIndividualFitness);
				  assert(tmpFitness <= *hostBestIndividualFitness);
					tmpFitness=*hostBestIndividualFitness;
//					printf("print best\n");
				  print_best(population1, io, deviceBestIndividualFitness, start_sample, stop_sample);
                  check_cuda_errors(__FILE__, __LINE__);
				  printf("best fitness: %f\n", *hostBestIndividualFitness);
              }

            g++;
            it++;
          } while (*hostBestIndividualFitness < 2);
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
#include "switch_in.dat"
      };
  const char* out[]={
#include "switch_out.dat"
      };
#else
      const char* in[]={
#include "bch_input.dat.3"
      };
      const char* out[]={
#include "bch_output.dat.3"
      };
#endif
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

	genetic(io, p1,p2, deviceBestIndividualFitness);

	cudaFree(p1);
	cudaFree(p2);
	cudaFree(io);
	cudaFree(hostBestIndividualFitness);
	return 0;
}
