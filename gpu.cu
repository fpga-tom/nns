#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <queue>		// std::priority_queue
#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <thrust/sort.h>
#include <thrust/functional.h>

#include <curand_kernel.h>
#include <assert.h>

using namespace std;

#define DEBUG

#define NEURON_NUM 8
#define CORTEX_NUM 16
#define THREAD_PER_NEURON 1
#define POPULATION_SIZE 256
#define MUTATION_PROB 0.15
#define CROSSOVER_PROB 0.8
#define BEST_INDIVIDUALS 2

#define NR_INPUTS 5
#define NR_OUTPUTS 15
#define SAMPLES 16

#define CHAR(x) ((char*)x)
#define TO_WEIGHT(x) (((x&0x0f)-8.)/7.)
#define WEIGHT(w,x,y) TO_WEIGHT((w[(y*NEURON_NUM+x)>>1]>>(((y*NEURON_NUM+x)<<2)&0x7))&0xf)
#define INTERCONNECT_INDEX(i,c,n) (((c*NEURON_NUM+n)*12/8))
#define INTERCONNECT_VALUE(i,c,n) ((((((short)i[INTERCONNECT_INDEX(i,c,n)])&0xff)|(((short)i[INTERCONNECT_INDEX(i,c,n)+1]&0xff)<<8))>>(((c*NEURON_NUM+n)*12)%8))&0xfff)/*((((((unsigned int*)(i))[INTERCONNECT_INDEX(i,c,n)]))>>(((c*NEURON_NUM+n)*16)%32)))*/
#define INTERCONNECT_WEIGHT(i,c,n) TO_WEIGHT((INTERCONNECT_VALUE(i,c,n)>>8)&0xf)
#define INTERCONNECT_CORTEX(i,c,n) (INTERCONNECT_VALUE(i,c,n)&0xf)
#define INTERCONNECT_NEURON(i,c,n) ((INTERCONNECT_VALUE(i,c,n)>>4)&0xf)


#define FITNESS(population, gIdx) ((population)->fitness[gIdx])
#define THREAD_ID (blockIdx.x*blockDim.x + threadIdx.x)
#define _THREAD_COUNT (blockDim.x*gridDim.x)

#define NEURON_IDX(output_neuron_idx) ((output_neuron_idx&0xf)%NEURON_NUM)
#define CORTEX_IDX(output_neuron_idx) (((output_neuron_idx>>4)&0xf)%CORTEX_NUM)

#define INPUT_WEIGHT(w,i,c,n) TO_WEIGHT((w[(i*(CORTEX_NUM*NEURON_NUM)+c*NEURON_NUM+n)*4/8] >> ((i*(CORTEX_NUM*NEURON_NUM)+c*NEURON_NUM+n)*4)%8))


typedef struct __attribute__ ((__packed__)) {
	char weight[NEURON_NUM*NEURON_NUM/2]; // weight is 4 bits
} cortex_t;

typedef struct __attribute__ ((__packed__)) {
	cortex_t cortex[CORTEX_NUM];
	char interconnect[CORTEX_NUM*NEURON_NUM*12/8]; // 16 bit value, use only 12
	unsigned char input_weight[NR_INPUTS*CORTEX_NUM*NEURON_NUM*4/8]; //4 bit weight;
	unsigned char output_neuron_idx[NR_OUTPUTS];
} genome_t;


typedef struct {
	genome_t genome[POPULATION_SIZE];
	float fitness[POPULATION_SIZE];
//	float total_fitness;
	float neuron_output[POPULATION_SIZE][CORTEX_NUM][NEURON_NUM];
	float input[NR_INPUTS];
	float output[NR_OUTPUTS];
	float error[POPULATION_SIZE];
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

//---------------------------------------------------------------------------------------------------

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
//	return __frsqrt_rn(1+__powf(signal,2));
	return signal / sqrtf(1+powf(signal, 2));
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

__global__ void qsort(population_t *population) {
	qs(population,population->map, 0, POPULATION_SIZE-1);
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

__device__ void mutate (population_t *p, double prob, genome_t * genome) {
  double r = my_random (p);
  if (r < prob)
    {
  int s = curand(&p->curandState[THREAD_ID]) % (sizeof (genome_t) * 8);
	assert(&((char*)genome)[s/8] < (char*)p->genome || &((char*)genome)[s/8] >= (char*)&p->genome[1]);
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
  genome_t *ng1 = &new_population->genome[c->getPopulationIndexInc()];
  assert(c->getPopulationIndex() < POPULATION_SIZE);
  genome_t *ng2 = &new_population->genome[c->getPopulationIndexInc()];


  for (int i = 0; i < POPULATION_SIZE; i++)
    {
      total += population->fitness[i];
    }
  genome_t *g1 = 0;
  genome_t *g2 = 0;
  while (g1 == g2) {
      float r1 = my_random (population) * total;
      float r2 = my_random (population) * total;
      float sum = 0;
      for (int i = 0; i < POPULATION_SIZE; i++)
	{
	  sum += population->fitness[i];
	  if (sum >= r1)
	    {
	      g1 = &population->genome[i];
	      break;
	    }
	}
      sum = 0;
      for (int i = 0; i < POPULATION_SIZE; i++)
	{
	  sum += population->fitness[i];
	  if (sum >= r2)
	    {
	      g2 = &population->genome[i];
	      break;
	    }
	}
  }

  if (r < prob)
    {

       int s= curand(&population->curandState[THREAD_ID]) % (sizeof(genome_t)*8);

//       genome_t ngg1,ngg2;

       cross(ng1,g1,ng2,g2,s,sizeof(genome_t));
       /*
       cross((char*)&ngg1, (char*)&g1->genome,(char*)&ngg2,(char*)&g2->genome,s,sizeof(genome_t));

       int s1 = (s==0?0:(curand(&THREAD(thread).curandState) % s));
       int new_size=(s+0)/8;

       if(new_size>0)
           cross((char*)&ng1->genome, (char*)&ngg1,(char*)&ng2->genome,(char*)&ngg2,s1,new_size);

       int s2 = curand(&THREAD(thread).curandState) % (sizeof(genome_t)*8-s);

       if(sizeof(genome_t)-new_size>0)
           cross(((char*)&ng1->genome)+new_size, ((char*)&ngg1)+new_size,((char*)&ng2->genome)+new_size,((char*)&ngg2)+new_size,s2,sizeof(genome_t)-new_size);
           */
    }
  else
    {
      memcpy (ng1, g1, sizeof (genome_t));
      memcpy (ng2, g2, sizeof (genome_t));
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

    while(c.getPopulationIndexEven() < POPULATION_SIZE && c.getPopulationIndexOdd() < POPULATION_SIZE) {
//		if(c.getPopulationIndexEven()==0 ||
//			c.getPopulationIndexEven()==16)
//		printf("cross na %d %d\n", c.getPopulationIndexEven(), c.getPopulationIndexOdd());
	     _crossover(&c, prob, population, new_population);
    }

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


//	printf("cIdx: %d nIdx: %d\n", cIdx,nIdx);

	while(g < POPULATION_SIZE) {
		__shared__ float neuron_output[CORTEX_NUM][NEURON_NUM];
		__shared__ float _signal[CORTEX_NUM][NEURON_NUM][THREAD_PER_NEURON];
//		__shared__ genome_t _genome;
		genome_t *genome=genome=&p->genome[g];

/*
		if(cIdx==0 && nIdx==0 && threadIdx.z==0) {
			memcpy(&_genome.interconnect,&genome->interconnect, sizeof(_genome.interconnect));
		}

		if(cIdx==0 && nIdx==1 && threadIdx.z==0) {
			memcpy(&_genome.input_weight,&genome->input_weight, sizeof(_genome.input_weight));
		}

		if(nIdx==1 && threadIdx.z==0) {
			memcpy(&_genome.cortex[cIdx],&genome->cortex[cIdx], sizeof(cortex_t));
		}
*/
			

		neuron_output[cIdx][nIdx]=p->neuron_output[g][cIdx][nIdx];
		_signal[cIdx][nIdx][threadIdx.z]=0.f;

		__syncthreads();

		float signal=0.;
		// toto je vypocet jedneho cortexu
		for(int n=threadIdx.z;n<NEURON_NUM;n+=blockDim.z) {
			float factor=WEIGHT(genome->cortex[cIdx].weight, nIdx, n);
			_signal[cIdx][nIdx][threadIdx.z]+=factor*neuron_output[cIdx][n];
		}
//		printf("sig1: %d %f", THREAD_ID, signal);
		__syncthreads();
		// toto su signaly z ostatnych cortexov
		for(int n=0;n<CORTEX_NUM;n++) {
			for(int m=threadIdx.z;m<NEURON_NUM;m+=blockDim.z) {
				if(INTERCONNECT_CORTEX(genome->interconnect,n,m)==cIdx)
					if( INTERCONNECT_NEURON(genome->interconnect,n,m)==nIdx) {
					float factor=INTERCONNECT_WEIGHT(genome->interconnect,n,m);
					_signal[cIdx][nIdx][threadIdx.z]+=factor*neuron_output[n][m];
//					printf("%f\n", p->neuron_output[g][n][m]);
				}
			}
		}
//		printf("sig2: %d %f", THREAD_ID, signal);
		__syncthreads();
		// toto su signaly z zo vstupov
		for(int n=threadIdx.z;n<NR_INPUTS;n+=blockDim.z) {
			float factor=INPUT_WEIGHT(genome->input_weight,n,cIdx,nIdx);
//			printf("p input: %f %f %d\n", p->input[n],factor, n);
			_signal[cIdx][nIdx][threadIdx.z]+=factor*p->input[n];
		}
//		printf("sig3: %d %f", THREAD_ID, signal);
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
		if(threadIdx.z==0)
			p->neuron_output[g][cIdx][nIdx]=neuron_output[cIdx][nIdx];
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
		genome_t *genome=&p->genome[g];
		int nIdx=NEURON_IDX(genome->output_neuron_idx[oIdx]);
		int cIdx=CORTEX_IDX(genome->output_neuron_idx[oIdx]);
		float output=p->neuron_output[g][cIdx][nIdx];
//		printf("neuron: %d %d %f\n", g, oIdx, output);
		p->outputs[g][oIdx]=output;
        int b1=p->output[oIdx]>0.?1:0;
        int b2=output>0.?1:0;
//		printf("e: %d %d %d %d %d %f %f\n",g, cIdx, nIdx, b1, b2, output, p->output[oIdx]); 
		error[oIdx]=/*.9*(b1^b2)+.1*/powf(p->output[oIdx] - output,2);//(SAMPLES*NR_OUTPUTS);
		
		for(int stride=16>>1;stride>0;stride>>=1) {
			__syncthreads();
			if(threadIdx.x<stride) {
				error[threadIdx.x]+=error[threadIdx.x+stride];
			}
		}
/*
		__syncthreads();
		if(threadIdx.x==0) {
			float e=0.;
			for(int i=0;i<NR_OUTPUTS;i++)
				e+=error[i];
			p->error[g]+=e;
			printf("error: %d %f\n", g, p->error[g]);
		}
*/
		__syncthreads();
		if(threadIdx.x==0) {
			p->error[g]+=error[0];
//				printf("error: %d %f\n", g, p->error[g]);
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

/*
__global__ void cuTotal(population_t *p) {
	int g=blockIdx.x;
	float sum=0.f;
	while(g < POPULATION_SIZE) {
		sum+=p->fitness[g];
		g+=gridDim.x;
	}
	p->total_fitness=sum;	
}
*/

__global__ void init_random_population (population_t* currentPopulation)
{
  int g=blockIdx.x;
  while(g < POPULATION_SIZE) {
      for (int j = 0; j < sizeof (genome_t); j++) {
    	  ((char *) &(currentPopulation->genome[g]))[j] = curand (&currentPopulation->curandState[g]) % ~(0U);
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
//		printf("copy na %d %f\n", pi, p1->fitness[p1->map[i]]);
          memcpy (&p2->genome[pi], &p1->genome[p1->map[i]], sizeof (genome_t));
		  
		  
        }
    }
}



void fitness(population_t *population1,IO_t *io, int sample) {
		cuInputs<<<1,NR_INPUTS>>>(population1, io,sample);
		check_cuda_errors(__FILE__, __LINE__);
		cuOutputs<<<1,NR_OUTPUTS>>>(population1, io,sample);
		check_cuda_errors(__FILE__, __LINE__);
		for(int i=0;i<8;i++) {
			cuExcite<<<POPULATION_SIZE,dim3(CORTEX_NUM,NEURON_NUM,THREAD_PER_NEURON)>>>(population1);
			check_cuda_errors(__FILE__, __LINE__);
		}
		cuError<<<256,NR_OUTPUTS>>>(population1);
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

//---------------------------------------------------------------------------------------------

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
#include "bch_input.dat.2"
      };
      const char* out[]={
#include "bch_output.dat.2"
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

	printf("sizeof(cortex_t)=%ld\n",sizeof(cortex_t));
	printf("sizeof(genome_t)=%ld\n",sizeof(genome_t));

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
