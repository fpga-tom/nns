#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <queue>		// std::priority_queue
#include <math.h>
#include <stdio.h>
#include <sys/time.h>

#include <curand_kernel.h>
//#include "cuda/util/printf.cu"

using namespace std;

//#define DEBUG

#define GRID_DIM 16
#define BLOCK_DIM 16

#define NODE_COUNT 16
#define CORTICAL_NUM 8
#define POPULATION_SIZE 512
#define MUTATION_PROB .15
#define CROSSOVER_PROB .80
#define BEST_INDIVIDUALS 2
#define THREAD_COUNT (GRID_DIM*BLOCK_DIM)
#define CORTICAL_IN_NUM 5
#define CORTICAL_OUT_NUM 15

#define NR_INPUTS 5
#define NR_OUTPUTS 15
#define SAMPLES 4


#define CHAR(x) ((char*)x)
#define EDGE(incidence, x, y) (incidence[(x*NODE_COUNT+y)/8] & (1<< ((x*NODE_COUNT+y)%8)))
#define IEDGE(inputs,i,n) (inputs[i][n/8] & (1<<(n%8)))
#define CORTICAL_INDEX(cortex1,cortex2) ((cortex1)*(CORTICAL_NUM) + (cortex2))
#define CORTICAL_INTERCONNECT(cortical_interconnect, cortex1,cortex2) (cortical_interconnect[CORTICAL_INDEX(cortex1,cortex2)/8] & (1<<(CORTICAL_INDEX(cortex1,cortex2)%8)))
#define ABS(x) ((x) < 0 ? -(x) : (x))

#define CORTICAL(val) ((((val))&0xff)%CORTICAL_NUM)
#define INPUT(val) ((((val)>>8)&0xff)%CORTICAL_IN_NUM)
#define OUTPUT(val) ((((val)>>8)&0xff)%CORTICAL_OUT_NUM)

#define WEIGHT_INDEX(x,y) ((x*NODE_COUNT+y)*4)
#define WEIGHT(weights,x,y) ((((((weights)[WEIGHT_INDEX(x,y)/8])>>(WEIGHT_INDEX(x,y)%8)) & 0x0f) -8)/(7.))
#define FITNESS(population, index) ((population)->individual[index]).fitness

#define THREAD_ID (blockIdx.x*blockDim.x + threadIdx.x)
#define THREAD(t) (t[THREAD_ID])



typedef struct __attribute__ ((__packed__))
{
//  char incidence[NODE_COUNT * NODE_COUNT / 8]; // bitmap, incidence matrix between neurons
  unsigned char weights[NODE_COUNT*NODE_COUNT/2]; // weight matrix of connections, one weight is 4 bits
//  unsigned char function[NODE_COUNT]; // activation function of neuron
//    char distance[NODE_COUNT][NODE_COUNT];
//  unsigned char inputs[CORTICAL_IN_NUM];//[NODE_COUNT / 8]; // inputs to network, 1st index is input number, 2nd index is neuron number to witch input belongs
  unsigned char input_weights[CORTICAL_IN_NUM/2];//[NODE_COUNT]; // weights of input connections
  unsigned char outputs[CORTICAL_OUT_NUM]; // neuron number of output shared between inner and outer corticals
} cortical_column_t;

typedef struct __attribute__ ((__packed__)) {
    cortical_column_t cortical[CORTICAL_NUM];
    char cortical_interconnect[CORTICAL_NUM*CORTICAL_NUM/8];
    unsigned char inputs[NR_INPUTS];
    unsigned short outputs[NR_OUTPUTS];
} genome_t;


typedef struct
{
  genome_t genome;
  double fitness;
  double node_outputs[CORTICAL_NUM][NODE_COUNT]; // output of every node in each cortical
  double input_values[NR_INPUTS];
  double output_values[NR_OUTPUTS];
  double output_errors[NR_OUTPUTS];
} genomef_t;

inline bool
operator< (const genomef_t & g1, const genomef_t & g2)
{
  /*
     return g1.output_errors[0] < g2.output_errors[0] && 
     g1.output_errors[1] < g2.output_errors[1];
   */
  return g1.fitness < g2.fitness;
}


typedef struct
{
  genomef_t individual[POPULATION_SIZE];
  genomef_t *best_individual;
  int map[POPULATION_SIZE];
  //int count;
} population_t;

typedef struct {
    double inputs[NR_INPUTS][SAMPLES];
    double outputs[NR_OUTPUTS][SAMPLES];
    double errors[NR_OUTPUTS];
} IO_t;

double *deviceBestIndividualFitness;
double hostBestIndividualFitness;


struct event_s
{
  int node;
  double value;
  long timestamp;
  double getValue () const
  {
    return value;
  };
};

typedef struct {
    //int thread_id;
    int count;
    IO_t io;
    __device__ int getPopulationIndex() const { int r=THREAD_COUNT*count+THREAD_ID; return r;};
    __device__ int getPopulationIndexInc() { int r=getPopulationIndex(); count++; return r; };
    __device__ int getPopulationIndexEven() { int r=THREAD_COUNT*count+THREAD_ID; return r; };
    __device__ int getPopulationIndexOdd() { int r=THREAD_COUNT*(count+1)+THREAD_ID; return r; };
    __device__ void resetCount() { count=0; };
    population_t *currentPopulation;
    curandState_t curandState;
} thread_local_t;

int current_sample=0;

thread_local_t threads[THREAD_COUNT];
//pthread_barrier_t barrier;

inline bool
operator< (const event_s & s1, const event_s & s2)
{
  return s1.getValue () < s2.getValue ();
}

priority_queue < struct event_s >events;

/*
population_t population;
population_t population1;
population_t *pPopulation;
population_t *pTmpPopulation;
*/

inline void check_cuda_errors(const char *filename, const int line_number)
{
#ifdef DEBUG
      cudaThreadSynchronize();
      cudaError_t error = cudaGetLastError();
      if(error != cudaSuccess) {
          printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
          exit(-1);
      }
#endif
}



typedef double (*function_t) (double);

__device__ double
sigmoid (double v)
{
  return v / sqrt (1 + v * v);
}

function_t node_functions[1] = { sigmoid };


void
init ()
{
  srand (time (NULL));
}


__device__ 
void
swap (int array[], int i, int j)
{
  int tmp = array[i];
  array[i] = array[j];
  array[j] = tmp;
}


int hamming(double *vec1, double *vec2, int size) {
    int r=0;
    for(int i=0;i<size;i++) {
        int b1=vec1[i]>.5?1:0;
        int b2=vec2[i]>.5?1:0;
        r+=b1^b2;
    }
    return r;
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
    /*
    bool change=true;
    while(change) {
        change=false;
        for(int i=0; i < POPULATION_SIZE-1;i++)
            if(FITNESS(population,population->map[i]) < FITNESS(population,population->map[i+1])) {
              swap(population->map,i,i+1);
              change=true;
            }
    }
    */
    /*
    for(int i=0;i<POPULATION_SIZE;i++) {
        printf("%lf,", population->individual[population->map[i]].fitness);
    }
    */
}


__device__ void reset_individual(genomef_t *genome) {
    memset(genome->node_outputs, 0, sizeof(genome->node_outputs));
}


__device__ inline double my_random (thread_local_t* thread) {
  double r = ((double) (curand (&THREAD(thread).curandState) / ((float)(0x0FFFFFFFFUL))));
  return r;
}

__device__ void mutate (thread_local_t *thread, double prob, genome_t * genome) {
  double r = my_random (thread);
  int s = curand(&THREAD(thread).curandState) % (sizeof (genome_t) * 8);
  if (r < prob)
    {
      ((char *) genome)[s / 8] ^= (1 << (s % 8));
    }
}



__device__ void cross(char* ng1, char* g1,char* ng2, char* g2, int s, int size) {

/*
      cout << (CHAR(ng1)+s/8+1) << endl;
      cout << (CHAR(g2)+s/8+1) << endl;
      */
//      assert(size-s/8 >= 0);
//      cout << (size-s/8-1) << endl;



      memcpy (ng1, g1, s / 8);
      memcpy (CHAR (ng1) + s / 8, CHAR (g2) + s / 8, size - s / 8);
      char mask = (1 << (s % 8)) - 1;
      CHAR (ng1)[s / 8] = (CHAR (g1)[s / 8] & ~mask) | (CHAR (g2)[s / 8] & mask);

      memcpy (ng2, g2, s / 8);
      memcpy (CHAR (ng2) + s / 8 , CHAR (g1) + s / 8 , size - s / 8);
      mask = (1 << (s % 8)) - 1;
      CHAR (ng1)[s / 8] = (CHAR (g2)[s / 8] & ~mask) | (CHAR (g1)[s / 8] & mask); }

__device__ void _crossover (thread_local_t *thread, double prob, population_t * population, population_t * new_population) {


  double r = my_random (thread);
  double total = 0;
//  assert(thread->getPopulationIndex() < POPULATION_SIZE);
  genomef_t *ng1 = &new_population->individual[THREAD(thread).getPopulationIndexInc()];
//  assert(thread->getPopulationIndex() < POPULATION_SIZE);
  genomef_t *ng2 = &new_population->individual[THREAD(thread).getPopulationIndexInc()];


  for (int i = 0; i < POPULATION_SIZE; i++)
    {
      total += population->individual[i].fitness;
    }
  genomef_t *g1 = 0;
  genomef_t *g2 = 0;
  while (g1 == g2)
    {
      double r1 = my_random (thread) * total;
      double r2 = my_random (thread) * total;
      double sum = 0;
      for (int i = 0; i < POPULATION_SIZE; i++)
	{
	  sum += population->individual[i].fitness;
	  if (sum >= r1)
	    {
	      g1 = &population->individual[i];
	      break;
	    }
	}
      sum = 0;
      for (int i = 0; i < POPULATION_SIZE; i++)
	{
	  sum += population->individual[i].fitness;
	  if (sum >= r2)
	    {
	      g2 = &population->individual[i];
	      break;
	    }
	}
    }

  if (r < prob)
    {

       int s= curand(&THREAD(thread).curandState) % (sizeof(genome_t)*8);

       genome_t ngg1,ngg2;

       //cross((char*)&ng1->genome, (char*)&g1->genome,(char*)&ng2->genome,(char*)&g2->genome,s,sizeof(genome_t));
       cross((char*)&ngg1, (char*)&g1->genome,(char*)&ngg2,(char*)&g2->genome,s,sizeof(genome_t));

       int s1 = (s==0?0:(curand(&THREAD(thread).curandState) % s));
       int new_size=(s+0)/8;

       if(new_size>0)
           cross((char*)&ng1->genome, (char*)&ngg1,(char*)&ng2->genome,(char*)&ngg2,s1,new_size);

       int s2 = curand(&THREAD(thread).curandState) % (sizeof(genome_t)*8-s);

       if(sizeof(genome_t)-new_size>0)
           cross(((char*)&ng1->genome)+new_size, ((char*)&ngg1)+new_size,((char*)&ng2->genome)+new_size,((char*)&ngg2)+new_size,s2,sizeof(genome_t)-new_size);
    }
  else
    {
      memcpy ((char*)&ng1->genome, (char*)&g1->genome, sizeof (genome_t));
      memcpy (&ng2->genome, &g2->genome, sizeof (genome_t));
    }

  mutate (thread, MUTATION_PROB, &ng1->genome);
  mutate (thread, MUTATION_PROB, &ng2->genome);
}

__global__ void crossover (thread_local_t *thread, double prob, population_t * population, population_t * new_population) {
    THREAD(thread).resetCount();
    while(THREAD(thread).getPopulationIndexEven() < POPULATION_SIZE && THREAD(thread).getPopulationIndexOdd() < POPULATION_SIZE) {
        _crossover(thread, prob, population, new_population);
    }
}

__device__ void calculate_cortical_outputs(genomef_t *individual,int cortical_index, cortical_column_t *cortical, double inputs[CORTICAL_IN_NUM]) {

    for(int i=0;i<NODE_COUNT;i++) {
        double signal = 0;

        for(int k=0;k<CORTICAL_IN_NUM;k++) {
            //if(cortical->inputs[k]%NODE_COUNT==i) {
            if(k%NODE_COUNT==i) {
              double factor = WEIGHT(cortical->input_weights,0, k);//(cortical->input_weights[k] / (double)((128/*32768*/)-1));
              signal+=inputs[k]*factor;
            }
        }

        for(int k=0;k<NODE_COUNT;k++) {
//              if (EDGE (cortical->incidence, i, k)) {
                  double factor = WEIGHT(cortical->weights,i,k);
                  signal += individual->node_outputs[cortical_index][k] * factor;
//              }
        }

        individual->node_outputs[cortical_index][i]=sigmoid(signal);
    }
}

__device__ void _fitness_individual(genomef_t *individual, bool p=false) {


    double inputs[CORTICAL_IN_NUM];
    memset(inputs, 0, sizeof(inputs));

    for(int i=0;i<CORTICAL_NUM;i++) {
        int j=0;
        for(int k=0;k<NR_INPUTS;k++) {
            if(j<CORTICAL_IN_NUM && CORTICAL(individual->genome.inputs[k])==i)
                 inputs[j++]=individual->input_values[k];
        }

        for(int k=0;k<CORTICAL_NUM;k++) {
            if(CORTICAL_INTERCONNECT(individual->genome.cortical_interconnect, k, i)) {
                for(int r=0;r<CORTICAL_OUT_NUM && j<CORTICAL_IN_NUM ;r++) {
                    inputs[j++]=individual->node_outputs[k][r];
                }
            }
        }
//        cout << j << endl;
        calculate_cortical_outputs(individual, i, &individual->genome.cortical[i], inputs);
    }



    for(int i=0;i<CORTICAL_NUM;i++) {
        for(int j=0;j<CORTICAL_OUT_NUM;j++) {
            for(int k=0;k<NR_OUTPUTS;k++) {
                if(CORTICAL(individual->genome.outputs[k])==i && OUTPUT(individual->genome.outputs[k])==j) {
                     individual->output_values[k]=individual->node_outputs[i][j];
                }
            }
        }
    }
   


/*
      cortical_column_t & g = individual->genome.cortical[c];

          for (int i = 0; i < NODE_COUNT; i++)
        {
          double signal = 0;
          for (int k = 0; k < NR_INPUTS; k++) {
              if (IEDGE (g.inputs, k, i))
            {
              double factor = (g.input_weights[k][i] / (double)((32768)-1));
              if(c==0) // 0 is input cortical
                  signal += individual->input_values[k] * factor;
              else
                  signal += values[k]*factor;
            }
          }
          for (int j = 0; j < NODE_COUNT; j++) {
              if (EDGE (g.incidence, i, j))
            {
              double factor = (g.weights[i][j] /  (double)((32768)-1));
              if(c==CORTICAL_NUM-1) // CORTICAL_NUM-1 is output cortical
                  signal += individual->node_outputs[j] * factor;
              else
                  signal += values[k] * factor;
            }
          }
          individual->node_outputs[i] =
            node_functions[g.function[i] %
                   (sizeof (node_functions) /
                    sizeof (function_t))] (signal);
        }
        */
}

__device__ void fitness_individual (genomef_t * individual, IO_t &io, bool p = false, int count=SAMPLES, bool forecast=false) {


//  cortical_column_t & g = individual->genome;

  for(int k=0;k<NR_OUTPUTS;k++) {
      io.errors[k]=0;
  }
  double cumulative_error=0;

      int ham=0;
  for (int q = 0; q < count; q++) {

//        reset_individual(individual);
      for(int k=0;k<NR_INPUTS;k++) {
        individual->input_values[k] = io.inputs[k][q];
      }

      if(p) {
          printf("input: ");
          for(int i=0;i<NR_INPUTS;i++) {
              if(individual->input_values[i]>0.) {
                  printf("1,");
              } else {
                  printf("0,");
              }
          }
          printf("\n");
      }

      _fitness_individual(individual, p);

      double oo[NR_OUTPUTS]={0.,};

      for(int k=0;k<NR_OUTPUTS;k++) {
          io.errors[k]=0;
      }

       
      int ham1=0;
      for(int k=0;k<NR_OUTPUTS;k++) {
//          int port=g.outputs[k]%NODE_COUNT;
          double output=individual->output_values[k];
          oo[k]=output;
          int b1=io.outputs[k][q]>0.?1:0;
          int b2=output>0.?1:0;
              ham1+=b1^b2;
              ham+=b1^b2;
          if(!forecast) {
                  io.errors[k]+=.9*(b1^b2)+.1*pow(io.outputs[k][q] - output,2);//(SAMPLES*NR_OUTPUTS);
          }
//              cout << io.errors[k];
      }
      double hamming_distance=0;
      for(int k=0;k<NR_OUTPUTS;k++) {
          hamming_distance+=io.errors[k];
      }
      cumulative_error+=hamming_distance;
      if (p){
          printf("fit:   ");
          for(int i=0;i<NR_OUTPUTS;i++) {
              printf("[");
              if(oo[i]>0.) {
                  printf("1,");
              } else {
                  printf("0,");
              }
              printf(" %d ]",CORTICAL(individual->genome.outputs[i]));
          }
          for(int i=0;i<NR_OUTPUTS;i++) {
                  printf("%f,", oo[i]);
          }
          printf("dist: %d\n", ham1);
      }

  }

  if(!forecast) {
//      double cumulative_error=0;
/*
      for(int k=0;k<NR_OUTPUTS;k++) {
          io.errors[k]/=SAMPLES;
          cumulative_error+=pow(io.errors[k],2)/4;
      }
      */
      individual->fitness = 1. / (cumulative_error+0.00001);
  }
  if (p) {
    printf("best individual fitness: %f\n",  individual->fitness); 
    printf("hamming distance: %d\n", ham );;
  }
    
}


__global__ void fitness (thread_local_t *thread, int current_sample) { /*, population_t * population, IO_t& io)*/

    int i=0;
    THREAD(thread).resetCount();
    while((i=THREAD(thread).getPopulationIndexInc()) < POPULATION_SIZE) {
       fitness_individual (&THREAD(thread).currentPopulation->individual[i],THREAD(thread).io, false, current_sample);
    }
}

__global__ void find_best_individual(thread_local_t *thread, population_t* population, double *deviceBestIndividualFitness) {
      __syncthreads();
  if(THREAD_ID==0) {
    qsort(population);
    population->best_individual = &population->individual[population->map[0]];
    *deviceBestIndividualFitness=population->best_individual->fitness;
//    printf("best: %lf\n", population->best_individual->fitness);
  }
  __syncthreads();
}

__global__ void init_population (thread_local_t *thread, population_t *population) {

    int i=0;
    THREAD(thread).resetCount();
    while((i=THREAD(thread).getPopulationIndexInc()) < POPULATION_SIZE) {
      population->individual[i].fitness = 0;
      population->map[i]=i;
      
      memset(population->individual[i].node_outputs, 0, sizeof(population->individual[i].node_outputs));
    }
    /*
    */
}

__global__ void init_random_population (thread_local_t *thread)
{
  //init_population(thread);
  int i=0;
  THREAD(thread).resetCount();
  while((i=THREAD(thread).getPopulationIndexInc()) < POPULATION_SIZE) {
      /*
      printf("%d\n", curand (&THREAD(thread).curandState) % 255);
      printf("%d\n", THREAD(thread).currentPopulation->individual);
      */
      for (int j = 0; j < sizeof (genome_t); j++) {
    	  ((char *) &(THREAD(thread).currentPopulation->individual[i].genome))[j] = curand (&THREAD(thread).curandState) % 255;
	}
 }
}

__global__ void copy_best_individuals (thread_local_t* thread, population_t * p1, population_t * p2)
{
    if(THREAD_ID==0) {
        THREAD(thread).resetCount();
        for(int i=0;i<BEST_INDIVIDUALS && i<POPULATION_SIZE;i++) {
          memcpy (&p2->individual[THREAD(thread).getPopulationIndexInc()].genome, &p1->individual[p1->map[i]], sizeof (genome_t));
        }
    }
}

__global__ void print_best_individual(thread_local_t* thread,int current_sample) {
    genomef_t *genome=&THREAD(thread).currentPopulation->individual[THREAD(thread).currentPopulation->map[0]];
    printf("best_fitness: %lf\n", genome->fitness);
    reset_individual(genome);
    fitness_individual(genome,THREAD(thread).io, true, current_sample, true);
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
      sscanf(in[k],"%lf,%lf,%lf,%lf,%lf", 
        &io.inputs[0][k],
        &io.inputs[1][k],
        &io.inputs[2][k],
        &io.inputs[3][k],
        &io.inputs[4][k]
        );
  //    cout << io.inputs[0][k] <<"," << io.inputs[1][k] << "," << io.inputs[2][k] << endl;
  }
  for(int k=0;k<SAMPLES;k++) {
      sscanf(out[k],"%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", 
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

__global__ void init_curand(thread_local_t* thread) {
    curand_init(1234,THREAD_ID, 0, &THREAD(thread).curandState);
}

__global__ void current_population(thread_local_t* thread, population_t* p) {
    THREAD(thread).currentPopulation=p;
}

#define START gettimeofday(&start, NULL);
#define STOP(f) \
              gettimeofday(&end, NULL); \
              seconds=end.tv_sec - start.tv_sec;\
              useconds=end.tv_usec - start.tv_usec;\
              mtime=(seconds)*1000000 + useconds;\
              printf( #f " elapsed time %ld\n", (mtime));

__host__ void genetic (thread_local_t *t, population_t* population1, population_t* population2) {

/*
  int thread_index=blockIdx.x*blockDim.x + threadIdx.x;

  population_t* pPopulation=population1;
  population_t *pTmpPopulation=population2;

  thread_local_t *thread=&t[thread_index];
  thread->thread_id=thread_index;

  thread->currentPopulation=pPopulation;
  */


  printf("setting current population...\n");
  current_population<<<GRID_DIM,BLOCK_DIM>>>(t,population1);
  printf("initializing curand...\n");
  init_curand<<<GRID_DIM,BLOCK_DIM>>>(t);
  printf("initializing current population...\n");
  init_population<<<GRID_DIM,BLOCK_DIM>>>(t, population1);
  printf("randomizing population...\n");
  init_random_population<<<GRID_DIM,BLOCK_DIM>>> (t);
  printf("fitnessing current population...\n");
  int g=0;
  current_sample=1;
  fitness<<<GRID_DIM,BLOCK_DIM>>> (t, current_sample);
  printf("setup done\n");
  double best_fitness=1;


    while(current_sample<=SAMPLES) {  
      printf("training %d samples...\n", current_sample);

      struct timeval start,end;
      long mtime,seconds,useconds;
      do {

          START;
          init_population<<<GRID_DIM,BLOCK_DIM>>> (t,population2);
          STOP("init_population");
          check_cuda_errors(__FILE__, __LINE__);
          START;
          copy_best_individuals<<<1,1>>> (t,population1, population2);
          STOP("copy_best_individual");
          check_cuda_errors(__FILE__, __LINE__);


          START;
          crossover<<<GRID_DIM,BLOCK_DIM>>> (t,CROSSOVER_PROB, population1, population2);
          STOP("crossover");
          check_cuda_errors(__FILE__, __LINE__);

          {
              population_t *p = population1;
              population1 = population2;
              population2 = p;
          }

          START;
          current_population<<<GRID_DIM,BLOCK_DIM>>>(t,population1);
          STOP("current_population");
          check_cuda_errors(__FILE__, __LINE__);

          START;
          fitness<<<GRID_DIM, BLOCK_DIM>>> (t, current_sample);
          STOP("fitness");
          check_cuda_errors(__FILE__, __LINE__);
          START;
          find_best_individual<<<1, 1>>>(t,population1,deviceBestIndividualFitness);
          STOP("find_best_individual");
          START;
          cudaMemcpy(&hostBestIndividualFitness, deviceBestIndividualFitness, sizeof(double), cudaMemcpyDeviceToHost);
          STOP("cudaMemcpy");
          check_cuda_errors(__FILE__, __LINE__);

          if(g++%10==0) {
              START;
              print_best_individual<<<1,1>>>(t, current_sample);
              STOP("print_best_individual");
    //            cudaDeviceSynchronize();
              check_cuda_errors(__FILE__, __LINE__);
              printf("best individual: %lf / %d\n", best_fitness, current_sample);
          }

      } while (hostBestIndividualFitness < .6);

      current_sample++;
    }
}


int
main ()
{

  cout << "cortical size: " <<  sizeof (cortical_column_t) << endl;
  cout << "genome size: " << sizeof (genome_t) << endl;
  cout << "threads: " << THREAD_COUNT << endl;

//  cudaPrintfInit();

  population_t* devicePopulation1=0;
  population_t* devicePopulation2=0;
  thread_local_t *deviceThread=0;

  cudaMalloc((void**)&devicePopulation1,sizeof(population_t));
  cudaMalloc((void**)&devicePopulation2,sizeof(population_t));
  cudaMalloc((void**)&deviceThread, sizeof(thread_local_t)*THREAD_COUNT);
  cudaMalloc((void**)&deviceBestIndividualFitness,sizeof(double));

  for(int i=0; i<GRID_DIM*BLOCK_DIM;i++) {
      read_io(threads[i].io);
  }

/*
  cudaMemcpy(devicePopulation1,&population, sizeof(population_t), cudaMemcpyHostToDevice);
  cudaMemcpy(devicePopulation2,&population1, sizeof(population_t), cudaMemcpyHostToDevice);
  */
  cudaMemcpy(deviceThread,&threads, sizeof(thread_local_t)*THREAD_COUNT, cudaMemcpyHostToDevice);

  genetic(deviceThread,devicePopulation1, devicePopulation2);

//  cudaPrintfDisplay();

//  cudaPrintfEnd();

  cudaFree(devicePopulation1);
  cudaFree(devicePopulation2);
  cudaFree(deviceThread);


  //init ();
  //signal(SIGINT, sighandler);
/*
  pPopulation = &population;
  pTmpPopulation = &population1;
  pthread_barrier_init(&barrier, NULL, THREAD_COUNT);

  for(int i=0; i<THREAD_COUNT;i++) {
      threads[i].thread_id=i;
      threads[i].count=0;
      int rc=pthread_create(&threads[i].thread, NULL, genetic, (void*)&threads[i]);
      if(rc) {
          printf("ERROR: return code from pthread_create is %d\n", rc);
          exit(-1);
      }
  }


  //genetic ();
  for(int i=0;i<THREAD_COUNT;i++) {
      pthread_join(threads[i].thread, NULL);
  }
  pthread_barrier_destroy(&barrier);
  pthread_exit(0);
  */
  return 0;
}
