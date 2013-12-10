#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <queue>		// std::priority_queue
#include <math.h>
#include <stdio.h>
#include <signal.h>
#include <pthread.h>

#include <assert.h>

using namespace std;

#define NODE_COUNT 8
#define CORTICAL_NUM 16
#define POPULATION_SIZE 2000
#define MUTATION_PROB .05
#define CROSSOVER_PROB .80
#define BEST_INDIVIDUALS 2
#define THREAD_COUNT 4
#define CORTICAL_IO_NUM 8 

#define NR_INPUTS 5
#define NR_OUTPUTS 15
#define SAMPLES 128


#define CHAR(x) ((char*)x)
#define EDGE(incidence, x, y) (incidence[(x*NODE_COUNT+y)/8] & (1<< ((x*NODE_COUNT+y)%8)))
#define IEDGE(inputs,i,n) (inputs[i][n/8] & (1<<(n%8)))
#define CORTICAL_INDEX(cortex1,output,cortex2,input) (cortex1*(CORTICAL_IO_NUM*CORTICAL_NUM*CORTICAL_IO_NUM) + output*(CORTICAL_NUM*CORTICAL_IO_NUM)+cortex2*(CORTICAL_IO_NUM) + input)
#define CORTICAL_INTERCONNECT(cortical_interconnect, cortex1,output,cortex2,input) (cortical_interconnect[CORTICAL_INDEX(cortex1,output,cortex2,input)/8] & (1<<(CORTICAL_INDEX(cortex1,output,cortex2,input)%8)))
#define ABS(x) ((x) < 0 ? -(x) : (x))



typedef struct __attribute__ ((__packed__))
{
  char incidence[NODE_COUNT * NODE_COUNT / 8]; // bitmap, incidence matrix between neurons
  short weights[NODE_COUNT][NODE_COUNT]; // weight matrix of connections
//  unsigned char function[NODE_COUNT]; // activation function of neuron
//    char distance[NODE_COUNT][NODE_COUNT];
  unsigned char inputs[CORTICAL_IO_NUM];//[NODE_COUNT / 8]; // inputs to network, 1st index is input number, 2nd index is neuron number to witch input belongs
  short input_weights[CORTICAL_IO_NUM][NODE_COUNT]; // weights of input connections
  unsigned char outputs[CORTICAL_IO_NUM]; // neuron number of output shared between inner and outer corticals
} cortical_column_t;

typedef struct __attribute__ ((__packed__)) {
    cortical_column_t cortical[CORTICAL_NUM];
    char cortical_interconnect[CORTICAL_NUM*CORTICAL_IO_NUM*CORTICAL_NUM*CORTICAL_IO_NUM/8];
    unsigned short inputs[NR_INPUTS];
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
    pthread_t thread;
    int thread_id;
    int count;
    IO_t io;
    int getPopulationIndex() const { int r=THREAD_COUNT*count+thread_id; return r;};
    int getPopulationIndexInc() { int r=getPopulationIndex(); count++; return r; };
    void resetCount() { count=0; };
} thread_local_t;

thread_local_t threads[THREAD_COUNT];
pthread_barrier_t barrier;

inline bool
operator< (const event_s & s1, const event_s & s2)
{
  return s1.getValue () < s2.getValue ();
}

priority_queue < struct event_s >events;

population_t population;
population_t population1;
population_t *pPopulation;
population_t *pTmpPopulation;

int current_sample=1;

typedef double (*function_t) (double);

double
sigmoid (double v)
{
  return v / sqrt (1 + v * v);
}

function_t node_functions[1] = { sigmoid };


string
printBitString (char *in, int len)
{
}

void
init ()
{
  srand (time (NULL));
}


void
swap (int array[], int i, int j)
{
  int tmp = array[i];
  array[i] = array[j];
  array[j] = tmp;
}

//#define INDEX(population,index) ((population)->individual[(population)->map[index]])
#define FITNESS(population, index) ((population)->individual[index]).fitness
void qsort_map(int *map, double *oo, int left, int right) {
  if (left < right)
    {
      double p = oo[map[left + (right - left) / 2]];
      int i = left;
      int j = right;

      while (i < j)
	{

	  while (oo[map[i]] > p && i < right)
	    i++;
	  while (oo[map[j]] < p && j > left)
	    j--;
	  if (i <= j)
	    {
	      swap (map, i, j);
	      i++;
	      j--;
	    }
	}

      qsort_map (map,oo, i, right);
      qsort_map (map,oo, left, j);
    }
}

int hamming(double *vec1, double *vec2, int size) {
    int r=0;
    for(int i=0;i<size;i++) {
        int b1=vec1[i]>.5?1:0;
        int b2=vec2[i]>.5?1:0;
        r+=b1^b2;
    }
}

int
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

void qsort(population_t *population) {
    qs(population, population->map, 0, POPULATION_SIZE-1);
}

void reset_individual(genomef_t *genome) {
    memset(genome->node_outputs, 0, sizeof(genome->node_outputs));
}


inline double
my_random ()
{
  double r = ((double) rand () / (RAND_MAX));
  return r;
}

void
mutate (double prob, genome_t * genome)
{
  double r = my_random ();
  int s = rand () % (sizeof (genome_t) * 8);
  if (r < prob)
    {
      ((char *) genome)[s / 8] ^= (1 << (s % 8));
    }
}

void cross(char* ng1, char* g1,char* ng2, char* g2, int s, int size) {
      memcpy (ng1, g1, s / 8);
      memcpy (CHAR (ng1) + s / 8 + 1, CHAR (g2) + s / 8 + 1, size - s / 8 - 1);
      char mask = (1 << (s % 8)) - 1;
      CHAR (ng1)[s / 8] = (CHAR (g1)[s / 8] & ~mask) | (CHAR (g2)[s / 8] & mask);

      memcpy (ng2, g2, s / 8);
      memcpy (CHAR (ng2) + s / 8 + 1, CHAR (g1) + s / 8 + 1, size - s / 8 - 1);
      mask = (1 << (s % 8)) - 1;
      CHAR (ng1)[s / 8] = (CHAR (g2)[s / 8] & ~mask) | (CHAR (g1)[s / 8] & mask);
}

void
crossover (thread_local_t *thread, double prob, population_t * population,
	   population_t * new_population)
{
  int s = rand () % (sizeof (cortical_column_t) * 8);
  double r = my_random ();
  double total = 0;
  assert(thread->getPopulationIndex() < POPULATION_SIZE);
  genomef_t *ng1 = &new_population->individual[thread->getPopulationIndexInc()];
  assert(thread->getPopulationIndex() < POPULATION_SIZE);
  genomef_t *ng2 = &new_population->individual[thread->getPopulationIndexInc()];


  for (int i = 0; i < POPULATION_SIZE; i++)
    {
      total += population->individual[i].fitness;
    }
  genomef_t *g1 = 0;
  genomef_t *g2 = 0;
  while (g1 == g2)
    {
      double r1 = my_random () * total;
      double r2 = my_random () * total;
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

    for(int k =0;k<CORTICAL_NUM;k++) {
       s= rand() % (sizeof(genome_t)*8);

       cross((char*)&ng1->genome, (char*)&g1->genome,(char*)&ng2->genome,(char*)&g2->genome,s,sizeof(genome_t));

        /*
       s= rand() % (sizeof(char)*NODE_COUNT*NODE_COUNT);
       cross((char*)&cg1->incidence,(char*) &g1->genome.incidence, (char*)&cg2->incidence, (char*)&g2->incidence,  s, sizeof(char)*NODE_COUNT*NODE_COUNT/8);
       s= rand() % (sizeof(short)*NODE_COUNT*NODE_COUNT*8);
       cross((char*)&cg1->genome.weights,(char*) &g1->genome.weights, (char*)&cg2->genome.weights, (char*)&g2->genome.weights,  s, sizeof(short)*NODE_COUNT*NODE_COUNT);
       s= rand() % (sizeof(char)*NODE_COUNT*8);
       cross((char*)&cg1->genome.function,(char*) &g1->genome.function, (char*)&cg2->genome.function, (char*)&g2->genome.function,  s, sizeof(char)*NODE_COUNT);
       s= rand() % (sizeof(char)*NR_INPUTS*NODE_COUNT);
       cross((char*)&cg1->genome.inputs,(char*) &g1->genome.inputs, (char*)&cg2->genome.inputs, (char*)&g2->genome.inputs,  s, sizeof(char)*NR_INPUTS*NODE_COUNT/8);
       s= rand() % (sizeof(short)*NR_INPUTS*NODE_COUNT*8);
       cross((char*)&cg1->genome.input_weights,(char*) &g1->genome.input_weights, (char*)&cg2->genome.input_weights, (char*)&g2->genome.input_weights,  s, sizeof(short)*NR_INPUTS*NODE_COUNT);
       s= rand() % (sizeof(char)*NR_OUTPUTS*8);
       cross((char*)&cg1->genome.outputs,(char*) &g1->genome.outputs, (char*)&cg2->genome.outputs, (char*)&g2->genome.outputs,  s, sizeof(char)*NR_OUTPUTS);
       */
    }

/*
      memcpy (&ng1->genome, &g1->genome, s / 8);
      memcpy (CHAR (&ng1->genome) + s / 8 + 1, CHAR (&g2->genome) + s / 8 + 1, sizeof (cortical_column_t) - s / 8 - 1);
      char mask = (1 << (s % 8)) - 1;
      CHAR (&ng1->genome)[s / 8] = (CHAR (&g1->genome)[s / 8] & ~mask) | (CHAR (&g2->genome)[s / 8] & mask);

      memcpy (&ng2->genome, &g2->genome, s / 8);
      memcpy (CHAR (&ng2->genome) + s / 8 + 1, CHAR (&g1->genome) + s / 8 + 1, sizeof (cortical_column_t) - s / 8 - 1);
      mask = (1 << (s % 8)) - 1;
      CHAR (&ng1->genome)[s / 8] = (CHAR (&g2->genome)[s / 8] & ~mask) | (CHAR (&g1->genome)[s / 8] & mask);

      */
    }
  else
    {
      memcpy (&ng1->genome, &g1->genome, sizeof (cortical_column_t));
      memcpy (&ng2->genome, &g2->genome, sizeof (cortical_column_t));
    }

  mutate (MUTATION_PROB, &ng1->genome);
  mutate (MUTATION_PROB, &ng2->genome);
}

void calculate_cortical_outputs(genomef_t *individual,int cortical_index, cortical_column_t *cortical, double inputs[CORTICAL_IO_NUM]) {

    for(int i=0;i<NODE_COUNT;i++) {
        double signal = 0;

        for(int k=0;k<CORTICAL_IO_NUM;k++) {
            if(cortical->inputs[k]%NODE_COUNT==i) {
              double factor = (cortical->input_weights[k][i] / (double)((32768)-1));
              signal+=inputs[k]*factor;
            }
        }

        for(int k=0;k<NODE_COUNT;k++) {
              if (EDGE (cortical->incidence, i, k)) {
                  double factor = (cortical->weights[i][k] /  (double)((32768)-1));
                  signal += individual->node_outputs[cortical_index][k] * factor;
              }
        }

        individual->node_outputs[cortical_index][i]=sigmoid(signal);
    }
}

void _fitness_individual(genomef_t *individual, bool p=false) {


    double inputs[CORTICAL_IO_NUM];
    memset(inputs, 0, sizeof(inputs));

    for(int i=0;i<CORTICAL_NUM;i++) {
        for(int j=0;j<CORTICAL_IO_NUM;j++) {
            for(int k=0;k<CORTICAL_NUM;k++) {
                for(int r=0;r<CORTICAL_IO_NUM;r++) {
                    if(CORTICAL_INTERCONNECT(individual->genome.cortical_interconnect, k,r, i,j)) {
                        inputs[j]=individual->node_outputs[k][r];
                    }
                }
            }
            for(int k=0;k<NR_INPUTS;k++) {
                if(individual->genome.inputs[k]/CORTICAL_NUM==i && individual->genome.inputs[k]%CORTICAL_NUM==j)
                     inputs[j]=individual->input_values[k];
            }
        }
        calculate_cortical_outputs(individual, i, &individual->genome.cortical[i], inputs);
    }



    for(int i=0;i<CORTICAL_NUM;i++) {
        for(int j=0;j<CORTICAL_IO_NUM;j++) {
            for(int k=0;k<NR_OUTPUTS;k++) {
                if(individual->genome.outputs[k]/CORTICAL_IO_NUM==i && individual->genome.outputs[k]%CORTICAL_IO_NUM==j)
                     individual->output_values[k]=individual->node_outputs[i][j];
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

void
fitness_individual (genomef_t * individual, IO_t &io, bool p = false, int count=SAMPLES, bool forecast=false) {


//  cortical_column_t & g = individual->genome;

  for(int k=0;k<NR_OUTPUTS;k++) {
      io.errors[k]=0;
  }
  double cumulative_error=0;

  for (int q = 0; q < count; q++) {

//        reset_individual(individual);
      for(int k=0;k<NR_INPUTS;k++) {
        individual->input_values[k] = io.inputs[k][q];
      }

      if(p) {
          cout << "input: ";
          for(int i=0;i<NR_INPUTS;i++) {
              if(individual->input_values[i]>0.5) {
                  cout << "1" << ",";
              } else {
                  cout << "0" << ",";
              }
          }
          cout << endl;
      }

      _fitness_individual(individual, p);

      double oo[NR_OUTPUTS]={0.,};

      for(int k=0;k<NR_OUTPUTS;k++) {
          io.errors[k]=0;
      }

      for(int k=0;k<NR_OUTPUTS;k++) {
//          int port=g.outputs[k]%NODE_COUNT;
          double output=individual->output_values[k];
          oo[k]=output;
          int b1=io.outputs[k][q]>.5?1:0;
          int b2=output>.5?1:0;
          if(!forecast)
              io.errors[k]+=(b1^b2)+pow(io.outputs[k][q] - output,2);
      }
      int hamming_distance=0;
      for(int k=0;k<NR_OUTPUTS;k++) {
          hamming_distance+=io.errors[k];
      }
      cumulative_error+=hamming_distance;
      if (p){
          cout << "fit:   ";
          for(int i=0;i<NR_OUTPUTS;i++) {
              if(oo[i]>0.50) {
                  cout << "1" <<  ",";
              } else {
                  cout << "0" <<  ",";
              }
          }
          cout << endl;
    	//cout << output << round (output) <<  endl;
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
  if (p)
    cout << "best individual fitness: " << individual->fitness << endl;
    
}


void
fitness (thread_local_t *thread, population_t * population, IO_t& io)
{

    int i=0;
    thread->resetCount();
    while((i=thread->getPopulationIndexInc()) < POPULATION_SIZE) {
       fitness_individual (&population->individual[i],io, false, current_sample);
    }
}

void find_best_individual(thread_local_t *thread, population_t* population) {
      pthread_barrier_wait(&barrier);
  if(thread->thread_id==0) {
    qsort(population);
    population->best_individual = &population->individual[population->map[0]];
  }
      pthread_barrier_wait(&barrier);
}

void
init_population (thread_local_t *thread, population_t * population)
{

    int i=0;
    thread->resetCount();
    while((i=thread->getPopulationIndexInc()) < POPULATION_SIZE) {
      population->individual[i].fitness = 0;
      population->map[i]=i;
      memset(population->individual[i].node_outputs, 0, sizeof(population->individual[i].node_outputs));
      /*
      for (int j = 0; j < NODE_COUNT; j++) {
	    population->individual[i].node_outputs[j] = 0;
	  }
      */
    }
}

void
init_random_population (thread_local_t *thread)
{
  init_population(thread, &population);
  int i=0;
  thread->resetCount();
  while((i=thread->getPopulationIndexInc()) < POPULATION_SIZE) {
      for (int j = 0; j < sizeof (cortical_column_t); j++) {
    	  ((char *) &population.individual[i].genome)[j] = rand () % 255;
	}
 }
}

void
copy_best_individuals (thread_local_t* thread, population_t * p1, population_t * p2)
{
      pthread_barrier_wait(&barrier);
    if(thread->thread_id==0) {
        for(int i=0;i<BEST_INDIVIDUALS && i<POPULATION_SIZE;i++) {
          memcpy (&p2->individual[thread->getPopulationIndexInc()].genome, &p1->individual[p1->map[i]], sizeof (genome_t));
        }
    }
      pthread_barrier_wait(&barrier);
}

void print(IO_t &io,genomef_t *genome) {
    cout << "best fitness: " << genome->fitness << endl;
    reset_individual(genome);
    fitness_individual(genome,io, true, current_sample, true);
}

void read_io(IO_t &io) {
#ifndef KENO
    /*
  char* in[]={
      "0,0,0",
      "0,0,1",
      "0,1,0",
      "0,1,1",
      "1,0,0",
      "1,0,1",
      "1,1,0",
      "1,1,1"};
  char* out[]={
      "0,0",
      "1,0",
      "0,1",
      "1,1",
      "0,0",
      "0,1",
      "1,0",
      "1,1"};
      */
      const char* in[]={
#include "bch_input.dat"
      };
      const char* out[]={
#include "bch_output.dat"
      };
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
#else
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
#endif

}

void *
genetic (void *t)
{
  thread_local_t *thread=(thread_local_t*)t;
  IO_t io=thread->io;
  read_io(io);
  init_random_population (thread);
  fitness (thread, pPopulation,io);

current_sample=SAMPLES;
//while(current_sample<SAMPLES) {  
  do
    {
      init_population (thread,pTmpPopulation);
      /*
      if(thread->thread_id==0) {
          for(int i=0;i<POPULATION_SIZE;i++) {
              cout << pTmpPopulation->map[i] << ",";
          }
          cout << endl;
      }
      */
        thread->resetCount();
      copy_best_individuals (thread,pPopulation, pTmpPopulation);

        while(thread->getPopulationIndex() < POPULATION_SIZE) {
          crossover (thread,CROSSOVER_PROB, pPopulation, pTmpPopulation);
        }

      pthread_barrier_wait(&barrier);

      if(thread->thread_id==0) {
          population_t *p = pPopulation;
          pPopulation = pTmpPopulation;
          pTmpPopulation = p;
      }

      pthread_barrier_wait(&barrier);

      fitness (thread,pPopulation, io);
      find_best_individual(thread, pPopulation);
      pthread_barrier_wait(&barrier);
      if(thread->thread_id==0)
          print (io,pPopulation->best_individual);
      pthread_barrier_wait(&barrier);

    }
  while (pPopulation->best_individual->fitness < .9999);
  reset_individual(pPopulation->best_individual);
  fitness_individual(pPopulation->best_individual, io, true);
  cout << pPopulation->best_individual->fitness << endl;
  /*
  pthread_barrier_wait(&barrier);
  if(thread->thread_id==0) {
      current_sample++;
      cout << "current sample: " << current_sample << endl;
  }
  pthread_barrier_wait(&barrier);
  */
//}
  pthread_exit(NULL);

}

/*
void sighandler(int arg) {
    cout << "Finish: " << endl;
    reset_individual(pPopulation->best_individual);
    fitness_individual(pPopulation->best_individual, io, true);
    exit(-1);
}
*/

int
main ()
{
        char cortical_incidence[CORTICAL_NUM*NR_OUTPUTS*CORTICAL_NUM*NR_INPUTS/8];

  cout << "cortical size: " <<  sizeof (cortical_column_t) << endl;
  cout << "genome size: " << sizeof (genome_t) << endl;
  cout << "cortical incidence: " << sizeof(cortical_incidence) << endl;

  init ();
  //signal(SIGINT, sighandler);
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
  return 0;
}
