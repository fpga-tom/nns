#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <queue>		// std::priority_queue
#include <math.h>


using namespace std;

#define NODE_COUNT 4
#define POPULATION_SIZE 2000
#define MUTATION_PROB .010
#define CROSSOVER_PROB .8
#define NR_INPUTS 3
#define NR_OUTPUTS 1
#define BEST_INDIVIDUALS 2

#define CHAR(x) ((char*)x)
#define EDGE(incidence, x, y) (incidence[(x*NODE_COUNT+y)/8] & (1<< ((x*NODE_COUNT+y)%8)))
#define IEDGE(inputs,i,n) (inputs[i][n/8] & (1<<(n%8)))

#define ABS(x) ((x) < 0 ? -(x) : (x))

typedef struct __attribute__ ((__packed__))
{
  char incidence[NODE_COUNT * NODE_COUNT / 8];
  short weights[NODE_COUNT][NODE_COUNT];
  unsigned char function[NODE_COUNT];
//    char distance[NODE_COUNT][NODE_COUNT];
  unsigned char inputs[NR_INPUTS][NODE_COUNT / 8];
  short input_weights[NR_INPUTS][NODE_COUNT];
  unsigned char outputs[NR_OUTPUTS];
} genome_t;

typedef struct
{
  genome_t genome;
  double fitness;
  double node_outputs[NODE_COUNT];
  double input_values[NR_INPUTS];
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
  int count;
} population_t;


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
    for(int i=0;i<NODE_COUNT;i++)
        genome->node_outputs[i]=0;
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

void
crossover (double prob, population_t * population,
	   population_t * new_population)
{
  int s = rand () % (sizeof (genome_t) * 8);
  double r = my_random ();
  double total = 0;
  genomef_t *ng1 = &new_population->individual[new_population->count++];
  genomef_t *ng2 = &new_population->individual[new_population->count++];

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
      //genomef_t *ng1=&new_population->individual[new_population->count++];
      //genomef_t *ng2=&new_population->individual[new_population->count++];

      memcpy (&ng1->genome, &g1->genome, s / 8);
      memcpy (CHAR (&ng1->genome) + s / 8 + 1, CHAR (&g2->genome) + s / 8 + 1,
	      sizeof (genome_t) - s / 8 - 1);
      char mask = (1 << (s % 8)) - 1;
      CHAR (&ng1->genome)[s / 8] =
	(CHAR (&g1->genome)[s / 8] & ~mask) | (CHAR (&g2->genome)[s / 8] &
					       mask);

      memcpy (&ng2->genome, &g2->genome, s / 8);
      memcpy (CHAR (&ng2->genome) + s / 8 + 1, CHAR (&g1->genome) + s / 8 + 1,
	      sizeof (genome_t) - s / 8 - 1);
      mask = (1 << (s % 8)) - 1;
      CHAR (&ng1->genome)[s / 8] =
	(CHAR (&g2->genome)[s / 8] & ~mask) | (CHAR (&g1->genome)[s / 8] &
					       mask);
    }
  else
    {
      memcpy (&ng1->genome, &g1->genome, sizeof (genome_t));
      memcpy (&ng2->genome, &g2->genome, sizeof (genome_t));
    }

  mutate (MUTATION_PROB, &ng1->genome);
  mutate (MUTATION_PROB, &ng2->genome);
}

void
fitness_individual (genomef_t * individual, bool p = false)
{

  genome_t & g = individual->genome;
  // training xor
  // A B Y
  // 0 0 0
  // 0 1 1
  // 1 0 1
  // 1 1 0

  // training switch
  // A B C S1 S2
  // 0 0 0 0 0
  // 0 0 1 1 0
  // 0 1 0 0 1
  // 0 1 1 1 1
  // 1 0 0 0 0
  // 1 0 1 0 1
  // 1 1 0 1 0
  // 1 1 1 1 1


/*
    double A[4] = {0,0,1,1};
    double B[4] = {0,1,0,1};
    double Y[4] = {0,1,1,0};
*/
  double A[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };
  double B[8] = { 0, 0, 1, 1, 0, 0, 1, 1 };
  double C[8] = { 0, 1, 0, 1, 0, 1, 0, 1 };
  double Y[8] = { 0, 1, 0, 1, 0, 0, 1, 1 };
  double Y1[8] = { 0, 0, 1, 1, 0, 1, 0, 1 };
  double error = 0;
  double error1 = 0;

  for (int q = 0; q < 8; q++)
    {

      individual->input_values[0] = A[q];
      individual->input_values[1] = B[q];
      individual->input_values[2] = C[q];

      for (int i = 0; i < NODE_COUNT; i++)
	{
	  double signal = 0;
	  for (int k = 0; k < NR_INPUTS; k++)
	    {
	      if (IEDGE (g.inputs, k, i))
		{
		  double factor = (g.input_weights[k][i] / 16384.);
		  signal += individual->input_values[k] * factor;
		}
	    }
	  for (int j = 0; j < NODE_COUNT; j++)
	    {
	      if (EDGE (g.incidence, i, j))
		{
		  double factor = (g.weights[i][j] / 16384.);
		  signal += individual->node_outputs[j] * factor;
		}
	    }
	  individual->node_outputs[i] =
	    node_functions[g.function[i] %
			   (sizeof (node_functions) /
			    sizeof (function_t))] (signal);
	}

      int out1 = g.outputs[0];
      double output = individual->node_outputs[out1 % NODE_COUNT];
      error += pow (Y[q] - output, 2);
      if (p)
	cout << A[q] << " " << B[q] << " " << output << " " << round (output)
	  << " " << error << endl;
    }

  error /= 8;
  individual->fitness = 1. / error;
  individual->output_errors[0] = error;
  if (p)
    cout << individual->fitness << endl;

}

void
fitness (population_t * population)
{

  for (int i = 0; i < POPULATION_SIZE; i++)
    {
      fitness_individual (&population->individual[i]);
    }

  qsort(population);
  population->best_individual = &population->individual[population->map[0]];
}

void
init_population (population_t * population)
{

  for (int i = 0; i < POPULATION_SIZE; i++)
    {
      population->individual[i].fitness = 0;
      population->map[i]=i;
      for (int j = 0; j < NODE_COUNT; j++)
	{
	  population->individual[i].node_outputs[j] = 0;
	}
    }
  population->count = 0;
}

void
init_random_population ()
{
  init ();
  init_population(&population);
  for (int i = 0; i < POPULATION_SIZE; i++)
    {
      for (int j = 0; j < sizeof (genome_t); j++)
	{
	  ((char *) &population.individual[i].genome)[j] = rand () % 255;
	}
    }
}

void
copy_best_individuals (population_t * p1, population_t * p2)
{
    for(int i=0;i<BEST_INDIVIDUALS && i<POPULATION_SIZE;i++) {
      memcpy (&p2->individual[p2->count++].genome, &p1->individual[p1->map[i]], sizeof (genome_t));
    }
}

void print(genomef_t *genome) {
    cout << "best fitness: " << genome->fitness << endl;
}


void
genetic ()
{
  init_random_population ();
  pPopulation = &population;
  pTmpPopulation = &population1;
  fitness (pPopulation);

  do
    {
      init_population (pTmpPopulation);
      copy_best_individuals (pPopulation, pTmpPopulation);

/*
      qsort(pPopulation);
      for(int i=0; i<POPULATION_SIZE;i++) {
         cout << FITNESS(pPopulation, pPopulation->map[i]) << ",";
        //cout << pPopulation->map[i] << ",";
      }
      cout << endl;
*/
      while (pTmpPopulation->count < POPULATION_SIZE)
	{
	  crossover (CROSSOVER_PROB, pPopulation, pTmpPopulation);
	}

      population_t *p = pPopulation;
      pPopulation = pTmpPopulation;
      pTmpPopulation = p;

      fitness (pPopulation);
      print (pPopulation->best_individual);

    }
  while (pPopulation->best_individual->fitness < 75);
  reset_individual(pPopulation->best_individual);
  fitness_individual(pPopulation->best_individual, true);
  cout << pPopulation->best_individual->fitness << endl;

}

int
main ()
{
  cout << sizeof (genome_t) << endl;
  genetic ();
  return 0;
}
