#include "layered.h"

const char *filename;
const char *infile;
const char *outfile;

// ----------------------------------------------------------------------------------------------------
//                       main function
// ----------------------------------------------------------------------------------------------------

void usage(char** argv) {
	fprintf(stderr,"Usage %s [-g genome] [-i indata] [-o outdata]\n", argv[0]);
	exit(-1);

}
int main(int argc, char **argv) {

	int opt;
	if(argc!=7) {
		usage(argv);
	}

	while((opt=getopt(argc, argv, "i:o:g:"))!=-1) {
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
			default:
				usage(argv);
		}
	}

	assert(NR_INPUTS<=NEURON_NUM);
	printf("population: %ld\n", sizeof(population_t)/1024/1024);
	printf("ga_genome_t %ld\n", sizeof(ga_genome_t));
	printf("bp_genome_t %ld\n", sizeof(bp_genome_t));

	population_t *p1,*p2;
	IO_t tmp;
	IO_t *io;

	read_io(tmp,infile, outfile);
	
//	cuAllocPopulation(&p1);
//	cuAllocPopulation(&p2);

	cudaMalloc((void**)&p1, sizeof(population_t));
	cudaMalloc((void**)&p2, sizeof(population_t));

  	cudaMalloc((void**)&deviceBestIndividualFitness, sizeof(double));
	cudaHostAlloc((void**)&hostBestIndividualFitness,sizeof(double), cudaHostAllocWriteCombined);
	cudaMalloc((void**)&io,sizeof(IO_t));

	cudaMemcpy(io,&tmp,sizeof(IO_t),cudaMemcpyHostToDevice);

	hoReset(p1);
	hoReset(p2);
	population_t *p=genetic(io, p1,p2, deviceBestIndividualFitness);

	int fd=open(filename, O_CREAT|O_WRONLY|O_TRUNC, 0444);
	writeBest(fd, p);
	close(fd);

	cudaFree(p1);
	cudaFree(p2);
	cudaFree(io);
	cudaFree(hostBestIndividualFitness);
	return 0;
}
