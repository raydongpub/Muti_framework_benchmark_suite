
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include "global.h"
#include "nw_cpu.h"
#include "nw_gpu.h"
#include "mpi.h"
#include <unistd.h>


#define WORKTAG 1
#define DIETAG 2

int totalpairs=1000;



extern char blosum62_cpu[24][24];
extern int tile_size;

void init_conf() {
	config.debug = false;
	config.device = 0;
	config.kernel = 0;
	config.num_blocks = 14;
	config.num_threads = 32;
	config.num_streams = 0;
	config.length = 1600;
	config.penalty = -10;
	config.repeat = 1;
}

void init_device(int device) {
	cudaSetDevice(device);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	char hostname[1024];
    	gethostname(hostname, 1024);
    	printf("GPU %d is set on %s host\n", device, hostname);
	cudaFree(0);
}

void usage(int argc, char **argv)
{
    fprintf(stderr, "\nUsage: %s [options]\n", argv[0]);
    fprintf(stderr, "\t[--length|-l <length> ] - x and y length (default: %d)\n",config.length);
    fprintf(stderr, "\t[--penalty|-p <penalty>] - penalty (negative integer, default: %d)\n",config.penalty);
    fprintf(stderr, "\t[--num_pair|-n <pair num>] - number of pairs per stream (default: %d)\n",config.num_streams);
    fprintf(stderr, "\t[--device|-d <device num> ]- device ID (default: %d)\n",config.device);
    fprintf(stderr, "\t[--kernel|-k <kernel type> ]- 0: diagonal 1: tile (default: %d)\n",config.kernel);
    fprintf(stderr, "\t[--num_blocks|-b <blocks> ]- blocks number per grid (default: %d)\n",config.num_blocks);
    fprintf(stderr, "\t[--num_threads|-t <threads> ]- threads number per block (default: %d)\n",config.num_threads);
    fprintf(stderr, "\t[--repeat|-r <num> ]- repeating number (default: %d)\n",config.repeat);
    fprintf(stderr, "\t[--debug]- 0: no validation 1: validation (default: %d)\n",config.debug);
    fprintf(stderr, "\t[--help|-h]- help information\n");
    exit(1);
}

void print_config()
{
	fprintf(stderr, "=============== Configuration ================\n");
    fprintf(stderr, "device = %d\n", config.device);
	fprintf(stderr, "kernel = %d\n", config.kernel);
	if ( config.kernel == 1 )
		fprintf(stderr, "tile size = %d\n", tile_size);
    fprintf(stderr, "stream number = %d\n", config.num_streams);
	for (int i=0; i<config.num_streams; ++i) {
    	fprintf(stderr, "Case %d - sequence number = %d\n", i, config.num_pairs[i]);
	}
	fprintf(stderr, "sequence length = %d\n", config.length);
    fprintf(stderr, "penalty = %d\n", config.penalty);
    fprintf(stderr, "block number = %d\n", config.num_blocks);
    fprintf(stderr, "thread number = %d\n", config.num_threads);
	if ( config.num_streams==0 ) {
		fprintf(stderr, "\nNot specify sequence length\n");
	}
    fprintf(stderr, "repeat = %d\n", config.repeat);
    fprintf(stderr, "debug = %d\n", config.debug);
	printf("==============================================\n");
}

void validate_config()
{
	if ( config.length % tile_size != 0 &&
		 config.kernel == 1 ) {
		fprintf(stderr, "Tile kernel used.\nSequence length should be multiple times of tile size %d\n", tile_size);
		exit(1);
	}
}

int validation(int *score_matrix_cpu, int *score_matrix, unsigned int length, int Rank)
{			

	FILE *fp;
	char filename[20]="first";
	sprintf(filename, "file_%d.out",Rank);
	fp = fopen(filename, "w");


	
	


	MPI_Status status;
   	 unsigned int i = 0;
	//printf("Length : %d\n", length);
    	while (i!=length){

        if ( score_matrix_cpu[i]==score_matrix[i] ){
		//	printf("On GPU: score_matrix[%d] = %d\n", i, score_matrix[i]);
		//	printf("On CPU: score_matrix[%d] = %d\n", i, score_matrix_cpu[i]);
			fprintf(fp, "Dumping result %d to file from rank %d\n", score_matrix[i], Rank);	
			//printf("Score-matrix is %d and Rank is %d\n", score_matrix[i], Rank);
            ++i;
            continue;
        }

        else {
            printf("On GPU: score_matrix[%d] = %d\n", i, score_matrix[i]);
            printf("On CPU: score_matrix[%d] = %d\n", i, score_matrix_cpu[i]);
           //++i;
			return 0;
        }	

   }

	fclose(fp);
    return 1;
}

void sendtosingleprocess(int *score_matrix, unsigned int length, int Rank, int NumberOfProcessors){




		FILE *fp2;
	char filename2[20]="second";
	sprintf(filename2, "centralnode.out",Rank);
	fp2 = fopen(filename2, "w");


	
		MPI_Status status; 	
		unsigned int i=0;
			
		int *central_node = (int *)malloc(NumberOfProcessors*pos_matrix[i][pair_num[i]]*sizeof(int));
		
		
		MPI_Gather(score_matrix, length, MPI_INT, central_node, length, MPI_INT, 0, MPI_COMM_WORLD);

		if (Rank==0){		
		while (i != NumberOfProcessors*length){
		
			fprintf(fp2, "Dumping central node result %d to file \n", central_node[i]);	
			++i; 
			}}

		fclose(fp2);
		
	
		
}



int parse_arguments(int argc, char **argv)
{
	int i = 1;
	if (argc<2) {
		usage(argc, argv);
		return 0;
	}
	while(i<argc) {
		if(strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--device") == 0){
			i++;
			if (i==argc){
				fprintf(stderr,"device number missing.\n");
				return 0 ;
			}
			config.device = atoi(argv[i]);
		}else if(strcmp(argv[i], "--debug") == 0){
			config.debug = 1;
		}else if(strcmp(argv[i], "-k") == 0 || strcmp(argv[i], "--kernel") == 0){
			i++;
			if (i==argc){
				fprintf(stderr,"device number missing.\n");
				return 0 ;
			}
			config.kernel = atoi(argv[i]);
		}else if(strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--repeat") == 0){
			i++;
			if (i==argc){
				fprintf(stderr,"repeating number missing.\n");
				return 0 ;
			}
			config.repeat = atoi(argv[i]);
		}else if(strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--num_threads") == 0){
			i++;
			if (i==argc){
				fprintf(stderr,"thread number missing.\n");
				return 0 ;
			}
			config.num_threads = atoi(argv[i]);
		}else if(strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--num_blocks") == 0){
			i++;
			if (i==argc){
				fprintf(stderr,"block number missing.\n");
				return 0 ;
			}
			config.num_blocks = atoi(argv[i]);
		}else if(strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--penalty") == 0){
			i++;
			if (i==argc){
				fprintf(stderr,"penalty score missing.\n");
				return 0 ;
			}
			config.penalty = atoi(argv[i]);
		}else if(strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--num_pairs") == 0){
			i++;
			if (i==argc){
				fprintf(stderr,"sequence length missing.\n");
				return 0 ;
			}
			config.num_pairs[ config.num_streams ] = atoi(argv[i]);
			if ( config.num_pairs[ config.num_streams ] > MAX_SEQ_NUM ) {
				fprintf(stderr, "The maximum sequence number per stream is %d\n", MAX_SEQ_NUM);
				return 0;
			} 
			config.num_streams++;
		}else if(strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--lengths") == 0){
			i++;
			if (i==argc){
				fprintf(stderr,"sequence length missing.\n");
				return 0 ;
			}
			config.length = atoi(argv[i]);
			if ( config.length > MAX_SEQ_LEN ) {
				fprintf(stderr,"The maximum seqence length is %d\n", MAX_SEQ_LEN);
				return 0;
			}
 		}else if(strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
			usage(argc, argv);
			return 0;
		}else {
			fprintf(stderr,"Unrecognized option : %s\nTry --help for more information\n", argv[i]);
			return 0;
		}
		i++;
	}
	return 1;
}





















/* Local functions */

static void master(void);
static void slave(int myrank);
static unit_of_work_t get_next_work_item(void);
static void process_results(int result);
static int do_work(int work);





int main(int argc, char **argv)
{


	double s_time, e_time;
	init_conf();
	while(!parse_arguments(argc, argv)) usage(argc, argv);

	print_config();
	validate_config();
	int dev_num = config.device;
	int	penalty = config.penalty;
	int seq1_len, seq2_len;
	int seq_len = config.length;
	DEBUG = config.debug;
	

 	 int myrank;

	  /* Initialize MPI */

 	 MPI_Init(&argc, &argv);

  	/* Find out my identity in the default communicator */

  	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  	if (myrank == 0) {
  	  master();
 	 } else {
 	   slave(myrank);
 	 }

 	/* Shut down MPI */

 	 MPI_Finalize();
 	 return 0;
}


static void master(void)
{
  int ntasks, rank;
  int work;
  int result;
  MPI_Status status;

  /* Find out how many processes there are in the default
     communicator */

  MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

  /* Seed the slaves; send one unit of work to each slave. */

  for (rank = 1; rank < ntasks; ++rank) {

    /* Find the next item of work to do */

    work = get_next_work_item();

    /* Send it to each rank */

    MPI_Send(&work,             /* message buffer */
             1,                 /* one data item */
             MPI_INT,           /* data item is an integer */
             rank,              /* destination process rank */
             WORKTAG,           /* user chosen message tag */
             MPI_COMM_WORLD);   /* default communicator */
  }

  /* Loop over getting new work requests until there is no more work
     to be done */

  work = get_next_work_item();
  while (work != NULL) {

    /* Receive results from a slave */

    MPI_Recv(&result,           /* message buffer */
             1,                 /* one data item */
             MPI_DOUBLE,        /* of type double real */
             MPI_ANY_SOURCE,    /* receive from any sender */
             MPI_ANY_TAG,       /* any type of message */
             MPI_COMM_WORLD,    /* default communicator */
             &status);          /* info about the received message */

    /* Send the slave a new work unit */

    MPI_Send(&work,             /* message buffer */
             1,                 /* one data item */
             MPI_INT,           /* data item is an integer */
             status.MPI_SOURCE, /* to who we just received from */
             WORKTAG,           /* user chosen message tag */
             MPI_COMM_WORLD);   /* default communicator */

    /* Get the next unit of work to be done */

    work = get_next_work_item();
  }

  /* There's no more work to be done, so receive all the outstanding
     results from the slaves. */

  for (rank = 1; rank < ntasks; ++rank) {
    MPI_Recv(&result, 1, MPI_DOUBLE, MPI_ANY_SOURCE,
             MPI_ANY_TAG, MPI_COMM_WORLD, &status);
  }

  /* Tell all the slaves to exit by sending an empty message with the
     DIETAG. */

  for (rank = 1; rank < ntasks; ++rank) {
    MPI_Send(0, 0, MPI_INT, rank, DIETAG, MPI_COMM_WORLD);
  }
}


static void slave(void, rank)
{
  int work;
  int  results;
  MPI_Status status;

  while (1) {

    /* Receive a message from the master */

    MPI_Recv(&work, 1, MPI_INT, 0, MPI_ANY_TAG,
             MPI_COMM_WORLD, &status);

    /* Check the tag of the received message. */

    if (status.MPI_TAG == DIETAG) {
      return;
    }

    /* Do the work */

    result = do_work(work, rank);

    /* Send the result back */

    MPI_Send(&result, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
}


static int get_next_work_item(void)
{
  /* Fill in with whatever is relevant to obtain a new unit of work
     suitable to be given to a slave. */

	
	int batch_size = 5; 
	totalpairs=totalpairs-batch_size;
	printf("%d\n", totalpairs);

	if (totalpairs<0)
	return NULL;
	else 
	return batch_size; 


}


static void process_results(int result)
{
  /* Fill in with whatever is relevant to process the results returned
     by the slave */
}


static int do_work(int work, int Rank)
{
  /* Fill in with whatever is necessary to process the work and
     generate a result */
	

	
	cudaStream_t stream[config.num_streams];

	int *score_matrix[config.num_streams];
	srand ( 7 );
	for (int k=0; k<config.num_streams;++k) {
    	pos_matrix[k][0] = pos1[k][0] = pos2[k][0] = 0;
		pair_num[k] = config.num_pairs[Rank];
		//printf("Pair num is %d and Rank is %d and pairnumrank is %d\n",pair_num[k],Rank,pair_num[Rank]);
    	for (int i=0; i<pair_num[k]; ++i){
        	//please define your own sequence 1
        	seq1_len = seq_len; //64+rand() % 20;
        	//printf("Seq1 length: %d\n", seq1_len);    
        	for (int j=0; j<seq1_len; ++j)
            	sequence_set1[k][ pos1[k][i] + j ] = rand() % 20 + 1;
        	pos1[k][i+1] = pos1[k][i] + seq1_len;
        	//please define your own sequence 2.
        	seq2_len = seq_len;//64+rand() % 20;       
        	//printf("Seq2 length: %d\n\n", seq2_len);      
        	for (int j=0; j<seq2_len; ++j)
            	sequence_set2[k][ pos2[k][i] +j ] = rand() % 20 + 1;
        	pos2[k][i+1] = pos2[k][i] + seq2_len;
        	//printf("Matrix size increase: %d\n", (seq1_len+1) * (seq2_len+1));
        	pos_matrix[k][i+1] = pos_matrix[k][i] + (seq1_len+1) * (seq2_len+1);
			dim_matrix[k][i] = (unsigned int)ceil( (float)seq_len/tile_size )*tile_size;
    	}
		// Pageable memory
		//score_matrix[k] = (int *)malloc( pos_matrix[k][pair_num[k]]*sizeof(int) );
		// Pinned memory
		cudaMallocHost( (void **)&score_matrix[k],  pos_matrix[k][pair_num[k]]*sizeof(int) );
	}
	/* initialize the GPU */
	s_time = gettime();
	//MPI_Barrier(MPI_COMM_WORLD);
	init_device( setdevice );
	//MPI_Barrier(MPI_COMM_WORLD);
	e_time = gettime();
	fprintf(stderr,"Initialize GPU : %fs\n", e_time - s_time);
	//printf("Rank %d reaches here\n", Rank);
	s_time = gettime();

	//MPI_Barrier(MPI_COMM_WORLD);
	for (int i=0; i<config.num_streams; ++i) {
		nw_gpu_allocate(i);
		cudaStreamCreate( &(stream[i]) );
		//printf("Cuda stream create by rank %d\n", Rank);
	}
	e_time = gettime();
	fprintf(stderr,"Memory allocation and copy on GPU : %f\n", e_time - s_time);
	
	for (int r=0; r<config.repeat; ++r) {
	fprintf(stderr, "Round #%d:\n", r);
	s_time = gettime();
	for (int i=0; i<config.num_streams; ++i ) {
		double stream_time_s, stream_time_e;
		stream_time_s = gettime();
		if (DEBUG) {
			fprintf(stderr,"Dataset[%d] starts\n", Rank);
		}
		//	MPI_Barrier(MPI_COMM_WORLD);
		nw_gpu(sequence_set1[i], sequence_set2[i], pos1[i], pos2[i], score_matrix[i], pos_matrix[i], pair_num[i], d_score_matrix[i], stream[0], i, config.kernel);
		nw_gpu_copyback(score_matrix[i], d_score_matrix[i], pos_matrix[i], pair_num[i], stream[0],i);
		
	


		cudaStreamSynchronize(stream[0]);
		stream_time_e = gettime();
		fprintf(stderr,"Dataset[%d] runtime on GPU : %fs\n", Rank, stream_time_e - stream_time_s);
	}	
	e_time = gettime();
	fprintf(stderr,"Total runtime on GPU : %fs\n", e_time - s_time);



	
	if (DEBUG) {
		for ( int i=0; i<config.num_streams; ++i) {
    		int *score_matrix_cpu = (int *)malloc( pos_matrix[i][pair_num[i]]*sizeof(int));
    		needleman_cpu(sequence_set1[i], sequence_set2[i], pos1[i], pos2[i], score_matrix_cpu, pos_matrix[i], pair_num[i], penalty);
	MPI_Barrier(MPI_COMM_WORLD);
	printf("Rank %d sending validation", Rank);
	MPI_Barrier(MPI_COMM_WORLD);

	if ( validation(score_matrix_cpu, score_matrix[i], pos_matrix[i][pair_num[i]], Rank) )
        		printf("Stream %d - Validation: PASS\n", i);
    		else
        		printf("Stream %d - Validation: FAIL\n", i);
			free(score_matrix_cpu);
		}
	}
	

}

	/*
	for ( int i=0; i<config.num_streams; ++i)
	sendtosingleprocess(score_matrix[i], pos_matrix[i][pair_num[i]], Rank, NumberOfProcessors);
	*/

	return 1; 




}
