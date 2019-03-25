#include "nw.decl.h"
#include "nw.h"
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include <iostream>

using namespace std;

_CONF_ config;

/*************** Init Function ****************/

void init_conf() {
    config.debug = false;
    config.device = 0;
    config.kernel = 0;
    config.num_blocks = 64;
    config.num_threads = 512;
    config.num_streams = 0;
    config.length = 1600;
    config.penalty = -10;
    config.repeat = 1;
    config.dataset = 50;
}

void usage(int argc, char **argv) {

    fprintf(stderr, "\nUsage: %s [options]\n", argv[0]);
    fprintf(stderr, "\t[--length|-l <length> ] - x and y length(default:");
    fprintf(stderr,"%d)\n",config.length);
    fprintf(stderr, "\t[--penalty|-p <penalty>] - penalty (negative");
    fprintf(stderr,"integer, default: %d)\n",config.penalty);
    fprintf(stderr, "\t[--num_pair|-n <pair num>] - number of pairs per");
    fprintf(stderr,"stream (default: %d)\n",config.num_streams);
    fprintf(stderr, "\t[--device|-d <device num> ]- device ID (default:");
    fprintf(stderr,"%d)\n",config.device);
    fprintf(stderr, "\t[--kernel|-k <kernel type> ]- 0: diagonal 1: tile");
    fprintf(stderr,"(default: %d)\n",config.kernel);
    fprintf(stderr, "\t[--num_blocks|-b <blocks> ]- blocks number per grid");
    fprintf(stderr,"(default: %d)\n",config.num_blocks);
    fprintf(stderr, "\t[--num_threads|-t <threads> ]- threads number per");
    fprintf(stderr,"block (default: %d)\n",config.num_threads);
    fprintf(stderr, "\t[--repeat|-r <num> ]- repeating number (default:");
    fprintf(stderr,"%d)\n",config.repeat);
    fprintf(stderr, "\t[--debug]- 0: no validation 1: validation (default:");
    fprintf(stderr,"%d)\n",config.debug);
    fprintf(stderr, "\t[--help|-h]- help information\n");
    exit(1);
}

void print_config() {

    fprintf(stderr, "=============== Configuration ================\n");
    fprintf(stderr, "device = %d\n", config.device);
    fprintf(stderr, "kernel = %d\n", config.kernel);
//    if ( config.kernel == 1 )
//        fprintf(stderr, "tile size = %d\n", tile_size);
    fprintf(stderr, "stream number = %d\n", config.num_streams);
    for (int i=0; i<config.num_streams; ++i) {
        fprintf(stderr, "Case %d - ", i );
        fprintf(stderr, "sequence number = %d\n", config.num_pairs);
    } // end for
    fprintf(stderr, "sequence length = %d\n", config.length);
    fprintf(stderr, "penalty = %d\n", config.penalty);
    fprintf(stderr, "block number = %d\n", config.num_blocks);
    fprintf(stderr, "thread number = %d\n", config.num_threads);
    if ( config.num_streams==0 ) {
        fprintf(stderr, "\nNot specify sequence length\n");
    } //end if
    fprintf(stderr, "repeat = %d\n", config.repeat);
    fprintf(stderr, "debug = %d\n", config.debug);
    fprintf(stderr, "div_fac = %d\n", config.div);
    fprintf(stderr, "chare_num = %d\n", config.chare_num);
    printf("==============================================\n");
    } // end print_config()


int parse_arguments(int argc, char **argv){
    int i = 1;
    if (argc<4) {
        usage(argc, argv);
        ckout << "Arg too small!" << endl; 
        return 0;
    } // end if
    while(i<argc) {
        if(strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--device") ==
          0){
            i++;
            if (i==argc){
                fprintf(stderr,"device number missing.\n");
                return 0 ;
                } // end if
            config.device = atoi(argv[i]);
        } // end if 
        else if(strcmp(argv[i], "--debug") == 0){
            config.debug = 1;
        } // end else if
        else if(strcmp(argv[i], "-k") == 0 || strcmp(argv[i],
                 "--kernel") == 0){
            i++;
            if (i==argc){
                fprintf(stderr,"device number missing.\n");
                return 0 ;
            }
            config.kernel = atoi(argv[i]);
        } // end elseif
        else if(strcmp(argv[i], "-r") == 0 || strcmp(argv[i],
                 "--repeat") == 0){
            i++;
            if (i==argc){
                fprintf(stderr,"repeating number missing.\n");
                return 0 ;
            }
            config.repeat = atoi(argv[i]);
        } // end elseif
        else if(strcmp(argv[i], "-t") == 0 || strcmp(argv[i],
                 "--num_threads") == 0){
            i++;
            if (i==argc){
                fprintf(stderr,"thread number missing.\n");
                return 0 ;
            }
            config.num_threads = atoi(argv[i]);
        } // end elseif
        else if(strcmp(argv[i], "-ds") == 0 || strcmp(argv[i],
                        "--dataset") == 0){
            i++;
            if (i==argc){
                fprintf(stderr,"Dataset size.\n");
                return 0 ;
            }
            config.dataset = atoi(argv[i]);
        } // end elseif
        else if(strcmp(argv[i], "-b") == 0 || strcmp(argv[i],                                    "--num_blocks") == 0){
            i++;
            if (i==argc){
                fprintf(stderr,"block number missing.\n");
                return 0 ;
            }
            config.num_blocks = atoi(argv[i]);
        } // end elseif
        else if(strcmp(argv[i], "-p") == 0 || strcmp(argv[i],
                 "--penalty") == 0){
            i++;
            if (i==argc){
                fprintf(stderr,"penalty score missing.\n");
                return 0 ;
            }
            config.penalty = atoi(argv[i]);
        } //end elseif
        else if(strcmp(argv[i], "-n") == 0 || strcmp(argv[i],
                "--num_pairs") == 0){
            i++;
            if (i==argc){
                fprintf(stderr,"sequence length missing.\n");
                return 0 ;
            }
            config.num_pairs = atoi(argv[i]);
            if ( config.num_pairs >
                        MAX_SEQ_NUM ) {
                fprintf(stderr, "The maximum sequence number");
                fprintf(stderr,"per stream is %d\n", MAX_SEQ_NUM);
                return 0;
            } // end if  
            config.num_streams++;
        } //end elseif
        else if(strcmp(argv[i], "-l") == 0 || strcmp(argv[i],
                 "--lengths") == 0){
            i++;
            if (i==argc){
                fprintf(stderr,"sequence length missing.\n");
                return 0 ;
            } // end if 
            config.length = atoi(argv[i]);
            if ( config.length > MAX_SEQ_LEN ) {
                fprintf(stderr,"The maximum seqence length is");
                                fprintf(stderr,"%d\n", MAX_SEQ_LEN);
                return 0;
            } // end if 
        }  // end elseif
        else if(strcmp(argv[i], "-h") == 0 || strcmp(argv[i],
                "--help") == 0) {
            usage(argc, argv);
            return 0;
        } //end else if 
        else if(strcmp(argv[i], "-s") == 0 || strcmp(argv[i],
                "--chare") == 0) {
            i++;
            config.chare_num = atoi(argv[i]);
       } //end else if 
         else if(strcmp(argv[i], "-f") == 0 || strcmp(argv[i],
                "--fac") == 0) {
            i++;
            config.div = atoi(argv[i]);
        } //end else if 

        else {
            fprintf(stderr,"Unrecognized option : %s\nTry --help\n", argv[i]);
            return 0;
        }
        i++;
    }  // end while
    return 1;
}  // end parse arguments

/**************** CUDA Func *******************/

extern void cudaMatMul(int num_threads, int num_blocks, int penalty, int pair_num, int offset, char * seq_data1, char * seq_data2, unsigned int * pos1, unsigned int * pos2, unsigned int * pos_matrix, int peid_, int pid);


/**************** Class Def *******************/

CProxy_Main mainProxy;

class Main : public CBase_Main {

public:

int penum, seq1_len, seq2_len, seq_len, iter, div_f;
double startTime;
CProxy_Data dnw;


    Main (CkArgMsg *m) {
        startTime = CkWallTimer();
ckout << "flag0!" << endl;
        init_conf();
ckout << "flag1!" << endl;
        iter = 0;
        while(!parse_arguments(m->argc, m->argv)) usage(m->argc, m->argv);

        print_config();

        mainProxy = thisProxy;

        penum = config.chare_num; 
        iter =0;
        div_f = config.div;
        if (config.num_pairs%penum != 0)
            CkAbort("Dimension is not dividable by PEs\n");

        dnw = CProxy_Data::ckNew(config, penum, penum);

        dnw.nwRun(CkCallback(CkReductionTarget(Main, done), thisProxy));
    }  
    
    void done(CkReductionMsg *msg) {
        CkPrintf("Resume iteration at step %d\n", iter);
        CkReduction::setElement *current = (CkReduction::setElement*) msg->getData();
        bool *result;
        if (current != NULL) {
            result = (bool *) &current->data;
        }
        iter++;
        if (iter < div_f) {
            //if (iter % 2)
                dnw.pauseForLB();
            //else
            //    dnw.nwRun(CkCallback(CkReductionTarget(Main, done), thisProxy));
        }
        else {
            double endTime = CkWallTimer();
            CkPrintf("Time: %f\n", endTime - startTime);
            CkExit();
        }
    }
    void resumeIter(CkReductionMsg *msg) {
        CkPrintf("Resume iteration at step %d\n", iter);
        CkReduction::setElement *current = (CkReduction::setElement*) msg->getData();
        bool *result;
        if (current != NULL) {
            result = (bool *) &current->data;
        }
        dnw.nwRun(CkCallback(CkReductionTarget(Main, done), thisProxy));
    }
};

class Data : public CBase_Data {

    _CONF_ conf;
    int seq_len, seq_size, pair_num, pos_size, pair_total;
    int penum, peid, pid, div_fac, iter;
//    int *score_matrix;
    char *sequence_set1, *sequence_set2;
    unsigned int *pos1, *pos2, *pos_matrix;
    int num_threads, num_blocks, penalty, dataset;
    bool ctype;
    Data_SDAG_CODE

public:
// Data Init
    Data (_CONF_ config, int penum_) {
    // load balance
        usesAtSync = true;
    // Init PE
        penum = penum_;
        peid  = thisIndex; 
        pid   = peid;
        iter  = 0;
    // Init arg
        conf.kernel      = config.kernel;
        conf.num_threads =  config.num_threads;
        conf.num_blocks  = config.num_blocks;
        conf.num_pairs   = config.num_pairs;
        conf.length      = config.length;
        conf.penalty     = config.penalty;
        conf.dataset     = config.dataset;
        conf.div         = config.div;
    // Init length
        div_fac     = conf.div;
        seq_len     = conf.length;
        pair_total  = conf.num_pairs / penum;
        pair_num    = pair_total / div_fac;
        seq_size    = seq_len * pair_num; 
        pos_size    = pair_num;
        num_threads = conf.num_threads;
        num_blocks  = conf.num_blocks;
        penalty     = conf.penalty;
    // alloc Data 
        sequence_set1 = new char[seq_len * pair_total];
        sequence_set2 = new char[seq_len * pair_total];
        pos1       = new unsigned int[pair_num+1];
        pos2       = new unsigned int[pair_num+1];
        pos_matrix = new unsigned int[pair_num+1];
    // Init sequence
        srand ( 7 );
        for (int i=0; i<seq_size; ++i) {
            sequence_set1[i]  = rand() % 20 + 1;
            sequence_set2[i]  = rand() % 20 + 1;
        }
    // Init Pos
        pos_matrix[0] = pos1[0] = pos2[0] = 0;
        for (int i=0; i<pair_num; ++i){
            pos1[i+1] = pos1[i] + seq_len;
            pos2[i+1] = pos2[i] + seq_len;
            pos_matrix[i+1] = pos_matrix[i] + (seq_len+1) * (seq_len+1);
        } 
    // Init Score
//        score_matrix = new int[pos_matrix[pair_num]*sizeof(int)];
            
    }

    void pup(PUP::er &p) {
        CBase_Data::pup(p);
        __sdag_pup(p);
        p|conf;
        p|seq_len; p|seq_size; p|pair_num; p|pos_size; p|pair_total;
        p|penum; p|peid; p|pid; p|div_fac; p|iter; p|ctype;
        p|num_threads; p|num_blocks; p|penalty; p|dataset;
        if (p.isUnpacking()) {
            sequence_set1 = new char[seq_len * pair_total];
            sequence_set2 = new char[seq_len * pair_total];
            pos1       = new unsigned int[pair_num+1];
            pos2       = new unsigned int[pair_num+1];
            pos_matrix = new unsigned int[pair_num+1];
        }
        p(sequence_set1, seq_len*pair_total);
        p(sequence_set2, seq_len*pair_total);
        p(pos1, pair_num+1);
        p(pos2, pair_num+1);
        p(pos_matrix, pair_num+1);

    }

    Data (CkMigrateMessage*) {}

    
    void doWork() {
        //if (iter < div_fac) {
        //    if (peid == 0)
        //        ckout << "\t\t******iteration " << iter << " ******" << endl;
//        int offset = pair_num * iter * seq_len;
        int offset = 0;

/*        ckout << "PE[" << pid << "/" << penum << "] " << "Pari_num: " << pair_num << " offset: " << offset << endl;
        ckout << "POS: " << endl;
        for (int i=0; i<(pair_num+1); i++) {
            ckout << pos1[i] << ":" << pos2[i] << " ";      
        }
        ckout << endl << "POS_MAT: " << endl;
        for (int i=0; i<(pair_num+1); i++) {
            ckout << pos_matrix[i] << " ";      
        }
        ckout << endl;*/
        cudaMatMul(num_threads, num_blocks, penalty, pair_num, offset, 
            sequence_set1, sequence_set2, pos1, pos2, pos_matrix, peid, /*pid*/CkMyPe());
       //     iter++;
        //    AtSync();
        //}
        //else {
        //    mainProxy.done();
        //}
    } 
    void pauseForLB() {
        CkPrintf("Element %d pause for LB on PE %d\n", thisIndex, CkMyPe());
        AtSync();
    }
    void ResumeFromSync() {
        CkCallback cbld(CkReductionTarget(Main, resumeIter), mainProxy);
        contribute(sizeof(bool), &ctype, CkReduction::set, cbld);
        //CkCallback cbld(CkReductionTarget(Main, done), mainProxy);
        //contribute(sizeof(bool), &ctype, CkReduction::set, cbld);
        //doWork();
    }
};



#include "nw.def.h"
