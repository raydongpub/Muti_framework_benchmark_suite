#ifndef _NBODYCONFIG_H_
#define _NBODYCONFIG_H_

#include "Config.h"

#define ENBC_SUCCESS            0x0100
#define ENBC_CLUSTER_KEY        0x0101
#define ENBC_NODECNT_KEY        0x0102
#define ENBC_NODENAME_KEY       0x0103
#define ENBC_NODEUSEGPU_KEY     0x0104
#define ENBC_NODEGPUCNT_KEY     0x0105
#define ENBC_GRAVITATIONAL_KEY  0x0106
#define ENBC_TIMERESOLUTION_KEY 0x0107
#define ENBC_DURATION_KEY       0x0108
#define ENBC_INITDATASET_KEY    0x0109
#define ENBC_LIBRARY_KEY        0x010a

#define _CLUSTER_               0x0000
#define _NODE_COUNT_            0x0001
#define _NODE_NAME_             0x0002
#define _NODE_USEGPU_           0x0003
#define _NODE_GPUCNT_           0x0004
#define _GRAVCONST_             0x0005
#define _TIMERES_               0x0006
#define _DURATION_              0x0007
#define _INITDATASET_           0x0008
#define _LIBRARY_               0x0009
#define _OSHMEM_                0x000A
#define _IVM_                   0x000B

#define NBC_TRUE                "true"
#define NBC_FALSE               "false"

class NbodyConfig : public Config {

private:
    void RetrieveConfigurations();
    int  RetrieveConfigurationsBody();

public:
    NbodyConfig(const char * filename) : Config(filename) { RetrieveConfigurations(); };
    ~NbodyConfig();
    const char        * GetValue(const char * key);
    const char        * GetValue(char * key);
    const char        * GetValue(string key);
    const char        * GetKeyString(int keyId);
    static PRECISION    GetTimeResolution(const char * timeRes);
    static const char * GetEMSG(int errId);
    static const char * GetClassID(int errId);

    typedef struct {
        char * nodeName;
        bool   useGPU;
        int    gpuCnt;
    } NodeSettings;

    typedef struct {
        bool           cluster;
        int            rank;
        int            commSize;
//        int            nodeCnt;
//        NodeSettings * nodeSettings;
        PRECISION      gravConstant;
        PRECISION      timeRes;
        PRECISION      duration;
        char         * initialDatasetFile;
        char         * library;
    } Params;

    Params      mParams;

private:

protected:

    static char mspNbodyKey[][MAX_TOK_LEN];
    static char mspNbodyEMSG[][ECFG_MSG_LEN];
};

#endif /* _NBODYCONFIG_H_ */

