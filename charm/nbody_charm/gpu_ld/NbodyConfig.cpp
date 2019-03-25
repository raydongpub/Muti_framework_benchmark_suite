#include "NbodyConfig.h"
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <iostream>

using namespace std;

NbodyConfig::~NbodyConfig() {
//    delete [] mParams.nodeSettings;
}

void NbodyConfig::RetrieveConfigurations() {
    int eid;

    eid = RetrieveConfigurationsBody();
    if (eid != ENBC_SUCCESS)
        throw eid;
}

int NbodyConfig::RetrieveConfigurationsBody() {

    char * cVal;
//    char   tempKey[MAX_TOK_LEN];
//    int    idx;

    cVal = const_cast<char *>(GetValue(mspNbodyKey[_CLUSTER_]));
    if (cVal == NULL)
        return ENBC_CLUSTER_KEY;

    if (!strcmp(cVal, NBC_TRUE)) {
        mParams.cluster = true;
/*
        cVal = const_cast<char *>(GetValue(mspNbodyKey[_NODE_COUNT_]));
        if (cVal == NULL)
            return ENBC_NODECNT_KEY;

        //Determine nodes count
        mParams.nodeCnt = atoi(cVal);
        if (mParams.nodeCnt <= 1)
            return ENBC_NODECNT_KEY;
        mParams.nodeSettings = new NodeSettings[mParams.nodeCnt];


        //Retrieve configurations for all the node specified
        for (idx=0;idx<mParams.nodeCnt;idx++) {

            //Determine the hostname for the node
            sprintf(tempKey, mspNbodyKey[_NODE_NAME_], idx);
            cVal = const_cast<char *>(GetValue(tempKey));
            if (cVal == NULL)
                return ENBC_NODENAME_KEY;
            mParams.nodeSettings[idx].nodeName = cVal;

            //Determine whether the node uses GPU
            sprintf(tempKey, mspNbodyKey[_NODE_USEGPU_], idx);
            cVal = const_cast<char *>(GetValue(tempKey));
            if (cVal == NULL)
                mParams.nodeSettings[idx].useGPU = false;
            else {
                if (!strcmp(cVal, NBC_TRUE)) 
                    mParams.nodeSettings[idx].useGPU = true;
                else if (!strcmp(cVal, NBC_FALSE))
                    mParams.nodeSettings[idx].useGPU = false;
                else
                    return ENBC_NODEUSEGPU_KEY;
            }

            //Determine GPUs count of the node
            sprintf(tempKey, mspNbodyKey[_NODE_GPUCNT_], idx);
            cVal = const_cast<char *>(GetValue(tempKey));
            if (cVal == NULL)
                return ENBC_NODEGPUCNT_KEY;
            mParams.nodeSettings[idx].gpuCnt = atoi(cVal);
            if (mParams.nodeSettings[idx].gpuCnt <= 0)
                return ENBC_NODEGPUCNT_KEY;
        }
*/
    }
    else if (!strcmp(cVal, NBC_FALSE)) 
        mParams.cluster = false;
    else 
        return ENBC_CLUSTER_KEY;

    //Retrieve gravitational constant
    cVal = const_cast<char *>(GetValue(mspNbodyKey[_GRAVCONST_]));
    if (cVal == NULL)
        return ENBC_GRAVITATIONAL_KEY;
    mParams.gravConstant = atof(cVal);
    gravCon = atof(cVal);

    //Retrieve time resolution
    cVal = const_cast<char *>(GetValue(mspNbodyKey[_TIMERES_]));
    if (cVal == NULL)
        return ENBC_TIMERESOLUTION_KEY;
    mParams.timeRes = GetTimeResolution(cVal);
    timeR = mParams.timeRes;    

    //Retrieve simulation duration
    cVal = const_cast<char *>(GetValue(mspNbodyKey[_DURATION_]));
    if (cVal == NULL)
        return ENBC_DURATION_KEY;
    mParams.duration = GetTimeResolution(cVal);
    duration = mParams.duration;
    //Retrieve initial dataset
    cVal = const_cast<char *>(GetValue(mspNbodyKey[_INITDATASET_]));
    if (cVal == NULL)
        return ENBC_INITDATASET_KEY;
    mParams.initialDatasetFile = cVal;

    //Retrieve library name
    cVal = const_cast<char *>(GetValue(mspNbodyKey[_LIBRARY_]));
    if (cVal == NULL)
        return ENBC_LIBRARY_KEY;
    mParams.library = cVal;

    return ENBC_SUCCESS;
}

const char * NbodyConfig::GetValue(const char * key) {
    return Config::GetValue(key);
}

const char * NbodyConfig::GetValue(char * key) {
    return Config::GetValue(key);
}

const char * NbodyConfig::GetValue(string key) {
    return Config::GetValue(key);
}

const char * NbodyConfig::GetKeyString(int keyId) {
    return NbodyConfig::mspNbodyKey[keyId];
}

#define TOK_SPC ' '
#define TOK_TAB '\t'
PRECISION NbodyConfig::GetTimeResolution(const char * timeRes) {
    char      digits[32];
    char      unit[32];
    PRECISION f_digits;
    PRECISION f_unit;
    int       digitsIdx, unitIdx;
    int       idx, len;

    char c;
    bool digit_s;

    len       = strlen(timeRes);
    digitsIdx = 0;
    unitIdx   = 0;
    memset(digits, 0, 32);
    memset(unit, 0, 32);

    digit_s = true;
    for (idx=0;idx<len;idx++) {
        c = timeRes[idx];

        switch (digit_s) {
        case true:
            if (((c >= '0') && (c <= '9')) || (c == '.'))
                digits[digitsIdx++] = c;
            else {
                digit_s         = false;
                unit[unitIdx++] = c;  
            }
            break;
        case false:
            if ((c != TOK_SPC) && (c != TOK_TAB)) 
                unit[unitIdx++] = c;
            break;
        }
    }

    f_digits = atof(digits);

    if (!strcmp(unit, "ms"))
        f_unit = 1e-3;
    else if (!strcmp(unit, "us"))
        f_unit = 1e-6;
    else if (!strcmp(unit, "us"))
        f_unit = 1e-6;
    else if (!strcmp(unit, "ns"))
        f_unit = 1e-9;
    else if (!strcmp(unit, "ps"))
        f_unit = 1e-12;
    else if (!strcmp(unit, "fs"))
        f_unit = 1e-15;
    else if (!strcmp(unit, "as"))
        f_unit = 1e-18;
    else if (!strcmp(unit, "zs"))
        f_unit = 1e-21;
    else if (!strcmp(unit, "ys"))
        f_unit = 1e-24;
    else
        f_unit = 1e+0;
    return (PRECISION) f_digits * f_unit;
}

const char * NbodyConfig::GetEMSG(int errId) {
    if (errId < ENBC_SUCCESS)
        return Config::GetEMSG(errId);
    return mspNbodyEMSG[errId - ENBC_SUCCESS];
}

const char * NbodyConfig::GetClassID(int errId) {
    if (errId < ENBC_SUCCESS)
        return Config::GetClassID();
    return "NbodyConfig";
}

char NbodyConfig::mspNbodyKey[][MAX_TOK_LEN] = {
    "Cluster",
    "ComputeNodesCount",
    "Node_%d_Name",
    "Node_%d_UseGPU",
    "Node_%d_GPUCount",
    "GravConstant",
    "TimeResolution",
    "Duration",
    "InitialDataset",
    "Library",
    "Oshmem",
    "IVM"
};

char NbodyConfig::mspNbodyEMSG[][ECFG_MSG_LEN] = {
    "Success",
    "Invalid cluster configuration",
    "Invalid node count",
    "Invalid node name",
    "Invalid node/gpu configuration",
    "Invalid gpu count",
    "Invalid gravitational constant",
    "Time resolution not specified",
    "Simulation duration not specified", 
    "Initial dataset not specified",
    "Library not specified"
};

