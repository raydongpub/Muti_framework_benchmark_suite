#ifndef _PARTICLEDATASET_H_
#define _PARTICLEDATASET_H_
#include "Config.h"
#include <fstream>
//#include "nbody.h"
#include "pup.h"

#define EDS_SUCCESS   0x0200
#define EDS_UNKNOWN   0x0201
#define EDS_FILE      0x0202
#define EDS_FILE_IO   0x0203
#define EDS_MEMALLOC  0x0204
#define EDS_INVFORMAT 0x0205

#define DS_MAGIC      0x1e4560f2
#define DS_VERSION    0x0d7f5092


using namespace std;

class ParticleDataset {
public:
    typedef struct {
        PRECISION xPos;
        PRECISION yPos;
        PRECISION zPos;
        PRECISION xVel;
        PRECISION yVel;
        PRECISION zVel;
        PRECISION xAcc;
        PRECISION yAcc;
        PRECISION zAcc;
        PRECISION mass;
        int       pid;
        int       localCnt;
        int       localDisp;
        void pup(PUP::er &p) {
            p|xPos;p|yPos;p|zPos;
            p|xVel;p|yVel;p|zVel;
            p|xAcc;p|yAcc;p|zAcc;
            p|mass;p|pid;p|localCnt;
            p|localDisp;

        } 
    } Particle;

    typedef struct {
        int magic;
        int version;
        int numParticles;
    } DatasetFileHeader;

    ParticleDataset();
    ParticleDataset(const char * filename);
    ParticleDataset(int numParticles);
    ~ParticleDataset();
    int                 CreateEmpty(int numParticles);
    int                 AddParticle(Particle * particle);
    int                 AddnParticles(Particle * particles, int numParticles);
    int                 SaveToFile(const char * filename);
    int                 SaveToFile();
    static const char * GetClassID(int errId);
    static const char * GetEMSG(int errId);

    int                 mNumParticles;
    Particle          * mpParticle;

/*    void pup(PUP::er &pp) {
        pp|mNumParticles;
        if (pp.isUnpacking()) {
            mpParticle = new Particle[mNumParticles];
        }
        PUParray(pp, mpParticle, mNumParticles);
    }*/
private:

    fstream       mDatasetFile;
    char        * mpFilename;
    size_t        mDatasetFileSize;
    int           mCapacity;
    int           mReallocCapacity;
    static char   mspEMSG[][ECFG_MSG_LEN];
};

#endif /* _PARTICLEDATASET_H_ */
