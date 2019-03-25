#include "ParticleDataset.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>

using namespace std;

ParticleDataset * pDataset;
/*
inline int SetPreference(int argc, char ** argv) {
#define IS_OPTION(str) (!strcmp(argv[idx], str))

    int idx = 1;

    while (idx < argc) {
        if IS_OPTION("-c") {
            if ((idx + 1) >= argc)
                return E_FILE_ARG;
            pConfigFile = argv[++idx];
        }
        else {
            return E_UNKNOWN_ARG;
        }

        idx++;
    }

    return E_SUCCESS;

#undef IS_OPTION
}
*/

int main(int argc, char ** argv) {

    ParticleDataset::Particle * particle;
    int                         idx, numParticles;

//    SetPreference(argc, argv);

    srand(time(NULL));

    numParticles = 1200000;
    pDataset     = new ParticleDataset();
    particle     = new ParticleDataset::Particle[numParticles];

    for (idx=0;idx<numParticles;idx++) {
        particle[idx].xPos = ((PRECISION) (rand()%900)) + 100.0;
        particle[idx].yPos = ((PRECISION) (rand()%900)) + 100.0;
        particle[idx].zPos = ((PRECISION) (rand()%900)) + 100.0;

        particle[idx].xVel = 0.0;
        particle[idx].yVel = 0.0;
        particle[idx].zVel = 0.0;

        particle[idx].xAcc = 0.0;
        particle[idx].yAcc = 0.0;
        particle[idx].zAcc = 0.0;

        particle[idx].mass = 500;
    }

    pDataset->AddnParticles(particle, numParticles);
    pDataset->SaveToFile("dset1.nn");

    delete    pDataset;
    delete [] particle;
}

