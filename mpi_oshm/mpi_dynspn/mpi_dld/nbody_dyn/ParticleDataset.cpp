#include "ParticleDataset.h"
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <stdio.h>

#define REALLOC_CAPACITY 100

ParticleDataset::ParticleDataset() {
    mNumParticles    = 0;
    mCapacity        = 0;
    mReallocCapacity = REALLOC_CAPACITY;
    mpParticle       = NULL;
    mpFilename       = NULL;
}

ParticleDataset::ParticleDataset(const char * filename) {
    DatasetFileHeader header;

    mDatasetFile.open(filename, ios_base::in | ios_base::binary);
    if (!mDatasetFile.is_open())
        throw (int) EDS_FILE;

    mDatasetFile.seekg(0, ios_base::end);
    mDatasetFileSize = mDatasetFile.tellg();
    mDatasetFile.seekg(0, ios_base::beg);
    mReallocCapacity = REALLOC_CAPACITY;

    mDatasetFile.read((char *) &header, sizeof(DatasetFileHeader));
    mNumParticles = header.numParticles;
    mCapacity     = mNumParticles;

    if ((mDatasetFileSize - sizeof(DatasetFileHeader)) !=
        (mNumParticles * sizeof(Particle)))
        throw (int) EDS_INVFORMAT;
        mpParticle = (Particle *) malloc(mNumParticles * sizeof(Particle));
    if (mpParticle == NULL)
        throw (int) EDS_MEMALLOC;
    mDatasetFile.read((char *) mpParticle, mNumParticles * sizeof(Particle));
    mDatasetFile.close();
    mpFilename = const_cast<char *>(filename);
}

ParticleDataset::ParticleDataset(int numParticles) {
    mNumParticles = numParticles;
    mCapacity     = numParticles;
    int rc = CreateEmpty(numParticles);
    if (rc != EDS_SUCCESS)
        throw (int) rc;
}


ParticleDataset::~ParticleDataset() {
    if (mpParticle != NULL) {
        free(mpParticle);
    }
}

int ParticleDataset::CreateEmpty(int numParticles) {
    mpParticle = (Particle *) malloc(numParticles * sizeof(Particle));
    if (mpParticle == NULL)
        return EDS_MEMALLOC;

    memset(mpParticle, 0, numParticles * sizeof(Particle));
    mNumParticles = numParticles;
    mCapacity     = numParticles;

    return EDS_SUCCESS;
}

int ParticleDataset::AddParticle(Particle * particle) {
    Particle * tempParticle;

    if ((mNumParticles + 1) > mCapacity) {
        if (mpParticle == NULL)
            tempParticle = (Particle *) malloc(
                           (mCapacity + mReallocCapacity) * sizeof(Particle));
        else
            tempParticle = (Particle *) realloc(mpParticle,
                           (mCapacity + mReallocCapacity) * sizeof(Particle));
        mpParticle = tempParticle;
    }
    memcpy(&(mpParticle[mNumParticles]), particle, sizeof(Particle));
    mNumParticles++;

    return EDS_SUCCESS;
}

int ParticleDataset::AddnParticles(Particle * particles, int numParticles) {
    Particle * tempParticle;
    int        chunk;

    if ((mNumParticles + numParticles) > mCapacity) {
        chunk = (numParticles / mReallocCapacity) + 1;

        if (mpParticle == NULL)
            tempParticle = (Particle *) malloc(
                           (mCapacity + (chunk * mReallocCapacity)) * sizeof(Particle));
        else
            tempParticle = (Particle *) realloc(mpParticle,
                           (mCapacity + (chunk * mReallocCapacity)) * sizeof(Particle));
        mpParticle = tempParticle;
    }
    memcpy(&(mpParticle[mNumParticles]), particles, numParticles * sizeof(Particle)),
    mNumParticles += numParticles;

    return EDS_SUCCESS;
}

int ParticleDataset::SaveToFile(const char * filename) {
    DatasetFileHeader header;

    mDatasetFile.open(filename, ios_base::out | ios_base::binary);
    if (!mDatasetFile.is_open())
        return EDS_FILE_IO;

    header.magic        = DS_MAGIC;
    header.version      = DS_VERSION;
    header.numParticles = mNumParticles;
    mDatasetFile.write((char *) &header, sizeof(DatasetFileHeader));
    mDatasetFile.write((char *) mpParticle, mNumParticles * sizeof(Particle));
    mDatasetFile.close();
    mpFilename = const_cast<char *>(filename);

    return EDS_SUCCESS;
}

int ParticleDataset::SaveToFile() {
    DatasetFileHeader header;

    if (mpFilename == NULL)
        return EDS_FILE_IO;

    mDatasetFile.open(mpFilename, ios_base::out | ios_base::binary);
    if (!mDatasetFile.is_open())
        return EDS_FILE_IO;

    header.magic        = DS_MAGIC;
    header.version      = DS_VERSION;
    header.numParticles = mNumParticles;
    mDatasetFile.write((char *) &header, sizeof(DatasetFileHeader));
    mDatasetFile.write((char *) mpParticle, mNumParticles * sizeof(Particle));
    mDatasetFile.close();

    return EDS_SUCCESS;
}

const char * ParticleDataset::GetClassID(int errId) {
    if (errId > EDS_SUCCESS)
        return "ParticleDataset";
    return NULL;
}

const char * ParticleDataset::GetEMSG(int errId) {
    if (errId > EDS_SUCCESS)
        return ParticleDataset::mspEMSG[errId - EDS_SUCCESS];
    return ParticleDataset::mspEMSG[EDS_UNKNOWN - EDS_SUCCESS];
}

char ParticleDataset::mspEMSG[][ECFG_MSG_LEN] = {
    "Success",
    "Unknown error",
    "Dataset not found",
    "File I/O fails",
    "Cannot allocate memory",
    "Invalid dataset file"
};

