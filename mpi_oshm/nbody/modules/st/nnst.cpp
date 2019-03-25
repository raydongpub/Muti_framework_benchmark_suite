#include "nbody_main.h"
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <termios.h>
#include <time.h>
#include <iostream>

using namespace std;

typedef struct {
    bool inProgress;
} ThreadSync;

NbodyConfig     * plConfig; 
ParticleDataset * plDataset;
ThreadSync        threadSync;

PRECISION     sec         = 0.0;

void * SingleThreadedNbody(void * arg);
void * ListeningThread(void * arg);

int LibSetup(NbodyConfig     * config,
             ParticleDataset * dataset) {

    plConfig  = config;
    plDataset = dataset;

    return 0;
}

void LibEntry(int argc, char **argv) {

    pthread_t      nbodyThread;
    pthread_t      listeningThread;
    pthread_attr_t threadAttr;

    threadSync.inProgress = true;

    pthread_attr_init(&threadAttr);
    pthread_attr_setdetachstate(&threadAttr, PTHREAD_CREATE_JOINABLE);
    pthread_create(&nbodyThread, &threadAttr,SingleThreadedNbody,
        (void *) &threadSync);
    pthread_create(&listeningThread, &threadAttr, ListeningThread,
        (void *) &threadSync);

    pthread_join(nbodyThread, NULL);
    pthread_join(listeningThread, NULL);

    pthread_attr_destroy(&threadAttr);
}

int LibCleanUp(void) {
    plDataset->SaveToFile("st.bin");
    cout << "Cleanup --> yes" << endl;

    return 0;
}

void * SingleThreadedNbody(void * arg) {
#define SET_PARTICLE(str1, str2)                    \
    x##str1##Pos = plDataset->mpParticle[str2].xPos; \
    y##str1##Pos = plDataset->mpParticle[str2].yPos; \
    z##str1##Pos = plDataset->mpParticle[str2].zPos; \
    x##str1##Vel = plDataset->mpParticle[str2].xVel; \
    y##str1##Vel = plDataset->mpParticle[str2].yVel; \
    z##str1##Vel = plDataset->mpParticle[str2].zVel; \
    x##str1##Acc = plDataset->mpParticle[str2].xAcc; \
    y##str1##Acc = plDataset->mpParticle[str2].yAcc; \
    z##str1##Acc = plDataset->mpParticle[str2].zAcc; \
    mass##str1 = plDataset->mpParticle[str2].mass

#define SET_IPARTICLE(str1, str2)                   \
    x##str1##Pos = plDataset->mpParticle[str2].xPos; \
    y##str1##Pos = plDataset->mpParticle[str2].yPos; \
    z##str1##Pos = plDataset->mpParticle[str2].zPos; \
    mass##str1 = plDataset->mpParticle[str2].mass

    PRECISION     duration     = plConfig->mParams.duration;
    PRECISION     step         = plConfig->mParams.timeRes;
    PRECISION     grav         = plConfig->mParams.gravConstant;
    int           idx, jIdx;
    int          numParticles  = plDataset->mNumParticles;
    ThreadSync * ts            = (ThreadSync *) arg;

    cout << endl;
    for (sec=0.0;sec<duration;sec+=step) {

        PRECISION x1Pos, y1Pos, z1Pos;
        PRECISION x1Vel, y1Vel, z1Vel;
        PRECISION x1Acc, y1Acc, z1Acc;
        PRECISION mass1;
        PRECISION x2Pos, y2Pos, z2Pos;
        PRECISION mass2;
        PRECISION force, force_x, force_y, force_z;
        PRECISION radius, radius_s;

        for (idx=0;idx<numParticles;idx++) {
            SET_PARTICLE(1, idx);

            force_x = 0; force_y = 0; force_z = 0;
            for (jIdx=0;jIdx<numParticles;jIdx++) {
                if (jIdx != idx) {
                    SET_IPARTICLE(2, jIdx);

                    radius_s  = ((x2Pos - x1Pos) * (x2Pos - x1Pos)) +
                                ((y2Pos - y1Pos) * (y2Pos - y1Pos)) +
                                ((z2Pos - z1Pos) * (z2Pos - z1Pos));
                    radius    = sqrt(radius_s);
                    force     = (grav * mass1 * mass2) / radius_s;
                    force_x  += force * ((x2Pos - x1Pos) / radius);
                    force_y  += force * ((y2Pos - y1Pos) / radius);
                    force_z  += force * ((z2Pos - z1Pos) / radius);

                }
            }

            x1Acc = force_x / mass1;
            y1Acc = force_y / mass1;
            z1Acc = force_z / mass1;
            x1Vel += x1Acc * step;
            y1Vel += y1Acc * step;
            z1Vel += z1Acc * step;
            x1Pos += x1Vel * step;
            y1Pos += y1Vel * step;
            z1Pos += z1Vel * step;

            plDataset->mpParticle[idx].xPos = x1Pos;
            plDataset->mpParticle[idx].yPos = y1Pos;
            plDataset->mpParticle[idx].zPos = z1Pos;
            plDataset->mpParticle[idx].xVel = x1Vel;
            plDataset->mpParticle[idx].yVel = y1Vel;
            plDataset->mpParticle[idx].zVel = z1Vel;
            plDataset->mpParticle[idx].xAcc = x1Acc;
            plDataset->mpParticle[idx].yAcc = y1Acc;
            plDataset->mpParticle[idx].zAcc = z1Acc;
        }
    }

    ts->inProgress = false;
    pthread_exit(NULL);
}

void * ListeningThread(void * arg) {

    ThreadSync * ts = (ThreadSync *) arg;
    int  cnt = 0;
    char c;

    struct termios prevTerm, newTerm;
    int    flag, n;

    tcgetattr(STDIN_FILENO, &prevTerm);
    newTerm = prevTerm;
    newTerm.c_lflag &= (~ICANON & ~ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newTerm);

    flag = fcntl(0, F_GETFL, 0);
    fcntl(0, F_SETFL, (flag | O_NDELAY));

    while (ts->inProgress) {
        n = read(0, &c, 1);
        if (n > 0)
            cout << endl << "YAY!" << endl;
        cout << "Time: " << sec << "\xd";
        cnt++;
        usleep(1000);
    }

    tcsetattr(STDIN_FILENO, TCSANOW, &prevTerm);
    fcntl(0, F_SETFL, flag);

    cout << endl << "Num: " << sec << endl;
    pthread_exit(NULL);
}

