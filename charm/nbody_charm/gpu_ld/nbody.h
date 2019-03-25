#ifndef _NBODY_H_
#define _NBODY_H_

#include "nbodycuda.cuh"
#include "NbodyConfig.h"
#define E_SUCCESS     0xFF00
#define E_UNKNOWN_ARG 0XFF01
#define E_FILE_ARG    0xFF02
#define E_LIB_ARG     0xFF03


//For building the version that uses various static libraries.
#ifdef _USE_STATIC_LIB
#ifdef _STATIC_MOD_ST
#elif  _STATIC_MOD_MPI
#elif  _STATIC_MOD_MPICUDA
#endif //_STATIC_MOD_XXXXX
#endif //_USE_STATIC_LIB

#endif
