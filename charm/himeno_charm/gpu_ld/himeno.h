#ifndef _HIMENO_H
#define _HIMENO_H

#include "commonBMT.h"
#include "himenocuda.cuh"

#define MODE_NORM 0x0000
#define MODE_HELP 0x0001

#define DSET_XS   0x0000
#define DSET_S    0x0001
#define DSET_M    0x0002
#define DSET_L    0x0003
#define DSET_XL   0x0004

#define E_SUCCESS 0x0000
#define E_NO_ARG  0x0001
#define E_UNKNOWN 0x0002
#define E_INV_PE  0x0003
#define E_INV_PEV 0x0004
#define E_INV_DS  0x0005
#define E_INV_DSV 0x0006

const char * e_msg[] = {
"No error",
"No arguments specified.",
"Unrecognized arguments presented.",
"Requires three PE numbers along the dimensions.",
"Invalid PE numbers specified.",
"Requires the size of dataset (xs, s, m, l, xl).",
"Unrecognized dataset size"
};

const char h_msg[] = {"\
Usage: %s [OPTIONS] [...]                                                   \n\
Options are available in both short and long format:                      \n\n\
\t-pe [pe_x pe_y pe_z]     Specify numbers of PEs along dimensions          \n\
\t-h, --help               Show this help message.                          \n\
"};

#define LEFT                    1
#define RIGHT                   2
#define TOP                     3
#define BOTTOM                  4
#define FRONT                   5
#define BACK                    6


#endif
