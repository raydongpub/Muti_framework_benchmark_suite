#ifndef __DMM_H__
#define __DMM_H__

#include <cstdio>
#include "legion.h"
#include <cmath>
using namespace Legion;
using namespace LegionRuntime::Accessor;


enum TaskIDs{
  TOP_LEVEL_TASK_ID,
  KERNEL_TASK_ID,
  CHECK_TASK_ID,
};

enum FieldIDs {
  MAT_A,
  MAT_B,
  MAT_C,
};

struct Sequence {
  //LogicalRegion matas_lr;
  //LogicalRegion matbs_lr;
  //LogicalRegion matcs_lr;
  ptr_t first_mata;
  ptr_t first_matb;
  ptr_t first_matc;
  int task_id;
  int dim;
  int dim_tile;
  int num_elem;
  int num_subrg;
  int num_threads;
  int num_blocks;
};
extern void kernel_cuda_task(const Sequence &piece, const std::vector<PhysicalRegion> &regions, int pid);
#endif
