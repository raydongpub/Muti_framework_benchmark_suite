#include "dmm.h"
#include "cuda_runtime.h"
#include <sys/time.h>

using namespace Legion;

__global__ void ComputeMatrix_Kernel(int h, int subnum, 
                                     RegionAccessor<AccessorType::SOA<sizeof(float)>, float> d_acc_mata, 
                                     RegionAccessor<AccessorType::SOA<sizeof(float)>, float> d_acc_matb,
                                     RegionAccessor<AccessorType::SOA<sizeof(float)>, float> d_acc_matc) {

    int row, col, idx;
    int tid        = (blockIdx.x * blockDim.x) + threadIdx.x;
    int gridsize   = gridDim.x * blockDim.x;
    float sum  = 0.0;
#if 0
    int stride = (subnum / gridsize) + 1;

    for (int i = 0; i < stride; i++) {
        idx = (i * gridsize) + tid;
#else
    for (int i = 0; i < subnum; i += gridsize) {
        idx = i + tid;
#endif
        row = idx / h;
        col = idx % h;
        if (idx < subnum) {
            for (int j = 0; j < h; j++) {
                sum += d_acc_mata.read(ptr_t(row*h + j)) * d_acc_matb.read(ptr_t(h*col + j));
            }
            d_acc_matc.write(ptr_t(idx), sum);
            sum = 0.0;
        }
    }
}

__host__
void kernel_cuda_task(const Sequence &piece, const std::vector<PhysicalRegion> &regions, int pid)
{
  RegionAccessor<AccessorType::Generic, float> acc_mata = regions[0].get_field_accessor(MAT_A).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> acc_matb = regions[1].get_field_accessor(MAT_B).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> acc_matc = regions[2].get_field_accessor(MAT_C).typeify<float>();
  // get argument
  timeval time_begin, time_end;
  double  time_period;
  int task_id = pid;
  ptr_t first_mata  = piece.first_mata;
  ptr_t first_matb  = piece.first_matb;
  ptr_t first_matc  = piece.first_matc;
  int dim         = piece.dim;
  int dim_tile    = piece.dim_tile;
  int num_subreg  = piece.num_subrg;
  int num_threads = piece.num_threads;
  int num_blocks  = piece.num_blocks;

  printf("task: %d, dim:%d, dim_tile: %d,  num_subregions:%d\n", task_id, dim, dim_tile,  num_subreg);
  // convert to SOA
  RegionAccessor<AccessorType::SOA<sizeof(float)>, float> d_acc_mata = acc_mata.convert<AccessorType::SOA<sizeof(float)> >();
  RegionAccessor<AccessorType::SOA<sizeof(float)>, float> d_acc_matb = acc_matb.convert<AccessorType::SOA<sizeof(float)> >();
  RegionAccessor<AccessorType::SOA<sizeof(float)>, float> d_acc_matc = acc_matc.convert<AccessorType::SOA<sizeof(float)> >();
  gettimeofday(&time_begin, NULL);
  ComputeMatrix_Kernel<<<num_blocks, num_threads>>> (dim, dim_tile*dim_tile, d_acc_mata, d_acc_matb, d_acc_matc);
  cudaDeviceSynchronize();
  gettimeofday(&time_end, NULL);
  time_period = (time_end.tv_sec + time_end.tv_usec * 1e-6) - (time_begin.tv_sec + time_begin.tv_usec * 1e-6);
  printf("kernel time: %lf\n", time_period);
}

