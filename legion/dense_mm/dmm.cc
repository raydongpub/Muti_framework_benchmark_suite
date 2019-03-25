#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include "legion.h"
#include "dmm.h"
#include "dmm_mapper.h"

void top_level_task(const Task *task, 
	      const std::vector<PhysicalRegion> &regions,
	      Context ctx, Runtime *runtime)
{



  int dim = 100;
//  int num_elements = dim * dim;
  int num_subregions = 4; // = number of tiles ( divide the output in tiles)
  int num_threads = 256;
  int num_blocks  = 32;
  {
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++) {
      if (!strcmp(command_args.argv[i],"-d"))
        dim = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-s"))
        num_subregions = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-b"))
        num_blocks = atoi(command_args.argv[++i]);
      if (!strcmp(command_args.argv[i],"-t"))
        num_threads = atoi(command_args.argv[++i]);
    }
  }
  int dim_tile =(int) dim/num_subregions; // dimension of the tile
  int num_elem_mat_a = dim * dim;
  int num_elem_mat_b = dim * dim;
  int num_elem_mat_c = dim_tile * dim;
  // make index space
  IndexSpace is_input_a = runtime->create_index_space(ctx, num_elem_mat_a);
  IndexSpace is_input_b = runtime->create_index_space(ctx, num_elem_mat_b);
  IndexSpace is_output  = runtime->create_index_space(ctx, num_elem_mat_c);
  // make field space ((PRECISION) (rand()%9)) + 1.0
  FieldSpace fs_input_a = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs_input_a);
    allocator.allocate_field(sizeof(float), MAT_A);
  }
  FieldSpace fs_input_b = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs_input_b);
    allocator.allocate_field(sizeof(float), MAT_B);
  }
  FieldSpace fs_output = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs_output);
    allocator.allocate_field(sizeof(float), MAT_C);
  }
  // create logic region
  LogicalRegion lr_input_a = runtime->create_logical_region(ctx, is_input_a, fs_input_a);
  LogicalRegion lr_input_b = runtime->create_logical_region(ctx, is_input_b, fs_input_b);
  LogicalRegion lr_output = runtime->create_logical_region(ctx, is_output, fs_output);
  // physical region acess to initialize
  RegionRequirement input_a_req(lr_input_a, READ_WRITE, EXCLUSIVE, lr_input_a);
  input_a_req.add_field(MAT_A);
  RegionRequirement input_b_req(lr_input_b, READ_WRITE, EXCLUSIVE, lr_input_b);
  input_b_req.add_field(MAT_B);
  RegionRequirement output_req(lr_output, READ_WRITE, EXCLUSIVE, lr_output);
  output_req.add_field(MAT_C);
  //InlineLauncher launcher(req);
  PhysicalRegion input_a_pr = runtime->map_region(ctx, input_a_req);
  PhysicalRegion input_b_pr = runtime->map_region(ctx, input_b_req);
  PhysicalRegion output_pr = runtime->map_region(ctx, output_req);
  /* partition work */
  Coloring coloring_input_a, coloring_input_b, coloring_output;
    int work_input_a = num_elem_mat_a / num_subregions;
#ifdef DEBUG_FLAG
  printf("flag1\n");
#endif
    /* Access region and partition for input_auence */
    input_a_pr.wait_until_valid();
    RegionAccessor<AccessorType::Generic, float> input_a_ra = input_a_pr.get_field_accessor(MAT_A).typeify<float>();
    {
      IndexAllocator input_a_allocator = runtime->create_index_allocator(ctx, lr_input_a.get_index_space());
      input_a_allocator.alloc(num_elem_mat_a);
    }
#ifdef DEBUG_FLAG
  printf("flag2\n");
#endif
    ptr_t *first_input_a = new ptr_t[num_subregions];
    {
      IndexIterator itr(runtime, ctx, lr_input_a.get_index_space());
      int count = 0;
      for (int i=0; i<num_subregions; i++) {
        for (int j=0; j<work_input_a; j++) {
          assert(itr.has_next());
          ptr_t ptr = itr.next();
          if (j==0)
            first_input_a[i] = ptr;
          input_a_ra.write(ptr, (((float) (rand()%9)) + 1.0));
          coloring_input_a[i].points.insert(ptr);
          count++;
        }
      }
    }
#ifdef DEBUG_FLAG
  printf("flag3\n");
#endif
    int work_input_b = num_elem_mat_b / num_subregions;
    input_b_pr.wait_until_valid();
    RegionAccessor<AccessorType::Generic, float> input_b_ra = input_b_pr.get_field_accessor(MAT_B).typeify<float>();
    {
      IndexAllocator input_b_allocator = runtime->create_index_allocator(ctx, lr_input_b.get_index_space());
      input_b_allocator.alloc(num_elem_mat_b);
    }
    ptr_t *first_input_b = new ptr_t[num_subregions];
    {
      IndexIterator itr(runtime, ctx, lr_input_b.get_index_space());
      int count_s = 0;
      for (int i=0; i<num_subregions; i++) {
        for (int j=0; j<work_input_b; j++) {
          assert(itr.has_next());
          ptr_t input_b_ptr = itr.next();
          if (j==0)
            first_input_b[i] = input_b_ptr;
          input_b_ra.write(input_b_ptr, (((float) (rand()%9)) + 1.0));
          coloring_input_b[i].points.insert(input_b_ptr);
          count_s++;
        }
      }
    }
    int work_output = num_elem_mat_c / num_subregions;
    output_pr.wait_until_valid();
    RegionAccessor<AccessorType::Generic, float> output_ra = output_pr.get_field_accessor(MAT_C).typeify<float>();
    {
      IndexAllocator output_allocator = runtime->create_index_allocator(ctx, lr_output.get_index_space());
      output_allocator.alloc(num_elem_mat_c);
    }
    ptr_t first_output;
    {
      IndexIterator itr(runtime, ctx, lr_output.get_index_space());
      for (int i=0; i<num_subregions; i++) {
          for (int j=0; j<work_output; j++) {
            assert(itr.has_next());
            ptr_t output_ptr = itr.next();
            if (j==0)
              first_output = output_ptr;
            output_ra.write(output_ptr, 0);
            coloring_output[i].points.insert(output_ptr);
          }
       }
    }
#ifdef DEBUG_FLAG
  printf("flag4\n");
#endif
#ifdef DEBUG_FLAG
  printf("flag4\n");
#endif
    //runtime->unmap_region(ctx, seqpr);
    //runtime->unmap_region(ctx, pospr);
    /* Index partion*/
    timeval time_begin, time_end;
    double  time_period;
    gettimeofday(&time_begin, NULL);
    /* create index partition */
    IndexPartition ip_input_a, ip_input_b, ip_output;
    ip_input_a = runtime->create_index_partition(ctx, is_input_a, coloring_input_a, true);
    ip_input_b = runtime->create_index_partition(ctx, is_input_b, coloring_input_b, true);
    ip_output = runtime->create_index_partition(ctx, is_output, coloring_output, true);
#ifdef DEBUG_FLAG
  printf("flag5\n");
#endif
    LogicalPartition lp_input_a = runtime->get_logical_partition(ctx, lr_input_a, ip_input_a);
    LogicalPartition lp_input_b = runtime->get_logical_partition(ctx, lr_input_b, ip_input_b);
    LogicalPartition lp_output = runtime->get_logical_partition(ctx, lr_output, ip_output);
#ifdef DEBUG_FLAG
  printf("flag6\n");
#endif
    Sequence sequence;
    sequence.task_id     = 0;
    sequence.first_mata  = first_input_a[0]; 
    sequence.first_matb  = first_input_b[0]; 
    sequence.first_matc  = first_output; 
    sequence.dim         = dim;
    sequence.dim_tile    = dim_tile;
    sequence.num_subrg   = num_subregions;
    sequence.num_threads = num_threads;
    sequence.num_blocks = num_blocks;
    // build argument map
#ifdef DEBUG_FLAG
  printf("flag7\n");
#endif
    ArgumentMap local_args;
    for (int idx = 0; idx < num_subregions;idx++)
    {
      DomainPoint point = DomainPoint::from_point<1>(Point<1>(idx));
      local_args.set_point(point, TaskArgument(&sequence, sizeof(Sequence)));
    }
    /* make partition */
    Rect<1> bounds(Point<1>(0), Point<1>(num_subregions-1));
    Domain launch_domain = Domain::from_rect<1>(bounds);
    // launch check_task
    IndexLauncher check_launcher(CHECK_TASK_ID, launch_domain, TaskArgument(NULL, 0), local_args);
    check_launcher.add_region_requirement(RegionRequirement(lp_input_a, 0, READ_ONLY, EXCLUSIVE, lr_input_a));
    check_launcher.region_requirements[0].add_field(MAT_A);
    check_launcher.add_region_requirement(RegionRequirement(lp_input_b, 0, READ_ONLY, EXCLUSIVE, lr_input_b));
    check_launcher.region_requirements[1].add_field(MAT_B);
    check_launcher.add_region_requirement(RegionRequirement(lp_output, 0, READ_WRITE, EXCLUSIVE, lr_output));
    check_launcher.region_requirements[2].add_field(MAT_C);
    FutureMap fm_check = runtime->execute_index_space(ctx, check_launcher);
    fm_check.wait_all_results();
#ifdef DEBUG_FLAG
  printf("flag8\n");
#endif
    IndexLauncher kernel_launcher(KERNEL_TASK_ID, launch_domain, TaskArgument(NULL, 0), local_args);
    kernel_launcher.add_region_requirement(RegionRequirement(lp_input_a, 0, READ_ONLY, EXCLUSIVE, lr_input_a));
    kernel_launcher.region_requirements[0].add_field(MAT_A);
    kernel_launcher.add_region_requirement(RegionRequirement(lp_input_b, 0, READ_ONLY, EXCLUSIVE, lr_input_b));
    kernel_launcher.region_requirements[1].add_field(MAT_B);
    kernel_launcher.add_region_requirement(RegionRequirement(lp_output, 0, READ_WRITE, EXCLUSIVE, lr_output));
    kernel_launcher.region_requirements[2].add_field(MAT_C);
#ifdef DEBUG_FLAG
  printf("flag9\n");
#endif
    for (int i=0; i<num_subregions; i++) {
        FutureMap fm_kernel = runtime->execute_index_space(ctx, kernel_launcher);
        fm_kernel.wait_all_results();
    }
    gettimeofday(&time_end, NULL);
    time_period = (time_end.tv_sec + time_end.tv_usec * 1e-6) - (time_begin.tv_sec + time_begin.tv_usec * 1e-6);
    printf("Time: %lf\n", time_period);
    runtime->destroy_logical_region(ctx, lr_input_a);
    runtime->destroy_logical_region(ctx, lr_input_b);
    runtime->destroy_logical_region(ctx, lr_output);
}

void check_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime) {

  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  assert(task->local_arglen == sizeof(Sequence));
  // get argument
  int task_id = task->index_point.point_data[0];
  printf("check Task ... taks: %d\n", task_id);
}

void kernel_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  // get argument
  //int task_id = task->index_point.point_data[0];
  const Sequence *piece = (Sequence*)task->local_args;
  int task_id = task->index_point.point_data[0];
  ptr_t first_mata  = piece->first_mata;
  ptr_t first_matb  = piece->first_matb;
  ptr_t first_matc  = piece->first_matc;
  int dim         = piece->dim;        
  int dim_tile    = piece->dim_tile;   
  int num_subreg  = piece->num_subrg;
  int num_threads = piece->num_threads;
  int num_blocks  = piece->num_blocks; 
  char hostname[256];
  gethostname(hostname, 256);
  printf("kernel Task begin... of task_id:%d on host: %s\n", task_id, hostname);
  //printf("task: %d, num_elements:%d, num_pairs: %d, length: %d, num_subregions:%d\n", task_id, num_elements, num_pairs, length, num_subreg);
  //for (int i=0; i<num_subreg; i++)
  kernel_cuda_task(*piece, regions, task_id);
}

int main(int argc, char **argv)
{

  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  Runtime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/, AUTO_GENERATE_ID, TaskConfigOptions(false/*leaf*/), "top_level");
  //{
  //  TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
  //  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  //  Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  //}
  Runtime::register_legion_task<check_task>(CHECK_TASK_ID,
      Processor::LOC_PROC, false/*single*/, true/*index*/, AUTO_GENERATE_ID, TaskConfigOptions(false/*leaf*/), "check task");
  Runtime::register_legion_task<kernel_task>(KERNEL_TASK_ID,
      Processor::TOC_PROC, false, true, AUTO_GENERATE_ID, TaskConfigOptions(false/*leaf*/), "kernel task");
/*
  Runtime::register_legion_task<gpu_mul_task>(MUL_GPU_TASK_ID,
      Processor::TOC_PROC, false, true);
*/
  Runtime::add_registration_callback(update_mappers);
  return Runtime::start(argc,argv);
}
