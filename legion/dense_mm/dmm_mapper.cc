/* Copyright 2017 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "dmm_mapper.h"

Logger log_mapper("mapper");

DmmMapper::DmmMapper(MapperRuntime *rt, Machine machine, Processor local,
                             const char *mapper_name,
                             std::vector<Processor>* _procs_list,
                             std::vector<Memory>* _sysmems_list,
                             std::map<Memory, std::vector<Processor> >* _sysmem_local_procs,
                             std::map<Processor, Memory>* _proc_sysmems,
                             std::map<Processor, Memory>* _proc_fbmems)
  : DefaultMapper(rt, machine, local, mapper_name),
    procs_list(*_procs_list),
    sysmems_list(*_sysmems_list),
    sysmem_local_procs(*_sysmem_local_procs),
    proc_sysmems(*_proc_sysmems),
    proc_fbmems(*_proc_fbmems)
{
}
void DmmMapper::map_task(const MapperContext      ctx,
                             const Task&              task,
                             const MapTaskInput&      input,
                                   MapTaskOutput&     output)
{
  #ifdef DEBUG_INFO
  printf("map_task begin...\n");
  printf("\tregion size: %d\n", task.regions.size());
  #endif
  if ((task.task_id != TOP_LEVEL_TASK_ID) &&
      (task.task_id != CHECK_TASK_ID))
  {
    Processor::Kind target_kind = task.target_proc.kind();
    #ifdef DEBUG_INFO
    if (target_kind == Processor::TOC_PROC) 
        printf("\tregion size: %d, GPU target!\n", task.regions.size());
    else
        printf("\tregion size: %d, CPU target!\n", task.regions.size());
    #endif
    VariantInfo chosen = default_find_preferred_variant(task, ctx,
                      true/*needs tight bound*/, true/*cache*/, target_kind);
    output.chosen_variant = chosen.variant;
    output.task_priority = 0;
    output.postmap_task = false;
    default_policy_select_target_processors(ctx, task, output.target_procs);

    bool map_to_gpu = task.target_proc.kind() == Processor::TOC_PROC;
    Memory sysmem = proc_sysmems[task.target_proc];
    Memory fbmem = proc_fbmems[task.target_proc];

    for (unsigned idx = 0; idx < task.regions.size(); idx++)
    {
      if ((task.regions[idx].privilege == NO_ACCESS) ||
          (task.regions[idx].privilege_fields.empty())) continue;

      Memory target_memory;
      if (!map_to_gpu) target_memory = sysmem;
      else {
        switch (task.task_id)
        {
          case KERNEL_TASK_ID:
            {
              target_memory = fbmem;
              break;
            }
          default:
            {
              assert(false);
              break;
            }
        }
      }
      const TaskLayoutConstraintSet &layout_constraints =
        runtime->find_task_layout_constraints(ctx,
                            task.task_id, output.chosen_variant);
      std::set<FieldID> fields(task.regions[idx].privilege_fields);
      if (!default_create_custom_instances(ctx, task.target_proc,
              target_memory, task.regions[idx], idx, fields,
              layout_constraints, true,
              output.chosen_instances[idx]))
      {
        printf("\t Size problem\n");
        default_report_failed_instance_creation(task, idx,
                                    task.target_proc, target_memory);
      }
    }
  }
  else
    DefaultMapper::map_task(ctx, task, input, output);
}

void DmmMapper::slice_task(const MapperContext      ctx,
                                   const Task&              task,
                                   const SliceTaskInput&    input,
                                         SliceTaskOutput&   output)
{
      log_mapper.spew("Default slice_task in %s", get_mapper_name());
      std::vector<VariantID> variants;
      runtime->find_valid_variants(ctx, task.task_id, variants);
      //printf("\tslice domain: %d, variant_size:%d\n", input.domain.get_dim(), variants.size());
      Processor::Kind target_kind =
        task.must_epoch_task ? local_proc.kind() : task.target_proc.kind();
      switch (target_kind)
      {
        case Processor::LOC_PROC:
          {
            default_slice_task(task, local_cpus, remote_cpus,
                               input, output, cpu_slices_cache);
            break;
          }
        case Processor::TOC_PROC:
          {
            default_slice_task(task, local_gpus, remote_gpus,
                               input, output, gpu_slices_cache);
            break;
          }
        case Processor::IO_PROC:
          {
            default_slice_task(task, local_ios, remote_ios,
                               input, output, io_slices_cache);
            break;
          }
        case Processor::PROC_SET:
          {
            default_slice_task(task, local_procsets, remote_procsets,
                               input, output, procset_slices_cache);
            break;
          }
        default:
          assert(false); // unimplemented processor kind
      }
}

void DmmMapper::default_slice_task(const Task &task,
                                           const std::vector<Processor> &local,
                                           const std::vector<Processor> &remote,
                                           const SliceTaskInput& input,
                                                 SliceTaskOutput &output,
                  std::map<Domain,std::vector<TaskSlice> > &cached_slices)
    //--------------------------------------------------------------------------
{
    //
      //printf("\t default_slice_task...\n");
      std::map<Domain,std::vector<TaskSlice> >::const_iterator finder =
        cached_slices.find(input.domain);
      if (finder != cached_slices.end()) {
        output.slices = finder->second;
        return;
      }
      // The two-level decomposition doesn't work so for now do a
      // simple one-level decomposition across all the processors.
      Machine::ProcessorQuery all_procs(machine);
      all_procs.only_kind(local[0].kind());
      std::vector<Processor> procs(all_procs.begin(), all_procs.end());
      //printf("\tall proc size: %d\n", procs.size());
      //printf("default slice domain: %d\n", input.domain.get_dim());
      switch (input.domain.get_dim())
      {
        case 1:
          {
            Rect<1> point_rect = input.domain.get_rect<1>();
            Point<1> num_blocks(procs.size());
            //printf("point size: %d\n", point_rect.volume());
            default_decompose_points<1>(point_rect, procs,
                  num_blocks, false/*recurse*/,
                  stealing_enabled, output.slices);
            break;
          }
      }
      cached_slices[input.domain] = output.slices;
}

void DmmMapper::map_inline(const MapperContext    ctx,
                               const InlineMapping&   inline_op,
                               const MapInlineInput&  input,
                                     MapInlineOutput& output)
{
  #ifdef DEBUG_INFO
  printf("map_inline begin...\n");
  #endif
  Memory target_memory =
    proc_sysmems[inline_op.parent_task->current_proc];
  bool force_create = false;
  LayoutConstraintID our_layout_id =
    default_policy_select_layout_constraints(ctx, target_memory,
        inline_op.requirement, INLINE_MAPPING, true, force_create);
  LayoutConstraintSet creation_constraints =
    runtime->find_layout_constraints(ctx, our_layout_id);
  std::set<FieldID> fields(inline_op.requirement.privilege_fields);
  creation_constraints.add_constraint(
      FieldConstraint(fields, false/*contig*/, false/*inorder*/));
  output.chosen_instances.resize(output.chosen_instances.size()+1);
  if (!default_make_instance(ctx, target_memory, creation_constraints,
        output.chosen_instances.back(), INLINE_MAPPING,
        force_create, true, inline_op.requirement))
  {
    log_mapper.error("Nw mapper failed allocation for region "
                 "requirement of inline mapping in task %s (UID %lld) "
                 "in memory " IDFMT "for processor " IDFMT ". This "
                 "means the working set of your application is too big "
                 "for the allotted capacity of the given memory under "
                 "the default mapper's mapping scheme. You have three "
                 "choices: ask Realm to allocate more memory, write a "
                 "custom mapper to better manage working sets, or find "
                 "a bigger machine. Good luck!",
                 inline_op.parent_task->get_task_name(),
                 inline_op.parent_task->get_unique_id(),
                 target_memory.id,
                 inline_op.parent_task->current_proc.id);
  }
}

void update_mappers(Machine machine, Runtime *runtime,
                    const std::set<Processor> &local_procs)
{
  #ifdef DEBUG_INFO
  printf("update_mappers begin...\n");
  #endif
  std::vector<Processor>* procs_list = new std::vector<Processor>();
  std::vector<Memory>* sysmems_list = new std::vector<Memory>();
  std::map<Memory, std::vector<Processor> >* sysmem_local_procs =
    new std::map<Memory, std::vector<Processor> >();
  std::map<Processor, Memory>* proc_sysmems = new std::map<Processor, Memory>();
  std::map<Processor, Memory>* proc_fbmems = new std::map<Processor, Memory>();

  std::vector<Machine::ProcessorMemoryAffinity> proc_mem_affinities;
  machine.get_proc_mem_affinity(proc_mem_affinities);

  for (unsigned idx = 0; idx < proc_mem_affinities.size(); ++idx) {
    Machine::ProcessorMemoryAffinity& affinity = proc_mem_affinities[idx];
    if (affinity.p.kind() == Processor::LOC_PROC) {
      if (affinity.m.kind() == Memory::SYSTEM_MEM) {
        printf("CPU tartget and system memrequest...\n");
        (*proc_sysmems)[affinity.p] = affinity.m;
      }
    }
    else if (affinity.p.kind() == Processor::TOC_PROC) {
      if (affinity.m.kind() == Memory::GPU_FB_MEM) {
        printf("GPU tartget and gmem memrequest...\n");
        (*proc_fbmems)[affinity.p] = affinity.m;
      }
    }
  }

  for (std::map<Processor, Memory>::iterator it = proc_sysmems->begin();
       it != proc_sysmems->end(); ++it) {
    procs_list->push_back(it->first);
    (*sysmem_local_procs)[it->second].push_back(it->first);
  }

  for (std::map<Memory, std::vector<Processor> >::iterator it =
        sysmem_local_procs->begin(); it != sysmem_local_procs->end(); ++it)
    sysmems_list->push_back(it->first);

  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    DmmMapper* mapper = new DmmMapper(runtime->get_mapper_runtime(),
                                              machine, *it, "dmm_mapper",
                                              procs_list,
                                              sysmems_list,
                                              sysmem_local_procs,
                                              proc_sysmems,
                                              proc_fbmems);
    runtime->replace_default_mapper(mapper, *it);
  }
}

