/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file auto_scheduler/transform_step.cc
 * \brief Transformation steps. For each schedule primitive, there is a corresponding transform
 * step.
 */

#include "transform_step.h"

#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>

#include <string>
#include <utility>
#include <vector>

#include "compute_dag.h"
#include "loop_state.h"
#include "utils.h"

namespace tvm {
namespace auto_scheduler {

// Update the te::stage to tir::IterVar axis mapping
void UpdateStageToAxesMap(const te::Stage& stage, StageToAxesMap* stage_to_axes) {
  if (auto pop = stage->op.as<te::ComputeOpNode>()) {
    Array<IterVar> axes;
    for (const auto& axis : pop->axis) {
      axes.push_back(axis);
    }
    for (const auto& axis : pop->reduce_axis) {
      axes.push_back(axis);
    }
    stage_to_axes->Set(stage, std::move(axes));
  } else if (stage->op->IsInstance<te::PlaceholderOpNode>()) {
    {}  // do nothing on Placeholder
  } else {
    LOG(FATAL) << "Invalid op " << stage->op;
  }
}

const char* IteratorAnnotationString[] = {
    "for",          // kNone = 0
    "unroll",       // kUnroll = 1
    "vectorize",    // kVectorize = 2
    "parallel",     // kParallel = 3
    "vthread",      // kVThread = 4
    "blockIdx.x",   // kBlockX = 5
    "threadIdx.x",  // kThreadX = 6
    "blockIdx.y",   // kBlockY = 7
    "threadIdx.y",  // kThreadY = 8
    "blockIdx.z",   // kBlockZ = 9
    "threadIdx.z",  // kThreadZ = 10
    "tensorize"     // kTensorized = 11
};

Step StepReadFromRecord(dmlc::JSONReader* reader) {
  std::string name;
  bool s;
  s = reader->NextArrayItem();
  CHECK(s);
  reader->Read(&name);
  if (name == AnnotationStepNode::record_prefix_str) {
    return AnnotationStep(reader);
  } else if (name == FuseStepNode::record_prefix_str) {
    return FuseStep(reader);
  } else if (name == ReorderStepNode::record_prefix_str) {
    return ReorderStep(reader);
  } else if (name == SplitStepNode::record_prefix_str) {
    return SplitStep(reader);
  } else if (name == ComputeAtStepNode::record_prefix_str) {
    return ComputeAtStep(reader);
  } else if (name == ComputeInlineStepNode::record_prefix_str) {
    return ComputeInlineStep(reader);
  } else if (name == ComputeRootStepNode::record_prefix_str) {
    return ComputeRootStep(reader);
  } else if (name == CacheReadStepNode::record_prefix_str) {
    return CacheReadStep(reader);
  } else if (name == CacheWriteStepNode::record_prefix_str) {
    return CacheWriteStep(reader);
  } else {
    LOG(FATAL) << "Invalid step format: " << name;
  }
  return Step();
}

void StepApplyToState(const Step& step, State* state, const ComputeDAG& dag) {
  if (auto ps = step.as<AnnotationStepNode>()) {
    ps->ApplyToState(state);
  } else if (auto ps = step.as<FuseStepNode>()) {
    ps->ApplyToState(state);
  } else if (auto ps = step.as<ReorderStepNode>()) {
    ps->ApplyToState(state);
  } else if (auto ps = step.as<SplitStepNode>()) {
    ps->ApplyToState(state);
  } else if (auto ps = step.as<ComputeAtStepNode>()) {
    ps->ApplyToState(state);
  } else if (auto ps = step.as<ComputeInlineStepNode>()) {
    ps->ApplyToState(state);
  } else if (auto ps = step.as<ComputeRootStepNode>()) {
    ps->ApplyToState(state);
  } else if (auto ps = step.as<CacheReadStepNode>()) {
    ps->ApplyToState(state, dag);
  } else if (auto ps = step.as<CacheWriteStepNode>()) {
    ps->ApplyToState(state, dag);
  } else {
    LOG(FATAL) << "Invalid step: " << step;
  }
}

void StepApplyToSchedule(const Step& step, Array<te::Stage>* stages, StageToAxesMap* stage_to_axes,
                         te::Schedule* schedule) {
  if (auto ps = step.as<AnnotationStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes);
  } else if (auto ps = step.as<FuseStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes);
  } else if (auto ps = step.as<ReorderStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes);
  } else if (auto ps = step.as<SplitStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes);
  } else if (auto ps = step.as<ComputeAtStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes);
  } else if (auto ps = step.as<ComputeInlineStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes);
  } else if (auto ps = step.as<ComputeRootStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes);
  } else if (auto ps = step.as<CacheReadStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes, schedule);
  } else if (auto ps = step.as<CacheWriteStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes, schedule);
  } else {
    LOG(FATAL) << "Invalid Step: " << step;
  }
}

String StepPrintAsPythonAPI(const Step& step, Array<te::Stage>* stages,
                            StageToAxesMap* stage_to_axes, te::Schedule* schedule) {
  if (auto ps = step.as<AnnotationStepNode>()) {
    return ps->PrintAsPythonAPI(stages, stage_to_axes);
  } else if (auto ps = step.as<FuseStepNode>()) {
    return ps->PrintAsPythonAPI(stages, stage_to_axes);
  } else if (auto ps = step.as<ReorderStepNode>()) {
    return ps->PrintAsPythonAPI(stages, stage_to_axes);
  } else if (auto ps = step.as<SplitStepNode>()) {
    return ps->PrintAsPythonAPI(stages, stage_to_axes);
  } else if (auto ps = step.as<ComputeAtStepNode>()) {
    return ps->PrintAsPythonAPI(stages, stage_to_axes);
  } else if (auto ps = step.as<ComputeInlineStepNode>()) {
    return ps->PrintAsPythonAPI(stages, stage_to_axes);
  } else if (auto ps = step.as<ComputeRootStepNode>()) {
    return ps->PrintAsPythonAPI(stages, stage_to_axes);
  } else if (auto ps = step.as<CacheReadStepNode>()) {
    return ps->PrintAsPythonAPI(stages, stage_to_axes, schedule);
  } else if (auto ps = step.as<CacheWriteStepNode>()) {
    return ps->PrintAsPythonAPI(stages, stage_to_axes, schedule);
  } else {
    LOG(FATAL) << "Invalid Step: " << step;
  }
  return "";
}

/********** Primitives working on single stage **********/

/********** Annotation **********/
AnnotationStep::AnnotationStep(int stage_id, int iter_id, IteratorAnnotation ann) {
  auto node = make_object<AnnotationStepNode>();
  node->stage_id = stage_id;
  node->iter_id = iter_id;
  node->annotation = ann;
  data_ = std::move(node);
}

AnnotationStep::AnnotationStep(dmlc::JSONReader* reader) {
  auto node = make_object<AnnotationStepNode>();
  bool s;
  s = reader->NextArrayItem();
  CHECK(s);
  reader->Read(&node->stage_id);
  s = reader->NextArrayItem();
  CHECK(s);
  reader->Read(&node->iter_id);
  s = reader->NextArrayItem();
  CHECK(s);
  int int_val;
  reader->Read(&int_val);
  node->annotation = IteratorAnnotation(int_val);
  data_ = std::move(node);
}

void AnnotationStepNode::WriteToRecord(dmlc::JSONWriter* writer) const {
  writer->WriteArraySeperator();
  writer->WriteString(record_prefix_str);
  writer->WriteArrayItem(stage_id);
  writer->WriteArrayItem(iter_id);
  writer->WriteArrayItem(static_cast<int>(annotation));
}

Iterator AnnotationStepNode::ApplyToState(State* state) const {
  const Stage& stage = (*state)->stages[stage_id];
  Iterator it = stage->iters[iter_id];

  CHECK(it->annotation == IteratorAnnotation::kNone);
  Iterator new_it = Iterator(it->name, it->range, it->iter_kind, annotation);
  Stage new_stage = stage;
  new_stage.CopyOnWrite()->iters.Set(iter_id, new_it);
  state->CopyOnWrite()->stages.Set(stage_id, std::move(new_stage));
  return new_it;
}

void AnnotationStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                         StageToAxesMap* stage_to_axes) const {
  te::Stage stage = (*stages)[stage_id];
  const Array<IterVar>& axes = (*stage_to_axes)[stage];

  switch (annotation) {
    case IteratorAnnotation::kUnroll:
      stage.unroll(axes[iter_id]);
      break;
    case IteratorAnnotation::kVectorize:
      stage.vectorize(axes[iter_id]);
      break;
    case IteratorAnnotation::kParallel:
      stage.parallel(axes[iter_id]);
      break;
    case IteratorAnnotation::kVThread:
    case IteratorAnnotation::kBlockX:
    case IteratorAnnotation::kBlockY:
    case IteratorAnnotation::kBlockZ:
    case IteratorAnnotation::kThreadX:
    case IteratorAnnotation::kThreadY:
    case IteratorAnnotation::kThreadZ:
      stage.bind(axes[iter_id],
                 te::thread_axis(Range(), IteratorAnnotationString[static_cast<int>(annotation)]));
      break;
    case IteratorAnnotation::kNone:
      break;
    default:
      LOG(FATAL) << "Invalid Annotation " << static_cast<int>(annotation);
      break;
  }

  stages->Set(stage_id, std::move(stage));
}

String AnnotationStepNode::PrintAsPythonAPI(Array<te::Stage>* stages,
                                            StageToAxesMap* stage_to_axes) const {
  std::stringstream ss;
  const auto& stage = (*stages)[stage_id];
  const auto& iter = (*stage_to_axes)[stage][iter_id];

  ss << "s[" << CleanName(stage->op->name) << "].";
  switch (annotation) {
    case IteratorAnnotation::kUnroll:
      ss << "unroll(";
      break;
    case IteratorAnnotation::kVectorize:
      ss << "vectorize(";
      break;
    case IteratorAnnotation::kParallel:
      ss << "parallel(";
      break;
    case IteratorAnnotation::kVThread:
    case IteratorAnnotation::kBlockX:
    case IteratorAnnotation::kBlockY:
    case IteratorAnnotation::kBlockZ:
    case IteratorAnnotation::kThreadX:
    case IteratorAnnotation::kThreadY:
    case IteratorAnnotation::kThreadZ:
      ss << "bind(";
      break;
    case IteratorAnnotation::kNone:
      break;
    default:
      LOG(FATAL) << "Invalid annotation " << static_cast<int>(annotation);
      break;
  }
  ss << CleanName(iter->var->name_hint);
  switch (annotation) {
    case IteratorAnnotation::kVThread:
    case IteratorAnnotation::kBlockX:
    case IteratorAnnotation::kBlockY:
    case IteratorAnnotation::kBlockZ:
    case IteratorAnnotation::kThreadX:
    case IteratorAnnotation::kThreadY:
    case IteratorAnnotation::kThreadZ:
      ss << ", tvm.thread_axis(\"" << IteratorAnnotationString[static_cast<int>(annotation)]
         << "\")";
      break;
    default:
      break;
  }
  ss << ")\n";

  ApplyToSchedule(stages, stage_to_axes);
  return ss.str();
}

/********** Fuse **********/
FuseStep::FuseStep(int stage_id, const Array<Integer>& fused_ids) {
  auto node = make_object<FuseStepNode>();
  node->stage_id = stage_id;
  for (const auto& x : fused_ids) {
    CHECK(x->IsInstance<IntImmNode>());
  }
  node->fused_ids = fused_ids;
  data_ = std::move(node);
}

FuseStep::FuseStep(dmlc::JSONReader* reader) {
  auto node = make_object<FuseStepNode>();
  bool s;
  s = reader->NextArrayItem();
  CHECK(s);
  reader->Read(&node->stage_id);
  std::vector<int> int_list;
  s = reader->NextArrayItem();
  CHECK(s);
  reader->Read(&int_list);
  ::tvm::Array<::tvm::Integer> fused_ids;
  for (const auto& i : int_list) {
    fused_ids.push_back(i);
  }
  node->fused_ids = fused_ids;
  data_ = std::move(node);
}

void FuseStepNode::WriteToRecord(dmlc::JSONWriter* writer) const {
  writer->WriteArraySeperator();
  writer->WriteString(record_prefix_str);
  writer->WriteArrayItem(stage_id);
  writer->WriteArrayItem(IntArrayToVector(fused_ids));
}

Iterator FuseStepNode::ApplyToState(State* state) const {
  const Stage& stage = (*state)->stages[stage_id];
  size_t old_iter_size = static_cast<int>(stage->iters.size());

  String new_name;
  PrimExpr new_extent = 1;
  IteratorKind new_iter_kind = IteratorKind::kSpecial;

  for (size_t i = 0; i < fused_ids.size(); ++i) {
    if (i > 0) {
      CHECK_EQ(fused_ids[i]->value, fused_ids[i - 1]->value + 1);
    }

    if (i != fused_ids.size() - 1) {
      const auto& iter_to_attached_stage = (*state)->attach_map->iter_to_attached_stages;
      if (iter_to_attached_stage.find(std::make_pair(stage_id, fused_ids[i])) !=
          iter_to_attached_stage.end()) {
        LOG(FATAL) << "Invalid Fuse. Trying to fuse iterators that have been attached by some "
                   << "stages. State before fusion:\n"
                   << (*state);
      }
    }

    const Iterator& it = stage->iters[fused_ids[i]];
    new_name = new_name + it->name + "@";

    if (it->range.defined() && new_extent.defined()) {
      new_extent = new_extent * it->range->extent;
    } else {
      new_extent = PrimExpr();
    }

    if (i == 0) {
      new_iter_kind = it->iter_kind;
    } else {
      if (new_iter_kind != it->iter_kind) {
        new_iter_kind = IteratorKind::kMixed;
      }
    }
  }

  Range range;
  if (new_extent.defined()) {
    range = Range::FromMinExtent(0, new_extent);
  }
  Iterator new_it = Iterator(new_name, range, new_iter_kind, IteratorAnnotation::kNone);
  Array<Iterator> new_iters;
  new_iters.insert(new_iters.end(), stage->iters.begin(), stage->iters.begin() + fused_ids.front());
  new_iters.push_back(new_it);
  new_iters.insert(new_iters.end(), stage->iters.begin() + fused_ids.back() + 1,
                   stage->iters.end());

  StateNode* pstate = state->CopyOnWrite();
  pstate->stages.Set(stage_id,
                     Stage(stage->op, stage->op_type, new_iters, stage->compute_at, stage->attrs));

  // Two vectors are used to represent the iterator relation before and after fuse
  // The original iterators in AttachMap will be updated with the new iterators
  std::vector<IterKey> from_iters;
  std::vector<IterKey> to_iters;
  const size_t begin_id = fused_ids.front(), end_id = fused_ids.back();
  for (size_t i = 0; i < old_iter_size; ++i) {
    if (i <= begin_id) {
      continue;
    } else if (i > end_id) {
      // move forward
      from_iters.emplace_back(stage_id, i);
      to_iters.emplace_back(stage_id, i - end_id + begin_id);
    } else {
      // move to the fused id
      from_iters.emplace_back(stage_id, i);
      to_iters.emplace_back(stage_id, begin_id);
    }
  }
  pstate->attach_map.UpdateIters(from_iters, to_iters);

  return new_it;
}

IterVar FuseStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                      StageToAxesMap* stage_to_axes) const {
  auto stage = (*stages)[stage_id];
  const Array<IterVar>& axes = stage_to_axes->at(stage);

  Array<IterVar> to_fuse;
  for (const auto& i : fused_ids) {
    to_fuse.push_back(axes[i]);
  }
  IterVar fused_axis;
  stage.fuse(to_fuse, &fused_axis);

  Array<IterVar> new_axes;
  new_axes.insert(new_axes.end(), axes.begin(), axes.begin() + fused_ids.front());
  new_axes.push_back(fused_axis);
  new_axes.insert(new_axes.end(), axes.begin() + fused_ids.back() + 1, axes.end());

  stage_to_axes->Set(stage, std::move(new_axes));
  stages->Set(stage_id, std::move(stage));
  return fused_axis;
}

String FuseStepNode::PrintAsPythonAPI(Array<te::Stage>* stages,
                                      StageToAxesMap* stage_to_axes) const {
  const auto& stage = (*stages)[stage_id];
  std::stringstream to_fuse;

  for (size_t i = 0; i < fused_ids.size(); ++i) {
    to_fuse << CleanName(stage_to_axes->at(stage)[fused_ids[i]]->var->name_hint);
    if (i != fused_ids.size() - 1) {
      to_fuse << ", ";
    }
  }

  std::stringstream ss;
  const auto& fused = ApplyToSchedule(stages, stage_to_axes);

  ss << CleanName(fused->var->name_hint) << " = s[" << CleanName(stage->op->name) << "].fuse("
     << to_fuse.str() << ")\n";

  return ss.str();
}

/********** Reorder **********/
ReorderStep::ReorderStep(int stage_id, const Array<Integer>& after_ids) {
  auto node = make_object<ReorderStepNode>();
  node->stage_id = stage_id;
  for (const auto& x : after_ids) {
    CHECK(x->IsInstance<IntImmNode>());
  }
  node->after_ids = after_ids;
  data_ = std::move(node);
}

ReorderStep::ReorderStep(dmlc::JSONReader* reader) {
  auto node = make_object<ReorderStepNode>();
  bool s;
  s = reader->NextArrayItem();
  CHECK(s);
  reader->Read(&node->stage_id);
  s = reader->NextArrayItem();
  CHECK(s);
  std::vector<int> int_list;
  reader->Read(&int_list);
  ::tvm::Array<::tvm::Integer> after_ids;
  for (const auto& i : int_list) {
    after_ids.push_back(i);
  }
  node->after_ids = after_ids;
  data_ = std::move(node);
}

void ReorderStepNode::WriteToRecord(dmlc::JSONWriter* writer) const {
  writer->WriteArraySeperator();
  writer->WriteString(record_prefix_str);
  writer->WriteArrayItem(stage_id);
  writer->WriteArrayItem(IntArrayToVector(after_ids));
}

void ReorderStepNode::ApplyToState(State* state) const {
  const Stage& stage = (*state)->stages[stage_id];
  Array<Iterator> iters;
  for (auto x : after_ids) {
    iters.push_back(stage->iters[x]);
  }
  state->CopyOnWrite()->stages.Set(
      stage_id, Stage(stage->op, stage->op_type, iters, stage->compute_at, stage->attrs));
}

void ReorderStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                      StageToAxesMap* stage_to_axes) const {
  auto stage = (*stages)[stage_id];
  const Array<IterVar>& axes = stage_to_axes->at(stage);
  CHECK_EQ(after_ids.size(), axes.size());

  Array<IterVar> new_axes;
  new_axes.reserve(axes.size());
  for (auto i : after_ids) {
    new_axes.push_back(axes[i]);
  }
  stage.reorder(new_axes);

  stage_to_axes->Set(stage, std::move(new_axes));
  stages->Set(stage_id, std::move(stage));
}

String ReorderStepNode::PrintAsPythonAPI(Array<te::Stage>* stages,
                                         StageToAxesMap* stage_to_axes) const {
  const auto& stage = (*stages)[stage_id];
  std::stringstream ss;

  ss << "s[" << CleanName(stage->op->name) << "].reorder(";
  for (size_t i = 0; i < after_ids.size(); ++i) {
    ss << CleanName((*stage_to_axes)[stage][after_ids[i]]->var->name_hint);
    if (i != after_ids.size() - 1) {
      ss << ", ";
    }
  }
  ss << ")\n";

  ApplyToSchedule(stages, stage_to_axes);
  return ss.str();
}

/********** Split **********/
// common part for SplitStep, FollowSplitStep, and FollowFusedSplitStep
Array<Iterator> ApplySplitToState(State* state, int stage_id, int iter_id,
                                  const Array<Optional<Integer>>& lengths, bool inner_to_outer) {
  const Stage& stage = (*state)->stages[stage_id];
  const Iterator& it = stage->iters[iter_id];
  size_t old_iter_size = stage->iters.size();
  bool concrete = true;

  Optional<PrimExpr> tosplit_min, tosplit_extent;
  if (it->range.defined()) {
    tosplit_min = it->range->min;
    tosplit_extent = it->range->extent;
  } else {
    tosplit_min = NullOpt;
    tosplit_extent = NullOpt;
  }

  Array<Iterator> outs;
  for (size_t i = 0; i < lengths.size(); ++i) {
    Optional<Integer> l;
    String name;
    if (inner_to_outer) {
      l = lengths[lengths.size() - i - 1];
      name = it->name + "." + std::to_string(lengths.size() - i);
    } else {
      l = lengths[i];
      name = it->name + "." + std::to_string(i);
    }
    Iterator res;
    if (l && tosplit_min && tosplit_extent) {
      res = Iterator(name, Range::FromMinExtent(tosplit_min.value(), l.value()), it->iter_kind,
                     IteratorAnnotation::kNone);
      tosplit_min = Integer(0);
      tosplit_extent = indexdiv(tosplit_extent.value() + l.value() - 1, l.value());
    } else {
      res = Iterator(name, Range(), it->iter_kind, IteratorAnnotation::kNone);
      tosplit_min = NullOpt;
      tosplit_extent = NullOpt;
      concrete = false;
    }
    outs.push_back(std::move(res));
  }

  Range range;
  if (tosplit_min && tosplit_extent) {
    range = Range::FromMinExtent(tosplit_min.value(), tosplit_extent.value());
  }
  if (inner_to_outer) {
    outs.push_back(Iterator(it->name + ".0", range, it->iter_kind, IteratorAnnotation::kNone));
    // Reverse the Iterator array
    Array<Iterator> temp(outs.rbegin(), outs.rend());
    outs = std::move(temp);
  } else {
    outs.push_back(Iterator(it->name + "." + std::to_string(lengths.size()), range, it->iter_kind,
                            IteratorAnnotation::kNone));
  }

  Array<Iterator> new_iters;
  new_iters.insert(new_iters.end(), stage->iters.begin(), stage->iters.begin() + iter_id);
  new_iters.insert(new_iters.end(), outs.begin(), outs.end());
  new_iters.insert(new_iters.end(), stage->iters.begin() + iter_id + 1, stage->iters.end());

  StateNode* pstate = state->CopyOnWrite();
  pstate->stages.Set(stage_id,
                     Stage(stage->op, stage->op_type, new_iters, stage->compute_at, stage->attrs));
  pstate->concrete &= concrete;

  // Two vectors are used to represent the iterator relation before and after split
  // The original iterators in AttachMap will be updated with the new iterators
  std::vector<IterKey> from_iters;
  std::vector<IterKey> to_iters;
  for (size_t i = iter_id; i < old_iter_size; ++i) {
    from_iters.emplace_back(stage_id, i);
    to_iters.emplace_back(stage_id, i + lengths.size());
  }
  pstate->attach_map.UpdateIters(from_iters, to_iters);

  return outs;
}

Array<IterVar> ApplySplitToSchedule(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes,
                                    int stage_id, int iter_id,
                                    const Array<Optional<Integer>>& lengths, bool inner_to_outer) {
  auto stage = (*stages)[stage_id];
  const Array<IterVar>& axes = stage_to_axes->at(stage);

  Array<IterVar> outs;
  if (inner_to_outer) {
    IterVar outer = axes[iter_id], inner;
    for (int i = static_cast<int>(lengths.size()) - 1; i >= 0; i--) {
      IterVar to_split = outer;
      stage.split(to_split, lengths[i].value(), &outer, &inner);
      outs.push_back(inner);
    }
    outs.push_back(outer);
  } else {
    IterVar outer, inner = axes[iter_id];
    for (size_t i = 0; i < lengths.size(); i++) {
      IterVar to_split = inner;
      stage.split_by_nparts(to_split, lengths[i].value(), &outer, &inner);
      outs.push_back(outer);
    }
    outs.push_back(inner);
  }

  Array<IterVar> new_axes;
  new_axes.insert(new_axes.end(), axes.begin(), axes.begin() + iter_id);
  if (inner_to_outer) {
    for (auto x = outs.rbegin(); x != outs.rend(); ++x) {
      new_axes.push_back((*x));
    }
  } else {
    for (const auto& x : outs) {
      new_axes.push_back(x);
    }
  }
  new_axes.insert(new_axes.end(), axes.begin() + iter_id + 1, axes.end());

  stage_to_axes->Set(stage, std::move(new_axes));
  stages->Set(stage_id, std::move(stage));
  return outs;
}

String PrintSplitAsPythonAPI(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes, int stage_id,
                             int iter_id, const Array<Optional<Integer>>& lengths,
                             bool inner_to_outer) {
  const auto& stage = (*stages)[stage_id];
  auto to_split = stage_to_axes->at(stage)[iter_id];
  const auto& func_name = CleanName(stage->op->name);
  const auto& outs =
      ApplySplitToSchedule(stages, stage_to_axes, stage_id, iter_id, lengths, inner_to_outer);
  CHECK_EQ(outs.size(), lengths.size() + 1);

  std::stringstream ss;
  int size = static_cast<int>(lengths.size());
  if (inner_to_outer) {
    for (int i = size - 1; i >= 0; i--) {
      ss << CleanName(outs[size - i]->var->name_hint) << ", "
         << CleanName(outs[size - i - 1]->var->name_hint) << " = s[" << func_name << "].split("
         << CleanName(to_split->var->name_hint) << ", factor=" << lengths[i] << ")\n";
      to_split = outs[size - i];
    }
  } else {
    for (int i = 0; i < size; i++) {
      ss << CleanName(outs[i]->var->name_hint) << ", " << CleanName(outs[i + 1]->var->name_hint)
         << " = s[" << func_name << "].split(" << CleanName(to_split->var->name_hint)
         << ", nparts=" << lengths[i] << ")\n";
      to_split = outs[i + 1];
    }
  }

  return ss.str();
}

SplitStep::SplitStep(int stage_id, int iter_id, Optional<PrimExpr> extent,
                     const Array<Optional<Integer>>& lengths, bool inner_to_outer) {
  auto node = make_object<SplitStepNode>();
  node->stage_id = stage_id;
  // Extent can be a unreducible expression in some special cases
  if (extent && extent.value()->IsInstance<IntImmNode>()) {
    node->extent = tvm::Downcast<Integer>(extent.value());
  }
  node->iter_id = iter_id;
  node->lengths = lengths;
  node->inner_to_outer = inner_to_outer;
  data_ = std::move(node);
}

SplitStep::SplitStep(dmlc::JSONReader* reader) {
  auto node = make_object<SplitStepNode>();
  bool s;
  s = reader->NextArrayItem();
  CHECK(s);
  reader->Read(&node->stage_id);
  s = reader->NextArrayItem();
  CHECK(s);
  reader->Read(&node->iter_id);
  int int_val;
  s = reader->NextArrayItem();
  CHECK(s);
  reader->Read(&int_val);
  if (int_val) {
    node->extent = Integer(int_val);
  }
  s = reader->NextArrayItem();
  CHECK(s);
  std::vector<int> int_list;
  reader->Read(&int_list);
  ::tvm::Array<::tvm::Optional<::tvm::Integer>> lengths;
  for (const auto& i : int_list) {
    lengths.push_back(::tvm::Integer(i));
  }
  node->lengths = lengths;
  s = reader->NextArrayItem();
  CHECK(s);
  reader->Read(&node->inner_to_outer);
  data_ = std::move(node);
}

void SplitStepNode::WriteToRecord(dmlc::JSONWriter* writer) const {
  writer->WriteArraySeperator();
  writer->WriteString(record_prefix_str);
  writer->WriteArrayItem(stage_id);
  writer->WriteArrayItem(iter_id);
  writer->WriteArrayItem(extent ? GetIntImm(extent.value()) : 0);
  writer->WriteArrayItem(IntArrayToVector(lengths));
  writer->WriteArrayItem(static_cast<int>(inner_to_outer));
}

Array<Iterator> SplitStepNode::ApplyToState(State* state) const {
  return ApplySplitToState(state, stage_id, iter_id, lengths, inner_to_outer);
}

Array<IterVar> SplitStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                              StageToAxesMap* stage_to_axes) const {
  return ApplySplitToSchedule(stages, stage_to_axes, stage_id, iter_id, lengths, inner_to_outer);
}

String SplitStepNode::PrintAsPythonAPI(Array<te::Stage>* stages,
                                       StageToAxesMap* stage_to_axes) const {
  return PrintSplitAsPythonAPI(stages, stage_to_axes, stage_id, iter_id, lengths, inner_to_outer);
}

/********** Primitives working on multiple stages **********/

/********** Compute At **********/
ComputeAtStep::ComputeAtStep(int stage_id, int target_stage_id, int target_iter_id) {
  auto node = make_object<ComputeAtStepNode>();
  node->stage_id = stage_id;
  node->target_stage_id = target_stage_id;
  node->target_iter_id = target_iter_id;
  data_ = std::move(node);
}

ComputeAtStep::ComputeAtStep(dmlc::JSONReader* reader) {
  auto node = make_object<ComputeAtStepNode>();
  bool s;
  s = reader->NextArrayItem();
  CHECK(s);
  reader->Read(&node->stage_id);
  s = reader->NextArrayItem();
  CHECK(s);
  reader->Read(&node->target_stage_id);
  s = reader->NextArrayItem();
  CHECK(s);
  reader->Read(&node->target_iter_id);
  data_ = std::move(node);
}

void ComputeAtStepNode::WriteToRecord(dmlc::JSONWriter* writer) const {
  writer->WriteArraySeperator();
  writer->WriteString(record_prefix_str);
  writer->WriteArrayItem(stage_id);
  writer->WriteArrayItem(target_stage_id);
  writer->WriteArrayItem(target_iter_id);
}
void ComputeAtStepNode::ApplyToState(State* state) const {
  const Stage& stage = (*state)->stages[stage_id];

  // Remove the bound information of each iterator since they may not be accurate after
  // compute at
  Array<Iterator> new_iters;
  for (const Iterator& it : stage->iters) {
    new_iters.push_back(Iterator(it->name, Range(), it->iter_kind, it->annotation));
  }

  StateNode* pstate = state->CopyOnWrite();
  pstate->stages.Set(stage_id, Stage(stage->op, stage->op_type, std::move(new_iters),
                                     ComputeAtKind::kIter, stage->attrs));
  // Update attach map
  pstate->attach_map.SetComputeAtIter(stage_id, target_stage_id, target_iter_id);
}

void ComputeAtStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                        StageToAxesMap* stage_to_axes) const {
  te::Stage stage = (*stages)[stage_id];
  const auto& target_stage = (*stages)[target_stage_id];
  const auto& target_axis = (*stage_to_axes)[target_stage][target_iter_id];
  stage.compute_at(target_stage, target_axis);

  stages->Set(stage_id, std::move(stage));
}

String ComputeAtStepNode::PrintAsPythonAPI(Array<te::Stage>* stages,
                                           StageToAxesMap* stage_to_axes) const {
  std::stringstream ss;
  const auto& stage = (*stages)[stage_id];
  const auto& target_stage = (*stages)[target_stage_id];
  ss << "s[" << CleanName(stage->op->name) << "].compute_at(s[" << CleanName(target_stage->op->name)
     << "], " << CleanName((*stage_to_axes)[target_stage][target_iter_id]->var->name_hint) << ")\n";
  ApplyToSchedule(stages, stage_to_axes);
  return ss.str();
}

/********** Compute Inline **********/
ComputeInlineStep::ComputeInlineStep(int stage_id) {
  auto node = make_object<ComputeInlineStepNode>();
  node->stage_id = stage_id;
  data_ = std::move(node);
}

ComputeInlineStep::ComputeInlineStep(dmlc::JSONReader* reader) {
  auto node = make_object<ComputeInlineStepNode>();
  bool s;
  s = reader->NextArrayItem();
  CHECK(s);
  reader->Read(&node->stage_id);
  data_ = std::move(node);
}

void ComputeInlineStepNode::WriteToRecord(dmlc::JSONWriter* writer) const {
  writer->WriteArraySeperator();
  writer->WriteString(record_prefix_str);
  writer->WriteArrayItem(stage_id);
}

void ComputeInlineStepNode::ApplyToState(State* state) const {
  const Stage& stage = (*state)->stages[stage_id];

  // Check the validity of compute_inline
  for (size_t i = 0; i < stage->iters.size(); ++i) {
    CHECK_EQ((*state)->attach_map->iter_to_attached_stages.count(std::make_pair(stage_id, i)), 0)
        << "Invalid compute_inline: There are some other stages that are attached to the "
        << "target stage";
  }

  StateNode* pstate = state->CopyOnWrite();
  auto new_stage = pstate->stages[stage_id];
  new_stage.CopyOnWrite()->compute_at = ComputeAtKind::kInlined;
  pstate->stages.Set(stage_id, std::move(new_stage));
  // Update attach map
  pstate->attach_map.DeleteStage(stage_id);
}

void ComputeInlineStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                            StageToAxesMap* stage_to_axes) const {
  auto stage = (*stages)[stage_id];
  stage.compute_inline();
  stages->Set(stage_id, std::move(stage));
}

String ComputeInlineStepNode::PrintAsPythonAPI(Array<te::Stage>* stages,
                                               StageToAxesMap* stage_to_axes) const {
  std::stringstream ss;
  const auto& stage = (*stages)[stage_id];
  ss << "s[" << CleanName(stage->op->name) << "].compute_inline()\n";
  ApplyToSchedule(stages, stage_to_axes);
  return ss.str();
}

/********** Compute Root **********/
ComputeRootStep::ComputeRootStep(int stage_id) {
  auto node = make_object<ComputeRootStepNode>();
  node->stage_id = stage_id;
  data_ = std::move(node);
}

ComputeRootStep::ComputeRootStep(dmlc::JSONReader* reader) {
  auto node = make_object<ComputeRootStepNode>();
  bool s;
  s = reader->NextArrayItem();
  CHECK(s);
  reader->Read(&node->stage_id);
  data_ = std::move(node);
}

void ComputeRootStepNode::WriteToRecord(dmlc::JSONWriter* writer) const {
  writer->WriteArraySeperator();
  writer->WriteString(record_prefix_str);
  writer->WriteArrayItem(stage_id);
}

void ComputeRootStepNode::ApplyToState(State* state) const {
  const Stage& stage = (*state)->stages[stage_id];

  // Remove the bound information of each iterator since they may not be accurate after
  // compute root
  Array<Iterator> new_iters;
  for (const Iterator& it : stage->iters) {
    new_iters.push_back(Iterator(it->name, Range(), it->iter_kind, it->annotation));
  }

  StateNode* pstate = state->CopyOnWrite();
  pstate->stages.Set(stage_id, Stage(stage->op, stage->op_type, std::move(new_iters),
                                     ComputeAtKind::kRoot, stage->attrs));
  // Update attach map
  pstate->attach_map.DeleteStage(stage_id);
}

void ComputeRootStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                          StageToAxesMap* stage_to_axes) const {
  auto stage = (*stages)[stage_id];
  stage.compute_root();
  stages->Set(stage_id, std::move(stage));
}

String ComputeRootStepNode::PrintAsPythonAPI(Array<te::Stage>* stages,
                                             StageToAxesMap* stage_to_axes) const {
  std::stringstream ss;
  const auto& stage = (*stages)[stage_id];
  ss << "s[" << CleanName(stage->op->name) << "].compute_root()\n";
  ApplyToSchedule(stages, stage_to_axes);
  return ss.str();
}

/********** Primitives adding new stages **********/

/*!
 * \brief Common part for steps that add new stages(e.g. CacheReadStep, CacheWriteStep,
 * RfactorStep). This will filter out all steps that can change the stages of ComputeDAG.
 */
Array<Step> GetStageModifiableSteps(Step current_step, const Array<Step>& transform_steps) {
  Array<Step> ret_steps;
  for (const Step& step : transform_steps) {
    if (step->IsInstance<CacheWriteStepNode>() || step->IsInstance<CacheReadStepNode>()) {
      ret_steps.push_back(step);
    }
    // TODO(jcf94): add rfactor support
    if (step.same_as(current_step)) {
      break;
    }
  }
  return ret_steps;
}

/********** Cache Read **********/
CacheReadStep::CacheReadStep(int stage_id, String scope_name,
                             const Array<Integer>& reader_stage_ids) {
  auto node = make_object<CacheReadStepNode>();
  node->stage_id = stage_id;
  node->scope_name = std::move(scope_name);
  node->reader_stage_ids = reader_stage_ids;
  data_ = std::move(node);
}

CacheReadStep::CacheReadStep(dmlc::JSONReader* reader) {
  auto node = make_object<CacheReadStepNode>();
  bool s;
  s = reader->NextArrayItem();
  CHECK(s);
  reader->Read(&node->stage_id);
  s = reader->NextArrayItem();
  CHECK(s);
  std::string string_value;
  reader->Read(&string_value);
  node->scope_name = std::move(string_value);
  s = reader->NextArrayItem();
  CHECK(s);
  std::vector<int> int_list;
  reader->Read(&int_list);
  Array<Integer> reader_stage_ids;
  for (int i : int_list) {
    reader_stage_ids.push_back(i);
  }
  node->reader_stage_ids = std::move(reader_stage_ids);
  data_ = std::move(node);
}

void CacheReadStepNode::WriteToRecord(dmlc::JSONWriter* writer) const {
  writer->WriteArraySeperator();
  writer->WriteString(record_prefix_str);
  writer->WriteArrayItem(stage_id);
  writer->WriteArraySeperator();
  writer->WriteString(scope_name);
  writer->WriteArrayItem(IntArrayToVector(reader_stage_ids));
}

int CacheReadStepNode::ApplyToState(State* state, const ComputeDAG& dag) const {
  StateNode* pstate = state->CopyOnWrite();
  const ComputeDAG& current_compute_dag =
      dag.ReplayAndGetDAG(GetStageModifiableSteps(GetRef<Step>(this), (*state)->transform_steps));

  // target_stage -> target_stage + target_store
  // Update the op of the target stage, insert a new cache read stage behind, update the op of
  // later stages, then update the stage_id mapping in AttachMap
  int added_stage_id = stage_id + 1;
  Stage tmp_stage = pstate->stages[stage_id];
  tmp_stage.CopyOnWrite()->op = current_compute_dag->ops[stage_id];
  pstate->stages.Set(stage_id, std::move(tmp_stage));
  pstate->stages.insert(pstate->stages.begin() + added_stage_id,
                        Stage(current_compute_dag->ops[added_stage_id]));
  for (size_t i = added_stage_id + 1; i < pstate->stages.size(); ++i) {
    tmp_stage = pstate->stages[i];
    tmp_stage.CopyOnWrite()->op = current_compute_dag->ops[i];
    pstate->stages.Set(i, std::move(tmp_stage));
  }
  pstate->attach_map = pstate->attach_map.ApplyStageIdOffset(added_stage_id);
  pstate->current_compute_dag = std::move(current_compute_dag);

  return added_stage_id;
}

te::Tensor CacheReadStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                              StageToAxesMap* stage_to_axes,
                                              te::Schedule* schedule) const {
  const te::Stage& stage = (*stages)[stage_id];
  Array<te::Operation> readers;
  for (const auto& i : reader_stage_ids) {
    readers.push_back((*stages)[i]->origin_op);
  }
  auto out = schedule->cache_read(stage->origin_op.output(0), scope_name, readers);

  const auto& new_stage = (*schedule)[out->op];
  UpdateStageToAxesMap(new_stage, stage_to_axes);
  stages->insert(stages->begin() + stage_id + 1, new_stage);

  return out;
}

String CacheReadStepNode::PrintAsPythonAPI(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes,
                                           te::Schedule* schedule) const {
  std::stringstream ss;
  // Since the original stage will be changed after schedule apply, keep a copy here
  // These information will be used to print Python API string later
  auto stage = (*stages)[stage_id];
  Array<te::Stage> reader_stages;
  for (size_t i = 0; i < reader_stage_ids.size(); ++i) {
    reader_stages.push_back((*stages)[reader_stage_ids[i]]);
  }
  auto out = ApplyToSchedule(stages, stage_to_axes, schedule);

  ss << CleanName(out->op->name) << " = "
     << "s.cache_read(" << CleanName(stage->op->name) << ", \"" << scope_name << "\", ["
     << CleanName(reader_stages[0]->op->name);
  for (size_t i = 1; i < reader_stage_ids.size(); ++i) {
    ss << ", " << CleanName(reader_stages[i]->op->name);
  }
  ss << "])\n";

  // Print the iterators of the new added stage
  const auto& iters = out->op->root_iter_vars();
  for (size_t i = 0; i < iters.size(); ++i) {
    ss << CleanName(iters[i]->var->name_hint);
    if (i != iters.size() - 1) {
      ss << ", ";
    }
  }
  ss << " = "
     << "tuple(" << CleanName(out->op->name) << ".op.axis)\n";

  return ss.str();
}

/********** Cache Write **********/
CacheWriteStep::CacheWriteStep(int stage_id, String scope_name) {
  auto node = make_object<CacheWriteStepNode>();
  node->stage_id = stage_id;
  node->scope_name = std::move(scope_name);
  data_ = std::move(node);
}

CacheWriteStep::CacheWriteStep(dmlc::JSONReader* reader) {
  auto node = make_object<CacheWriteStepNode>();
  bool s;
  s = reader->NextArrayItem();
  CHECK(s);
  reader->Read(&node->stage_id);
  s = reader->NextArrayItem();
  CHECK(s);
  std::string string_value;
  reader->Read(&string_value);
  node->scope_name = std::move(string_value);
  data_ = std::move(node);
}

void CacheWriteStepNode::WriteToRecord(dmlc::JSONWriter* writer) const {
  writer->WriteArraySeperator();
  writer->WriteString(record_prefix_str);
  writer->WriteArrayItem(stage_id);
  writer->WriteArraySeperator();
  writer->WriteString(scope_name);
}

int CacheWriteStepNode::ApplyToState(State* state, const ComputeDAG& dag) const {
  StateNode* pstate = state->CopyOnWrite();
  int last_dag_op_size = pstate->current_compute_dag.defined()
                             ? pstate->current_compute_dag.as<ComputeDAGNode>()->ops.size()
                             : dag->ops.size();
  const ComputeDAG& current_compute_dag =
      dag.ReplayAndGetDAG(GetStageModifiableSteps(GetRef<Step>(this), (*state)->transform_steps));
  int added_ops = current_compute_dag->ops.size() - last_dag_op_size;
  // TODO(jcf94): Update this check to equal after fixing the cache write bug in TVM
  CHECK_GE(added_ops, 1);

  // target_stage -> cache_write_stage + target_stage
  // Assume no step has been applied to the target stage before cache write.
  // Insert a new cache write stage ahead, update the op of the target stage and later stages, then
  // update the stage_id mapping in AttachMap
  pstate->stages.insert(pstate->stages.begin() + stage_id,
                        Stage(current_compute_dag->ops[stage_id]));
  pstate->stages.Set(stage_id + 1, Stage(current_compute_dag->ops[stage_id + 1]));
  int next_stage_id = stage_id + 2;
  // TODO(jc94): Fix the cache write bug in TVM and remove added_op == 2 support.
  // TVM's cache_write has a bug with multi outputs. See
  // `tests/python/unittest/test_auto_scheduler_loop_state.py::test_cache_read_write` test
  // for more details
  if (added_ops == 2) {
    pstate->stages.insert(pstate->stages.begin() + next_stage_id,
                          Stage(current_compute_dag->ops[next_stage_id]));
    next_stage_id++;
  } else if (added_ops > 2) {
    LOG(ERROR) << "Unexpected behavior of CacheWrite.";
  }
  for (size_t i = next_stage_id; i < current_compute_dag->ops.size(); ++i) {
    Stage tmp_stage = pstate->stages[i];
    tmp_stage.CopyOnWrite()->op = current_compute_dag->ops[i];
    pstate->stages.Set(i, std::move(tmp_stage));
  }
  pstate->attach_map = pstate->attach_map.ApplyStageIdOffset(stage_id, added_ops);
  pstate->current_compute_dag = std::move(current_compute_dag);

  return stage_id;
}

Array<te::Tensor> CacheWriteStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                                      StageToAxesMap* stage_to_axes,
                                                      te::Schedule* schedule) const {
  const te::Stage& stage = (*stages)[stage_id];
  Array<te::Tensor> tensor_array;
  // If the target stage has multi outputs, TVM requires to cache_write
  // all of them or schedule.cache_write will raise an error
  for (auto i = 0; i < stage->op->num_outputs(); ++i) {
    tensor_array.push_back(stage->origin_op.output(i));
  }
  auto outs = schedule->cache_write(tensor_array, scope_name);

  UpdateStageToAxesMap(stage, stage_to_axes);
  // Even if there is multi outputs, TVM schedule only generate one
  // new stage
  const auto& new_stage = (*schedule)[outs[0]->op];
  UpdateStageToAxesMap(new_stage, stage_to_axes);
  stages->insert(stages->begin() + stage_id, new_stage);

  return outs;
}

String CacheWriteStepNode::PrintAsPythonAPI(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes,
                                            te::Schedule* schedule) const {
  std::stringstream ss;
  // Since the original stage will be changed after schedule apply, keep a copy here
  // These information will be used to print Python API string later
  te::Stage stage = (*stages)[stage_id];
  auto outs = ApplyToSchedule(stages, stage_to_axes, schedule);

  for (size_t i = 0; i < outs.size(); ++i) {
    ss << CleanName(outs[i]->op->name) << ", ";
  }
  ss << "= "
     << "s.cache_write([" << CleanName(stage->op.output(0)->op->name);
  for (auto i = 1; i < stage->op->num_outputs(); ++i) {
    ss << ", " << CleanName(stage->op.output(i)->op->name);
  }
  ss << "], \"" << scope_name << "\")\n";

  // Print the iterators of the new added stage
  for (const auto& out : outs) {
    const auto& iters = out->op->root_iter_vars();
    for (size_t i = 0; i < iters.size(); ++i) {
      ss << CleanName(iters[i]->var->name_hint);
      if (i != iters.size() - 1) {
        ss << ", ";
      }
    }
    ss << " = "
       << "tuple(" << CleanName(out->op->name) << ".op.axis)"
       << " + "
       << "tuple(" << CleanName(out->op->name) << ".op.reduce_axis)\n";
  }

  return ss.str();
}

}  // namespace auto_scheduler
}  // namespace tvm
