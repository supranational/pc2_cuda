// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cassert>

#include "poseidon_device.cu"

template <int t>
void hash_row_device(fr_t* in_ptr, fr_t* aux_ptr, fr_t* out_ptr,
                     const Poseidon& poseidon, const size_t next_num_sub_trees,
                     const size_t next_row_len, const bool first,
                     const cudaStream_t& stream, const bool mont) {

  const int hash_1_3_block_size = (256 / t) * t;
  const int hash_1_3_num_hashes_in_block = hash_1_3_block_size / t;
  const int hash_2_4_block_size = 128;

  int hash_1_3_thread_count = next_num_sub_trees * next_row_len * t;
  int hash_1_3_block_count =
    hash_1_3_thread_count / hash_1_3_block_size +
    static_cast<bool>(hash_1_3_thread_count % hash_1_3_block_size);

  int hash_2_4_block_count =
    (next_num_sub_trees * next_row_len) / hash_2_4_block_size +
    static_cast<bool>
      ((next_num_sub_trees * next_row_len) % hash_2_4_block_size);

  if (first) {
    poseidon_hash_1_0<<<hash_1_3_block_count, hash_1_3_block_size,
      sizeof(fr_t) * hash_1_3_num_hashes_in_block * t, stream>>>
      (in_ptr, aux_ptr, poseidon.poseidon_constants.domain_tag,
      poseidon.poseidon_constants.poseidon_constants_device, 0, 0,
      hash_1_3_thread_count, mont);
  }
  else {
    poseidon_hash_1_1<<<hash_1_3_block_count, hash_1_3_block_size,
      sizeof(fr_t) * hash_1_3_num_hashes_in_block * t, stream>>>
      (in_ptr, aux_ptr, poseidon.poseidon_constants.domain_tag,
      poseidon.poseidon_constants.poseidon_constants_device, 0, 0,
      hash_1_3_thread_count);
  }

  poseidon_hash_2<t><<<hash_2_4_block_count, hash_2_4_block_size, 0, stream>>>
    (aux_ptr, poseidon.poseidon_constants.poseidon_constants_device,
    t * (poseidon.half_full_rounds + 1), poseidon.half_full_rounds,
    next_num_sub_trees * next_row_len);

  poseidon_hash_3<<<hash_1_3_block_count, hash_1_3_block_size,
    sizeof(fr_t) * hash_1_3_num_hashes_in_block * t, stream>>>
    (aux_ptr, poseidon.poseidon_constants.poseidon_constants_device,
    t * (poseidon.half_full_rounds + 1) + poseidon.partial_rounds,
    poseidon.half_full_rounds + poseidon.partial_rounds,
    hash_1_3_thread_count);

  poseidon_hash_4<t><<<hash_2_4_block_count, hash_2_4_block_size, 0, stream>>>
    (aux_ptr, out_ptr,
    poseidon.poseidon_constants.poseidon_constants_device.mds_matrix,
    next_num_sub_trees * next_row_len);
}

static constexpr size_t ONE_GiB = 1073741824ull;

template <int t>
void tree_builder_device_internal(fr_t* leaves, const size_t leaves_len,
                                  fr_t* digests, const size_t digests_len,
                                  const Poseidon& poseidon,
                                  const size_t cutoff_leaves_len,
                                  const size_t cutoff_digests_len,
                                  const size_t sub_tree_len,
                                  const size_t sub_digests_len,
                                  const bool mont) {

  const size_t sub_tree_device_len = (sub_tree_len / poseidon.arity) * t;

  const size_t sub_tree_digests_req_mem =
    sizeof(fr_t) * (sub_tree_device_len + sub_digests_len);

  size_t device_avail_mem, device_total_mem;
  cudaMemGetInfo(&device_avail_mem, &device_total_mem);
  device_avail_mem -= ONE_GiB / 8; // leave 128 MiB free just in case

  const size_t num_sub_trees =
    ONE_GiB / sub_tree_digests_req_mem; // batches of 1 GiB
  const size_t num_streams = device_avail_mem / ONE_GiB;

  const size_t total_num_sub_trees = num_sub_trees * num_streams;

  cudaStream_t* streams =
    static_cast<cudaStream_t*>(malloc(sizeof(cudaStream_t) * num_streams));
  for (size_t i = 0; i < num_streams; i++) {
    cudaStreamCreate(&streams[i]);
  }

  fr_t* d_leaves, * d_digests;
  cudaMalloc(&d_leaves, sizeof(fr_t) *
                        std::min(total_num_sub_trees, cutoff_leaves_len) *
                        sub_tree_device_len);
  cudaMalloc(&d_digests, sizeof(fr_t) *
                         std::min(total_num_sub_trees, cutoff_leaves_len) *
                         sub_digests_len);

  fr_t* current_leaves_ptr = leaves;

  size_t loops = 0;
  for (size_t remaining_sub_trees = cutoff_leaves_len,
    processed_sub_trees = 0;
    remaining_sub_trees > 0;
    processed_sub_trees += std::min(num_sub_trees, remaining_sub_trees),
    remaining_sub_trees -= std::min(num_sub_trees, remaining_sub_trees)) {

    const size_t stream_idx = loops % num_streams;

    const size_t next_num_sub_trees =
      std::min(num_sub_trees, remaining_sub_trees);

    fr_t* in_ptr = d_leaves + num_sub_trees * sub_tree_device_len * stream_idx;
    fr_t* aux_ptr = in_ptr;
    fr_t* out_ptr = d_digests + num_sub_trees * sub_digests_len * stream_idx;

    cudaMemcpy2DAsync(in_ptr, sizeof(fr_t) * t, current_leaves_ptr,
                      sizeof(fr_t) * poseidon.arity,
                      sizeof(fr_t) * poseidon.arity,
                      next_num_sub_trees * sub_tree_len / poseidon.arity,
                      cudaMemcpyHostToDevice,
                      streams[stream_idx]);

    size_t digests_segment = leaves_len / poseidon.arity;
    size_t total_digests_offset = 0;

    for (size_t next_row_len = sub_tree_len / poseidon.arity;
       next_row_len > 0;
       next_row_len /= poseidon.arity) {

      const bool first = next_row_len == sub_tree_len / poseidon.arity;
      hash_row_device<t>(in_ptr, aux_ptr, out_ptr, poseidon,
                         next_num_sub_trees, next_row_len,
                         first, streams[stream_idx], mont && first);

      fr_t* current_digests_ptr = digests + total_digests_offset +
                                  next_row_len * processed_sub_trees;

      cudaMemcpyAsync(current_digests_ptr, out_ptr,
                      sizeof(fr_t) * next_row_len * next_num_sub_trees,
                      cudaMemcpyDeviceToHost,
                      streams[stream_idx]);

      in_ptr = out_ptr;
      out_ptr += next_row_len * next_num_sub_trees;
      total_digests_offset += digests_segment;
      digests_segment /= poseidon.arity;
    }

    current_leaves_ptr += sub_tree_len * next_num_sub_trees;

    loops++;
  }

  cudaFree(d_leaves);
  cudaFree(d_digests);
  for (size_t i = 0; i < num_streams; i++) {
    cudaStreamDestroy(streams[i]);
  }
  free(streams);
}

extern "C"
void tree_builder_device(fr_t* leaves, const size_t leaves_len, fr_t* digests,
                         const size_t digests_len, const Poseidon& poseidon,
                         const size_t cutoff_leaves_len,
                         const size_t cutoff_digests_len,
                         const size_t sub_tree_len,
                         const size_t sub_digests_len, const bool mont) {

  #define ARGUMENTS leaves, leaves_len, digests, digests_len, poseidon, \
            cutoff_leaves_len, cutoff_digests_len, sub_tree_len, \
            sub_digests_len, mont

  switch(poseidon.arity) {
    case 2:
      tree_builder_device_internal<3>(ARGUMENTS);
      break;
    case 4:
      tree_builder_device_internal<5>(ARGUMENTS);
      break;
    case 8:
      tree_builder_device_internal<9>(ARGUMENTS);
      break;
    case 11:
      tree_builder_device_internal<12>(ARGUMENTS);
      break;
    case 16:
      tree_builder_device_internal<17>(ARGUMENTS);
      break;
    case 24:
      tree_builder_device_internal<25>(ARGUMENTS);
      break;
    case 36:
      tree_builder_device_internal<37>(ARGUMENTS);
      break;
    default:
      // Only arities {2, 4, 8, 11, 16, 24, 36} are supported
      assert(false);
      break;
  }

  #undef ARGUMENTS
}

template <int column_t, int tree_t>
void column_tree_builder_device_internal(fr_t* leaves, const size_t leaves_len,
                                         fr_t* digests,
                                         const size_t digests_len,
                                         const Poseidon& column_poseidon,
                                         const Poseidon& tree_poseidon,
                                         const size_t cutoff_leaves_len,
                                         const size_t cutoff_digests_len,
                                         const size_t sub_tree_len,
                                         const size_t sub_digests_len,
                                         const bool mont) {

  const size_t sub_tree_device_len =
    (sub_tree_len / column_poseidon.arity) * column_t;

  const size_t sub_tree_digests_req_mem =
    sizeof(fr_t) * (sub_tree_device_len + sub_digests_len);

  size_t device_avail_mem, device_total_mem;
  cudaMemGetInfo(&device_avail_mem, &device_total_mem);
  device_avail_mem -= ONE_GiB / 8; // leave 128 MiB free just in case

  const size_t num_sub_trees =
    ONE_GiB / sub_tree_digests_req_mem; // batches of 1 GiB
  const size_t num_streams = device_avail_mem / ONE_GiB;

  const size_t total_num_sub_trees = num_sub_trees * num_streams;

  cudaStream_t* streams =
    static_cast<cudaStream_t*>(malloc(sizeof(cudaStream_t) * num_streams));
  for (size_t i = 0; i < num_streams; i++) {
    cudaStreamCreate(&streams[i]);
  }

  fr_t* d_leaves, * d_digests;
  cudaMalloc(&d_leaves, sizeof(fr_t) *
                        std::min(total_num_sub_trees, cutoff_leaves_len) *
                        sub_tree_device_len);
  cudaMalloc(&d_digests, sizeof(fr_t) *
                         std::min(total_num_sub_trees, cutoff_leaves_len) *
                         sub_digests_len);

  fr_t* current_leaves_ptr = leaves;

  size_t loops = 0;
  for (size_t remaining_sub_trees = cutoff_leaves_len,
    processed_sub_trees = 0;
    remaining_sub_trees > 0;
    processed_sub_trees += std::min(num_sub_trees, remaining_sub_trees),
    remaining_sub_trees -= std::min(num_sub_trees, remaining_sub_trees)) {

    const size_t stream_idx = loops % num_streams;

    const size_t next_num_sub_trees = std::min(num_sub_trees,
                                               remaining_sub_trees);

    fr_t* in_ptr = d_leaves + num_sub_trees * sub_tree_device_len * stream_idx;
    fr_t* aux_ptr = in_ptr;
    fr_t* out_ptr = d_digests + num_sub_trees * sub_digests_len * stream_idx;

    cudaMemcpy2DAsync(in_ptr, sizeof(fr_t) * column_t, current_leaves_ptr,
                      sizeof(fr_t) * column_poseidon.arity,
                      sizeof(fr_t) * column_poseidon.arity,
                      next_num_sub_trees *
                      sub_tree_len / column_poseidon.arity,
                      cudaMemcpyHostToDevice,
                      streams[stream_idx]);


    size_t digests_segment = leaves_len / column_poseidon.arity;
    size_t total_digests_offset = 0;

    {
      size_t next_row_len = sub_tree_len / column_poseidon.arity;

      hash_row_device<column_t>(in_ptr, aux_ptr, out_ptr, column_poseidon,
                                next_num_sub_trees, next_row_len, true,
                                streams[stream_idx], mont);

      fr_t* current_digests_ptr = digests + total_digests_offset +
                                  next_row_len * processed_sub_trees;

      cudaMemcpyAsync(current_digests_ptr, out_ptr,
                      sizeof(fr_t) * next_row_len * next_num_sub_trees,
                      cudaMemcpyDeviceToHost, streams[stream_idx]);

      in_ptr = out_ptr;
      out_ptr += next_row_len * next_num_sub_trees;
      total_digests_offset += digests_segment;
      digests_segment /= tree_poseidon.arity;
    }

    for (size_t next_row_len =
       sub_tree_len / column_poseidon.arity / tree_poseidon.arity;
       next_row_len > 0;
       next_row_len /= tree_poseidon.arity) {

      hash_row_device<tree_t>(in_ptr, aux_ptr, out_ptr, tree_poseidon,
                              next_num_sub_trees, next_row_len, false,
                              streams[stream_idx], false);

      fr_t* current_digests_ptr = digests + total_digests_offset +
                                  next_row_len * processed_sub_trees;

      cudaMemcpyAsync(current_digests_ptr, out_ptr,
                      sizeof(fr_t) * next_row_len * next_num_sub_trees,
                      cudaMemcpyDeviceToHost,
                      streams[stream_idx]);

      in_ptr = out_ptr;
      out_ptr += next_row_len * next_num_sub_trees;
      total_digests_offset += digests_segment;
      digests_segment /= tree_poseidon.arity;
    }

    current_leaves_ptr += sub_tree_len * next_num_sub_trees;

    loops++;
  }

  cudaFree(d_leaves);
  cudaFree(d_digests);
  for (size_t i = 0; i < num_streams; i++) {
    cudaStreamDestroy(streams[i]);
  }
  free(streams);
}

template <int column_t>
void column_tree_builder_device_1(fr_t* leaves, const size_t leaves_len,
                                  fr_t* digests, const size_t digests_len,
                                  const Poseidon& column_poseidon,
                                  const Poseidon& tree_poseidon,
                                  const size_t cutoff_leaves_len,
                                  const size_t cutoff_digests_len,
                                  const size_t sub_tree_len,
                                  const size_t sub_digests_len,
                                  const bool mont) {

  #define ARGUMENTS leaves, leaves_len, digests, digests_len, \
            column_poseidon, tree_poseidon, cutoff_leaves_len, \
            cutoff_digests_len, sub_tree_len, sub_digests_len, \
            mont

  switch(tree_poseidon.arity) {
    case 2:
      column_tree_builder_device_internal<column_t, 3>(ARGUMENTS);
      break;
    case 4:
      column_tree_builder_device_internal<column_t, 5>(ARGUMENTS);
      break;
    case 8:
      column_tree_builder_device_internal<column_t, 9>(ARGUMENTS);
      break;
    case 11:
      column_tree_builder_device_internal<column_t, 12>(ARGUMENTS);
      break;
    case 16:
      column_tree_builder_device_internal<column_t, 17>(ARGUMENTS);
      break;
    case 24:
      column_tree_builder_device_internal<column_t, 25>(ARGUMENTS);
      break;
    case 36:
      column_tree_builder_device_internal<column_t, 37>(ARGUMENTS);
      break;
    default:
      // Only arities {2, 4, 8, 11, 16, 24, 36} are supported
      assert(false);
      break;
  }

  #undef ARGUMENTS
}

extern "C"
void column_tree_builder_device(fr_t* leaves, const size_t leaves_len,
                                fr_t* digests, const size_t digests_len,
                                const Poseidon& column_poseidon,
                                const Poseidon& tree_poseidon,
                                const size_t cutoff_leaves_len,
                                const size_t cutoff_digests_len,
                                const size_t sub_tree_len,
                                const size_t sub_digests_len,
                                const bool mont) {

  #define ARGUMENTS leaves, leaves_len, digests, digests_len, \
            column_poseidon, tree_poseidon, cutoff_leaves_len, \
            cutoff_digests_len, sub_tree_len, sub_digests_len, \
            mont

  switch(column_poseidon.arity) {
    case 2:
      column_tree_builder_device_1<3>(ARGUMENTS);
      break;
    case 4:
      column_tree_builder_device_1<5>(ARGUMENTS);
      break;
    case 8:
      column_tree_builder_device_1<9>(ARGUMENTS);
      break;
    case 11:
      column_tree_builder_device_1<12>(ARGUMENTS);
      break;
    case 16:
      column_tree_builder_device_1<17>(ARGUMENTS);
      break;
    case 24:
      column_tree_builder_device_1<25>(ARGUMENTS);
      break;
    case 36:
      column_tree_builder_device_1<37>(ARGUMENTS);
      break;
    default:
      // Only arities {2, 4, 8, 11, 16, 24, 36} are supported
      assert(false);
      break;
  }

  #undef ARGUMENTS
}
