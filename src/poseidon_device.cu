// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "poseidon.cu"

extern __shared__ fr_t scratchpad[];

__device__ __forceinline__
fr_t pow_5(const fr_t& element) {
  fr_t tmp = sqr(element);
  tmp = sqr(tmp);
  return element * tmp;
}

__device__ __forceinline__
void quintic_s_box(fr_t& element, const fr_t& round_constant) {
  element = pow_5(element);
  element += round_constant;
}

__device__ __forceinline__
void partial_quintic_s_box(fr_t& element) {
  element = pow_5(element);
}

__device__ __forceinline__
void add_full_round_constants(fr_t& element, const fr_t& round_constant) {
  element += round_constant;
}

__device__ __forceinline__
void matrix_mul(fr_t& element, const fr_t* matrix, const int t,
                const int thread_pos, const int shared_pos) {
  scratchpad[threadIdx.x] = element;
  __syncthreads();

  element = scratchpad[shared_pos] * matrix[thread_pos];

  for (int j = 1; j < t; j++) {
    element += scratchpad[shared_pos + j] * matrix[j * t + thread_pos];
  }
  __syncthreads();
}

__device__ __forceinline__
fr_t last_matrix_mul(const fr_t* elements, const fr_t* matrix, const int t) {
  fr_t tmp = elements[0] * matrix[1];

  #pragma unroll
  for (int j = 1; j < t; j++) {
    tmp += elements[j] * matrix[j * t + 1];
  }

  return tmp;
}

__device__ __forceinline__
void scalar_product(fr_t* elements, const fr_t* sparse_matrix, const int t) {
  elements[0] *= sparse_matrix[0];

  #pragma unroll
  for (int i = 1; i < t; i++) {
    elements[0] += elements[i] * sparse_matrix[i];
  }
}

__device__ __forceinline__
void sparse_matrix_mul(fr_t* elements, const fr_t* sparse_matrix,
                       const int t) {
  fr_t element0 = elements[0];

  scalar_product(elements, sparse_matrix, t);

  #pragma unroll
  for (int i = 1; i < t; i++) {
    elements[i] += element0 * sparse_matrix[t + i - 1];
  }
}

__device__ __forceinline__
void round_matrix_mul(fr_t& element, const PoseidonConstantsDevice constants,
                      const int current_round, const int thread_pos,
                      const int shared_pos) {
  if (current_round == constants.half_full_rounds - 1) {
    matrix_mul(element, constants.pre_sparse_matrix, constants.t,
               thread_pos, shared_pos);
  }
  else {
    matrix_mul(element, constants.mds_matrix, constants.t, thread_pos,
               shared_pos);
  }
}

__device__ __forceinline__
void full_round(fr_t& element, const PoseidonConstantsDevice constants,
                int& rk_offset, int& current_round, const int thread_pos,
                const int shared_pos) {
  quintic_s_box(element, constants.round_constants[rk_offset]);
  rk_offset += constants.t;

  round_matrix_mul(element, constants, current_round, thread_pos, shared_pos);
  current_round++;
}

__device__ __forceinline__
void partial_round(fr_t* elements, const int t,
                   const PoseidonConstantsDevice constants,
                   int& rk_offset, int& current_round) {
  quintic_s_box(elements[0], constants.round_constants[rk_offset]);
  rk_offset += 1;

  sparse_matrix_mul(elements,constants.sparse_matrices +
                    (t * 2 - 1) *
                    (current_round - constants.half_full_rounds), t);
  current_round++;
}


__global__
void poseidon_hash_1_0(const fr_t* in_ptr, fr_t* aux_ptr,
                       const fr_t domain_tag,
                       const PoseidonConstantsDevice constants, int rk_offset,
                       int current_round, const int batch_size,
                       const bool mont) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= batch_size) {
    return;
  }

  int thread_pos = threadIdx.x % constants.t;
  int shared_pos = (threadIdx.x / constants.t) * constants.t;
  idx = blockIdx.x * (blockDim.x / constants.t) + threadIdx.x / constants.t;

  fr_t element;

  if (thread_pos == 0) {
    element = domain_tag;
  }
  else {
    element = in_ptr[idx * constants.t + thread_pos - 1];
    if (mont) {
      element.to();
    }
  }

  rk_offset += thread_pos;

  add_full_round_constants(element, constants.round_constants[rk_offset]);
  rk_offset += constants.t;

  for (int i = 0; i < constants.half_full_rounds; i++) {
    full_round(element, constants, rk_offset, current_round, thread_pos,
               shared_pos);
  }

  aux_ptr[idx * constants.t + thread_pos] = element;
}

__global__
void poseidon_hash_1_1(const fr_t* in_ptr, fr_t* aux_ptr,
                       const fr_t domain_tag,
                       const PoseidonConstantsDevice constants, int rk_offset,
                       int current_round, const int batch_size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= batch_size) {
    return;
  }

  int thread_pos = threadIdx.x % constants.t;
  int shared_pos = (threadIdx.x / constants.t) * constants.t;
  idx = blockIdx.x * (blockDim.x / constants.t) + threadIdx.x / constants.t;

  fr_t element;

  if (thread_pos == 0) {
    element = domain_tag;
  }
  else {
    element = in_ptr[idx * (constants.t - 1) + thread_pos - 1];
  }

  rk_offset += thread_pos;

  add_full_round_constants(element, constants.round_constants[rk_offset]);
  rk_offset += constants.t;

  for (int i = 0; i < constants.half_full_rounds; i++) {
    full_round(element, constants, rk_offset, current_round, thread_pos,
               shared_pos);
  }

  aux_ptr[idx * constants.t + thread_pos] = element;
}

template<int t> __global__
void poseidon_hash_2(fr_t* aux_ptr, const PoseidonConstantsDevice constants,
                     int rk_offset, int current_round, const int batch_size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= batch_size) {
    return;
  }

  aux_ptr += idx * t;

  fr_t elements[t];

  for (int i = 0; i < t; i++) {
    elements[i] = aux_ptr[i];
  }

  for (int i = 0; i < constants.partial_rounds; i++) {
    partial_round(elements, t, constants, rk_offset, current_round);
  }

  for (int i = 0; i < t; i++) {
    aux_ptr[i] = elements[i];
  }
}

__global__
void poseidon_hash_3(fr_t* aux_ptr, const PoseidonConstantsDevice constants,
                     int rk_offset, int current_round, const int batch_size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= batch_size) {
    return;
  }

  int thread_pos = threadIdx.x % constants.t;
  int shared_pos = (threadIdx.x / constants.t) * constants.t;
  idx = blockIdx.x * (blockDim.x / constants.t) + threadIdx.x / constants.t;

  rk_offset += thread_pos;

  fr_t element = aux_ptr[idx * constants.t + thread_pos];

  for (int i = 0; i < constants.half_full_rounds - 1; i++) {
    full_round(element, constants, rk_offset, current_round, thread_pos,
               shared_pos);
  }

  partial_quintic_s_box(element);

  aux_ptr[idx * constants.t + thread_pos] = element;
}

template<int t> __global__
void poseidon_hash_4(const fr_t* aux_ptr, fr_t* out_ptr,
                     const fr_t* mds_matrix, const int batch_size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= batch_size) {
    return;
  }

  aux_ptr += idx * t;

  out_ptr[idx] = last_matrix_mul(aux_ptr, mds_matrix, t);
}
