// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <map>

#include <bls12-381.hpp>
#include "../utils/parser.cpp"

static int get_partial_rounds(const int arity) {
  const std::map<int, int> partial_rounds_map = {
    {2, 55},
    {4, 56},
    {8, 57},
    {11, 57},
    {16, 59},
    {24, 59},
    {36, 60}
  };

  std::map<int, int>::const_iterator partial_rounds =
    partial_rounds_map.find(arity);
  assert(partial_rounds != partial_rounds_map.end());

  return partial_rounds->second;
}

struct PoseidonConstantsDevice {

  int t, partial_rounds, half_full_rounds;
  fr_t *round_constants, *mds_matrix, *pre_sparse_matrix, *sparse_matrices;
};

class PoseidonConstants {
public:
  fr_t domain_tag; // 2^arity - 1
  fr_t *round_constants, *mds_matrix, *pre_sparse_matrix, *sparse_matrices;
  PoseidonConstantsDevice poseidon_constants_device;

  PoseidonConstants(const int arity,
                    const int partial_rounds,
                    const int half_full_rounds = 4) {

    const int t = arity + 1;

    this->concatenated_constants = read_constants_from_file(arity);
    this->domain_tag = calculate_domain_tag(arity);

    this->round_constants_len = t * half_full_rounds * 2 + partial_rounds;
    this->mds_matrix_len = t * t;
    this->pre_sparse_matrix_len = this->mds_matrix_len;
    this->sparse_matrices_len = (t * 2 - 1) * partial_rounds;

    this->round_constants_offset = 0;
    this->mds_matrix_offset = this->round_constants_len;
    this->pre_sparse_matrix_offset = this->round_constants_len +
                                     this->mds_matrix_len;
    this->sparse_matrices_offset = this->round_constants_len +
                                   this->mds_matrix_len +
                                   this->pre_sparse_matrix_len;

    this->round_constants = this->concatenated_constants;
    this->mds_matrix = this->concatenated_constants +
                       this->mds_matrix_offset;
    this->pre_sparse_matrix = this->concatenated_constants +
                              this->pre_sparse_matrix_offset;
    this->sparse_matrices = this->concatenated_constants +
                            this->sparse_matrices_offset;

    this->poseidon_constants_device.t = t;
    this->poseidon_constants_device.partial_rounds = partial_rounds;
    this->poseidon_constants_device.half_full_rounds = half_full_rounds;

    cudaMalloc(&this->poseidon_constants_device.round_constants,
               sizeof(fr_t) * this->round_constants_len);
    cudaMalloc(&this->poseidon_constants_device.mds_matrix,
               sizeof(fr_t) * this->mds_matrix_len);
    cudaMalloc(&this->poseidon_constants_device.pre_sparse_matrix,
               sizeof(fr_t) * this->pre_sparse_matrix_len);
    cudaMalloc(&this->poseidon_constants_device.sparse_matrices,
               sizeof(fr_t) * this->sparse_matrices_len);

    cudaMemcpy(this->poseidon_constants_device.round_constants,
               this->round_constants,
               sizeof(fr_t) * this->round_constants_len,
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->poseidon_constants_device.mds_matrix,
               this->mds_matrix, sizeof(fr_t) * this->mds_matrix_len,
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->poseidon_constants_device.pre_sparse_matrix,
               this->pre_sparse_matrix,
               sizeof(fr_t) * this->pre_sparse_matrix_len,
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->poseidon_constants_device.sparse_matrices,
               this->sparse_matrices,
               sizeof(fr_t) * this->sparse_matrices_len,
               cudaMemcpyHostToDevice);
  }

  ~PoseidonConstants() {
    free(concatenated_constants);
    cudaFree(this->poseidon_constants_device.round_constants);
    cudaFree(this->poseidon_constants_device.mds_matrix);
    cudaFree(this->poseidon_constants_device.pre_sparse_matrix);
    cudaFree(this->poseidon_constants_device.sparse_matrices);
  }

private:
  // Main array where all constants are stored. Host pointers for constants
  // point to locations inside this array.
  fr_t* concatenated_constants;
  int round_constants_len, mds_matrix_len;
  int pre_sparse_matrix_len, sparse_matrices_len;
  size_t round_constants_offset, mds_matrix_offset;
  size_t pre_sparse_matrix_offset, sparse_matrices_offset;

  static fr_t calculate_domain_tag(const int arity) {
    fr_t domain_tag;
    domain_tag = domain_tag.one();
    domain_tag = (domain_tag << arity) - domain_tag;
    return domain_tag;
  }
};

class Poseidon {
public:
  const int half_full_rounds = 4;
  int arity, t, partial_rounds;
  PoseidonConstants poseidon_constants;

  Poseidon(const int arity):
    poseidon_constants(arity, get_partial_rounds(arity)) {

    this->arity = arity;
    this->t = arity + 1;
    this->partial_rounds = get_partial_rounds(arity);
  }
};
