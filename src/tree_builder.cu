// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "poseidon.cu"

extern "C"
void tree_builder_device(fr_t*, const size_t, fr_t*, const size_t,
                         const Poseidon&, const size_t, const size_t,
                         const size_t, const size_t, const bool);

extern "C"
void column_tree_builder_device(fr_t*, const size_t, fr_t*, const size_t,
                                const Poseidon&, const Poseidon&, const size_t,
                                const size_t, const size_t, const size_t,
                                const bool);

#ifndef __CUDA_ARCH__

#include <math.h>
#include <omp.h>

#include <iostream>
#include <algorithm>

#include "poseidon_host.cpp"

class TreeBuilder {
public:
  TreeBuilder (const int arity): poseidon(arity) {
    this->cutoff_leaves_len =
      static_cast<size_t>(pow(static_cast<double>(arity), cutoff_rows));
    this->cutoff_digests_len =
      calc_digests_len(this->cutoff_leaves_len, arity);
  }

  void build_tree_with_preimages(fr_t* leaves, const size_t leaves_len,
                                 fr_t* digests, const bool mont) {

    size_t digests_len = calc_digests_len(leaves_len, this->poseidon.arity);

    bool cuda = leaves_len / this->poseidon.arity >= this->cutoff_leaves_len;

    size_t next_row_len = leaves_len / this->poseidon.arity;

    if (cuda) {
      const size_t sub_tree_len = leaves_len / this->cutoff_leaves_len;
      const size_t sub_digests_len = calc_digests_len(sub_tree_len,
                                                      this->poseidon.arity);
      tree_builder_device(leaves, leaves_len, digests, digests_len,
                          this->poseidon, this->cutoff_leaves_len,
                          this->cutoff_digests_len, sub_tree_len,
                          sub_digests_len, mont);
    }

    fr_t* in_ptr, * out_ptr;

    if (cuda) {
      next_row_len = cutoff_leaves_len / this->poseidon.arity;
      in_ptr = digests + digests_len - this->cutoff_digests_len -
               this->cutoff_leaves_len;
      out_ptr = digests + digests_len - this->cutoff_digests_len;
    }
    else {
      in_ptr = leaves;
      out_ptr = digests;
    }

    this->build_tree_with_preimages_host(in_ptr, out_ptr, next_row_len);
  }

  static size_t calc_digests_len(size_t leaves_len, const int _arity) {
    assert(leaves_len != 0);

    size_t arity = static_cast<size_t>(_arity);

    size_t digests_len = 0;

    while (leaves_len > 1) {
      assert(leaves_len % arity == 0);
      leaves_len /= arity;
      digests_len += leaves_len;
    }

    return digests_len;
  }

private:
  Poseidon poseidon;
  size_t cutoff_leaves_len, cutoff_digests_len;
  static const int cutoff_rows = 1;

  void build_tree_with_preimages_host(fr_t*& in_ptr, fr_t*& out_ptr,
                                      size_t& next_row_len) {
    while (next_row_len > 1) {
      this->hash_row_with_preimages_host(in_ptr, out_ptr, next_row_len);
    }

    poseidon_hash(in_ptr, out_ptr, this->poseidon);
  }

  void hash_row_with_preimages_host(fr_t*& in_ptr, fr_t*& out_ptr,
                                    size_t& next_row_len) {
    #pragma omp parallel for
    for (size_t i = 0; i < next_row_len; i++) {
      poseidon_hash(in_ptr + i * this->poseidon.arity, out_ptr + i,
                    this->poseidon);
    }

    in_ptr = out_ptr;
    out_ptr += next_row_len;
    next_row_len /= this->poseidon.arity;
  }
};

class ColumnTreeBuilder {

public:

  ColumnTreeBuilder (const int column_arity, const int tree_arity):
    column_poseidon(column_arity), tree_poseidon(tree_arity) {

    this->cutoff_leaves_len =
      static_cast<size_t>(pow(static_cast<double>(tree_arity), cutoff_rows));
    this->cutoff_digests_len = calc_digests_len(this->cutoff_leaves_len,
                                                tree_arity);
  }

  void build_column_tree_with_preimages(fr_t* leaves, const size_t leaves_len,
                                        fr_t* digests, const bool mont) {
    size_t column_digests_len =
      calc_column_digests_len(leaves_len,
                              this->column_poseidon.arity,
                              this->tree_poseidon.arity);

    bool cuda =
      leaves_len / this->column_poseidon.arity >= this->cutoff_leaves_len;

    size_t next_row_len = leaves_len / this->column_poseidon.arity;

    if (cuda) {
      const size_t sub_tree_len = leaves_len / this->cutoff_leaves_len;
      const size_t sub_digests_len =
        calc_column_digests_len(sub_tree_len,
                                this->column_poseidon.arity,
                                this->tree_poseidon.arity);
      column_tree_builder_device(leaves, leaves_len, digests,
                                 column_digests_len,
                                 this->column_poseidon,
                                 this->tree_poseidon,
                                 this->cutoff_leaves_len,
                                 this->cutoff_digests_len, sub_tree_len,
                                 sub_digests_len, mont);
    }

    fr_t* in_ptr, * out_ptr;

    if (cuda) {
      next_row_len = cutoff_leaves_len / this->tree_poseidon.arity;
      in_ptr = digests + column_digests_len - this->cutoff_digests_len -
               this->cutoff_leaves_len;
      out_ptr = digests + column_digests_len - this->cutoff_digests_len;
    }
    else {
      in_ptr = leaves;
      out_ptr = digests;
    }

    this->build_column_tree_with_preimages_host(in_ptr, out_ptr,
                                                next_row_len, !cuda);
  }

  static size_t calc_digests_len(size_t leaves_len, const int _arity) {
    assert(leaves_len != 0);

    size_t arity = static_cast<size_t>(_arity);

    size_t digests_len = 0;

    while (leaves_len > 1) {
      assert(leaves_len % arity == 0);
      leaves_len /= arity;
      digests_len += leaves_len;
    }

    return digests_len;
  }

  static size_t calc_column_digests_len(size_t leaves_len,
                                        const int _column_arity,
                                        const int _tree_arity) {
    assert(leaves_len != 0);

    size_t column_arity = _column_arity;
    size_t tree_arity = _tree_arity;

    size_t digests_len = 0;

    assert(leaves_len % column_arity == 0);
    leaves_len /= column_arity;
    digests_len += leaves_len;

    digests_len += calc_digests_len(leaves_len, _tree_arity);

    return digests_len;
  }

private:
  Poseidon column_poseidon, tree_poseidon;
  size_t cutoff_leaves_len, cutoff_digests_len;
  static const int cutoff_rows = 1;

  void build_column_tree_with_preimages_host(fr_t*& in_ptr, fr_t*& out_ptr,
                                             size_t& next_row_len,
                                             const bool column) {
    if (column) {
      this->hash_row_with_preimages_host(in_ptr, out_ptr, next_row_len,
                                         this->column_poseidon);
    }

    while (next_row_len > 1) {
      this->hash_row_with_preimages_host(in_ptr, out_ptr, next_row_len,
                                         this->tree_poseidon);
    }

    poseidon_hash(in_ptr, out_ptr, this->tree_poseidon);
  }

  void hash_row_with_preimages_host(fr_t*& in_ptr, fr_t*& out_ptr,
                                    size_t& next_row_len,
                                    const Poseidon& poseidon) {
    #pragma omp parallel for
    for (size_t i = 0; i < next_row_len; i++) {
      poseidon_hash(in_ptr + i * poseidon.arity, out_ptr + i, poseidon);
    }

    in_ptr = out_ptr;
    out_ptr += next_row_len;
    next_row_len /= poseidon.arity;
  }
};

#endif
