// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

inline fr_t pow_5(const fr_t& element) {
  fr_t tmp = sqr(element);
  tmp = sqr(tmp);
  return element * tmp;
}

inline void quintic_s_box(fr_t& element, const fr_t& round_constant) {
  element = pow_5(element);
  element += round_constant;
}

inline void partial_quintic_s_box(fr_t& element) {
  element = pow_5(element);
}

inline void add_full_round_constants(fr_t* elements,
                                     const fr_t* round_constants,
                                     const int t) {
  for (int i = 0; i < t; i++) {
    elements[i] += round_constants[i];
  }
}

inline void matrix_mul(fr_t* elements, const fr_t* matrix, const int t) {
  fr_t* tmp = static_cast<fr_t*>(malloc(sizeof(fr_t) * t));

  for (int i = 0; i < t; i++) {
    tmp[i] = elements[0] * matrix[i];

    for (int j = 1; j < t; j++) {
      tmp[i] += elements[j] * matrix[j * t + i];
    }
  }

  for (int i = 0; i < t; i++) {
    elements[i] = tmp[i];
  }

  free(tmp);
}

inline void last_matrix_mul(fr_t* elements, const fr_t* matrix, const int t) {
  fr_t tmp = elements[0] * matrix[1];

  for (int j = 1; j < t; j++) {
    tmp += elements[j] * matrix[j * t + 1];
  }

  elements[1] = tmp;
}

inline void scalar_product(fr_t* elements, const fr_t* sparse_matrix,
                           const int t) {
  elements[0] *= sparse_matrix[0];

  for (int i = 1; i < t; i++) {
    elements[0] += elements[i] * sparse_matrix[i];
  }
}

inline void sparse_matrix_mul(fr_t* elements, const fr_t* sparse_matrix,
                              const int t) {
  fr_t element0 = elements[0];

  scalar_product(elements, sparse_matrix, t);

  for (int i = 1; i < t; i++) {
    elements[i] += element0 * sparse_matrix[t + i - 1];
  }
}

inline void round_matrix_mul(fr_t* elements, const Poseidon& poseidon,
                             const int current_round) {
  if (current_round == 3) {
    matrix_mul(elements, poseidon.poseidon_constants.pre_sparse_matrix,
               poseidon.t);
  }
  else if ((current_round > 3) &&
           (current_round < poseidon.half_full_rounds +
            poseidon.partial_rounds)) {
    int index = current_round - poseidon.half_full_rounds;
    sparse_matrix_mul(elements,
                      poseidon.poseidon_constants.sparse_matrices +
                      (poseidon.t * 2 - 1) * index, poseidon.t);
  }
  else {
    matrix_mul(elements, poseidon.poseidon_constants.mds_matrix, poseidon.t);
  }
}

inline void full_round(fr_t* elements, const Poseidon& poseidon,
                       int& rk_offset, int& current_round) {
  for (int i = 0; i < poseidon.t; i++) {
    quintic_s_box(elements[i],
                  poseidon.poseidon_constants.round_constants[rk_offset + i]);
  }
  rk_offset += poseidon.t;

  round_matrix_mul(elements, poseidon, current_round);
  current_round++;
}

inline void last_full_round(fr_t* elements, const fr_t* mds_matrix,
                            const int t) {
  for (int i = 0; i < t; i++) {
    partial_quintic_s_box(elements[i]);
  }

  // Neptune returns the 2nd element of the resulting t element array as the
  // digest. We perform multiplication only on a single row instead of the
  // entire matrix.
  matrix_mul(elements, mds_matrix, t);
}

inline void partial_round(fr_t* elements, const Poseidon& poseidon,
                          int& rk_offset, int& current_round) {
  quintic_s_box(elements[0],
                poseidon.poseidon_constants.round_constants[rk_offset]);
  rk_offset += 1;

  round_matrix_mul(elements, poseidon, current_round);
  current_round++;
}


inline void poseidon_hash(const fr_t* in_ptr, fr_t* out_ptr,
                          const Poseidon& poseidon) {
  fr_t* elements = static_cast<fr_t*>(malloc(sizeof(fr_t) * poseidon.t));

  elements[0] = poseidon.poseidon_constants.domain_tag;

  for (int i = 0; i < poseidon.t - 1; i++) {
    elements[i + 1] = in_ptr[i];
  }

  add_full_round_constants(elements,
                           poseidon.poseidon_constants.round_constants,
                           poseidon.t);

  int rk_offset = poseidon.t;
  int current_round = 0;

  for (int i = 0; i < poseidon.half_full_rounds; i++) {
    full_round(elements, poseidon, rk_offset, current_round);
  }

  for (int i = 0; i < poseidon.partial_rounds; i++) {
    partial_round(elements, poseidon, rk_offset, current_round);
  }

  for (int i = 0; i < poseidon.half_full_rounds - 1; i++) {
    full_round(elements, poseidon, rk_offset, current_round);
  }

  last_full_round(elements, poseidon.poseidon_constants.mds_matrix,
          poseidon.t);

  out_ptr[0] = elements[1];

  free(elements);
}
