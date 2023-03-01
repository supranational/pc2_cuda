// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <all_gpus.cpp>

#ifndef __CUDA_ARCH__

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>

#include "tree_builder.cu"
#include "parameters.cpp"
#include "thread_pool_t.hpp"

static const size_t page_size = sysconf(_SC_PAGE_SIZE);

std::string get_layer_file_name(const int layer_idx) {
  return "/sc-02-data-layer-" + std::to_string(layer_idx + 1) + ".dat";
}

std::string get_tree_c_file_name(const int config_idx, const int configs) {
  return "/sc-02-data-tree-c" +
    (configs == 1 ? "" : "-" + std::to_string(config_idx)) + ".dat";
}

std::string get_tree_r_file_name(const int config_idx, const int configs) {
  return "/sc-02-data-tree-r-last" +
    (configs == 1 ? "" : "-" + std::to_string(config_idx)) + ".dat";
}

void read_file(const size_t file_size, fr_t* ptr, const std::string file_name,
               const size_t offset = 0) {
  int file = open(file_name.c_str(), O_RDONLY);
  fr_t* file_ptr = reinterpret_cast<fr_t*>(mmap(NULL, file_size,
    PROT_READ, MAP_PRIVATE, file, offset));

  #pragma omp parallel for
  for (size_t i = 0; i < file_size / sizeof(fr_t); i++) {
    ptr[i] = file_ptr[i];
  }

  munmap(reinterpret_cast<char*>(file_ptr), file_size);

  close(file);
}

void write_to_file(const std::string file_name, const fr_t* ptr,
                   const size_t len) {
  int file = open(file_name.c_str(), O_CREAT | O_RDWR,
    S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);

  posix_fallocate(file, 0, sizeof(fr_t) * len);

  fr_t* file_ptr = reinterpret_cast<fr_t*>(mmap(NULL, sizeof(fr_t) * len,
    PROT_WRITE, MAP_SHARED, file, 0));

  #pragma omp parallel for
  for (size_t i = 0; i < len; i++) {
    fr_t tmp = ptr[i];
    tmp.from();
    file_ptr[i] = tmp;
  }

  munmap(reinterpret_cast<char*>(file_ptr), sizeof(fr_t) * len);

  close(file);
}

inline void tree_c_prepare_data(fr_t* leaves, const size_t sector_size,
                                const int configs, const int config_idx,
                                const int layers, const int layer_idx,
                                const int batches, const int batch_idx,
                                std::string cache_path) {
  size_t bytes_to_read = sector_size / configs / batches;
  const size_t leaves_to_read = bytes_to_read / sizeof(fr_t);
  size_t file_offset = (sector_size / configs) * config_idx +
             bytes_to_read * batch_idx;

  size_t fallback_offset = 0;
  if (file_offset % page_size != 0) {
    fallback_offset = file_offset / sizeof(fr_t);
    file_offset   = 0;
    bytes_to_read   = sector_size;
  }

  const std::string layer_file = cache_path + get_layer_file_name(layer_idx);
  int file = open(layer_file.c_str(), O_RDONLY);

  fr_t* file_ptr = reinterpret_cast<fr_t*>(mmap(NULL,
    bytes_to_read, PROT_READ, MAP_PRIVATE, file, file_offset));

  #pragma omp parallel for
  for (size_t i = 0; i < leaves_to_read; i++) {
    leaves[i * layers + layer_idx] = file_ptr[fallback_offset + i];
  }

  munmap(reinterpret_cast<char*>(file_ptr), bytes_to_read);
  close(file);
}

void write_tree_c_to_disk(const int config_idx, const int configs,
                          const fr_t* digests, const size_t digests_len,
                          const size_t partial_digests_len,
                          const int batch_idx, const int batches,
                          const size_t _offset_start, const int tree_arity,
                          std::string output_path) {
  const std::string output_file =
    output_path + get_tree_c_file_name(config_idx, configs);

  int file = open(output_file.c_str(), O_CREAT | O_RDWR,
    S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);

  if (batch_idx == 0)
    posix_fallocate(file, 0, sizeof(fr_t) * digests_len);

  size_t map_size =
    digests_len == partial_digests_len ? digests_len : digests_len - 1;

  fr_t* file_ptr = reinterpret_cast<fr_t*>(mmap(NULL,
    sizeof(fr_t) * map_size, PROT_WRITE, MAP_SHARED, file, 0));

  if (digests_len == partial_digests_len) {
    #pragma omp parallel for
    for (size_t i = 0; i < partial_digests_len; i++) {
      fr_t tmp = digests[i];
      tmp.from();
      file_ptr[i] = tmp;
    }
  }
  else {
    #pragma omp parallel for
    for (size_t i = 0; i < partial_digests_len; i++) {
      size_t file_offset = 0;

      size_t offset_end = i;
      size_t offset_start = _offset_start;
      while (offset_end >= offset_start / tree_arity) {
        file_offset += offset_start;
        offset_end %= offset_start / tree_arity;
        offset_start /= tree_arity;
      }
      file_offset += offset_end + offset_start * batch_idx / batches;

      fr_t tmp = digests[i];
      tmp.from();
      file_ptr[file_offset] = tmp;
    }
  }

  munmap(reinterpret_cast<char*>(file_ptr), sizeof(fr_t) * map_size);

  close(file);
}

void write_tree_c_root_to_disk(const int config_idx, const int configs,
                               const size_t digests_len, const fr_t& root,
                               std::string output_path) {
  const std::string output_file =
    output_path + get_tree_c_file_name(config_idx, configs);

  int file = open(output_file.c_str(), O_RDWR);

  fr_t* file_ptr = reinterpret_cast<fr_t*>(mmap(NULL,
    sizeof(fr_t) * digests_len, PROT_WRITE, MAP_SHARED, file, 0));

  fr_t tmp = root;
  tmp.from();
  file_ptr[digests_len - 1] = tmp;

  munmap(reinterpret_cast<char*>(file_ptr), sizeof(fr_t) * digests_len);

  close(file);
}

void tree_r_prepare_data(fr_t* leaves, const size_t sector_size,
                         const int configs, const int config_idx,
                         const int layers, std::string cache_path) {
  size_t bytes_to_read = sector_size / configs;
  const size_t leaves_to_read = bytes_to_read / sizeof(fr_t);
  size_t file_offset = bytes_to_read * config_idx;

  size_t fallback_offset = 0;
  if (file_offset % page_size != 0) {
    fallback_offset = file_offset / sizeof(fr_t);
    file_offset   = 0;
    bytes_to_read   = sector_size;
  }

  const std::string layer_file = cache_path + get_layer_file_name(layers - 1);
  int file = open(layer_file.c_str(), O_RDONLY);
  fr_t* file_ptr = reinterpret_cast<fr_t*>(mmap(NULL, bytes_to_read,
    PROT_READ, MAP_PRIVATE, file, file_offset));

  #pragma omp parallel for
  for (size_t i = 0; i < leaves_to_read; i++) {
    leaves[i] = file_ptr[fallback_offset + i];
  }

  munmap(reinterpret_cast<char*>(file_ptr), bytes_to_read);

  close(file);
}

void write_tree_r_to_disk(const int config_idx, const int configs,
                          const fr_t* digests, const size_t digests_len,
                          std::string output_path) {
  const std::string output_file =
    output_path + get_tree_r_file_name(config_idx, configs);

  write_to_file(output_file, digests, digests_len);
}

extern "C"
void pc2(const SectorParameters sector_parameters, fr_t* leaves, fr_t* digests,
         fr_t* digests_r, fr_t roots[3],
         std::string output_path, std::string cache_path) {
  const gpu_t& gpu = select_gpu((size_t)0);

  const size_t sector_size   = sector_parameters.sector_size;
  const int  layers          = sector_parameters.layers;
  const int  column_arity    = sector_parameters.column_arity;
  const int  tree_arity      = sector_parameters.tree_arity;
  const int  configs         = sector_parameters.configs;
  const int  rows_to_discard = sector_parameters.rows_to_discard;

  const size_t leaves_len    = sector_size / sizeof(fr_t);
  const size_t tree_c_leaves_len = leaves_len * layers / configs;
  const size_t tree_r_leaves_len = leaves_len      / configs;

  ColumnTreeBuilder column_tree_builder(column_arity, tree_arity);
  TreeBuilder tree_builder(tree_arity);

  const size_t tree_c_digests_len =
    column_tree_builder.calc_column_digests_len(tree_c_leaves_len,
                                                column_arity, tree_arity);
  const size_t tree_r_digests_len =
    tree_builder.calc_digests_len(tree_r_leaves_len, tree_arity);

  size_t batches = tree_arity;
  size_t tree_c_batch_leaves_len = tree_c_leaves_len / batches;

  size_t tree_c_batch_digests_len =
    column_tree_builder.calc_column_digests_len(tree_c_batch_leaves_len,
                                                column_arity, tree_arity);

  size_t rows_to_discard_offset = 0;
  for (size_t i = 0; i < rows_to_discard; i++) {
    rows_to_discard_offset +=
      tree_r_leaves_len / pow(static_cast<double>(tree_arity), i + 1);
  }

  fr_t* tree_c_2nd_last_row = (fr_t*)malloc(configs * sizeof(fr_t));
  fr_t* tree_r_2nd_last_row = (fr_t*)malloc(configs * sizeof(fr_t));

  fr_t* tree_c_merge = (fr_t*)malloc((batches * configs) * sizeof(fr_t));

  channel_t<fr_t*> tree_c_memory_channel, tree_c_compute_channel;
  channel_t<fr_t*> tree_r_memory_channel, tree_r_compute_channel;
  channel_t<int>   file_write_channel, all_complete;

  gpu.spawn([&, leaves]() {
    for (int batch_idx = 0; batch_idx < batches; batch_idx++) {
      for (int layer_idx = 0; layer_idx < layers; layer_idx++) {
        tree_c_prepare_data(
          leaves + tree_c_batch_leaves_len * batch_idx,
          sector_size,
          configs,
          0,
          layers,
          layer_idx,
          batches,
          batch_idx,
          cache_path
        );
      }

      tree_c_memory_channel.send(leaves + tree_c_batch_leaves_len * batch_idx);
    }

    for (int config_idx = 1; config_idx < configs; config_idx++) {
      for (int batch_idx = 0; batch_idx < batches; batch_idx++) {
        file_write_channel.recv();

        for (int layer_idx = 0; layer_idx < layers; layer_idx++) {
          tree_c_prepare_data(
            leaves + tree_c_batch_leaves_len * batch_idx,
            sector_size,
            configs,
            config_idx,
            layers,
            layer_idx,
            batches,
            batch_idx,
            cache_path
          );
        }

        tree_c_memory_channel.send(leaves +
                                   tree_c_batch_leaves_len * batch_idx);
      }
    }

    for (int batch_idx = 0; batch_idx < batches; batch_idx++)
      file_write_channel.recv();

    for (int config_idx = 0; config_idx < configs; config_idx++) {
      tree_r_prepare_data(
        leaves + tree_r_leaves_len * config_idx,
        sector_size,
        configs,
        config_idx,
        layers,
        cache_path
      );

      tree_r_memory_channel.send(leaves + tree_r_leaves_len * config_idx);
    }
  });

  gpu.spawn([&, leaves]() {
    for (int config_idx = 0; config_idx < configs; config_idx++) {
      for (int batch_idx = 0; batch_idx < batches; batch_idx++) {
        fr_t* cur_leaves = tree_c_memory_channel.recv();

        column_tree_builder.build_column_tree_with_preimages(
          cur_leaves,
          tree_c_batch_leaves_len,
          digests + batch_idx * tree_c_batch_digests_len,
          true
        );
        cudaDeviceSynchronize();

        tree_c_merge[config_idx * batches + batch_idx] =
          digests[batch_idx * tree_c_batch_digests_len +
                  tree_c_batch_digests_len - 1];

        tree_c_compute_channel.send(digests +
                                    batch_idx * tree_c_batch_digests_len);
      }
    }

    for (int config_idx = 0; config_idx < configs; config_idx++) {
      fr_t* current_leaves = tree_r_memory_channel.recv();
      tree_builder.build_tree_with_preimages(
        current_leaves,
        tree_r_leaves_len,
        digests_r + config_idx * tree_r_digests_len,
        true
      );

      cudaDeviceSynchronize();

      tree_r_2nd_last_row[config_idx] =
        digests_r[config_idx * tree_r_digests_len + tree_r_digests_len - 1];

      tree_r_compute_channel.send(digests_r + config_idx * tree_r_digests_len);
    }
  });

  gpu.spawn([&, leaves]() {
    for (int config_idx = 0; config_idx < configs; config_idx++) {
      for (int batch_idx = 0; batch_idx < batches; batch_idx++) {
        fr_t* cur_digests = tree_c_compute_channel.recv();

        write_tree_c_to_disk(
          config_idx,
          configs,
          cur_digests,
          tree_c_digests_len,
          tree_c_batch_digests_len,
          batch_idx,
          batches,
          tree_c_leaves_len / column_arity,
          tree_arity,
          output_path
        );

        file_write_channel.send(0);
        all_complete.send(0);
      }
    }

    for (int config_idx = 0; config_idx < configs; config_idx++) {
      fr_t* current_digests = tree_r_compute_channel.recv();
      write_tree_r_to_disk(config_idx, configs,
                           current_digests + rows_to_discard_offset,
                           tree_r_digests_len - rows_to_discard_offset,
                           output_path);

      all_complete.send(0);
    }
  });

  for (int all_receive = 0; all_receive < configs * (batches + 1);
      all_receive++)
    all_complete.recv();

  if (batches > 1) {
    for (int config_idx = 0; config_idx < configs; config_idx++) {
      fr_t tree_c_this_batch_root;

      tree_builder.build_tree_with_preimages(
        tree_c_merge + config_idx * batches,
        batches,
        &tree_c_2nd_last_row[config_idx],
        false
      );

      write_tree_c_root_to_disk(
        config_idx,
        configs,
        tree_c_digests_len,
        tree_c_2nd_last_row[config_idx],
        output_path
      );
    }
  }
  else {
    for (int config_idx = 0; config_idx < configs; config_idx++) {
      tree_c_2nd_last_row[config_idx] = tree_c_merge[config_idx];
    }
  }

  if (configs > 1) {
    TreeBuilder tree_builder_last(configs);
    tree_builder_last.build_tree_with_preimages(
      tree_c_2nd_last_row,
      configs,
      &roots[0],
      false
    );
    tree_builder_last.build_tree_with_preimages(
      tree_r_2nd_last_row,
      configs,
      &roots[1],
      false
    );
  }
  else {
    roots[0] = tree_c_2nd_last_row[0];
    roots[1] = tree_r_2nd_last_row[0];
  }

  TreeBuilder comm_r_builder(2);
  comm_r_builder.build_tree_with_preimages(roots, 2, &roots[2], false);

  free(tree_c_2nd_last_row);
  free(tree_r_2nd_last_row);
  free(tree_c_merge);
}

extern "C"
void test_output_data(const SectorParameters sector_parameters,
                      std::string output_path, std::string cache_path) {

  const size_t sector_size = sector_parameters.sector_size;
  const int layers = sector_parameters.layers;
  const int column_arity = sector_parameters.column_arity;
  const int tree_arity = sector_parameters.tree_arity;
  const int configs = sector_parameters.configs;
  const int rows_to_discard = sector_parameters.rows_to_discard;

  const size_t leaves_len = sector_size / sizeof(fr_t);
  // number of leaf nodes for one configs'th of tree C

  const size_t tree_c_leaves_len = leaves_len * layers / configs;
  const size_t tree_r_leaves_len = sector_size / sizeof(fr_t) / configs;

  ColumnTreeBuilder column_tree_builder(column_arity, tree_arity);
  TreeBuilder tree_builder(tree_arity);

  size_t tree_c_digests_len =
    column_tree_builder.calc_column_digests_len(tree_c_leaves_len,
                                                column_arity, tree_arity);

  size_t tree_r_digests_len =
    tree_builder.calc_digests_len(tree_r_leaves_len, tree_arity);

  size_t rows_to_discard_offset = 0;
  for (size_t i = 0; i < rows_to_discard; i++) {
    rows_to_discard_offset +=
      tree_r_leaves_len / pow(static_cast<double>(tree_arity), i + 1);
  }

  tree_r_digests_len -= rows_to_discard_offset;

  fr_t* _correct_ptr = (fr_t*)malloc(sizeof(fr_t) * tree_c_digests_len);
  fr_t* _test_ptr = (fr_t*)malloc(sizeof(fr_t) * tree_c_digests_len);

  for (int config_idx = 0; config_idx < configs; config_idx++) {
    read_file(tree_c_digests_len * sizeof(fr_t), _correct_ptr,
              cache_path + get_tree_c_file_name(config_idx, configs));
    read_file(tree_c_digests_len * sizeof(fr_t), _test_ptr,
              output_path + get_tree_c_file_name(config_idx, configs));

    uint32_t* correct_ptr = reinterpret_cast<uint32_t*>(_correct_ptr);
    uint32_t* test_ptr = reinterpret_cast<uint32_t*>(_test_ptr);

    for (size_t i = 0;
         i < tree_c_digests_len * sizeof(fr_t) / sizeof(uint32_t); i++) {
      if (correct_ptr[i] != test_ptr[i]) {
        printf("tree c failure at index %zu/%zu, config %d\n",
               i / (sizeof(fr_t) / sizeof(uint32_t)),
               i % (sizeof(uint32_t)), config_idx);
        exit(2);
      }
    }

    read_file(tree_r_digests_len * sizeof(fr_t), _correct_ptr,
              cache_path + get_tree_r_file_name(config_idx, configs));
    read_file(tree_r_digests_len * sizeof(fr_t), _test_ptr,
              output_path + get_tree_r_file_name(config_idx, configs));

    for (size_t i = 0; i < tree_r_digests_len; i++) {
      if (correct_ptr[i] != test_ptr[i]) {
        printf("tree r failure at index %zu/%zu, config %d\n",
               i / (sizeof(fr_t) / sizeof(uint32_t)), i % (sizeof(uint32_t)),
               config_idx);
        exit(3);
      }
    }
  }

  free(_correct_ptr);
  free(_test_ptr);
}

extern "C"
void allocate_pinned_memory(const SectorParameters sector_parameters,
                            fr_t*& main_ptr, fr_t*& digests_ptr,
                            fr_t*& digests_r_ptr) {
  const size_t leaves_len = sector_parameters.sector_size / sizeof(fr_t);
  const size_t tree_c_leaves_len = leaves_len *
                                   sector_parameters.layers /
                                   sector_parameters.configs;
  const size_t tree_r_leaves_len = leaves_len / sector_parameters.configs;

  cudaMallocHost(&main_ptr,
                 std::max(tree_c_leaves_len, leaves_len) * sizeof(fr_t));

  ColumnTreeBuilder column_tree_builder(sector_parameters.column_arity,
                                        sector_parameters.tree_arity);
  TreeBuilder tree_builder(sector_parameters.tree_arity);

  size_t tree_c_digests_len =
    column_tree_builder.calc_column_digests_len(tree_c_leaves_len,
                                                sector_parameters.column_arity,
                                                sector_parameters.tree_arity);
  size_t tree_r_digests_len =
    tree_builder.calc_digests_len(tree_r_leaves_len,
                                  sector_parameters.tree_arity);

  cudaMallocHost(&digests_ptr, sizeof(fr_t) * tree_c_digests_len);
  cudaMallocHost(&digests_r_ptr,
                sizeof(fr_t) * tree_r_digests_len * sector_parameters.configs);
}

extern "C"
void free_pinned_memory(fr_t*& main_ptr, fr_t*& digests_ptr,
                        fr_t*& digests_r_ptr) {
  cudaFreeHost(main_ptr);
  cudaFreeHost(digests_ptr);
  cudaFreeHost(digests_r_ptr);
}

#endif
