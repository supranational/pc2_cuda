// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <bls12-381.hpp>
#include <string>

#include "src/parameters.cpp"

extern "C" void allocate_pinned_memory(const SectorParameters,
                                       fr_t*&, fr_t*&, fr_t*&);
extern "C" void free_pinned_memory(fr_t*&, fr_t*&, fr_t*&);
extern "C" void test_output_data(const SectorParameters,
                                 std::string, std::string);
extern "C" void pc2(const SectorParameters, fr_t*, fr_t*, fr_t*, fr_t[3],
                    std::string, std::string);

#ifndef __CUDA_ARCH__

#include <chrono>

#define TIME_TYPE std::chrono::time_point<std::chrono::high_resolution_clock>
#define TIME_NOW std::chrono::high_resolution_clock::now()
#define TIME_ELAPSED(start, end) \
  std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()

#define TIME_INIT TIME_TYPE start; TIME_TYPE end;
#define TIME_START start = TIME_NOW
#define TIME_STOP(str) \
  end = TIME_NOW; \
  std::cout << str << " :" << TIME_ELAPSED(start, end) << " ms" << std::endl;

#include <vect.h>
#include <unistd.h>
#include <iostream>
#include <sstream>

std::string fr_to_hex(const fr_t fr) {
  limb_t v[sizeof(fr_t) / sizeof(limb_t)];
  fr.store(v);

  std::string ret = "0x";

  for (int i = (sizeof(fr_t) / sizeof(limb_t)) - 1; i >= 0; i--) {
    std::stringstream ss;
    ss << std::hex << v[i];
    std::string hex = ss.str();
    if (hex.length() < sizeof(limb_t) / 2) {
      hex.insert(0, sizeof(limb_t) / 2 - hex.length(), '0');
    }
    ret += hex;
  }

  return ret;
}

int main(int argc, char* argv[]) {
  int  opt   = 0;
  std::string cache_path  = "./";
  std::string output_path = "./";
  std::string sector_size = "2KiB";

  while ((opt = getopt(argc, argv, "i:o:s:h")) != -1) {
    switch(opt) {
      case 'i':
        std::cout << "input_path input " << optarg << std::endl;
        cache_path = optarg;
        break;
      case 'o':
        std::cout << "output_path input " << optarg << std::endl;
        output_path = optarg;
        break;
      case 's':
        std::cout << "sector_size input " << optarg << std::endl;
        sector_size = optarg;
        break;
      case 'h':
      case ':':
      case '?':
        std::cout << "Sealing Client" << std::endl;
        std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl;
        std::cout << "-h        Print help message" << std::endl;
        std::cout << "-i <path> Path to cached layer data files " << std::endl;
        std::cout << "-o <path> Path to place tree files" << std::endl;
        std::cout << "-s <size> Sector Size (2KiB, 32GiB, etc) " << std::endl;
        break;
    }
  }

  std::cout << "input_path  = " << cache_path << std::endl;
  std::cout << "output_path = " << output_path << std::endl;
  std::cout << "sector_size = " << sector_size << std::endl;

  fr_t* leaves = nullptr, * digests_c = nullptr, * digests_r = nullptr;
  fr_t roots[3];

  SectorParameters sector_parameters = get_sector_parameters(sector_size);

  TIME_INIT;

  TIME_START;
  allocate_pinned_memory(sector_parameters, leaves, digests_c, digests_r);
  TIME_STOP("Pinned memory allocation");

  TIME_START;
  pc2(sector_parameters, leaves, digests_c, digests_r, roots,
      output_path, cache_path);
  TIME_STOP("Pre-commit phase 2");

  TIME_START;
  free_pinned_memory(leaves, digests_c, digests_r);
  TIME_STOP("Pinned memory deallocation");

  test_output_data(sector_parameters, output_path, cache_path);

  roots[0].from();
  roots[1].from();
  roots[2].from();

  std::cout << "CommC = " << fr_to_hex(roots[0]) << std::endl;
  std::cout << "RootR = " << fr_to_hex(roots[1]) << std::endl;
  std::cout << "CommR = " << fr_to_hex(roots[2]) << std::endl;

  return 0;
}

#endif
