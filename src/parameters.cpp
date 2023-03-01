// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <string>

constexpr size_t ONE_KiB = 1024ull; // 1 KiB in bytes
constexpr size_t ONE_MiB = ONE_KiB * 1024; // 1 MiB in bytes
constexpr size_t ONE_GiB = ONE_MiB * 1024; // 1 GiB in bytes

struct SectorParameters {
  size_t sector_size;
  int layers, column_arity, tree_arity, configs, rows_to_discard;
  SectorParameters(size_t _sector_size, int _layers, int _configs,
    int _rows_to_discard = 2, int _tree_arity = 8) {

    sector_size = _sector_size;
    layers = _layers;
    column_arity = _layers;
    configs = _configs;
    rows_to_discard = _rows_to_discard;
    tree_arity = _tree_arity;
  }
};

// values from
// https://github.com/filecoin-project/rust-fil-proofs/blob/128f7209ec583e023f04630102ef1dd17fbe2370/filecoin-proofs/src/constants.rs
// where |sector_shape| corresponds to |configs|. base = 1, sub2 = 2, sub8 = 8, top2 = 16

static SectorParameters Sector2KiB(ONE_KiB * 2, 2, 1, 1);
static SectorParameters Sector4KiB(ONE_KiB * 4, 2, 2, 1);
static SectorParameters Sector16KiB(ONE_KiB * 16, 2, 8, 1);
static SectorParameters Sector32KiB(ONE_KiB * 32, 2, 16, 1);

static SectorParameters Sector8MiB(ONE_MiB * 8, 2, 1);
static SectorParameters Sector16MiB(ONE_MiB * 16, 2, 2);
static SectorParameters Sector512MiB(ONE_MiB * 512, 2, 1);

static SectorParameters Sector1GiB(ONE_GiB * 1, 2, 2);
static SectorParameters Sector32GiB(ONE_GiB * 32, 11, 8);
static SectorParameters Sector64GiB(ONE_GiB * 64, 11, 16);

static SectorParameters get_sector_parameters(const std::string size_string) {
  if (size_string == "2KiB") return Sector2KiB;
  if (size_string == "4KiB") return Sector4KiB;
  if (size_string == "16KiB") return Sector16KiB;
  if (size_string == "32KiB") return Sector32KiB;

  if (size_string == "8MiB") return Sector8MiB;
  if (size_string == "16MiB") return Sector16MiB;
  if (size_string == "512MiB") return Sector512MiB;

  if (size_string == "1GiB") return Sector1GiB;
  if (size_string == "32GiB") return Sector32GiB;
  if (size_string == "64GiB") return Sector64GiB;

  printf("Sector size argument does not correspond to any valid sector\n");
  exit(1);
}
