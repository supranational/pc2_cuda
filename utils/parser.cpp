// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __PARSER_CPP__
#define __PARSER_CPP__

#include <cassert>
#include <fstream>
#include <string>

static fr_t* read_constants_from_file(const int arity) {
  std::ifstream file(std::string("poseidon-constants/constants_") +
                     std::to_string(arity), std::ios::binary | std::ios::ate);
  assert(file.is_open());

  size_t size = file.tellg();
  fr_t* constants = static_cast<fr_t*>(malloc(size));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(constants), size);
  file.close();

  return constants;
}

#endif /* __PARSER_CPP__ */
