// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <test/test_gpu_validator.h>
#include <test/test_utils.h>
#include <union_find.h>

#include <algorithm>
#include <complex>
#include <iostream>
#include <sstream>
#include <thread>

namespace nvfuser {

using namespace at::indexing;

TEST_F(NVFuserTest, FusionUnionFind) {
  UnionFind<uint8_t> uf;

  uf.enlarge(5);
  uf.merge(3, 4);

  assert(uf.equiv(3, 4));
  assert(!uf.equiv(2, 3));

  uf.enlarge(8);
  assert(!uf.equiv(3, 7));

  EXPECT_ANY_THROW(uf.enlarge(270); // Try to enlarge past capacity of IndexType
  );

  assert(uf.size() == 8);
  EXPECT_ANY_THROW(uf.find(8); // Try to index past current size
  );
}

} // namespace nvfuser