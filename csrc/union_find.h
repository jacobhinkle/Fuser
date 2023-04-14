// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <c10/util/Exception.h>

#include <iostream>
#include <typeinfo>
#include <vector>

namespace nvfuser {

//! A Union-Find (by rank) data structure on integers
//! The template parameter IndexType dictates the maximum number of elements
//! that can be used.
template <typename IndexType>
class UnionFind {
 public:
  UnionFind(size_t size) : parent_(size), rank_(size, 0) {
    for (size_t i = 0; i < size; ++i) {
      parent_[i] = (IndexType)i;
    }
  }

  //! Resize the data-structure to equal or larger size than current
  void enlarge(size_t new_size) {
    TORCH_CHECK(new_size >= size(), "Cannot shrink a UnionFind");
    TORCH_CHECK(
        new_size <= std::numeric_limits<IndexType>::max() + 1,
        "Tried to enlarge UnionFind to size ",
        new_size,
        " which is greater than this IndexType's capacity of ",
        std::to_string(std::numeric_limits<IndexType>::max() + 1));
    auto old_size = parent_.size();
    parent_.resize(new_size);
    rank_.resize(new_size, 0);
    for (auto i = old_size; i < new_size; ++i) {
      parent_[i] = (IndexType)i;
    }
  }

  //! Return the number of elements in this data structure.
  size_t size() const {
    return parent_.size();
  }

  //! Determine root of element a
  IndexType find(IndexType a) {
    TORCH_CHECK(
        a < size(),
        "Tried to find root of element ",
        a,
        " but total size of UnionFind is ",
        size());
    // This implementation avoids recursion by doing two passes
    // The equivalent recursive definition is:
    //   auto p = parent_[a];
    //   if (p == a) {
    //     return a;
    //   } else {
    //     // Path compression step. Next call will shortcut directly to root.
    //     return parent_[a] = find(p);
    //   }

    // First find the root without path compression
    auto p = a;
    auto root = parent_[p];
    while (p != root) {
      p = root;
      root = parent_[p];
    }

    // Path compression
    // Loop again to set parents along the path equal to root.
    // On the next call, both loops will not be entered.
    while (a != root) {
      p = parent_[a];
      parent_[a] = root;
      a = p;
    }

    return root;
  }

  //! Test whether two elements are equivalent
  bool equiv(IndexType a, IndexType b) {
    return find(a) == find(b);
  }

  //! Merge classes of a and b so that they will share a root.
  //! Returns the new root
  IndexType merge(IndexType a, IndexType b) {
    auto root_a = find(a);
    auto root_b = find(b);
    if (root_a == root_b) {
      return root_a;
    }
    // Rank is a surrogate for "height" of each subtree. It is actually an
    // upper bound on height, since path compression can reduce the height of a
    // subtree without updating rank_. When merging trees, we try to place the
    // "shorter" tree inside the "taller" one since this would not increase the
    // larger tree's height. If they are equal, we point b's root at a's root.
    // Note that in that case the rank (height) of a must be incremented as it
    // is now equal to b's height plus the new step linking it with a.
    auto rank_a = rank_[root_a];
    auto rank_b = rank_[rank_a];
    if (rank_a == rank_b) {
      rank_[root_a]++;
      return parent_[root_b] = root_a;
    } else {
      if (rank_a < rank_b) {
        std::swap(root_a, root_b);
      }
      return parent_[root_b] = root_a;
    }
  }

  //! Resize to zero losing all merge information without altering reserved
  //! capacity
  void clear() {
    parent_.clear();
    rank_.clear();
  }

 private:
  std::vector<IndexType> parent_{std::vector<IndexType>()};
  std::vector<IndexType> rank_{std::vector<IndexType>()};
};

} // namespace nvfuser
