// Copyright 2017, Brown University, Providence, RI.
// MGARD: MultiGrid Adaptive Reduction of Data
// Authors: Mark Ainsworth, Ozan Tugluk, Ben Whitney
// Corresponding Author: Ben Whitney, Qing Liu
//
// version: 0.1.0
// See LICENSE for details.
#ifndef MGARD_API_TPP
#define MGARD_API_TPP

#include <cstddef>

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

#include "TensorMultilevelCoefficientQuantizer.hpp"
#include "TensorNorms.hpp"
#include "mgard.hpp"
#include "shuffle.hpp"

namespace mgard {

template <std::size_t N, typename Real>
RefactoredDataset<N, Real>::RefactoredDataset(
    const TensorMeshHierarchy<N, Real> &hierarchy,
    std::vector<Real *> data, std::vector<std::size_t> sizes)
    : hierarchy(hierarchy),
      data_(data), sizes_(sizes) {}

template <std::size_t N, typename Real>
std::vector<Real *> RefactoredDataset<N, Real>::data() const {
  return data_;
}

template <std::size_t N, typename Real>
std::vector<std::size_t> RefactoredDataset<N, Real>::sizes() const {
  return sizes_;
}

template <std::size_t N, typename Real>
RecomposedDataset<N, Real>::RecomposedDataset(
    const RefactoredDataset<N, Real> &refactored, Real const *const data)
    : hierarchy(refactored.hierarchy), data_(data) {}

template <std::size_t N, typename Real>
Real const *RecomposedDataset<N, Real>::data() const {
  return data_.get();
}

template <std::size_t N, typename Real>
CompressedDataset<N, Real>::CompressedDataset(
    const TensorMeshHierarchy<N, Real> &hierarchy, const Real s,
    const Real tolerance, void const *const data, const std::size_t size)
    : hierarchy(hierarchy), s(s), tolerance(tolerance),
      data_(static_cast<unsigned char const *>(data)), size_(size) {}

template <std::size_t N, typename Real>
void const *CompressedDataset<N, Real>::data() const {
  return data_.get();
}

template <std::size_t N, typename Real>
std::size_t CompressedDataset<N, Real>::size() const {
  return size_;
}

template <std::size_t N, typename Real>
DecompressedDataset<N, Real>::DecompressedDataset(
    const CompressedDataset<N, Real> &compressed, Real const *const data)
    : hierarchy(compressed.hierarchy), s(compressed.s),
      tolerance(compressed.tolerance), data_(data) {}

template <std::size_t N, typename Real>
Real const *DecompressedDataset<N, Real>::data() const {
  return data_.get();
}

using DEFAULT_INT_T = long int;

template <std::size_t N, typename Real>
RefactoredDataset<N, Real>
refactor(const TensorMeshHierarchy<N, Real> &hierarchy, Real *const v) {
  std::vector<Real *> r;
  std::vector<std::size_t> sizes;

  const std::size_t ndof = hierarchy.ndof();
  // TODO: Can be smarter about copies later.
  Real *const u = static_cast<Real *>(std::malloc(ndof * sizeof(Real)));
  shuffle(hierarchy, v, u);
  decompose(hierarchy, u);

  std::cout << "size = " << r.size();

  Real const *p = u;

  for (std::size_t l = 0; l <= hierarchy.L; ++l) {
    std::size_t dndof = (l > 0) ? hierarchy.ndof(l) - hierarchy.ndof(l - 1) : hierarchy.ndof(0);
    r.push_back(static_cast<Real *>(std::malloc(dndof * sizeof(Real))));
    std::copy(p, p + dndof, r.at(l));
    sizes.push_back(dndof);
    p += dndof;
  }

  std::free(u);

  return RefactoredDataset<N, Real>(hierarchy, r, sizes);
}

template <std::size_t N, typename Real>
CompressedDataset<N, Real>
compress(const TensorMeshHierarchy<N, Real> &hierarchy, Real *const v,
         const Real s, const Real tolerance) {
  const std::size_t ndof = hierarchy.ndof();
  // TODO: Can be smarter about copies later.
  Real *const u = static_cast<Real *>(std::malloc(ndof * sizeof(Real)));
  shuffle(hierarchy, v, u);
  decompose(hierarchy, u);

  using Qntzr = TensorMultilevelCoefficientQuantizer<N, Real, DEFAULT_INT_T>;
  const Qntzr quantizer(hierarchy, s, tolerance);
  using It = typename Qntzr::iterator;
  const RangeSlice<It> quantized_range = quantizer(u);
  const std::vector<DEFAULT_INT_T> quantized(quantized_range.begin(),
                                             quantized_range.end());
  std::free(u);

  std::vector<std::uint8_t> z_output;
  // TODO: Check whether `compress_memory_z` changes its input.
  compress_memory_z(
      const_cast<void *>(static_cast<void const *>(quantized.data())),
      sizeof(DEFAULT_INT_T) * hierarchy.ndof(), z_output);
  // Possibly we should check that `sizeof(std::uint8_t)` is `1`.
  const std::size_t size = z_output.size();

  void *const buffer = new unsigned char[size];
  std::copy(z_output.begin(), z_output.end(),
            static_cast<unsigned char *>(buffer));
  return CompressedDataset<N, Real>(hierarchy, s, tolerance, buffer, size);
}

template <std::size_t N, typename Real>
RecomposedDataset<N, Real>
recompose(const RefactoredDataset<N, Real> &refactored) {
  const std::size_t ndof = refactored.hierarchy.ndof();
  std::vector<Real *> r = refactored.data();

  Real *const buffer = static_cast<Real *>(std::malloc(ndof * sizeof(Real)));
  Real zero_buffer[ndof] = {0};
  std::memcpy(buffer, zero_buffer, ndof * sizeof(Real)); 
  Real *p = buffer;

  for (std::size_t l = 0; l < refactored.data().size(); ++l) {
    std::size_t dndof = (l > 0) ? refactored.hierarchy.ndof(l) - refactored.hierarchy.ndof(l - 1) : refactored.hierarchy.ndof(0);
    std::copy(r.at(l), r.at(l) + dndof, p);
    p += dndof;
  }

  recompose(refactored.hierarchy, buffer);
  Real *const v = new Real[ndof];
  unshuffle(refactored.hierarchy, buffer, v);
  std::free(buffer);

  return RecomposedDataset<N, Real>(refactored, v);
}

template <std::size_t N, typename Real>
DecompressedDataset<N, Real>
decompress(const CompressedDataset<N, Real> &compressed) {
  const std::size_t ndof = compressed.hierarchy.ndof();
  DEFAULT_INT_T *const quantized =
      static_cast<DEFAULT_INT_T *>(std::malloc(ndof * sizeof(*quantized)));
  // TODO: Figure out all these casts here and above.
  decompress_memory_z(const_cast<void *>(compressed.data()), compressed.size(),
                      reinterpret_cast<int *>(quantized),
                      ndof * sizeof(*quantized));

  using Dqntzr = TensorMultilevelCoefficientDequantizer<N, DEFAULT_INT_T, Real>;
  const Dqntzr dequantizer(compressed.hierarchy, compressed.s,
                           compressed.tolerance);
  using It = typename Dqntzr::template iterator<DEFAULT_INT_T *>;
  const RangeSlice<It> dequantized_range =
      dequantizer(quantized, quantized + ndof);

  // TODO: Can be smarter about copies later.
  Real *const buffer = static_cast<Real *>(std::malloc(ndof * sizeof(Real)));
  std::copy(dequantized_range.begin(), dequantized_range.end(), buffer);
  std::free(quantized);

  recompose(compressed.hierarchy, buffer);
  Real *const v = new Real[ndof];
  unshuffle(compressed.hierarchy, buffer, v);
  std::free(buffer);
  return DecompressedDataset<N, Real>(compressed, v);
}

} // namespace mgard

#endif
