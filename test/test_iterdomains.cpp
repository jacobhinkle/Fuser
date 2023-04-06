#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <executor.h>
#include <executor_utils.h>
#include <fusion.h>
#include <inlining.h>
#include <kernel_cache.h>
#include <ops/all_ops.h>
#include <scheduler/utils.h>
#include <test/test_gpu_validator.h>
#include <test/test_utils.h>

namespace nvfuser {

//! This test just prints the kernel before and after a call to rFactor.
TEST_F(NVFuserTest, FusionRFactorPrint_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  auto tv1 = sum(tv0, {0});
  fusion.addInput(tv0);
  fusion.addOutput(tv1);

  fusion.printMath();
  fusion.printTransforms(); // A
  fusion.printKernel();

  std::cout << "^^A" << std::endl;
  tv1->split(0, 8);
  fusion.printTransforms(); // B
  fusion.printKernel();
  std::cout << "^^B" << std::endl;

  auto tv2 = tv1->rFactor({1});
  fusion.printTransforms(); // C
  fusion.printKernel();
  std::cout << "^^C" << std::endl;

  tv2->split(1, 2);
  fusion.printTransforms(); // D
  fusion.printKernel();
  std::cout << "^^D" << std::endl;
}

//! This test just prints the kernel after a call to permute.
TEST_F(NVFuserTest, FusionPermutePrint_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);
  fusion.addInput(tv0);
  auto tv1 = permute(tv0, {1, 0}); 
  fusion.addOutput(tv1);

  fusion.printMath();
  fusion.printTransforms();
  fusion.printKernel();
}

//! This test just prints the kernel after a call to reshape.
TEST_F(NVFuserTest, FusionReshapePrint_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto tv0 = makeSymbolicTensor(2);

  auto tv1 = reshape(tv0, {4, 6}, {3, 8});
  fusion.addInput(tv0);
  fusion.addOutput(tv1);

  fusion.printMath();
  fusion.printTransforms();
  fusion.printKernel();
}


} // namespace nvfuser
