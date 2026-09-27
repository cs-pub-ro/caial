cmake_minimum_required(VERSION 3.20)

if(NOT CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/riscv64-clang.cmake")
endif()

project(caial C ASM)

set(CAIAL_FLOAT_TYPES ieee posit1 posit2 morris morrisHeb morrisUnaryHeb morrisBiasHeb)
set(CAIAL_FLOAT_TYPE "ieee" CACHE STRING "Float representation: ieee or one of the NRS pass types")
set_property(CACHE CAIAL_FLOAT_TYPE PROPERTY STRINGS ${CAIAL_FLOAT_TYPES})
if(NOT CAIAL_FLOAT_TYPE IN_LIST CAIAL_FLOAT_TYPES)
  message(FATAL_ERROR "CAIAL_FLOAT_TYPE must be one of: ${CAIAL_FLOAT_TYPES}")
endif()

set(CAIAL_OPT_LEVEL "2" CACHE STRING "Optimization level (0-3) for the test sources")

set(NRS_PASS_PLUGIN "/workspace/shared/NRSSL-LLVMPass/src/build/libNRSConversionPass.so"
    CACHE FILEPATH "NRS conversion pass plugin")
set(NRSSL_JARS "/workspace/shared/NRSSL-LLVMPass/src/jars/" CACHE PATH "Directory with nrssl.jar")
set(JVM_LIB "/usr/lib/jvm/java-17-openjdk-amd64/lib/server/libjvm.so" CACHE FILEPATH "libjvm.so preloaded by opt")

set(ROCKET_CONFIG "DefaultConfig" CACHE STRING "rocket-chip emulator config")
set(EMULATOR "/workspace/shared/rocket-chip/out/emulator/freechips.rocketchip.system.TestHarness/freechips.rocketchip.system.${ROCKET_CONFIG}/verilator/elf.dest/emulator"
    CACHE FILEPATH "Verilator emulator binary")

set(OPT "${LLVM_BIN_DIR}/opt")
set(LLC "${LLVM_BIN_DIR}/llc")

include(cmake/caial.cmake)
enable_testing()

add_subdirectory(runtime)

caial_add_test(E E/E.c)
caial_add_test(PI PI/PI.c)
caial_add_test(MSEQ MSEQ/MSEQ.c)
caial_add_test(MM MM/MM.c)
caial_add_test(KNN KNN/KNN.c)
caial_add_test(KMEANS KMEANS/KMEANS.c)
caial_add_test(LR LR/LR.c)
caial_add_test(NB NB/NB.c)
caial_add_test(SVM SVM/SVM.c)
caial_add_test(CT CT/CT.c)
caial_add_test(DNN DNN/DNN.c)
caial_add_test(DNN2 DNN2/DNN2.c)
caial_add_test(CNN CNN/CNN.c)
caial_add_test(SimpleFPUTest SimpleFPUTest/SimpleFPUTest.c)
