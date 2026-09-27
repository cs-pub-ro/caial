# CAIAL

Small floating-point workloads (series, sequences and simple AI algorithms)
used to measure the accuracy of alternative number representations against
IEEE 754 `float`. Each test is built either as plain IEEE code or through the
[NRSSL LLVM pass](https://github.com/Earthbert/NRSSL-LLVMPass), which lowers
the float operations to the custom NRS instructions of our rocket-chip, and
runs bare metal on the rocket-chip Verilator emulator.

## Requirements

Everything is expected to run inside the `racheta` container (the `shared/`
directory mounted at `/workspace/shared`):

- clang/opt/llc/ld.lld from `shared/llvm-project/build`
- the RISC-V GCC toolchain in `/opt/riscv` (only for headers, libgcc and objdump)
- the pass plugin and NRSSL jars from `shared/NRSSL-LLVMPass/src`
- the emulator from `shared/rocket-chip/out/emulator/...DefaultConfig...`

All paths are CMake cache variables with these defaults, see
`CMakeLists.txt` and `cmake/riscv64-clang.cmake` to override them.

## Building and running

There is one preset (and build directory) per representation: `ieee`,
`posit1`, `posit2`, `morris`, `morrisHeb`, `morrisUnaryHeb`, `morrisBiasHeb`.

```sh
cmake --preset posit1           # configure build/posit1
cmake --build --preset posit1   # build all tests
ctest --preset posit1           # run all tests on the emulator
ninja -C build/posit1 run-E     # run one test
ninja -C build/posit1 trace-E   # run with +verbose, trace in build/posit1/E/E.trace
```

Without presets: `cmake -B build/x -G Ninja -DCAIAL_FLOAT_TYPE=posit2 -DCAIAL_OPT_LEVEL=0`.

Every test is compiled as `clang -> [NRS pass] -> opt -O<n> -> llc`, so the IEEE
and NRS builds differ only by the pass. `build/<type>/<test>/` keeps all the
intermediate files:

| File | Content |
| --- | --- |
| `<test>.ll` | clang output, unoptimized |
| `<test>.nrs.ll` | after the NRS pass (NRS builds only), pass output in `<test>.pass.log` |
| `<test>.opt.ll` | after `opt -O<n>` |
| `<test>.s` | assembly |
| `<test>.elf`, `<test>.dump` | binary and its disassembly |

## Runtime

`runtime/` holds a minimal bare-metal runtime: startup code, HTIF console and
exit, a trap handler and a tiny libc (`printf` without floating point, `mem*`,
`str*`). There is no newlib: the one shipped with the toolchain is built for
medlow and cannot be linked at 0x80000000. Floats have to be printed as bits,
since the pass does not convert library code anyway.

## Adding a test

Put the sources in a new directory and register it in `CMakeLists.txt`:

```cmake
caial_add_test(NEWTEST NEWTEST/NEWTEST.c)
```

## Authors

* **Ciocîrlan Ștefan-Dan** - original algorithms and build system - [sdcioc](https://github.com/sdcioc)

## License

BSD 3-Clause, see [LICENSE](LICENSE).
