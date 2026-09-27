# caial_add_test(<name> <source>)
#
# Builds one test through an explicit clang -> opt -> llc pipeline, so that the
# IEEE and NRS builds differ only by the conversion pass:
#
#   <name>.ll      clang output, unoptimized (no optnone)
#   <name>.nrs.ll  after the NRS pass (NRS builds only)
#   <name>.opt.ll  after opt -O<level>
#   <name>.s/.o    llc output
#   <name>.elf     linked with the bare-metal runtime, plus <name>.dump
#
# The pass runs before any optimization: running it on optimized IR would let
# InstCombine & co. fold constants and rewrite float ops with IEEE semantics.
# It must also run exactly once, since it rewrites constants in place.

set(CAIAL_C_FLAGS -std=gnu11 -g -ffp-contract=off -fno-math-errno -Wall)

separate_arguments(_caial_arch_flags UNIX_COMMAND "${RISCV_ARCH_FLAGS}")
set(_caial_clang
  ${CMAKE_C_COMPILER} --target=${CMAKE_C_COMPILER_TARGET} --sysroot=${CMAKE_SYSROOT}
  --gcc-toolchain=${RISCV_GCC_TOOLCHAIN} ${_caial_arch_flags} ${CAIAL_C_FLAGS})

if(CAIAL_OPT_LEVEL STREQUAL "0")
  set(_caial_frontend_opt -O0 -Xclang -disable-O0-optnone)
else()
  set(_caial_frontend_opt -O${CAIAL_OPT_LEVEL} -Xclang -disable-llvm-passes)
endif()

set(_caial_llc ${LLC} -O${CAIAL_OPT_LEVEL} -march=riscv64 -mattr=+m,+a,+f,+d
  -target-abi=lp64d -code-model=medium)

function(caial_add_test name source)
  set(out "${CMAKE_BINARY_DIR}/${name}")
  set(src "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
  file(MAKE_DIRECTORY "${out}")

  add_custom_command(
    OUTPUT "${out}/${name}.ll"
    COMMAND ${_caial_clang} ${_caial_frontend_opt} -I${CMAKE_SOURCE_DIR}
            -MD -MF "${out}/${name}.d" -S -emit-llvm -o "${out}/${name}.ll" "${src}"
    DEPENDS "${src}"
    DEPFILE "${out}/${name}.d"
    COMMENT "[${name}] clang -> ${name}.ll"
    VERBATIM)

  set(ir "${out}/${name}.ll")
  if(NOT CAIAL_FLOAT_TYPE STREQUAL "ieee")
    add_custom_command(
      OUTPUT "${out}/${name}.nrs.ll"
      COMMAND ${CMAKE_COMMAND} -E env LD_PRELOAD=${JVM_LIB} NRSSL_JARS=${NRSSL_JARS}
              ${OPT} -load-pass-plugin=${NRS_PASS_PLUGIN} -passes=ieee-to-posit
              --float_type=${CAIAL_FLOAT_TYPE} -S -o "${out}/${name}.nrs.ll" "${ir}"
              > "${out}/${name}.pass.log"
      DEPENDS "${ir}" "${NRS_PASS_PLUGIN}"
      COMMENT "[${name}] NRS pass (${CAIAL_FLOAT_TYPE}) -> ${name}.nrs.ll"
      VERBATIM)
    set(ir "${out}/${name}.nrs.ll")
  endif()

  add_custom_command(
    OUTPUT "${out}/${name}.opt.ll"
    COMMAND ${OPT} -passes=default<O${CAIAL_OPT_LEVEL}> -S -o "${out}/${name}.opt.ll" "${ir}"
    DEPENDS "${ir}"
    COMMENT "[${name}] opt -O${CAIAL_OPT_LEVEL} -> ${name}.opt.ll"
    VERBATIM)

  add_custom_command(
    OUTPUT "${out}/${name}.o" "${out}/${name}.s"
    COMMAND ${_caial_llc} -filetype=obj -o "${out}/${name}.o" "${out}/${name}.opt.ll"
    COMMAND ${_caial_llc} -filetype=asm -o "${out}/${name}.s" "${out}/${name}.opt.ll"
    DEPENDS "${out}/${name}.opt.ll"
    COMMENT "[${name}] llc -> ${name}.o, ${name}.s"
    VERBATIM)

  set_source_files_properties("${out}/${name}.o" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
  add_executable(${name} "${out}/${name}.o")
  target_link_libraries(${name} PRIVATE caial_runtime)
  set_target_properties(${name} PROPERTIES
    LINKER_LANGUAGE C
    OUTPUT_NAME "${name}.elf"
    RUNTIME_OUTPUT_DIRECTORY "${out}")
  add_custom_command(TARGET ${name} POST_BUILD
    COMMAND ${CMAKE_OBJDUMP} -d -S $<TARGET_FILE:${name}> > "${out}/${name}.dump"
    VERBATIM)

  # run-<name>: plain run; trace-<name>: +verbose, disassembled trace in <name>.trace
  add_custom_target(run-${name}
    COMMAND ${EMULATOR} $<TARGET_FILE:${name}>
    DEPENDS ${name}
    USES_TERMINAL
    VERBATIM)
  add_custom_target(trace-${name}
    COMMAND bash -c "'${EMULATOR}' +verbose '$<TARGET_FILE:${name}>' 2> >(spike-dasm > '${out}/${name}.trace')"
    DEPENDS ${name}
    USES_TERMINAL
    VERBATIM)

  add_test(NAME ${name} COMMAND ${EMULATOR} $<TARGET_FILE:${name}>)
endfunction()
