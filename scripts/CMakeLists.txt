#-------------------------------------------------------
# Slurm job launching script generator
#
# We would like to generate a script which can be used to generate batch jobs
# to submit benchmarks to slurm using many combinations of settings.
#
# We add a custom command which takes our template script and
# expands out all the variables we need to pass into it.
# Unfortunately, due to the way cmake works, some variables are only known
# at build time, and not at cmake configure time. Using a custom command which
# calls cmake to run our script at build time, allows us to pass variables
# into the final script which is placed in our build dir.
#
# Note that we generate these scripts in the build dir instead
# of the install dir as they are intended for development testing.
# A version could be supported for installation later.
#-------------------------------------------------------

set(TSGEMM_BENCHMARK_SCRIPTS_PATH
    "${CMAKE_BINARY_DIR}/scripts"
    CACHE PATH
    "Directory to place generated scripts in")

# Make sure scripts dir exists
execute_process(COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/scripts")

#--------------------------------------------------
# Slurm script generator for
# Standalone Bandwidth test
#--------------------------------------------------
set(SCRIPTS "tsgemm-bench")

foreach(script ${SCRIPTS})
  ADD_CUSTOM_COMMAND(
    DEPENDS   "${CMAKE_CURRENT_SOURCE_DIR}/${script}.sh.in"
    COMMAND   "${CMAKE_COMMAND}"
    ARGS      -DEXE_V1="$<TARGET_FILE:v1>"
              -DEXE_V2="$<TARGET_FILE:v2>"
              -DMPIEXEC="${MPIEXEC}"
              -DSCRIPT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
              -DSCRIPT_NAME=${script}
              -DSCRIPT_DEST_DIR="${TSGEMM_BENCHMARK_SCRIPTS_PATH}"
              -DJOB_OPTIONS1="${SLURM_JOB_OPTIONS1}"
              -P "${CMAKE_CURRENT_SOURCE_DIR}/copy_script.cmake"
              OUTPUT    "${TSGEMM_BENCHMARK_SCRIPTS_PATH}/${script}.sh"
    VERBATIM
  )

  add_custom_target(script-${script}
    DEPENDS "${TSGEMM_BENCHMARK_SCRIPTS_PATH}/${script}.sh"
  )
endforeach(script)
