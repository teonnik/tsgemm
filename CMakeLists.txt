cmake_minimum_required(VERSION 3.13)

project(tsgemm VERSION 0.1 LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(MPI REQUIRED)
find_package(HPX REQUIRED)
find_package(MKL REQUIRED COMPONENTS BLAS_32BIT_SEQ)

# set the CXX standard to the same as that used by HPX
set(CMAKE_CXX_STANDARD ${HPX_CXX_STANDARD})

add_subdirectory(src/tsgemm)
add_subdirectory(apps)
add_subdirectory(scripts)
