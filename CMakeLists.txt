cmake_minimum_required(VERSION 3.14)

project(tsgemm VERSION 0.1 LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(MPI REQUIRED)
find_package(HPX REQUIRED)
find_package(MKL REQUIRED COMPONENTS BLAS_32BIT_SEQ)

add_library(tsgemm INTERFACE)
target_include_directories(tsgemm INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)
target_compile_features(tsgemm INTERFACE cxx_std_14)
target_link_libraries(tsgemm INTERFACE MPI::MPI_CXX mkl::blas_32bit_seq hpx)

add_subdirectory(apps)
