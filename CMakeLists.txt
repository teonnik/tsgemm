cmake_minimum_required(VERSION 3.14)

project(tsgemm VERSION 0.1 LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

find_package(MPI REQUIRED)
find_package(HPX REQUIRED)
find_package(MKL REQUIRED COMPONENTS BLAS_32BIT_OMP)

function(add_tsgemm_target name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE MPI::MPI_CXX mkl::blas_32bit_omp hpx)
    target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_features(${name} PRIVATE cxx_std_14)
endfunction()

add_tsgemm_target(v1)
add_tsgemm_target(v2)
add_tsgemm_target(v3)
