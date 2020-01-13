cmake_minimum_required(VERSION 3.14)

project(tsgemm VERSION 0.1 LANGUAGES CXX)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

find_package(MPI REQUIRED)
find_package(HPX REQUIRED)
find_package(MKL REQUIRED COMPONENTS BLAS_32BIT_OMP)

#add_executable(v1 v1.cpp) 
#target_link_libraries(v1 PRIVATE ${deps})
#
#add_executable(v2 v2.cpp)
#target_link_libraries(v2 PRIVATE ${deps})

add_executable(v3 v3.cpp)
target_link_libraries(v3 PRIVATE MPI::MPI_CXX mkl::blas_32bit_omp hpx)
target_include_directories(v3 PRIVATE ${PROJECT_SOURCE_DIR})
target_compile_features(v3 PRIVATE cxx_std_14)

