# based on John's mpi futures
add_executable(v1 v1.cpp)
target_link_libraries(v1 PRIVATE tsgemm)

# based on manual yielding
add_executable(v2 v2.cpp)
target_link_libraries(v2 PRIVATE tsgemm)

add_executable(v2_pool v2.cpp)
target_link_libraries(v2_pool PRIVATE tsgemm)
target_compile_definitions(v2_pool PRIVATE TSGEMM_USE_MPI_POOL)
