# based on John's mpi futures
add_executable(v1 v1.cpp)
target_link_libraries(v1 PRIVATE tsgemm)

# based on manual yielding
add_executable(v2 v2.cpp)
target_link_libraries(v2 PRIVATE tsgemm)
