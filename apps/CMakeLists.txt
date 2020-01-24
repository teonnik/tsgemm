add_executable(v1 v1.cpp)
target_link_libraries(v1 PRIVATE tsgemm)

add_executable(v2 v2.cpp)
target_link_libraries(v2 PRIVATE tsgemm)

add_executable(v3 v3.cpp)
target_link_libraries(v3 PRIVATE tsgemm)
