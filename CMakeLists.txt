cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 20)
enable_testing()
project(oneMKL_Example)
find_package(MKL CONFIG REQUIRED)
message(STATUS "${MKL_IMPORTED_TARGETS}") #Provides available list of targets based on input

# add_compile_options(-mfma -mavx2 -mavx512f)
# add_compile_options(-mavx2 )
add_compile_options(-mfma -mavx2)

add_executable(myapp app.cpp)
target_compile_options(myapp PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
include_directories( $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(myapp PUBLIC $<LINK_ONLY:MKL::MKL>)
# add_test(NAME mytest COMMAND myapp)

if(MKL_ENV)
  set_tests_properties(mytest PROPERTIES ENVIRONMENT "${MKL_ENV}")
endif()


add_executable(matrix_test test/matrix_test.cpp)
add_executable(simd_support test/simd_support.cpp)
