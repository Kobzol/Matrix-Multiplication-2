cmake_minimum_required(VERSION 3.5)
project(mpi_test)

set(dir ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${dir})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${dir})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${dir})

add_executable(mpi_test
        matrix.cpp
        MPI.cpp
        mat.h
        main.cpp
)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
include_directories(MPI_INCLUDE_PATH)
target_link_libraries(mpi_test ${MPI_LIBRARIES})

SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")