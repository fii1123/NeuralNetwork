cmake_minimum_required(VERSION 3.5)

project(FIA_NN LANGUAGES C)

add_library(LIB STATIC FIA_NN.c FIA_NN.h NN_func.h)

add_executable(FIA_NN main.c)

target_link_libraries(FIA_NN LIB)

target_compile_options(FIA_NN PRIVATE
    -ffloat-store
    -O3
    )
