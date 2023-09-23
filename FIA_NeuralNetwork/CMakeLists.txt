cmake_minimum_required(VERSION 3.5)

project(FIA_NeuralNetwork LANGUAGES C)

add_library(LIB STATIC FIA_NeuralNetwork.c)

add_executable(FIA_NeuralNetwork main.c)

target_link_libraries(FIA_NeuralNetwork LIB)
