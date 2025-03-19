cmake_minimum_required(VERSION 3.10)
project(wasi_nn_llama C CXX)

include(FetchContent)

# Setup cJSON
FetchContent_Declare(
    cjson
    GIT_REPOSITORY https://github.com/DaveGamble/cJSON.git
    GIT_TAG v1.7.18
)
set(ENABLE_CJSON_TEST OFF CACHE INTERNAL "")
set(ENABLE_CJSON_UNINSTALL OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(cjson)

# Setup llama.cpp
FetchContent_Declare(
    llamacpp
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../_build/llama/llama.cpp
)
set(LLAMA_BUILD_TESTS OFF)
set(LLAMA_BUILD_EXAMPLES OFF)
set(LLAMA_BUILD_SERVER OFF)
FetchContent_MakeAvailable(llamacpp)

# Create wasi_nn_llama library
add_library(wasi_nn_llama SHARED
    src/wasi_nn_llamacpp.c
)

target_include_directories(wasi_nn_llama PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${cjson_SOURCE_DIR}
    ${llamacpp_SOURCE_DIR}
    ${llamacpp_SOURCE_DIR}/build
    ${llamacpp_SOURCE_DIR}/examples
    ${llamacpp_SOURCE_DIR}/common
)

target_link_libraries(wasi_nn_llama
    llama
    cjson
)

# Install headers
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)