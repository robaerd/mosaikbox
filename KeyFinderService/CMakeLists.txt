cmake_minimum_required(VERSION 3.22)
project(KeyFinderService)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# Path to manually installed libraries
# Replace "/User/keyfinder" with the actual installation path
set(KEYFINDER_PATH "/Users/robertsowula/DEV-local/TU/masterthesis/libkeyfinder/tmp_install" CACHE PATH "Path to manually installed keyfinder libraries")
message(STATUS "Using keyfinder library at path: ${KEYFINDER_PATH}")

# Find the libraries and headers
set(LIB_SUFFIX "" CACHE STRING "Suffice of libraries, e.g. 64 for lib64")
if (LIB_SUFFIX)
    set(CMAKE_LIBRARY_PATH /usr/local/lib${LIB_SUFFIX} ${CMAKE_LIBRARY_PATH})
endif ()

find_library(KEYFINDER_LIB keyfinder HINTS ${KEYFINDER_PATH}/lib${LIB_SUFFIX})
find_path(KEYFINDER_INCLUDE_DIR keyfinder.h HINTS ${KEYFINDER_PATH}/include/keyfinder)
message(STATUS "Found keyfinder library: ${KEYFINDER_LIB}")
message(STATUS "Found KEYFINDER_INCLUDE_DIR library: ${KEYFINDER_INCLUDE_DIR}")

# ------
# Logging
find_package(spdlog 1.12.0 QUIET)

if(NOT spdlog_FOUND)
    message(STATUS "spdlog not found, fetching it...")
    FetchContent_Declare(
            spdlog
            GIT_REPOSITORY https://github.com/gabime/spdlog.git
            GIT_TAG v1.12.0
    )
    FetchContent_MakeAvailable(spdlog)
endif()
# ------

# Find required protobuf package
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${protobuf_VERSION}")

set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)

# Find required gRPC package
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")

set(_GRPC_GRPCPP gRPC::grpc++_unsecure)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)

# Proto file
get_filename_component(hw_proto "proto/keyfinder.proto" ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)

# Generated sources
set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/keyfinder.pb.cc")
set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/keyfinder.pb.h")
set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/keyfinder.grpc.pb.cc")
set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/keyfinder.grpc.pb.h")

add_custom_command(
        OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${hw_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${hw_proto}"
        DEPENDS "${hw_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# Targets
add_library(keyfinder_proto_lib STATIC
        ${hw_proto_srcs}
        ${hw_proto_hdrs}
        ${hw_grpc_srcs}
        ${hw_grpc_hdrs})
target_include_directories(keyfinder_proto_lib PRIVATE $ENV{MY_INSTALL_DIR}/include)


add_executable(server src/server.cpp src/key_utils.cpp src/key_utils.h)

set(CMAKE_LIBRARY_PATH /usr/local/lib64 ${CMAKE_LIBRARY_PATH})

set(SHOULD_INCLUDE_FFTW3 OFF CACHE BOOL "Should include FFTW3 library")

if (SHOULD_INCLUDE_FFTW3)
    find_package(FFTW3)
    message(STATUS "Found FFTW3: ${FFTW3_INCLUDE_DIRS}")
    set(_FFTW3_LIB FFTW3::fftw3)
else ()
    message(WARNING "FFTW3 not found. Assuming keyfinder is installed as shared library.")
    set(_FFTW3_LIB "")
endif ()
#target_link_libraries(keyfinder PUBLIC FFTW3::fftw3)

target_link_libraries(server
        keyfinder_proto_lib
        ${KEYFINDER_LIB}
        ${_GRPC_GRPCPP}
        ${_PROTOBUF_LIBPROTOBUF}
        spdlog::spdlog
        ${_FFTW3_LIB}
        )
target_include_directories(server PRIVATE ${KEYFINDER_INCLUDE_DIR})
