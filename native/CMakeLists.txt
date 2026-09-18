cmake_minimum_required(VERSION 3.22)
project(microduck_policy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(GODOTCPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.deps/godot-cpp" CACHE PATH "Pinned godot-cpp checkout")
set(ORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.deps/onnxruntime-linux-aarch64-1.29.0" CACHE PATH "ONNX Runtime C/C++ SDK")
set(GODOTCPP_BUILD_PROFILE "${CMAKE_CURRENT_SOURCE_DIR}/build_profile.json" CACHE FILEPATH "Minimal Godot API surface")
set(GODOTCPP_DISABLE_EXCEPTIONS OFF CACHE BOOL "ORT C++ API reports recoverable failures with exceptions" FORCE)
set(GODOTCPP_TARGET "template_debug" CACHE STRING "Must match the Godot editor/debug or release allocator ABI")
set(GODOTCPP_API_VERSION "4.7" CACHE STRING "Pinned Godot extension API bundled by godot-cpp")
set(MICRODUCK_PRECISION "single" CACHE STRING "Godot ABI precision: single or double")
set(GODOTCPP_PRECISION "${MICRODUCK_PRECISION}" CACHE STRING "" FORCE)
add_subdirectory("${GODOTCPP_DIR}" godot-cpp)
add_library(microduck_policy SHARED src/policy_runtime.cpp)
target_include_directories(microduck_policy PRIVATE "${ORT_DIR}/include")
target_link_libraries(microduck_policy PRIVATE godot-cpp "${ORT_DIR}/lib/libonnxruntime.so")
set_target_properties(microduck_policy PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../godot/native/bin"
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE)
if(MICRODUCK_PRECISION STREQUAL "double")
    set(MICRODUCK_PRECISION_SUFFIX ".double")
else()
    set(MICRODUCK_PRECISION_SUFFIX "")
endif()
if(GODOTCPP_TARGET STREQUAL "template_release")
    set_target_properties(microduck_policy PROPERTIES OUTPUT_NAME "microduck_policy.linux${MICRODUCK_PRECISION_SUFFIX}.release.arm64")
else()
    set_target_properties(microduck_policy PROPERTIES OUTPUT_NAME "microduck_policy.linux${MICRODUCK_PRECISION_SUFFIX}.debug.arm64")
endif()
add_custom_command(TARGET microduck_policy POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ORT_DIR}/lib/libonnxruntime.so.1.29.0"
        "$<TARGET_FILE_DIR:microduck_policy>/libonnxruntime.so.1.29.0"
    COMMAND ${CMAKE_COMMAND} -E create_symlink "libonnxruntime.so.1.29.0"
        "$<TARGET_FILE_DIR:microduck_policy>/libonnxruntime.so.1")
