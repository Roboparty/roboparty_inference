include(FetchContent)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(ONNXRUNTIME_NAME "onnxruntime-linux-x64-1.21.0")
    set(ONNXRUNTIME_HASH "SHA256=7485c7e7aac6501b27e353dcbe068e45c61ab51fbaf598d13970dfae669d20bf")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(ONNXRUNTIME_NAME "onnxruntime-linux-aarch64-1.21.0")
    set(ONNXRUNTIME_HASH "SHA256=4508084bde1232ee1ab4b6fad2155be0ea2ccab1c1aae9910ddb3fb68a60805e")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

message(STATUS "Architecture detected: ${CMAKE_SYSTEM_PROCESSOR}. Using ${ONNXRUNTIME_NAME}")

set(ONNXRUNTIME_LOCAL "${CMAKE_CURRENT_LIST_DIR}/${ONNXRUNTIME_NAME}.tgz")
set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/${ONNXRUNTIME_NAME}.tgz")

if(EXISTS "${ONNXRUNTIME_LOCAL}")
    message(STATUS "ONNX Runtime: using local tarball ${ONNXRUNTIME_LOCAL}")
    set(ONNXRUNTIME_URL "${ONNXRUNTIME_LOCAL}")
endif()

FetchContent_Declare(
    onnxruntime
    URL      "${ONNXRUNTIME_URL}"
    URL_HASH ${ONNXRUNTIME_HASH}
    SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/onnxruntime"
)

FetchContent_MakeAvailable(onnxruntime)

set(ONNXRUNTIME_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/onnxruntime/include")
set(ONNXRUNTIME_LIB_DIR "${CMAKE_CURRENT_LIST_DIR}/onnxruntime/lib")
set(ONNXRUNTIME_LIB_FILE "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so")

set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_INCLUDE_DIR} PARENT_SCOPE)
set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_LIB_DIR} PARENT_SCOPE)
set(ONNXRUNTIME_LIB_FILE ${ONNXRUNTIME_LIB_FILE} PARENT_SCOPE)

# --- yaml-cpp ---
set(YAML_CPP_TGZ "${CMAKE_CURRENT_LIST_DIR}/yaml-cpp-0.9.0.tar.gz")
if(EXISTS "${YAML_CPP_TGZ}")
    message(STATUS "yaml-cpp: using local tarball ${YAML_CPP_TGZ}")
    FetchContent_Declare(
        yaml-cpp
        URL      "${YAML_CPP_TGZ}"
        URL_HASH SHA256=298593d9c440fd9034b8b193d96318b76d49bc97c6ceadb7b0836edf0b6d7539
    )
else()
    message(STATUS "yaml-cpp: downloading from GitHub")
    FetchContent_Declare(
        yaml-cpp
        URL      https://github.com/jbeder/yaml-cpp/releases/download/yaml-cpp-0.9.0/yaml-cpp-yaml-cpp-0.9.0.tar.gz
        URL_HASH SHA256=298593d9c440fd9034b8b193d96318b76d49bc97c6ceadb7b0836edf0b6d7539
    )
endif()
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
option(YAML_BUILD_SHARED_LIBS "Build shared library" OFF)
option(YAML_CPP_BUILD_TESTS "Build tests" OFF)
option(YAML_CPP_BUILD_TOOLS "Build tools" OFF)
FetchContent_MakeAvailable(yaml-cpp)

# --- cnpy ---
find_package(ZLIB REQUIRED)

add_library(cnpy SHARED "cnpy/cnpy.cpp")
target_include_directories(cnpy PUBLIC ${CMAKE_CURRENT_LIST_DIR}/cnpy ${ZLIB_INCLUDE_DIRS})
target_link_libraries(cnpy ${ZLIB_LIBRARIES})

add_library(cnpy-static STATIC "cnpy/cnpy.cpp")
set_target_properties(cnpy-static PROPERTIES OUTPUT_NAME "cnpy")
target_include_directories(cnpy-static PUBLIC ${CMAKE_CURRENT_LIST_DIR}/cnpy ${ZLIB_INCLUDE_DIRS})

install(TARGETS "cnpy" LIBRARY DESTINATION lib PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
install(TARGETS "cnpy-static" ARCHIVE DESTINATION lib)
install(FILES "cnpy/cnpy.h" DESTINATION include)

set(CNPY_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/cnpy" PARENT_SCOPE)

# Don't build yaml-cpp tools (sandbox, parse, read)
foreach(_tgt yaml-cpp-sandbox yaml-cpp-parse yaml-cpp-read)
    if(TARGET ${_tgt})
        set_target_properties(${_tgt} PROPERTIES EXCLUDE_FROM_ALL TRUE)
    endif()
endforeach()
