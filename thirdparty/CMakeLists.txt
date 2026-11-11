include(FetchContent)

include(${CMAKE_CURRENT_LIST_DIR}/onnxruntime.cmake)
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_INCLUDE_DIR} PARENT_SCOPE)
set(ONNXRUNTIME_LIB_FILE ${ONNXRUNTIME_LIB_FILE} PARENT_SCOPE)
set(ONNXRUNTIME_BUNDLED ${ONNXRUNTIME_BUNDLED} PARENT_SCOPE)

# --- yaml-cpp (fallback when the system package is unavailable) ---
if(NOT yaml-cpp_FOUND AND NOT TARGET yaml-cpp::yaml-cpp AND NOT TARGET yaml-cpp)
    set(YAML_CPP_TGZ "${CMAKE_CURRENT_LIST_DIR}/yaml-cpp-0.9.0.tar.gz")
    set(YAML_CPP_URL "https://github.com/jbeder/yaml-cpp/releases/download/yaml-cpp-0.9.0/yaml-cpp-yaml-cpp-0.9.0.tar.gz")

    if(EXISTS "${YAML_CPP_TGZ}")
        message(STATUS "yaml-cpp: system package not found, using local tarball ${YAML_CPP_TGZ}")
        set(YAML_CPP_URL "${YAML_CPP_TGZ}")
    else()
        message(STATUS "yaml-cpp: system package not found, downloading from GitHub")
    endif()

    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        yaml-cpp
        URL      "${YAML_CPP_URL}"
        URL_HASH SHA256=298593d9c440fd9034b8b193d96318b76d49bc97c6ceadb7b0836edf0b6d7539
        SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/yaml-cpp-0.9.0"
    )
    FetchContent_MakeAvailable(yaml-cpp)
endif()

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
