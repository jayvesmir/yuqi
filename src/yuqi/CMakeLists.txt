cmake_minimum_required(VERSION 3.14)

file(
    GLOB_RECURSE
    SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp"
)

add_library(
    ${PROJECT_NAME} ${SOURCES}
)

set_target_properties(
    ${PROJECT_NAME} PROPERTIES

    CXX_EXTENSIONS false
    CXX_SCAN_FOR_MODULES false
)

target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../../include")