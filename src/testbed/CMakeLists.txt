cmake_minimum_required(VERSION 3.14)

file(
    GLOB_RECURSE
    SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp"
)

add_executable(
    __${PROJECT_NAME}_testbed ${SOURCES}
)

set_target_properties(
    __${PROJECT_NAME}_testbed PROPERTIES

    OUTPUT_NAME "testbed"
)

target_link_libraries(__${PROJECT_NAME}_testbed PRIVATE ${PROJECT_NAME})