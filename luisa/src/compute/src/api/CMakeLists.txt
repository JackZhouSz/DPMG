if (LUISA_COMPUTE_ENABLE_RUST)
    # option(LUISA_COMPUTE_RUST OFF)
    # if(NOT LUISA_COMPUTE_RUST)
    # execute_process(
    #         COMMAND cargo build
    #         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/luisa_compute_api_types)
    # endif()
    set(LUISA_COMPUTE_API_SOURCES
            runtime.cpp
            logging.cpp
            )
    add_library(luisa-compute-api SHARED ${LUISA_COMPUTE_API_SOURCES})
    target_link_libraries(luisa-compute-api PRIVATE
            luisa-compute-runtime
            luisa-compute-backends
            luisa-compute-ir
            )
    target_link_libraries(luisa-compute-api INTERFACE
            luisa-compute-rust-meta
            )
    set_target_properties(luisa-compute-api PROPERTIES
            UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD}
            OUTPUT_NAME lc-api)

    target_compile_definitions(luisa-compute-api PUBLIC LUISA_ENABLE_API)

    luisa_compute_install(api SOURCES ${LUISA_COMPUTE_API_SOURCES})

    # find_package(Python3 COMPONENTS Interpreter)
    # if (Python3_Interpreter_FOUND AND Python3_VERSION VERSION_GREATER_EQUAL 3.6)
    #     message(STATUS "Found suitable Python interpreter: ${Python3_VERSION}")
    #     set(LUISA_COMPUTE_API_PYTHON_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/generate_python_library.py")
    #     add_custom_target(luisa-compute-api-python ALL
    #             COMMAND ${Python3_EXECUTABLE} ${LUISA_COMPUTE_API_PYTHON_SCRIPT}
    #             DEPENDS ${LUISA_COMPUTE_API_PYTHON_SCRIPT}
    #             COMMENT "Updating Python bindings")
    #     add_dependencies(luisa-compute-api luisa-compute-api-python)
    # endif ()
else ()
    add_library(luisa-compute-api INTERFACE)
    luisa_compute_install(api)
endif ()
