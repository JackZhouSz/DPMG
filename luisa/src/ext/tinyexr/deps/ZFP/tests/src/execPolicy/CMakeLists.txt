add_executable(testOmp testOmp.c)
target_link_libraries(testOmp cmocka zfp)
add_test(NAME testOmp COMMAND testOmp)
if(ZFP_WITH_OPENMP)
  target_link_libraries(testOmp OpenMP::OpenMP_C)
  set_property(TEST testOmp PROPERTY RUN_SERIAL TRUE)
endif()

if(ZFP_WITH_OPENMP)
  add_executable(testOmpInternal testOmpInternal.c)
  target_link_libraries(testOmpInternal cmocka zfp OpenMP::OpenMP_C)
  add_test(NAME testOmpInternal COMMAND testOmpInternal)
endif()

if(ZFP_WITH_CUDA AND NOT DEFINED ZFP_OMP_TESTS_ONLY)
  add_executable(testCuda testCuda.c)
  target_link_libraries(testCuda cmocka zfp)
  add_test(NAME testCuda COMMAND testCuda)
endif()
