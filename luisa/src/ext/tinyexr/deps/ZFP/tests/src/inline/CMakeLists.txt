add_executable(testBitstream testBitstream.c)
target_link_libraries(testBitstream cmocka zfp)
add_test(NAME testBitstream COMMAND testBitstream)

add_executable(testBitstreamSmallWsize testBitstreamSmallWsize.c)
target_link_libraries(testBitstreamSmallWsize cmocka)
add_test(NAME testBitstreamSmallWsize COMMAND testBitstreamSmallWsize)

add_executable(testBitstreamStrided testBitstreamStrided.c)
target_link_libraries(testBitstreamStrided cmocka)
add_test(NAME testBitstreamStrided COMMAND testBitstreamStrided)
