## This file should be placed in the root directory of your project.
## Then modify the CMakeLists.txt file in the root directory of your
## project to incorporate the testing dashboard.
## # The following are required to uses Dart and the Cdash dashboard
##   ENABLE_TESTING()
##   INCLUDE(CTest)
set(CTEST_PROJECT_NAME "ZFP")
set(CTEST_NIGHTLY_START_TIME "01:00:00 UTC")

set(CTEST_DROP_METHOD "https")
set(CTEST_DROP_SITE "open.cdash.org")
set(CTEST_DROP_LOCATION "/submit.php?project=zfp")
set(CTEST_DROP_SITE_CDASH TRUE)

# Test Options
set(MEMORYCHECK_COMMAND_OPTIONS "--show-reachable=no")
