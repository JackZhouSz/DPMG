#include "src/encode1l.c"

#include "constants/1dInt64.h"
#include "serialExecBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    #include "testcases/serial.c"
  };

  return cmocka_run_group_tests(tests, setupRandomData, teardownRandomData);
}
