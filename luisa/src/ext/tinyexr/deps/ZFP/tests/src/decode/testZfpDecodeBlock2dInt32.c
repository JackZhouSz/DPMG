#include "src/decode2i.c"

#include "constants/2dInt32.h"
#include "utils/rand32.h"
#include "zfpDecodeBlockBase.c"

int main()
{
  const struct CMUnitTest tests[] = {
    #include "testcases/block.c"
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
