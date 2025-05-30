#include "zfp.hpp"
using namespace zfp;

extern "C" {
  #include "constants/3dFloat.h"
  #include "utils/rand32.h"
}

#define ZFP_FIELD_FUNC zfp_field_3d
#define ZFP_ENCODE_BLOCK_FUNC zfp_encode_block_float_3
#define ZFP_ENCODE_BLOCK_STRIDED_FUNC zfp_encode_block_strided_float_3
#define ZFP_ENCODE_PARTIAL_BLOCK_STRIDED_FUNC zfp_encode_partial_block_strided_float_3

#define SCALAR float
#define DIMS 3

#include "testTemplatedEncodeBase.cpp"
