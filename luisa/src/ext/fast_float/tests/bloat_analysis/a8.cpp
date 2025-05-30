#include "fast_float/fast_float.h"


double get8(const char* input) {
    double result_value;
    auto result = fast_float::from_chars(input, input + strlen(input), result_value);
    if (result.ec != std::errc()) { return 8; }
    return result_value;
}