#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "fast_float/fast_float.h"
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <ios>
#include <limits>
#include <string>
#include <system_error>
#include <type_traits>
#include <cfenv>

#if FASTFLOAT_IS_CONSTEXPR
#ifndef FASTFLOAT_CONSTEXPR_TESTS
#define FASTFLOAT_CONSTEXPR_TESTS 1
#endif // #ifndef FASTFLOAT_CONSTEXPR_TESTS
#endif // FASTFLOAT_IS_CONSTEXPR

#if FASTFLOAT_HAS_BIT_CAST
#include <bit>
#endif

#ifndef SUPPLEMENTAL_TEST_DATA_DIR
#define SUPPLEMENTAL_TEST_DATA_DIR "data/"
#endif

#ifndef __cplusplus
#error fastfloat requires a C++ compiler
#endif

#ifndef FASTFLOAT_CPLUSPLUS
#if defined(_MSVC_LANG) && !defined(__clang__)
#define FASTFLOAT_CPLUSPLUS (_MSC_VER == 1900 ? 201103L : _MSVC_LANG)
#else
#define FASTFLOAT_CPLUSPLUS __cplusplus
#endif
#endif


#if defined(__CYGWIN__) || defined(__MINGW32__) || defined(__MINGW64__) || defined(sun) || defined(__sun)
#define FASTFLOAT_ODDPLATFORM 1
#endif
#if defined __has_include
#if __has_include (<filesystem>)
#else
// filesystem is not available
#define FASTFLOAT_ODDPLATFORM 1
#endif
#else
// __has_include is not available
#define FASTFLOAT_ODDPLATFORM 1
#endif


#define iHexAndDec(v) std::hex << "0x" << (v) << " (" << std::dec << (v) << ")"
#define fHexAndDec(v) std::hexfloat << (v) << " (" << std::defaultfloat << (v) << ")"


const char * round_name(int d) {
  switch(d) {
    case FE_UPWARD:
      return "FE_UPWARD";
    case FE_DOWNWARD:
      return "FE_DOWNWARD";
    case FE_TOWARDZERO:
      return "FE_TOWARDZERO";
    case FE_TONEAREST:
      return "FE_TONEAREST";
    default:
      return "UNKNOWN";
  }
}


#define FASTFLOAT_STR(x)   #x
#define SHOW_DEFINE(x) printf("%s='%s'\n", #x, FASTFLOAT_STR(x))

TEST_CASE("system_info") {
    std::cout << "system info:" << std::endl;
#ifdef FASTFLOAT_CONSTEXPR_TESTS
    SHOW_DEFINE(FASTFLOAT_CONSTEXPR_TESTS);

#endif
#ifdef _MSC_VER
    SHOW_DEFINE(_MSC_VER);
#endif
#ifdef FASTFLOAT_64BIT_LIMB
    SHOW_DEFINE(FASTFLOAT_64BIT_LIMB);
#endif
#ifdef __clang__
    SHOW_DEFINE(__clang__);
#endif
#ifdef FASTFLOAT_VISUAL_STUDIO
    SHOW_DEFINE(FASTFLOAT_VISUAL_STUDIO);
#endif
#ifdef FASTFLOAT_IS_BIG_ENDIAN
    #if FASTFLOAT_IS_BIG_ENDIAN
      printf("big endian\n");
    #else
      printf("little endian\n");
    #endif
#endif
#ifdef FASTFLOAT_32BIT
    SHOW_DEFINE(FASTFLOAT_32BIT);
#endif
#ifdef FASTFLOAT_64BIT
    SHOW_DEFINE(FASTFLOAT_64BIT);
#endif
#ifdef FLT_EVAL_METHOD
    SHOW_DEFINE(FLT_EVAL_METHOD);
#endif
#ifdef _WIN32
    SHOW_DEFINE(_WIN32);
#endif
#ifdef _WIN64
    SHOW_DEFINE(_WIN64);
#endif
    std::cout << "fegetround() = " << round_name(fegetround()) << std::endl;
    std::cout << std::endl;

}


TEST_CASE("rounds_to_nearest") {
  //
  // If this function fails, we may be left in a non-standard rounding state.
  //
  static volatile float fmin = std::numeric_limits<float>::min();
  fesetround(FE_UPWARD);
  std::cout << "FE_UPWARD: fmin + 1.0f = " << iHexAndDec(fmin + 1.0f) << " 1.0f - fmin = " << iHexAndDec(1.0f - fmin) << std::endl;
  CHECK(fegetround() == FE_UPWARD);
  CHECK(fast_float::detail::rounds_to_nearest() == false);

  fesetround(FE_DOWNWARD);
  std::cout << "FE_DOWNWARD: fmin + 1.0f = " << iHexAndDec(fmin + 1.0f) << " 1.0f - fmin = " << iHexAndDec(1.0f - fmin) << std::endl;
  CHECK(fegetround() == FE_DOWNWARD);
  CHECK(fast_float::detail::rounds_to_nearest() == false);

  fesetround(FE_TOWARDZERO);
  std::cout << "FE_TOWARDZERO: fmin + 1.0f = " << iHexAndDec(fmin + 1.0f) << " 1.0f - fmin = " << iHexAndDec(1.0f - fmin) << std::endl;
  CHECK(fegetround() == FE_TOWARDZERO);
  CHECK(fast_float::detail::rounds_to_nearest() == false);

  fesetround(FE_TONEAREST);
  std::cout << "FE_TONEAREST: fmin + 1.0f = " << iHexAndDec(fmin + 1.0f) << " 1.0f - fmin = " << iHexAndDec(1.0f - fmin) << std::endl;
  CHECK(fegetround() == FE_TONEAREST);
#if (FLT_EVAL_METHOD == 1) || (FLT_EVAL_METHOD == 0)
  CHECK(fast_float::detail::rounds_to_nearest() == true);
#endif
}

TEST_CASE("parse_zero") {
  //
  // If this function fails, we may be left in a non-standard rounding state.
  //
  const char * zero = "0";
  uint64_t float64_parsed;
  double f = 0;
  ::memcpy(&float64_parsed, &f, sizeof(f));
  CHECK(float64_parsed == 0);

  fesetround(FE_UPWARD);
  auto r1 = fast_float::from_chars(zero, zero + 1, f);
  CHECK(r1.ec == std::errc());
  std::cout << "FE_UPWARD parsed zero as " << iHexAndDec(f) << std::endl;
  CHECK(f == 0);
  ::memcpy(&float64_parsed, &f, sizeof(f));
  std::cout << "double as uint64_t is " << float64_parsed << std::endl;
  CHECK(float64_parsed == 0);

  fesetround(FE_TOWARDZERO);
  auto r2 = fast_float::from_chars(zero, zero + 1, f);
  CHECK(r2.ec == std::errc());
  std::cout << "FE_TOWARDZERO parsed zero as " << iHexAndDec(f) << std::endl;
  CHECK(f == 0);
  ::memcpy(&float64_parsed, &f, sizeof(f));
  std::cout << "double as uint64_t is " << float64_parsed << std::endl;
  CHECK(float64_parsed == 0);

  fesetround(FE_DOWNWARD);
  auto r3 = fast_float::from_chars(zero, zero + 1, f);
  CHECK(r3.ec == std::errc());
  std::cout << "FE_DOWNWARD parsed zero as " << iHexAndDec(f) << std::endl;
  CHECK(f == 0);
  ::memcpy(&float64_parsed, &f, sizeof(f));
  std::cout << "double as uint64_t is " << float64_parsed << std::endl;
  CHECK(float64_parsed == 0);

  fesetround(FE_TONEAREST);
  auto r4 = fast_float::from_chars(zero, zero + 1, f);
  CHECK(r4.ec == std::errc());
  std::cout << "FE_TONEAREST parsed zero as " << iHexAndDec(f) << std::endl;
  CHECK(f == 0);
  ::memcpy(&float64_parsed, &f, sizeof(f));
  std::cout << "double as uint64_t is " << float64_parsed << std::endl;
  CHECK(float64_parsed == 0);
}

TEST_CASE("parse_negative_zero") {
  //
  // If this function fails, we may be left in a non-standard rounding state.
  //
  const char * negative_zero = "-0";
  uint64_t float64_parsed;
  double f = -0.;
  ::memcpy(&float64_parsed, &f, sizeof(f));
  CHECK(float64_parsed == 0x8000'0000'0000'0000);

  fesetround(FE_UPWARD);
  auto r1 = fast_float::from_chars(negative_zero, negative_zero + 2, f);
  CHECK(r1.ec == std::errc());
  std::cout << "FE_UPWARD parsed negative zero as " << iHexAndDec(f) << std::endl;
  CHECK(f == 0);
  ::memcpy(&float64_parsed, &f, sizeof(f));
  std::cout << "double as uint64_t is " << float64_parsed << std::endl;
  CHECK(float64_parsed == 0x8000'0000'0000'0000);

  fesetround(FE_TOWARDZERO);
  auto r2 = fast_float::from_chars(negative_zero, negative_zero + 2, f);
  CHECK(r2.ec == std::errc());
  std::cout << "FE_TOWARDZERO parsed negative zero as " << iHexAndDec(f) << std::endl;
  CHECK(f == 0);
  ::memcpy(&float64_parsed, &f, sizeof(f));
  std::cout << "double as uint64_t is " << float64_parsed << std::endl;
  CHECK(float64_parsed == 0x8000'0000'0000'0000);

  fesetround(FE_DOWNWARD);
  auto r3 = fast_float::from_chars(negative_zero, negative_zero + 2, f);
  CHECK(r3.ec == std::errc());
  std::cout << "FE_DOWNWARD parsed negative zero as " << iHexAndDec(f) << std::endl;
  CHECK(f == 0);
  ::memcpy(&float64_parsed, &f, sizeof(f));
  std::cout << "double as uint64_t is " << float64_parsed << std::endl;
  CHECK(float64_parsed == 0x8000'0000'0000'0000);

  fesetround(FE_TONEAREST);
  auto r4 = fast_float::from_chars(negative_zero, negative_zero + 2, f);
  CHECK(r4.ec == std::errc());
  std::cout << "FE_TONEAREST parsed negative zero as " << iHexAndDec(f) << std::endl;
  CHECK(f == 0);
  ::memcpy(&float64_parsed, &f, sizeof(f));
  std::cout << "double as uint64_t is " << float64_parsed << std::endl;
  CHECK(float64_parsed == 0x8000'0000'0000'0000);
}

// C++ 17 because it is otherwise annoying to browse all files in a directory.
// We also only run these tests on little endian systems.
#if (FASTFLOAT_CPLUSPLUS >= 201703L) && (FASTFLOAT_IS_BIG_ENDIAN == 0) && !defined(FASTFLOAT_ODDPLATFORM)

#include <iostream>
#include <filesystem>
#include <charconv>



// return true on success
bool check_file(std::string file_name) {
  std::cout << "Checking " << file_name << std::endl;
  // We check all rounding directions, for each file.
  std::vector<int> directions = {FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO, FE_TONEAREST};
  for (int d : directions) {
    std::cout << "fesetround to " << round_name(d) << std::endl;
    fesetround(d);
    size_t number{0};
    std::fstream newfile(file_name, std::ios::in);
    if (newfile.is_open()) {
      std::string str;
      while (std::getline(newfile, str)) {
        if (str.size() > 0) {
          // Read 32-bit hex
          uint32_t float32;
          auto r32 = std::from_chars(str.data() + 5, str.data() + str.size(),
                                   float32, 16);
          if(r32.ec != std::errc()) { std::cerr << "32-bit parsing failure\n"; return false; }
          // Read 64-bit hex
          uint64_t float64;
          auto r64 = std::from_chars(str.data() + 14, str.data() + str.size(),
                                   float64, 16);
          if(r64.ec != std::errc()) { std::cerr << "64-bit parsing failure\n"; return false; }
          // The string to parse:
          const char *number_string = str.data() + 31;
          const char *end_of_string = str.data() + str.size();
          // Parse as 32-bit float
          float parsed_32;
          auto fast_float_r32 = fast_float::from_chars(number_string, end_of_string, parsed_32);
          if(fast_float_r32.ec != std::errc() && fast_float_r32.ec != std::errc::result_out_of_range) {std::cerr << "32-bit fast_float parsing failure for: " + str + "\n"; return false; }
          // Parse as 64-bit float
          double parsed_64;
          auto fast_float_r64 = fast_float::from_chars(number_string, end_of_string, parsed_64);
          if(fast_float_r64.ec != std::errc() && fast_float_r32.ec != std::errc::result_out_of_range) { std::cerr << "64-bit fast_float parsing failure: " + str + "\n"; return false; }
          // Convert the floats to unsigned ints.
          uint32_t float32_parsed;
          uint64_t float64_parsed;
          ::memcpy(&float32_parsed, &parsed_32, sizeof(parsed_32));
          ::memcpy(&float64_parsed, &parsed_64, sizeof(parsed_64));
          // Compare with expected results
          if (float32_parsed != float32) {
            std::cout << "bad 32 " << str << std::endl;
            std::cout << "parsed as " << iHexAndDec(parsed_32) << std::endl;
            std::cout << "as  raw uint32_t, parsed = " << float32_parsed << ", expected = " << float32 << std::endl;
            std::cout << "fesetround: " << round_name(d) << std::endl;
            fesetround(FE_TONEAREST);
            return false;
          }
          if (float64_parsed != float64) {
            std::cout << "bad 64 " << str << std::endl;
            std::cout << "parsed as " << iHexAndDec(parsed_64) << std::endl;
            std::cout << "as raw uint64_t, parsed = " << float64_parsed << ", expected = " << float64 << std::endl;
            std::cout << "fesetround: " << round_name(d) << std::endl;
            fesetround(FE_TONEAREST);
            return false;
          }
          number++;
        }
      }
      std::cout << "checked " << std::defaultfloat << number << " values" << std::endl;
      newfile.close(); // close the file object
    } else {
      std::cout << "Could not read  " << file_name << std::endl;
      fesetround(FE_TONEAREST);
      return false;
    }
  }
  fesetround(FE_TONEAREST);
  return true;
}


TEST_CASE("supplemental") {
    std::string path = SUPPLEMENTAL_TEST_DATA_DIR;
    for (const auto & entry : std::filesystem::directory_iterator(path)) {
        CHECK(check_file(entry.path().string()));
    }
}
#endif

TEST_CASE("leading_zeroes") {
  constexpr const uint64_t bit = 1;
  CHECK(fast_float::leading_zeroes(bit <<  0) == 63);
  CHECK(fast_float::leading_zeroes(bit <<  1) == 62);
  CHECK(fast_float::leading_zeroes(bit <<  2) == 61);
  CHECK(fast_float::leading_zeroes(bit << 61) ==  2);
  CHECK(fast_float::leading_zeroes(bit << 62) ==  1);
  CHECK(fast_float::leading_zeroes(bit << 63) ==  0);
}

void test_full_multiplication(uint64_t lhs, uint64_t rhs, uint64_t expected_lo, uint64_t expected_hi) {
  fast_float::value128 v;
  v = fast_float::full_multiplication(lhs, rhs);
  INFO("lhs=" << iHexAndDec(lhs) << "  " << "rhs=" << iHexAndDec(rhs)
       << "\n  actualLo=" << iHexAndDec(v.low) << "  " << "actualHi=" << iHexAndDec(v.high)
       << "\n  expectedLo=" << iHexAndDec(expected_lo) << "  " << "expectedHi=" << iHexAndDec(expected_hi));
  CHECK_EQ(v.low, expected_lo);
  CHECK_EQ(v.high, expected_hi);
  v = fast_float::full_multiplication(rhs, lhs);
  CHECK_EQ(v.low, expected_lo);
  CHECK_EQ(v.high, expected_hi);
}

TEST_CASE("full_multiplication") {
  constexpr const uint64_t bit = 1;
  //                       lhs        rhs        lo         hi
  test_full_multiplication(bit << 0 , bit <<  0, 1u       , 0u);
  test_full_multiplication(bit << 0 , bit << 63, bit << 63, 0u);
  test_full_multiplication(bit << 1 , bit << 63, 0u       , 1u);
  test_full_multiplication(bit << 63, bit <<  0, bit << 63, 0u);
  test_full_multiplication(bit << 63, bit <<  1, 0u       , 1u);
  test_full_multiplication(bit << 63, bit <<  2, 0u       , 2u);
  test_full_multiplication(bit << 63, bit << 63, 0u       , bit << 62);
}


TEST_CASE("issue8") {
  const char* s =
      "3."
      "141592653589793238462643383279502884197169399375105820974944592307816406"
      "286208998628034825342117067982148086513282306647093844609550582231725359"
      "408128481117450284102701938521105559644622948954930381964428810975665933"
      "446128475648233786783165271201909145648566923460348610454326648213393607"
      "260249141273724587006606315588174881520920962829254091715364367892590360"
      "011330530548820466521384146951941511609433057270365759591953092186117381"
      "932611793105118548074462379962749567351885752724891227938183011949129833"
      "673362440656643086021394946395224737190702179860943702770539217176293176"
      "752384674818467669405132000568127145263560827785771342757789609173637178"
      "721468440901224953430146549585371050792279689258923542019956112129021960"
      "864034418159813629774771309960518707211349999998372978";
  for (int i = 0; i < 16; i++) {
    // Parse all but the last i chars. We should still get 3.141ish.
    double d = 0.0;
    auto answer = fast_float::from_chars(s, s + strlen(s) - i, d);
    CHECK_MESSAGE(answer.ec == std::errc(), "i=" << i);
    CHECK_MESSAGE(d == 0x1.921fb54442d18p+1, "i=" << i << "\n"
                  << std::string(s, strlen(s) - size_t(i)) << "\n"
                  << std::hexfloat << d << "\n"
                  << std::defaultfloat << "\n");
  }
}

TEST_CASE("check_behavior") {
  const std::string input =  "abc";
  double result;
  auto answer = fast_float::from_chars(input.data(), input.data()+input.size(), result);
  CHECK_MESSAGE(answer.ec != std::errc(), "expected parse failure");
  CHECK_MESSAGE(answer.ptr == input.data(), "If there is no pattern match, we should have ptr equals first");
}


TEST_CASE("decimal_point_parsing") {
  double result;
  fast_float::parse_options options{};
  {
    const std::string input = "1,25";
    auto answer = fast_float::from_chars_advanced(input.data(), input.data()+input.size(), result, options);
    CHECK_MESSAGE(answer.ec == std::errc(), "expected parse success");
    CHECK_MESSAGE(answer.ptr == input.data() + 1,
                  "Parsing should have stopped at comma");
    CHECK_EQ(result, 1.0);

    options.decimal_point = ',';
    answer = fast_float::from_chars_advanced(input.data(), input.data()+input.size(), result, options);
    CHECK_MESSAGE(answer.ec == std::errc(), "expected parse success");
    CHECK_MESSAGE(answer.ptr == input.data() + input.size(),
                  "Parsing should have stopped at end");
    CHECK_EQ(result, 1.25);
  }
  {
    const std::string input = "1.25";
    auto answer = fast_float::from_chars_advanced(input.data(), input.data()+input.size(), result, options);
    CHECK_MESSAGE(answer.ec == std::errc(), "expected parse success");
    CHECK_MESSAGE(answer.ptr == input.data() + 1,
                  "Parsing should have stopped at dot");
    CHECK_EQ(result, 1.0);

    options.decimal_point = '.';
    answer = fast_float::from_chars_advanced(input.data(), input.data()+input.size(), result, options);
    CHECK_MESSAGE(answer.ec == std::errc(), "expected parse success");
    CHECK_MESSAGE(answer.ptr == input.data() + input.size(),
                  "Parsing should have stopped at end");
    CHECK_EQ(result, 1.25);
  }
}

TEST_CASE("issue19") {
  const std::string input =   "234532.3426362,7869234.9823,324562.645";
  double result;
  auto answer = fast_float::from_chars(input.data(), input.data()+input.size(), result);
  CHECK_MESSAGE(answer.ec == std::errc(), "We want to parse up to 234532.3426362\n");
  CHECK_MESSAGE(answer.ptr == input.data() + 14,
                "Parsed the number " << result
                << " and stopped at the wrong character: after " << (answer.ptr - input.data()) << " characters");
  CHECK_MESSAGE(result == 234532.3426362, "We want to parse234532.3426362\n");
  CHECK_MESSAGE(answer.ptr[0] == ',', "We want to parse up to the comma\n");

  answer = fast_float::from_chars(answer.ptr + 1, input.data()+input.size(), result);
  CHECK_MESSAGE(answer.ec == std::errc(), "We want to parse 7869234.9823\n");
  CHECK_MESSAGE(answer.ptr == input.data() + 27,
                "Parsed the number " << result
                << " and stopped at the wrong character " << (answer.ptr - input.data()));
  CHECK_MESSAGE(answer.ptr[0] == ',', "We want to parse up to the comma\n");
  CHECK_MESSAGE(result == 7869234.9823, "We want to parse up 7869234.9823\n");

  answer = fast_float::from_chars(answer.ptr + 1, input.data()+input.size(), result);
  CHECK_MESSAGE(answer.ec == std::errc(), "We want to parse 324562.645\n");
  CHECK_MESSAGE(answer.ptr == input.data() + 38,
                "Parsed the number " << result
                << " and stopped at the wrong character " << (answer.ptr - input.data()));
  CHECK_MESSAGE(result == 324562.645, "We want to parse up 7869234.9823\n");
}

TEST_CASE("issue19") {
  const std::string input =  "3.14e";
  double result;
  auto answer = fast_float::from_chars(input.data(), input.data()+input.size(), result);
  CHECK_MESSAGE(answer.ec == std::errc(), "We want to parse up to 3.14\n");
  CHECK_MESSAGE(answer.ptr == input.data() + 4,
                "Parsed the number " << result
                << " and stopped at the wrong character: after " << (answer.ptr - input.data()) << " characters");
}


TEST_CASE("scientific_only") {
    // first, we try with something that should fail...
    std::string input =  "3.14";
    double result;
    auto answer = fast_float::from_chars(input.data(), input.data()+input.size(), result, fast_float::chars_format::scientific);
    CHECK_MESSAGE(answer.ec != std::errc(), "It is not scientific! Parsed: " << result);

    input = "3.14e10";
    answer = fast_float::from_chars(input.data(), input.data()+input.size(), result, fast_float::chars_format::scientific);
    CHECK_MESSAGE(answer.ec == std::errc(), "It is scientific! Parsed: " << result);
    CHECK_MESSAGE(answer.ptr == input.data() + input.size(),
                  "Parsed the number " << result
                  << " and stopped at the wrong character: after " << (answer.ptr - input.data()) << " characters");
}


TEST_CASE("test_fixed_only") {
    const std::string input =  "3.14e10";
    double result;
    auto answer = fast_float::from_chars(input.data(), input.data()+input.size(), result, fast_float::chars_format::fixed);
    CHECK_MESSAGE(answer.ec == std::errc(), "We want to parse up to 3.14; parsed: " << result);
    CHECK_MESSAGE(answer.ptr == input.data() + 4,
                  "Parsed the number " << result
                  << " and stopped at the wrong character: after " << (answer.ptr - input.data()) << " characters");
}


static const double testing_power_of_ten[] = {
      1e-323, 1e-322, 1e-321, 1e-320, 1e-319, 1e-318, 1e-317, 1e-316, 1e-315,
      1e-314, 1e-313, 1e-312, 1e-311, 1e-310, 1e-309, 1e-308,

      1e-307, 1e-306, 1e-305, 1e-304, 1e-303, 1e-302, 1e-301, 1e-300, 1e-299,
      1e-298, 1e-297, 1e-296, 1e-295, 1e-294, 1e-293, 1e-292, 1e-291, 1e-290,
      1e-289, 1e-288, 1e-287, 1e-286, 1e-285, 1e-284, 1e-283, 1e-282, 1e-281,
      1e-280, 1e-279, 1e-278, 1e-277, 1e-276, 1e-275, 1e-274, 1e-273, 1e-272,
      1e-271, 1e-270, 1e-269, 1e-268, 1e-267, 1e-266, 1e-265, 1e-264, 1e-263,
      1e-262, 1e-261, 1e-260, 1e-259, 1e-258, 1e-257, 1e-256, 1e-255, 1e-254,
      1e-253, 1e-252, 1e-251, 1e-250, 1e-249, 1e-248, 1e-247, 1e-246, 1e-245,
      1e-244, 1e-243, 1e-242, 1e-241, 1e-240, 1e-239, 1e-238, 1e-237, 1e-236,
      1e-235, 1e-234, 1e-233, 1e-232, 1e-231, 1e-230, 1e-229, 1e-228, 1e-227,
      1e-226, 1e-225, 1e-224, 1e-223, 1e-222, 1e-221, 1e-220, 1e-219, 1e-218,
      1e-217, 1e-216, 1e-215, 1e-214, 1e-213, 1e-212, 1e-211, 1e-210, 1e-209,
      1e-208, 1e-207, 1e-206, 1e-205, 1e-204, 1e-203, 1e-202, 1e-201, 1e-200,
      1e-199, 1e-198, 1e-197, 1e-196, 1e-195, 1e-194, 1e-193, 1e-192, 1e-191,
      1e-190, 1e-189, 1e-188, 1e-187, 1e-186, 1e-185, 1e-184, 1e-183, 1e-182,
      1e-181, 1e-180, 1e-179, 1e-178, 1e-177, 1e-176, 1e-175, 1e-174, 1e-173,
      1e-172, 1e-171, 1e-170, 1e-169, 1e-168, 1e-167, 1e-166, 1e-165, 1e-164,
      1e-163, 1e-162, 1e-161, 1e-160, 1e-159, 1e-158, 1e-157, 1e-156, 1e-155,
      1e-154, 1e-153, 1e-152, 1e-151, 1e-150, 1e-149, 1e-148, 1e-147, 1e-146,
      1e-145, 1e-144, 1e-143, 1e-142, 1e-141, 1e-140, 1e-139, 1e-138, 1e-137,
      1e-136, 1e-135, 1e-134, 1e-133, 1e-132, 1e-131, 1e-130, 1e-129, 1e-128,
      1e-127, 1e-126, 1e-125, 1e-124, 1e-123, 1e-122, 1e-121, 1e-120, 1e-119,
      1e-118, 1e-117, 1e-116, 1e-115, 1e-114, 1e-113, 1e-112, 1e-111, 1e-110,
      1e-109, 1e-108, 1e-107, 1e-106, 1e-105, 1e-104, 1e-103, 1e-102, 1e-101,
      1e-100, 1e-99,  1e-98,  1e-97,  1e-96,  1e-95,  1e-94,  1e-93,  1e-92,
      1e-91,  1e-90,  1e-89,  1e-88,  1e-87,  1e-86,  1e-85,  1e-84,  1e-83,
      1e-82,  1e-81,  1e-80,  1e-79,  1e-78,  1e-77,  1e-76,  1e-75,  1e-74,
      1e-73,  1e-72,  1e-71,  1e-70,  1e-69,  1e-68,  1e-67,  1e-66,  1e-65,
      1e-64,  1e-63,  1e-62,  1e-61,  1e-60,  1e-59,  1e-58,  1e-57,  1e-56,
      1e-55,  1e-54,  1e-53,  1e-52,  1e-51,  1e-50,  1e-49,  1e-48,  1e-47,
      1e-46,  1e-45,  1e-44,  1e-43,  1e-42,  1e-41,  1e-40,  1e-39,  1e-38,
      1e-37,  1e-36,  1e-35,  1e-34,  1e-33,  1e-32,  1e-31,  1e-30,  1e-29,
      1e-28,  1e-27,  1e-26,  1e-25,  1e-24,  1e-23,  1e-22,  1e-21,  1e-20,
      1e-19,  1e-18,  1e-17,  1e-16,  1e-15,  1e-14,  1e-13,  1e-12,  1e-11,
      1e-10,  1e-9,   1e-8,   1e-7,   1e-6,   1e-5,   1e-4,   1e-3,   1e-2,
      1e-1,   1e0,    1e1,    1e2,    1e3,    1e4,    1e5,    1e6,    1e7,
      1e8,    1e9,    1e10,   1e11,   1e12,   1e13,   1e14,   1e15,   1e16,
      1e17,   1e18,   1e19,   1e20,   1e21,   1e22,   1e23,   1e24,   1e25,
      1e26,   1e27,   1e28,   1e29,   1e30,   1e31,   1e32,   1e33,   1e34,
      1e35,   1e36,   1e37,   1e38,   1e39,   1e40,   1e41,   1e42,   1e43,
      1e44,   1e45,   1e46,   1e47,   1e48,   1e49,   1e50,   1e51,   1e52,
      1e53,   1e54,   1e55,   1e56,   1e57,   1e58,   1e59,   1e60,   1e61,
      1e62,   1e63,   1e64,   1e65,   1e66,   1e67,   1e68,   1e69,   1e70,
      1e71,   1e72,   1e73,   1e74,   1e75,   1e76,   1e77,   1e78,   1e79,
      1e80,   1e81,   1e82,   1e83,   1e84,   1e85,   1e86,   1e87,   1e88,
      1e89,   1e90,   1e91,   1e92,   1e93,   1e94,   1e95,   1e96,   1e97,
      1e98,   1e99,   1e100,  1e101,  1e102,  1e103,  1e104,  1e105,  1e106,
      1e107,  1e108,  1e109,  1e110,  1e111,  1e112,  1e113,  1e114,  1e115,
      1e116,  1e117,  1e118,  1e119,  1e120,  1e121,  1e122,  1e123,  1e124,
      1e125,  1e126,  1e127,  1e128,  1e129,  1e130,  1e131,  1e132,  1e133,
      1e134,  1e135,  1e136,  1e137,  1e138,  1e139,  1e140,  1e141,  1e142,
      1e143,  1e144,  1e145,  1e146,  1e147,  1e148,  1e149,  1e150,  1e151,
      1e152,  1e153,  1e154,  1e155,  1e156,  1e157,  1e158,  1e159,  1e160,
      1e161,  1e162,  1e163,  1e164,  1e165,  1e166,  1e167,  1e168,  1e169,
      1e170,  1e171,  1e172,  1e173,  1e174,  1e175,  1e176,  1e177,  1e178,
      1e179,  1e180,  1e181,  1e182,  1e183,  1e184,  1e185,  1e186,  1e187,
      1e188,  1e189,  1e190,  1e191,  1e192,  1e193,  1e194,  1e195,  1e196,
      1e197,  1e198,  1e199,  1e200,  1e201,  1e202,  1e203,  1e204,  1e205,
      1e206,  1e207,  1e208,  1e209,  1e210,  1e211,  1e212,  1e213,  1e214,
      1e215,  1e216,  1e217,  1e218,  1e219,  1e220,  1e221,  1e222,  1e223,
      1e224,  1e225,  1e226,  1e227,  1e228,  1e229,  1e230,  1e231,  1e232,
      1e233,  1e234,  1e235,  1e236,  1e237,  1e238,  1e239,  1e240,  1e241,
      1e242,  1e243,  1e244,  1e245,  1e246,  1e247,  1e248,  1e249,  1e250,
      1e251,  1e252,  1e253,  1e254,  1e255,  1e256,  1e257,  1e258,  1e259,
      1e260,  1e261,  1e262,  1e263,  1e264,  1e265,  1e266,  1e267,  1e268,
      1e269,  1e270,  1e271,  1e272,  1e273,  1e274,  1e275,  1e276,  1e277,
      1e278,  1e279,  1e280,  1e281,  1e282,  1e283,  1e284,  1e285,  1e286,
      1e287,  1e288,  1e289,  1e290,  1e291,  1e292,  1e293,  1e294,  1e295,
      1e296,  1e297,  1e298,  1e299,  1e300,  1e301,  1e302,  1e303,  1e304,
      1e305,  1e306,  1e307,  1e308};


TEST_CASE("powers_of_ten") {
    char buf[1024];
    WARN_MESSAGE(1e-308 == std::pow(10, -308), "On your system, the pow function is busted. Sorry about that.");
    bool is_pow_correct{1e-308 == std::pow(10,-308)};
    // large negative values should be zero.
    int start_point = is_pow_correct ? -1000 : -307;
    for (int i = start_point; i <= 308; ++i) {
      INFO("i=" << i);
      size_t n = size_t(snprintf(buf, sizeof(buf), "1e%d", i));
      REQUIRE(n < sizeof(buf)); // if false, fails the test and exits
      double actual;
      auto result = fast_float::from_chars(buf, buf + 1000, actual);
      double expected = ((i >= -323) ? testing_power_of_ten[i + 323] : std::pow(10, i));
      auto expected_ec = (i < -323 || i > 308) ? std::errc::result_out_of_range : std::errc();
      CHECK_MESSAGE(result.ec == expected_ec, " I could not parse " << buf);
      CHECK_MESSAGE(actual == expected, "String '" << buf << "'parsed to " << actual);
    }
}


template <typename T> std::string to_string(T d) {
  std::string s(64, '\0');
  auto written = std::snprintf(&s[0], s.size(), "%.*e",
                               std::numeric_limits<T>::max_digits10 - 1, d);
  s.resize(size_t(written));
  return s;
}

template <typename T> std::string to_long_string(T d) {
  std::string s(4096, '\0');
  auto written = std::snprintf(&s[0], s.size(), "%.*e",
                               std::numeric_limits<T>::max_digits10 * 10, d);
  s.resize(size_t(written));
  return s;
}

uint32_t get_mantissa(float f) {
  uint32_t m;
  memcpy(&m, &f, sizeof(f));
  return (m & ((uint32_t(1)<<23)-1));
}

uint64_t get_mantissa(double f) {
  uint64_t m;
  memcpy(&m, &f, sizeof(f));
  return (m & ((uint64_t(1)<<57)-1));
}


std::string append_zeros(std::string str, size_t number_of_zeros) {
  std::string answer(str);
  for(size_t i = 0; i < number_of_zeros; i++) { answer += "0"; }
  return answer;
}

namespace {

enum class Diag { runtime, comptime };

} // anonymous namespace

template <Diag diag, class T, typename result_type, typename stringtype>
constexpr void check_basic_test_result(stringtype str,
                                       result_type result,
                                       T actual, T expected, std::errc expected_ec) {
  if constexpr (diag == Diag::runtime) {
      INFO(
          "str=" << str << "\n"
                 << "  expected=" << fHexAndDec(expected) << "\n"
                 << "  ..actual=" << fHexAndDec(actual) << "\n"
                 << "  expected mantissa=" << iHexAndDec(get_mantissa(expected))
                 << "\n"
                 << "  ..actual mantissa=" << iHexAndDec(get_mantissa(actual)));
  }

  struct ComptimeDiag {
      // Purposely not constexpr
      static void error_not_equal() {}
  };

#define FASTFLOAT_CHECK_EQ(...)                                                \
  if constexpr (diag == Diag::runtime) {                                       \
    CHECK_EQ(__VA_ARGS__);                                                     \
  } else {                                                                     \
    if ([](const auto &lhs, const auto &rhs) {                                 \
          return lhs != rhs;                                                   \
        }(__VA_ARGS__)) {                                                      \
      ComptimeDiag::error_not_equal();                                         \
    }                                                                          \
  }

  auto copysign = [](double x, double y) -> double {
#if FASTFLOAT_HAS_BIT_CAST
    if (fast_float::cpp20_and_in_constexpr()) {
      using equiv_int = std::make_signed_t<
          typename fast_float::binary_format<double>::equiv_uint>;
      const auto i = std::bit_cast<equiv_int>(y);
      if (i < 0) {
        return -x;
      }
      return x;
    }
#endif
    return std::copysign(x, y);
  };

  auto isnan = [](double x) -> bool {
    return x != x;
  };

  FASTFLOAT_CHECK_EQ(result.ec, expected_ec);
  FASTFLOAT_CHECK_EQ(result.ptr, str.data() + str.size());
  FASTFLOAT_CHECK_EQ(copysign(1, actual), copysign(1, expected));
  FASTFLOAT_CHECK_EQ(isnan(actual), isnan(expected));
  FASTFLOAT_CHECK_EQ(actual, expected);

#undef FASTFLOAT_CHECK_EQ
}


template<Diag diag, class T>
constexpr void basic_test(std::string_view str, T expected, std::errc expected_ec = std::errc()) {
  T actual;
  auto result = fast_float::from_chars(str.data(), str.data() + str.size(), actual);
  check_basic_test_result<diag>(str, result, actual, expected, expected_ec);
  constexpr size_t global_string_capacity = 2048;

  if(str.size() > global_string_capacity) {
    return;
  }
  // We give plenty of memory: 2048 characters.
  char16_t u16[global_string_capacity]{};

  for (size_t i = 0; i < str.size(); i++) {
    u16[i] = char16_t(str[i]);
  }
  auto result16 = fast_float::from_chars(u16, u16 + str.size(), actual);
  check_basic_test_result<diag>(std::u16string_view(u16, str.size()), result16, actual, expected, expected_ec);

  char32_t u32[global_string_capacity]{};

  for (size_t i = 0; i < str.size(); i++) {
    u32[i] = char32_t(str[i]);
  }
  auto result32 = fast_float::from_chars(u32, u32 + str.size(), actual);
  check_basic_test_result<diag>(std::u32string_view(u32, str.size()), result32, actual, expected, expected_ec);
}

template<Diag diag, class T>
constexpr void basic_test(std::string_view str, T expected, fast_float::parse_options options) {
  T actual;
  auto result = fast_float::from_chars_advanced(str.data(), str.data() + str.size(), actual, options);
  check_basic_test_result<diag>(str, result, actual, expected, std::errc());
}

template<Diag diag, class T>
constexpr void basic_test(std::string_view str, T expected, std::errc expected_ec, fast_float::parse_options options) {
  T actual;
  auto result = fast_float::from_chars_advanced(str.data(), str.data() + str.size(), actual, options);
  check_basic_test_result<diag>(str, result, actual, expected, expected_ec);
}

void basic_test(float val) {
  {
      std::string long_vals = to_long_string(val);
      INFO("long vals: " << long_vals);
      basic_test<Diag::runtime, float>(long_vals, val);
  }
  {
      std::string vals = to_string(val);
      INFO("vals: " << vals);
      basic_test<Diag::runtime, float>(vals, val);
  }
}

#define verify_runtime(...)                                                    \
  do {                                                                         \
    basic_test<Diag::runtime>(__VA_ARGS__);                                    \
  } while (false)

#define verify_comptime(...)                                                   \
  do {                                                                         \
    constexpr int verify_comptime_var =                                        \
        (basic_test<Diag::comptime>(__VA_ARGS__), 0);                          \
    (void)verify_comptime_var;                                                 \
  } while (false)

#define verify_options_runtime(...)                                            \
  do {                                                                         \
    basic_test<Diag::runtime>(__VA_ARGS__, options);                           \
  } while (false)

#define verify_options_comptime(...)                                           \
  do {                                                                         \
    constexpr int verify_options_comptime_var =                                \
        (basic_test<Diag::comptime>(__VA_ARGS__, options), 0);                 \
    (void)verify_options_comptime_var;                                         \
  } while (false)

#if defined(FASTFLOAT_CONSTEXPR_TESTS)
#if !FASTFLOAT_IS_CONSTEXPR
#error "from_chars must be constexpr for constexpr tests"
#endif

#define verify(...)                                                            \
  do {                                                                         \
    verify_runtime(__VA_ARGS__);                                               \
    verify_comptime(__VA_ARGS__);                                              \
  } while (false)

#define verify_options(...)                                                    \
  do {                                                                         \
    verify_options_runtime(__VA_ARGS__);                                       \
    verify_options_comptime(__VA_ARGS__);                                      \
  } while (false)

#else
#define verify verify_runtime
#define verify_options verify_options_runtime
#endif

#define verify32(val) { INFO(#val); basic_test(val); }

TEST_CASE("64bit.inf") {
  verify("INF", std::numeric_limits<double>::infinity());
  verify("-INF", -std::numeric_limits<double>::infinity());
  verify("INFINITY", std::numeric_limits<double>::infinity());
  verify("-INFINITY", -std::numeric_limits<double>::infinity());
  verify("infinity", std::numeric_limits<double>::infinity());
  verify("-infinity", -std::numeric_limits<double>::infinity());
  verify("inf", std::numeric_limits<double>::infinity());
  verify("-inf", -std::numeric_limits<double>::infinity());
  verify("1234456789012345678901234567890e9999999999999999999999999999", std::numeric_limits<double>::infinity(), std::errc::result_out_of_range);
  verify("-2139879401095466344511101915470454744.9813888656856943E+272", -std::numeric_limits<double>::infinity(), std::errc::result_out_of_range);
  verify("1.8e308", std::numeric_limits<double>::infinity(), std::errc::result_out_of_range);
  verify("1.832312213213213232132132143451234453123412321321312e308", std::numeric_limits<double>::infinity(), std::errc::result_out_of_range);
  verify("2e30000000000000000", std::numeric_limits<double>::infinity(), std::errc::result_out_of_range);
  verify("2e3000", std::numeric_limits<double>::infinity(), std::errc::result_out_of_range);
  verify("1.9e308", std::numeric_limits<double>::infinity(), std::errc::result_out_of_range);
}

TEST_CASE("64bit.general") {
  verify("22250738585072012e-324",0x1p-1022); /* limit between normal and subnormal*/
  verify("-22250738585072012e-324",-0x1p-1022); /* limit between normal and subnormal*/
  verify("-1e-999", -0.0, std::errc::result_out_of_range);
  verify("-2.2222222222223e-322",-0x1.68p-1069);
  verify("9007199254740993.0", 0x1p+53);
  verify("860228122.6654514319E+90", 0x1.92bb20990715fp+328);
  verify_runtime(append_zeros("9007199254740993.0",1000), 0x1p+53);
  verify("10000000000000000000", 0x1.158e460913dp+63);
  verify("10000000000000000000000000000001000000000000", 0x1.cb2d6f618c879p+142);
  verify("10000000000000000000000000000000000000000001", 0x1.cb2d6f618c879p+142);
  verify("1.1920928955078125e-07", 1.1920928955078125e-07);
  verify("9355950000000000000.00000000000000000000000000000000001844674407370955161600000184467440737095516161844674407370955161407370955161618446744073709551616000184467440737095516166000001844674407370955161618446744073709551614073709551616184467440737095516160001844674407370955161601844674407370955674451616184467440737095516140737095516161844674407370955161600018446744073709551616018446744073709551611616000184467440737095001844674407370955161600184467440737095516160018446744073709551168164467440737095516160001844073709551616018446744073709551616184467440737095516160001844674407536910751601611616000184467440737095001844674407370955161600184467440737095516160018446744073709551616184467440737095516160001844955161618446744073709551616000184467440753691075160018446744073709",0x1.03ae05e8fca1cp+63);
  verify("-0",-0.0);
  verify("2.22507385850720212418870147920222032907240528279439037814303133837435107319244194686754406432563881851382188218502438069999947733013005649884107791928741341929297200970481951993067993290969042784064731682041565926728632933630474670123316852983422152744517260835859654566319282835244787787799894310779783833699159288594555213714181128458251145584319223079897504395086859412457230891738946169368372321191373658977977723286698840356390251044443035457396733706583981055420456693824658413747607155981176573877626747665912387199931904006317334709003012790188175203447190250028061277777916798391090578584006464715943810511489154282775041174682194133952466682503431306181587829379004205392375072083366693241580002758391118854188641513168478436313080237596295773983001708984375e-308", 0x1.0000000000002p-1022);
  verify("1.0000000000000006661338147750939242541790008544921875",1.0000000000000007);
  verify("1090544144181609348835077142190",0x1.b8779f2474dfbp+99);
  verify("2.2250738585072013e-308", 2.2250738585072013e-308);
  verify("-92666518056446206563E3", -92666518056446206563E3);
  verify("-92666518056446206563E3", -92666518056446206563E3);
  verify("-42823146028335318693e-128",-42823146028335318693e-128);
  verify("90054602635948575728E72",90054602635948575728E72);
  verify("1.00000000000000188558920870223463870174566020691753515394643550663070558368373221972569761144603605635692374830246134201063722058e-309", 1.00000000000000188558920870223463870174566020691753515394643550663070558368373221972569761144603605635692374830246134201063722058e-309);
  verify("0e9999999999999999999999999999", 0.0);
  verify("-2402844368454405395.2", -2402844368454405395.2);
  verify("2402844368454405395.2", 2402844368454405395.2);
  verify("7.0420557077594588669468784357561207962098443483187940792729600000e+59", 7.0420557077594588669468784357561207962098443483187940792729600000e+59);
  verify("7.0420557077594588669468784357561207962098443483187940792729600000e+59", 7.0420557077594588669468784357561207962098443483187940792729600000e+59);
  verify("-1.7339253062092163730578609458683877051596800000000000000000000000e+42", -1.7339253062092163730578609458683877051596800000000000000000000000e+42);
  verify("-2.0972622234386619214559824785284023792871122537545728000000000000e+52", -2.0972622234386619214559824785284023792871122537545728000000000000e+52);
  verify("-1.0001803374372191849407179462120053338028379051879898808320000000e+57", -1.0001803374372191849407179462120053338028379051879898808320000000e+57);
  verify("-1.8607245283054342363818436991534856973992070520151142825984000000e+58", -1.8607245283054342363818436991534856973992070520151142825984000000e+58);
  verify("-1.9189205311132686907264385602245237137907390376574976000000000000e+52", -1.9189205311132686907264385602245237137907390376574976000000000000e+52);
  verify("-2.8184483231688951563253238886553506793085187889855201280000000000e+54", -2.8184483231688951563253238886553506793085187889855201280000000000e+54);
  verify("-1.7664960224650106892054063261344555646357024359107788800000000000e+53", -1.7664960224650106892054063261344555646357024359107788800000000000e+53);
  verify("-2.1470977154320536489471030463761883783915110400000000000000000000e+45", -2.1470977154320536489471030463761883783915110400000000000000000000e+45);
  verify("-4.4900312744003159009338275160799498340862630046359789166919680000e+61", -4.4900312744003159009338275160799498340862630046359789166919680000e+61);
  verify("1", 1.0);
  verify("1.797693134862315700000000000000001e308", 1.7976931348623157e308);
  verify("3e-324", 0x0.0000000000001p-1022);
  verify("1.00000006e+09", 0x1.dcd651ep+29);
  verify("4.9406564584124653e-324", 0x0.0000000000001p-1022);
  verify("4.9406564584124654e-324", 0x0.0000000000001p-1022);
  verify("2.2250738585072009e-308", 0x0.fffffffffffffp-1022);
  verify("2.2250738585072014e-308", 0x1p-1022);
  verify("1.7976931348623157e308", 0x1.fffffffffffffp+1023);
  verify("1.7976931348623158e308", 0x1.fffffffffffffp+1023);
  verify("4503599627370496.5", 4503599627370496.5);
  verify("4503599627475352.5", 4503599627475352.5);
  verify("4503599627475353.5", 4503599627475353.5);
  verify("2251799813685248.25", 2251799813685248.25);
  verify("1125899906842624.125", 1125899906842624.125);
  verify("1125899906842901.875", 1125899906842901.875);
  verify("2251799813685803.75", 2251799813685803.75);
  verify("4503599627370497.5", 4503599627370497.5);
  verify("45035996.273704995", 45035996.273704995);
  verify("45035996.273704985", 45035996.273704985);
  verify("0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000044501477170144022721148195934182639518696390927032912960468522194496444440421538910330590478162701758282983178260792422137401728773891892910553144148156412434867599762821265346585071045737627442980259622449029037796981144446145705102663115100318287949527959668236039986479250965780342141637013812613333119898765515451440315261253813266652951306000184917766328660755595837392240989947807556594098101021612198814605258742579179000071675999344145086087205681577915435923018910334964869420614052182892431445797605163650903606514140377217442262561590244668525767372446430075513332450079650686719491377688478005309963967709758965844137894433796621993967316936280457084866613206797017728916080020698679408551343728867675409720757232455434770912461317493580281734466552734375", 0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000044501477170144022721148195934182639518696390927032912960468522194496444440421538910330590478162701758282983178260792422137401728773891892910553144148156412434867599762821265346585071045737627442980259622449029037796981144446145705102663115100318287949527959668236039986479250965780342141637013812613333119898765515451440315261253813266652951306000184917766328660755595837392240989947807556594098101021612198814605258742579179000071675999344145086087205681577915435923018910334964869420614052182892431445797605163650903606514140377217442262561590244668525767372446430075513332450079650686719491377688478005309963967709758965844137894433796621993967316936280457084866613206797017728916080020698679408551343728867675409720757232455434770912461317493580281734466552734375);
  verify("0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000022250738585072008890245868760858598876504231122409594654935248025624400092282356951787758888037591552642309780950434312085877387158357291821993020294379224223559819827501242041788969571311791082261043971979604000454897391938079198936081525613113376149842043271751033627391549782731594143828136275113838604094249464942286316695429105080201815926642134996606517803095075913058719846423906068637102005108723282784678843631944515866135041223479014792369585208321597621066375401613736583044193603714778355306682834535634005074073040135602968046375918583163124224521599262546494300836851861719422417646455137135420132217031370496583210154654068035397417906022589503023501937519773030945763173210852507299305089761582519159720757232455434770912461317493580281734466552734375", 0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000022250738585072008890245868760858598876504231122409594654935248025624400092282356951787758888037591552642309780950434312085877387158357291821993020294379224223559819827501242041788969571311791082261043971979604000454897391938079198936081525613113376149842043271751033627391549782731594143828136275113838604094249464942286316695429105080201815926642134996606517803095075913058719846423906068637102005108723282784678843631944515866135041223479014792369585208321597621066375401613736583044193603714778355306682834535634005074073040135602968046375918583163124224521599262546494300836851861719422417646455137135420132217031370496583210154654068035397417906022589503023501937519773030945763173210852507299305089761582519159720757232455434770912461317493580281734466552734375);
  verify("1438456663141390273526118207642235581183227845246331231162636653790368152091394196930365828634687637948157940776599182791387527135353034738357134110310609455693900824193549772792016543182680519740580354365467985440183598701312257624545562331397018329928613196125590274187720073914818062530830316533158098624984118889298281371812288789537310599037529113415438738954894752124724983067241108764488346454376699018673078404751121414804937224240805993123816932326223683090770561597570457793932985826162604255884529134126396282202126526253389383421806727954588525596114379801269094096329805054803089299736996870951258573010877404407451953846698609198213926882692078557033228265259305481198526059813164469187586693257335779522020407645498684263339921905227556616698129967412891282231685504660671277927198290009824680186319750978665734576683784255802269708917361719466043175201158849097881370477111850171579869056016061666173029059588433776015644439705050377554277696143928278093453792803846252715966016733222646442382892123940052441346822429721593884378212558701004356924243030059517489346646577724622498919752597382095222500311124181823512251071356181769376577651390028297796156208815375089159128394945710515861334486267101797497111125909272505194792870889617179758703442608016143343262159998149700606597792535574457560429226974273443630323818747730771316763398572110874959981923732463076884528677392654150010269822239401993427482376513231389212353583573566376915572650916866553612366187378959554983566712767093372906030188976220169058025354973622211666504549316958271880975697143546564469806791358707318873075708383345004090151974068325838177531266954177406661392229801349994695941509935655355652985723782153570084089560139142231.738475042362596875449154552392299548947138162081694168675340677843807613129780449323363759027012972466987370921816813162658754726545121090545507240267000456594786540949605260722461937870630634874991729398208026467698131898691830012167897399682179601734569071423681e-733", std::numeric_limits<double>::infinity(), std::errc::result_out_of_range);
  verify("-2240084132271013504.131248280843119943687942846658579428", -0x1.f1660a65b00bfp+60);
}

TEST_CASE("64bit.decimal_point") {
  constexpr auto options = []{
    fast_float::parse_options ret{};
    ret.decimal_point = ',';
    return ret;
  }();

  // infinities
  verify_options("1,8e308", std::numeric_limits<double>::infinity(), std::errc::result_out_of_range);
  verify_options("1,832312213213213232132132143451234453123412321321312e308", std::numeric_limits<double>::infinity(), std::errc::result_out_of_range);
  verify_options("2e30000000000000000", std::numeric_limits<double>::infinity(), std::errc::result_out_of_range);
  verify_options("2e3000", std::numeric_limits<double>::infinity(), std::errc::result_out_of_range);
  verify_options("1,9e308", std::numeric_limits<double>::infinity(), std::errc::result_out_of_range);

  // finites
  verify_options("-2,2222222222223e-322",-0x1.68p-1069);
  verify_options("9007199254740993,0", 0x1p+53);
  verify_options("860228122,6654514319E+90", 0x1.92bb20990715fp+328);
  verify_options_runtime(append_zeros("9007199254740993,0",1000), 0x1p+53);
  verify_options("1,1920928955078125e-07", 1.1920928955078125e-07);
  verify_options("9355950000000000000,00000000000000000000000000000000001844674407370955161600000184467440737095516161844674407370955161407370955161618446744073709551616000184467440737095516166000001844674407370955161618446744073709551614073709551616184467440737095516160001844674407370955161601844674407370955674451616184467440737095516140737095516161844674407370955161600018446744073709551616018446744073709551611616000184467440737095001844674407370955161600184467440737095516160018446744073709551168164467440737095516160001844073709551616018446744073709551616184467440737095516160001844674407536910751601611616000184467440737095001844674407370955161600184467440737095516160018446744073709551616184467440737095516160001844955161618446744073709551616000184467440753691075160018446744073709",0x1.03ae05e8fca1cp+63);
  verify_options("2,22507385850720212418870147920222032907240528279439037814303133837435107319244194686754406432563881851382188218502438069999947733013005649884107791928741341929297200970481951993067993290969042784064731682041565926728632933630474670123316852983422152744517260835859654566319282835244787787799894310779783833699159288594555213714181128458251145584319223079897504395086859412457230891738946169368372321191373658977977723286698840356390251044443035457396733706583981055420456693824658413747607155981176573877626747665912387199931904006317334709003012790188175203447190250028061277777916798391090578584006464715943810511489154282775041174682194133952466682503431306181587829379004205392375072083366693241580002758391118854188641513168478436313080237596295773983001708984375e-308", 0x1.0000000000002p-1022);
  verify_options("1,0000000000000006661338147750939242541790008544921875",1.0000000000000007);
  verify_options("2,2250738585072013e-308", 2.2250738585072013e-308);
  verify_options("1,00000000000000188558920870223463870174566020691753515394643550663070558368373221972569761144603605635692374830246134201063722058e-309", 1.00000000000000188558920870223463870174566020691753515394643550663070558368373221972569761144603605635692374830246134201063722058e-309);
  verify_options("-2402844368454405395,2", -2402844368454405395.2);
  verify_options("2402844368454405395,2", 2402844368454405395.2);
  verify_options("7,0420557077594588669468784357561207962098443483187940792729600000e+59", 7.0420557077594588669468784357561207962098443483187940792729600000e+59);
  verify_options("7,0420557077594588669468784357561207962098443483187940792729600000e+59", 7.0420557077594588669468784357561207962098443483187940792729600000e+59);
  verify_options("-1,7339253062092163730578609458683877051596800000000000000000000000e+42", -1.7339253062092163730578609458683877051596800000000000000000000000e+42);
  verify_options("-2,0972622234386619214559824785284023792871122537545728000000000000e+52", -2.0972622234386619214559824785284023792871122537545728000000000000e+52);
  verify_options("-1,0001803374372191849407179462120053338028379051879898808320000000e+57", -1.0001803374372191849407179462120053338028379051879898808320000000e+57);
  verify_options("-1,8607245283054342363818436991534856973992070520151142825984000000e+58", -1.8607245283054342363818436991534856973992070520151142825984000000e+58);
  verify_options("-1,9189205311132686907264385602245237137907390376574976000000000000e+52", -1.9189205311132686907264385602245237137907390376574976000000000000e+52);
  verify_options("-2,8184483231688951563253238886553506793085187889855201280000000000e+54", -2.8184483231688951563253238886553506793085187889855201280000000000e+54);
  verify_options("-1,7664960224650106892054063261344555646357024359107788800000000000e+53", -1.7664960224650106892054063261344555646357024359107788800000000000e+53);
  verify_options("-2,1470977154320536489471030463761883783915110400000000000000000000e+45", -2.1470977154320536489471030463761883783915110400000000000000000000e+45);
  verify_options("-4,4900312744003159009338275160799498340862630046359789166919680000e+61", -4.4900312744003159009338275160799498340862630046359789166919680000e+61);
  verify_options("1", 1.0);
  verify_options("1,797693134862315700000000000000001e308", 1.7976931348623157e308);
  verify_options("3e-324", 0x0.0000000000001p-1022);
  verify_options("1,00000006e+09", 0x1.dcd651ep+29);
  verify_options("4,9406564584124653e-324", 0x0.0000000000001p-1022);
  verify_options("4,9406564584124654e-324", 0x0.0000000000001p-1022);
  verify_options("2,2250738585072009e-308", 0x0.fffffffffffffp-1022);
  verify_options("2,2250738585072014e-308", 0x1p-1022);
  verify_options("1,7976931348623157e308", 0x1.fffffffffffffp+1023);
  verify_options("1,7976931348623158e308", 0x1.fffffffffffffp+1023);
  verify_options("4503599627370496,5", 4503599627370496.5);
  verify_options("4503599627475352,5", 4503599627475352.5);
  verify_options("4503599627475353,5", 4503599627475353.5);
  verify_options("2251799813685248,25", 2251799813685248.25);
  verify_options("1125899906842624,125", 1125899906842624.125);
  verify_options("1125899906842901,875", 1125899906842901.875);
  verify_options("2251799813685803,75", 2251799813685803.75);
  verify_options("4503599627370497,5", 4503599627370497.5);
  verify_options("45035996,273704995", 45035996.273704995);
  verify_options("45035996,273704985", 45035996.273704985);
  verify_options("0,000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000044501477170144022721148195934182639518696390927032912960468522194496444440421538910330590478162701758282983178260792422137401728773891892910553144148156412434867599762821265346585071045737627442980259622449029037796981144446145705102663115100318287949527959668236039986479250965780342141637013812613333119898765515451440315261253813266652951306000184917766328660755595837392240989947807556594098101021612198814605258742579179000071675999344145086087205681577915435923018910334964869420614052182892431445797605163650903606514140377217442262561590244668525767372446430075513332450079650686719491377688478005309963967709758965844137894433796621993967316936280457084866613206797017728916080020698679408551343728867675409720757232455434770912461317493580281734466552734375", 0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000044501477170144022721148195934182639518696390927032912960468522194496444440421538910330590478162701758282983178260792422137401728773891892910553144148156412434867599762821265346585071045737627442980259622449029037796981144446145705102663115100318287949527959668236039986479250965780342141637013812613333119898765515451440315261253813266652951306000184917766328660755595837392240989947807556594098101021612198814605258742579179000071675999344145086087205681577915435923018910334964869420614052182892431445797605163650903606514140377217442262561590244668525767372446430075513332450079650686719491377688478005309963967709758965844137894433796621993967316936280457084866613206797017728916080020698679408551343728867675409720757232455434770912461317493580281734466552734375);
  verify_options("0,000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000022250738585072008890245868760858598876504231122409594654935248025624400092282356951787758888037591552642309780950434312085877387158357291821993020294379224223559819827501242041788969571311791082261043971979604000454897391938079198936081525613113376149842043271751033627391549782731594143828136275113838604094249464942286316695429105080201815926642134996606517803095075913058719846423906068637102005108723282784678843631944515866135041223479014792369585208321597621066375401613736583044193603714778355306682834535634005074073040135602968046375918583163124224521599262546494300836851861719422417646455137135420132217031370496583210154654068035397417906022589503023501937519773030945763173210852507299305089761582519159720757232455434770912461317493580281734466552734375", 0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000022250738585072008890245868760858598876504231122409594654935248025624400092282356951787758888037591552642309780950434312085877387158357291821993020294379224223559819827501242041788969571311791082261043971979604000454897391938079198936081525613113376149842043271751033627391549782731594143828136275113838604094249464942286316695429105080201815926642134996606517803095075913058719846423906068637102005108723282784678843631944515866135041223479014792369585208321597621066375401613736583044193603714778355306682834535634005074073040135602968046375918583163124224521599262546494300836851861719422417646455137135420132217031370496583210154654068035397417906022589503023501937519773030945763173210852507299305089761582519159720757232455434770912461317493580281734466552734375);
}


TEST_CASE("32bit.inf") {
  verify("INF", std::numeric_limits<float>::infinity());
  verify("-INF", -std::numeric_limits<float>::infinity());
  verify("INFINITY", std::numeric_limits<float>::infinity());
  verify("-INFINITY", -std::numeric_limits<float>::infinity());
  verify("infinity", std::numeric_limits<float>::infinity());
  verify("-infinity", -std::numeric_limits<float>::infinity());
  verify("inf", std::numeric_limits<float>::infinity());
  verify("-inf", -std::numeric_limits<float>::infinity());
  verify("1234456789012345678901234567890e9999999999999999999999999999", std::numeric_limits<float>::infinity(), std::errc::result_out_of_range);
  verify("2e3000", std::numeric_limits<float>::infinity(), std::errc::result_out_of_range);
  verify("3.5028234666e38", std::numeric_limits<float>::infinity(), std::errc::result_out_of_range);
}

TEST_CASE("32bit.general") {
  verify("-1e-999", -0.0f, std::errc::result_out_of_range);
  verify("1.1754941406275178592461758986628081843312458647327962400313859427181746759860647699724722770042717456817626953125", 0x1.2ced3p+0f);
  verify("1.1754941406275178592461758986628081843312458647327962400313859427181746759860647699724722770042717456817626953125e-38", 0x1.fffff8p-127f);
  verify_runtime(append_zeros("1.1754941406275178592461758986628081843312458647327962400313859427181746759860647699724722770042717456817626953125",655), 0x1.2ced3p+0f);
  verify_runtime(append_zeros("1.1754941406275178592461758986628081843312458647327962400313859427181746759860647699724722770042717456817626953125",656), 0x1.2ced3p+0f);
  verify_runtime(append_zeros("1.1754941406275178592461758986628081843312458647327962400313859427181746759860647699724722770042717456817626953125",1000), 0x1.2ced3p+0f);
  std::string test_string;
  test_string = append_zeros("1.1754941406275178592461758986628081843312458647327962400313859427181746759860647699724722770042717456817626953125",655) + std::string("e-38");
  verify_runtime(test_string, 0x1.fffff8p-127f);
  test_string = append_zeros("1.1754941406275178592461758986628081843312458647327962400313859427181746759860647699724722770042717456817626953125",656) + std::string("e-38");
  verify_runtime(test_string, 0x1.fffff8p-127f);
  test_string = append_zeros("1.1754941406275178592461758986628081843312458647327962400313859427181746759860647699724722770042717456817626953125",1000) + std::string("e-38");
  verify_runtime(test_string, 0x1.fffff8p-127f);
  verify32(1.00000006e+09f);
  verify32(1.4012984643e-45f);
  verify32(1.1754942107e-38f);
  verify32(1.1754943508e-45f);
  verify("-0", -0.0f);
  verify("1090544144181609348835077142190", 0x1.b877ap+99f);
  verify("1.1754943508e-38", 1.1754943508e-38f);
  verify("30219.0830078125", 30219.0830078125f);
  verify("16252921.5", 16252921.5f);
  verify("5322519.25", 5322519.25f);
  verify("3900245.875", 3900245.875f);
  verify("1510988.3125", 1510988.3125f);
  verify("782262.28125", 782262.28125f);
  verify("328381.484375", 328381.484375f);
  verify("156782.0703125", 156782.0703125f);
  verify("85003.24609375", 85003.24609375f);
  verify("17419.6494140625", 17419.6494140625f);
  verify("15498.36376953125", 15498.36376953125f);
  verify("6318.580322265625", 6318.580322265625f);
  verify("2525.2840576171875", 2525.2840576171875f);
  verify("1370.9265747070312", 1370.9265747070312f);
  verify("936.3702087402344", 936.3702087402344f);
  verify("411.88682556152344", 411.88682556152344f);
  verify("206.50310516357422", 206.50310516357422f);
  verify("124.16878890991211", 124.16878890991211f);
  verify("50.811574935913086", 50.811574935913086f);
  verify("17.486443519592285", 17.486443519592285f);
  verify("13.91745138168335", 13.91745138168335f);
  verify("7.5464513301849365", 0x1.e2f90ep+2f);
  verify("2.687217116355896", 2.687217116355896f);
  verify("1.1877630352973938", 0x1.30113ep+0f);
  verify("0.7622503340244293", 0.7622503340244293f);
  verify("0.30531780421733856", 0x1.38a53ap-2f);
  verify("0.21791061013936996", 0x1.be47eap-3f);
  verify("0.09289376810193062", 0x1.7c7e2ep-4f);
  verify("0.03706067614257336", 0.03706067614257336f);
  verify("0.028068351559340954", 0.028068351559340954f);
  verify("0.012114629615098238", 0x1.8cf8e2p-7f);
  verify("0.004221370676532388", 0x1.14a6dap-8f);
  verify("0.002153817447833717", 0.002153817447833717f);
  verify("0.0015924838953651488", 0x1.a175cap-10f);
  verify("0.0008602388261351734", 0.0008602388261351734f);
  verify("0.00036393293703440577", 0x1.7d9c82p-12f);
  verify("0.00013746770127909258", 0.00013746770127909258f);
  verify("16407.9462890625", 16407.9462890625f);
  verify("1.1754947011469036e-38", 0x1.000006p-126f);
  verify("7.0064923216240854e-46", 0x1p-149f);
  verify("8388614.5", 8388614.5f);
  verify("0e9999999999999999999999999999", 0.f);
  verify("4.7019774032891500318749461488889827112746622270883500860350068251e-38",4.7019774032891500318749461488889827112746622270883500860350068251e-38f);
  verify("3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679", 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679f);
  verify("2.3509887016445750159374730744444913556373311135441750430175034126e-38", 2.3509887016445750159374730744444913556373311135441750430175034126e-38f);
  verify("1", 1.f);
  verify("7.0060e-46", 0.f, std::errc::result_out_of_range);
  verify("3.4028234664e38", 0x1.fffffep+127f);
  verify("3.4028234665e38", 0x1.fffffep+127f);
  verify("3.4028234666e38", 0x1.fffffep+127f);
  verify("0.000000000000000000000000000000000000011754943508222875079687365372222456778186655567720875215087517062784172594547271728515625", 0.000000000000000000000000000000000000011754943508222875079687365372222456778186655567720875215087517062784172594547271728515625);
  verify("0.00000000000000000000000000000000000000000000140129846432481707092372958328991613128026194187651577175706828388979108268586060148663818836212158203125", 0.00000000000000000000000000000000000000000000140129846432481707092372958328991613128026194187651577175706828388979108268586060148663818836212158203125f);
  verify("0.00000000000000000000000000000000000002350988561514728583455765982071533026645717985517980855365926236850006129930346077117064851336181163787841796875", 0.00000000000000000000000000000000000002350988561514728583455765982071533026645717985517980855365926236850006129930346077117064851336181163787841796875f);
  verify("0.00000000000000000000000000000000000001175494210692441075487029444849287348827052428745893333857174530571588870475618904265502351336181163787841796875", 0.00000000000000000000000000000000000001175494210692441075487029444849287348827052428745893333857174530571588870475618904265502351336181163787841796875f);
}

TEST_CASE("32bit.decimal_point") {
  constexpr auto options = [] {
    fast_float::parse_options ret{};
    ret.decimal_point = ',';
    return ret;
  }();

  // infinity
  verify_options("3,5028234666e38", std::numeric_limits<float>::infinity(), std::errc::result_out_of_range);

  // finites
  verify_options("1,1754941406275178592461758986628081843312458647327962400313859427181746759860647699724722770042717456817626953125", 0x1.2ced3p+0f);
  verify_options("1,1754941406275178592461758986628081843312458647327962400313859427181746759860647699724722770042717456817626953125e-38", 0x1.fffff8p-127f);
  verify_options_runtime(append_zeros("1,1754941406275178592461758986628081843312458647327962400313859427181746759860647699724722770042717456817626953125",655), 0x1.2ced3p+0f);
  verify_options_runtime(append_zeros("1,1754941406275178592461758986628081843312458647327962400313859427181746759860647699724722770042717456817626953125",656), 0x1.2ced3p+0f);
  verify_options_runtime(append_zeros("1,1754941406275178592461758986628081843312458647327962400313859427181746759860647699724722770042717456817626953125",1000), 0x1.2ced3p+0f);
  std::string test_string;
  test_string = append_zeros("1,1754941406275178592461758986628081843312458647327962400313859427181746759860647699724722770042717456817626953125",655) + std::string("e-38");
  verify_options_runtime(test_string, 0x1.fffff8p-127f);
  test_string = append_zeros("1,1754941406275178592461758986628081843312458647327962400313859427181746759860647699724722770042717456817626953125",656) + std::string("e-38");
  verify_options_runtime(test_string, 0x1.fffff8p-127f);
  test_string = append_zeros("1,1754941406275178592461758986628081843312458647327962400313859427181746759860647699724722770042717456817626953125",1000) + std::string("e-38");
  verify_options_runtime(test_string, 0x1.fffff8p-127f);
  verify_options("1,1754943508e-38", 1.1754943508e-38f);
  verify_options("30219,0830078125", 30219.0830078125f);
  verify_options("1,1754947011469036e-38", 0x1.000006p-126f);
  verify_options("7,0064923216240854e-46", 0x1p-149f);
  verify_options("8388614,5", 8388614.5f);
  verify_options("0e9999999999999999999999999999", 0.f);
  verify_options("4,7019774032891500318749461488889827112746622270883500860350068251e-38",4.7019774032891500318749461488889827112746622270883500860350068251e-38f);
  verify_options("3,1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679", 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679f);
  verify_options("2,3509887016445750159374730744444913556373311135441750430175034126e-38", 2.3509887016445750159374730744444913556373311135441750430175034126e-38f);
  verify_options("1", 1.f);
  verify_options("7,0060e-46", 0.f, std::errc::result_out_of_range);
  verify_options("3,4028234664e38", 0x1.fffffep+127f);
  verify_options("3,4028234665e38", 0x1.fffffep+127f);
  verify_options("3,4028234666e38", 0x1.fffffep+127f);
  verify_options("0,000000000000000000000000000000000000011754943508222875079687365372222456778186655567720875215087517062784172594547271728515625", 0.000000000000000000000000000000000000011754943508222875079687365372222456778186655567720875215087517062784172594547271728515625);
  verify_options("0,00000000000000000000000000000000000000000000140129846432481707092372958328991613128026194187651577175706828388979108268586060148663818836212158203125", 0.00000000000000000000000000000000000000000000140129846432481707092372958328991613128026194187651577175706828388979108268586060148663818836212158203125f);
  verify_options("0,00000000000000000000000000000000000002350988561514728583455765982071533026645717985517980855365926236850006129930346077117064851336181163787841796875", 0.00000000000000000000000000000000000002350988561514728583455765982071533026645717985517980855365926236850006129930346077117064851336181163787841796875f);
  verify_options("0,00000000000000000000000000000000000001175494210692441075487029444849287348827052428745893333857174530571588870475618904265502351336181163787841796875", 0.00000000000000000000000000000000000001175494210692441075487029444849287348827052428745893333857174530571588870475618904265502351336181163787841796875f);
}
