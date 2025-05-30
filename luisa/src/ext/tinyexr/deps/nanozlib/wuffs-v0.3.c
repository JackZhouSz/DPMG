#ifndef WUFFS_INCLUDE_GUARD
#define WUFFS_INCLUDE_GUARD

// Wuffs ships as a "single file C library" or "header file library" as per
// https://github.com/nothings/stb/blob/master/docs/stb_howto.txt
//
// To use that single file as a "foo.c"-like implementation, instead of a
// "foo.h"-like header, #define WUFFS_IMPLEMENTATION before #include'ing or
// compiling it.

// Wuffs' C code is generated automatically, not hand-written. These warnings'
// costs outweigh the benefits.
//
// The "elif defined(__clang__)" isn't redundant. While vanilla clang defines
// __GNUC__, clang-cl (which mimics MSVC's cl.exe) does not.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunreachable-code"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#if defined(__cplusplus)
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-fallthrough"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wunreachable-code"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-parameter"
#if defined(__cplusplus)
#pragma clang diagnostic ignored "-Wold-style-cast"
#endif
#endif

// Copyright 2017 The Wuffs Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
#if (__cplusplus >= 201103L) || defined(_MSC_VER)
#include <memory>
#define WUFFS_BASE__HAVE_EQ_DELETE
#define WUFFS_BASE__HAVE_UNIQUE_PTR
// The "defined(__clang__)" isn't redundant. While vanilla clang defines
// __GNUC__, clang-cl (which mimics MSVC's cl.exe) does not.
#elif defined(__GNUC__) || defined(__clang__)
#warning "Wuffs' C++ code expects -std=c++11 or later"
#endif

extern "C" {
#endif

// ---------------- Version

// WUFFS_VERSION is the major.minor.patch version, as per https://semver.org/,
// as a uint64_t. The major number is the high 32 bits. The minor number is the
// middle 16 bits. The patch number is the low 16 bits. The pre-release label
// and build metadata are part of the string representation (such as
// "1.2.3-beta+456.20181231") but not the uint64_t representation.
//
// WUFFS_VERSION_PRE_RELEASE_LABEL (such as "", "beta" or "rc.1") being
// non-empty denotes a developer preview, not a release version, and has no
// backwards or forwards compatibility guarantees.
//
// WUFFS_VERSION_BUILD_METADATA_XXX, if non-zero, are the number of commits and
// the last commit date in the repository used to build this library. Within
// each major.minor branch, the commit count should increase monotonically.
//
// WUFFS_VERSION was overridden by "wuffs gen -version" based on revision
// 00d5e35865a2f2718f4bb2596adaaa54bd639bbe committed on 2023-04-08.
#define WUFFS_VERSION 0x000030003
#define WUFFS_VERSION_MAJOR 0
#define WUFFS_VERSION_MINOR 3
#define WUFFS_VERSION_PATCH 3
#define WUFFS_VERSION_PRE_RELEASE_LABEL ""
#define WUFFS_VERSION_BUILD_METADATA_COMMIT_COUNT 3399
#define WUFFS_VERSION_BUILD_METADATA_COMMIT_DATE 20230408
#define WUFFS_VERSION_STRING "0.3.3+3399.20230408"

// ---------------- Configuration

// Define WUFFS_CONFIG__AVOID_CPU_ARCH to avoid any code tied to a specific CPU
// architecture, such as SSE SIMD for the x86 CPU family.
#if defined(WUFFS_CONFIG__AVOID_CPU_ARCH)  // (#if-chain ref AVOID_CPU_ARCH_0)
// No-op.
#else  // (#if-chain ref AVOID_CPU_ARCH_0)

// The "defined(__clang__)" isn't redundant. While vanilla clang defines
// __GNUC__, clang-cl (which mimics MSVC's cl.exe) does not.
#if defined(__GNUC__) || defined(__clang__)
#define WUFFS_BASE__MAYBE_ATTRIBUTE_TARGET(arg) __attribute__((target(arg)))
#else
#define WUFFS_BASE__MAYBE_ATTRIBUTE_TARGET(arg)
#endif  // defined(__GNUC__) || defined(__clang__)

#if defined(__GNUC__)  // (#if-chain ref AVOID_CPU_ARCH_1)

// To simplify Wuffs code, "cpu_arch >= arm_xxx" requires xxx but also
// unaligned little-endian load/stores.
#if defined(__ARM_FEATURE_UNALIGNED) && !defined(__native_client__) && \
    defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
// Not all gcc versions define __ARM_ACLE, even if they support crc32
// intrinsics. Look for __ARM_FEATURE_CRC32 instead.
#if defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#define WUFFS_BASE__CPU_ARCH__ARM_CRC32
#endif  // defined(__ARM_FEATURE_CRC32)
#if defined(__ARM_NEON)
#include <arm_neon.h>
#define WUFFS_BASE__CPU_ARCH__ARM_NEON
#endif  // defined(__ARM_NEON)
#endif  // defined(__ARM_FEATURE_UNALIGNED) etc

// Similarly, "cpu_arch >= x86_sse42" requires SSE4.2 but also PCLMUL and
// POPCNT. This is checked at runtime via cpuid, not at compile time.
//
// Likewise, "cpu_arch >= x86_avx2" also requires PCLMUL, POPCNT and SSE4.2.
#if defined(__i386__) || defined(__x86_64__)
#if !defined(__native_client__)
#include <cpuid.h>
#include <x86intrin.h>
// X86_FAMILY means X86 (32-bit) or X86_64 (64-bit, obviously).
#define WUFFS_BASE__CPU_ARCH__X86_FAMILY
#endif  // !defined(__native_client__)
#endif  // defined(__i386__) || defined(__x86_64__)

#elif defined(_MSC_VER)  // (#if-chain ref AVOID_CPU_ARCH_1)

#if defined(_M_IX86) || defined(_M_X64)
#if defined(__AVX__) || defined(__clang__)

// We need <intrin.h> for the __cpuid function.
#include <intrin.h>
// That's not enough for X64 SIMD, with clang-cl, if we want to use
// "__attribute__((target(arg)))" without e.g. "/arch:AVX".
//
// Some web pages suggest that <immintrin.h> is all you need, as it pulls in
// the earlier SIMD families like SSE4.2, but that doesn't seem to work in
// practice, possibly for the same reason that just <intrin.h> doesn't work.
#include <immintrin.h>  // AVX, AVX2, FMA, POPCNT
#include <nmmintrin.h>  // SSE4.2
#include <wmmintrin.h>  // AES, PCLMUL
// X86_FAMILY means X86 (32-bit) or X86_64 (64-bit, obviously).
#define WUFFS_BASE__CPU_ARCH__X86_FAMILY

#else  // defined(__AVX__) || defined(__clang__)

// clang-cl (which defines both __clang__ and _MSC_VER) supports
// "__attribute__((target(arg)))".
//
// For MSVC's cl.exe (unlike clang or gcc), SIMD capability is a compile-time
// property of the source file (e.g. a /arch:AVX or -mavx compiler flag), not
// of individual functions (that can be conditionally selected at runtime).
#pragma message("Wuffs with MSVC+IX86/X64 needs /arch:AVX for best performance")

#endif  // defined(__AVX__) || defined(__clang__)
#endif  // defined(_M_IX86) || defined(_M_X64)

#endif  // (#if-chain ref AVOID_CPU_ARCH_1)
#endif  // (#if-chain ref AVOID_CPU_ARCH_0)

// --------

// Define WUFFS_CONFIG__STATIC_FUNCTIONS (combined with WUFFS_IMPLEMENTATION)
// to make all of Wuffs' functions have static storage.
//
// This can help the compiler ignore or discard unused code, which can produce
// faster compiles and smaller binaries. Other motivations are discussed in the
// "ALLOW STATIC IMPLEMENTATION" section of
// https://raw.githubusercontent.com/nothings/stb/master/docs/stb_howto.txt
#if defined(WUFFS_CONFIG__STATIC_FUNCTIONS)
#define WUFFS_BASE__MAYBE_STATIC static
#else
#define WUFFS_BASE__MAYBE_STATIC
#endif  // defined(WUFFS_CONFIG__STATIC_FUNCTIONS)

// ---------------- CPU Architecture

static inline bool  //
wuffs_base__cpu_arch__have_arm_crc32() {
#if defined(WUFFS_BASE__CPU_ARCH__ARM_CRC32)
  return true;
#else
  return false;
#endif  // defined(WUFFS_BASE__CPU_ARCH__ARM_CRC32)
}

static inline bool  //
wuffs_base__cpu_arch__have_arm_neon() {
#if defined(WUFFS_BASE__CPU_ARCH__ARM_NEON)
  return true;
#else
  return false;
#endif  // defined(WUFFS_BASE__CPU_ARCH__ARM_NEON)
}

static inline bool  //
wuffs_base__cpu_arch__have_x86_avx2() {
#if defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
  // GCC defines these macros but MSVC does not.
  //  - bit_AVX2 = (1 <<  5)
  const unsigned int avx2_ebx7 = 0x00000020;
  // GCC defines these macros but MSVC does not.
  //  - bit_PCLMUL = (1 <<  1)
  //  - bit_POPCNT = (1 << 23)
  //  - bit_SSE4_2 = (1 << 20)
  const unsigned int avx2_ecx1 = 0x00900002;

  // clang defines __GNUC__ and clang-cl defines _MSC_VER (but not __GNUC__).
#if defined(__GNUC__)
  unsigned int eax7 = 0;
  unsigned int ebx7 = 0;
  unsigned int ecx7 = 0;
  unsigned int edx7 = 0;
  if (__get_cpuid_count(7, 0, &eax7, &ebx7, &ecx7, &edx7) &&
      ((ebx7 & avx2_ebx7) == avx2_ebx7)) {
    unsigned int eax1 = 0;
    unsigned int ebx1 = 0;
    unsigned int ecx1 = 0;
    unsigned int edx1 = 0;
    if (__get_cpuid(1, &eax1, &ebx1, &ecx1, &edx1) &&
        ((ecx1 & avx2_ecx1) == avx2_ecx1)) {
      return true;
    }
  }
#elif defined(_MSC_VER)  // defined(__GNUC__)
  int x7[4];
  __cpuidex(x7, 7, 0);
  if ((((unsigned int)(x7[1])) & avx2_ebx7) == avx2_ebx7) {
    int x1[4];
    __cpuid(x1, 1);
    if ((((unsigned int)(x1[2])) & avx2_ecx1) == avx2_ecx1) {
      return true;
    }
  }
#else
#error "WUFFS_BASE__CPU_ARCH__ETC combined with an unsupported compiler"
#endif  // defined(__GNUC__); defined(_MSC_VER)
#endif  // defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
  return false;
}

static inline bool  //
wuffs_base__cpu_arch__have_x86_bmi2() {
#if defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
  // GCC defines these macros but MSVC does not.
  //  - bit_BMI2 = (1 <<  8)
  const unsigned int bmi2_ebx7 = 0x00000100;

  // clang defines __GNUC__ and clang-cl defines _MSC_VER (but not __GNUC__).
#if defined(__GNUC__)
  unsigned int eax7 = 0;
  unsigned int ebx7 = 0;
  unsigned int ecx7 = 0;
  unsigned int edx7 = 0;
  if (__get_cpuid_count(7, 0, &eax7, &ebx7, &ecx7, &edx7) &&
      ((ebx7 & bmi2_ebx7) == bmi2_ebx7)) {
    return true;
  }
#elif defined(_MSC_VER)  // defined(__GNUC__)
  int x7[4];
  __cpuidex(x7, 7, 0);
  if ((((unsigned int)(x7[1])) & bmi2_ebx7) == bmi2_ebx7) {
    return true;
  }
#else
#error "WUFFS_BASE__CPU_ARCH__ETC combined with an unsupported compiler"
#endif  // defined(__GNUC__); defined(_MSC_VER)
#endif  // defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
  return false;
}

static inline bool  //
wuffs_base__cpu_arch__have_x86_sse42() {
#if defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
  // GCC defines these macros but MSVC does not.
  //  - bit_PCLMUL = (1 <<  1)
  //  - bit_POPCNT = (1 << 23)
  //  - bit_SSE4_2 = (1 << 20)
  const unsigned int sse42_ecx1 = 0x00900002;

  // clang defines __GNUC__ and clang-cl defines _MSC_VER (but not __GNUC__).
#if defined(__GNUC__)
  unsigned int eax1 = 0;
  unsigned int ebx1 = 0;
  unsigned int ecx1 = 0;
  unsigned int edx1 = 0;
  if (__get_cpuid(1, &eax1, &ebx1, &ecx1, &edx1) &&
      ((ecx1 & sse42_ecx1) == sse42_ecx1)) {
    return true;
  }
#elif defined(_MSC_VER)  // defined(__GNUC__)
  int x1[4];
  __cpuid(x1, 1);
  if ((((unsigned int)(x1[2])) & sse42_ecx1) == sse42_ecx1) {
    return true;
  }
#else
#error "WUFFS_BASE__CPU_ARCH__ETC combined with an unsupported compiler"
#endif  // defined(__GNUC__); defined(_MSC_VER)
#endif  // defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
  return false;
}

// ---------------- Fundamentals

// Wuffs assumes that:
//  - converting a uint32_t to a size_t will never overflow.
//  - converting a size_t to a uint64_t will never overflow.
#if defined(__WORDSIZE)
#if (__WORDSIZE != 32) && (__WORDSIZE != 64)
#error "Wuffs requires a word size of either 32 or 64 bits"
#endif
#endif

// The "defined(__clang__)" isn't redundant. While vanilla clang defines
// __GNUC__, clang-cl (which mimics MSVC's cl.exe) does not.
#if defined(__GNUC__) || defined(__clang__)
#define WUFFS_BASE__POTENTIALLY_UNUSED __attribute__((unused))
#define WUFFS_BASE__WARN_UNUSED_RESULT __attribute__((warn_unused_result))
#else
#define WUFFS_BASE__POTENTIALLY_UNUSED
#define WUFFS_BASE__WARN_UNUSED_RESULT
#endif

// --------

// Options (bitwise or'ed together) for wuffs_foo__bar__initialize functions.

#define WUFFS_INITIALIZE__DEFAULT_OPTIONS ((uint32_t)0x00000000)

// WUFFS_INITIALIZE__ALREADY_ZEROED means that the "self" receiver struct value
// has already been set to all zeroes.
#define WUFFS_INITIALIZE__ALREADY_ZEROED ((uint32_t)0x00000001)

// WUFFS_INITIALIZE__LEAVE_INTERNAL_BUFFERS_UNINITIALIZED means that, absent
// WUFFS_INITIALIZE__ALREADY_ZEROED, only some of the "self" receiver struct
// value will be set to all zeroes. Internal buffers, which tend to be a large
// proportion of the struct's size, will be left uninitialized. Internal means
// that the buffer is contained by the receiver struct, as opposed to being
// passed as a separately allocated "work buffer".
//
// For more detail, see:
// https://github.com/google/wuffs/blob/main/doc/note/initialization.md
#define WUFFS_INITIALIZE__LEAVE_INTERNAL_BUFFERS_UNINITIALIZED \
  ((uint32_t)0x00000002)

// --------

// wuffs_base__empty_struct is used when a Wuffs function returns an empty
// struct. In C, if a function f returns void, you can't say "x = f()", but in
// Wuffs, if a function g returns empty, you can say "y = g()".
typedef struct wuffs_base__empty_struct__struct {
  // private_impl is a placeholder field. It isn't explicitly used, except that
  // without it, the sizeof a struct with no fields can differ across C/C++
  // compilers, and it is undefined behavior in C99. For example, gcc says that
  // the sizeof an empty struct is 0, and g++ says that it is 1. This leads to
  // ABI incompatibility if a Wuffs .c file is processed by one compiler and
  // its .h file with another compiler.
  //
  // Instead, we explicitly insert an otherwise unused field, so that the
  // sizeof this struct is always 1.
  uint8_t private_impl;
} wuffs_base__empty_struct;

static inline wuffs_base__empty_struct  //
wuffs_base__make_empty_struct() {
  wuffs_base__empty_struct ret;
  ret.private_impl = 0;
  return ret;
}

// wuffs_base__utility is a placeholder receiver type. It enables what Java
// calls static methods, as opposed to regular methods.
typedef struct wuffs_base__utility__struct {
  // private_impl is a placeholder field. It isn't explicitly used, except that
  // without it, the sizeof a struct with no fields can differ across C/C++
  // compilers, and it is undefined behavior in C99. For example, gcc says that
  // the sizeof an empty struct is 0, and g++ says that it is 1. This leads to
  // ABI incompatibility if a Wuffs .c file is processed by one compiler and
  // its .h file with another compiler.
  //
  // Instead, we explicitly insert an otherwise unused field, so that the
  // sizeof this struct is always 1.
  uint8_t private_impl;
} wuffs_base__utility;

typedef struct wuffs_base__vtable__struct {
  const char* vtable_name;
  const void* function_pointers;
} wuffs_base__vtable;

// --------

// See https://github.com/google/wuffs/blob/main/doc/note/statuses.md
typedef struct wuffs_base__status__struct {
  const char* repr;

#ifdef __cplusplus
  inline bool is_complete() const;
  inline bool is_error() const;
  inline bool is_note() const;
  inline bool is_ok() const;
  inline bool is_suspension() const;
  inline const char* message() const;
#endif  // __cplusplus

} wuffs_base__status;

extern const char wuffs_base__note__i_o_redirect[];
extern const char wuffs_base__note__end_of_data[];
extern const char wuffs_base__note__metadata_reported[];
extern const char wuffs_base__suspension__even_more_information[];
extern const char wuffs_base__suspension__mispositioned_read[];
extern const char wuffs_base__suspension__mispositioned_write[];
extern const char wuffs_base__suspension__short_read[];
extern const char wuffs_base__suspension__short_write[];
extern const char wuffs_base__error__bad_i_o_position[];
extern const char wuffs_base__error__bad_argument_length_too_short[];
extern const char wuffs_base__error__bad_argument[];
extern const char wuffs_base__error__bad_call_sequence[];
extern const char wuffs_base__error__bad_data[];
extern const char wuffs_base__error__bad_receiver[];
extern const char wuffs_base__error__bad_restart[];
extern const char wuffs_base__error__bad_sizeof_receiver[];
extern const char wuffs_base__error__bad_vtable[];
extern const char wuffs_base__error__bad_workbuf_length[];
extern const char wuffs_base__error__bad_wuffs_version[];
extern const char wuffs_base__error__cannot_return_a_suspension[];
extern const char wuffs_base__error__disabled_by_previous_error[];
extern const char wuffs_base__error__initialize_falsely_claimed_already_zeroed[];
extern const char wuffs_base__error__initialize_not_called[];
extern const char wuffs_base__error__interleaved_coroutine_calls[];
extern const char wuffs_base__error__no_more_information[];
extern const char wuffs_base__error__not_enough_data[];
extern const char wuffs_base__error__out_of_bounds[];
extern const char wuffs_base__error__unsupported_method[];
extern const char wuffs_base__error__unsupported_option[];
extern const char wuffs_base__error__unsupported_pixel_swizzler_option[];
extern const char wuffs_base__error__too_much_data[];

static inline wuffs_base__status  //
wuffs_base__make_status(const char* repr) {
  wuffs_base__status z;
  z.repr = repr;
  return z;
}

static inline bool  //
wuffs_base__status__is_complete(const wuffs_base__status* z) {
  return (z->repr == NULL) || ((*z->repr != '$') && (*z->repr != '#'));
}

static inline bool  //
wuffs_base__status__is_error(const wuffs_base__status* z) {
  return z->repr && (*z->repr == '#');
}

static inline bool  //
wuffs_base__status__is_note(const wuffs_base__status* z) {
  return z->repr && (*z->repr != '$') && (*z->repr != '#');
}

static inline bool  //
wuffs_base__status__is_ok(const wuffs_base__status* z) {
  return z->repr == NULL;
}

static inline bool  //
wuffs_base__status__is_suspension(const wuffs_base__status* z) {
  return z->repr && (*z->repr == '$');
}

// wuffs_base__status__message strips the leading '$', '#' or '@'.
static inline const char*  //
wuffs_base__status__message(const wuffs_base__status* z) {
  if (z->repr) {
    if ((*z->repr == '$') || (*z->repr == '#') || (*z->repr == '@')) {
      return z->repr + 1;
    }
  }
  return z->repr;
}

#ifdef __cplusplus

inline bool  //
wuffs_base__status::is_complete() const {
  return wuffs_base__status__is_complete(this);
}

inline bool  //
wuffs_base__status::is_error() const {
  return wuffs_base__status__is_error(this);
}

inline bool  //
wuffs_base__status::is_note() const {
  return wuffs_base__status__is_note(this);
}

inline bool  //
wuffs_base__status::is_ok() const {
  return wuffs_base__status__is_ok(this);
}

inline bool  //
wuffs_base__status::is_suspension() const {
  return wuffs_base__status__is_suspension(this);
}

inline const char*  //
wuffs_base__status::message() const {
  return wuffs_base__status__message(this);
}

#endif  // __cplusplus

// --------

// WUFFS_BASE__RESULT is a result type: either a status (an error) or a value.
//
// A result with all fields NULL or zero is as valid as a zero-valued T.
#define WUFFS_BASE__RESULT(T)  \
  struct {                     \
    wuffs_base__status status; \
    T value;                   \
  }

typedef WUFFS_BASE__RESULT(double) wuffs_base__result_f64;
typedef WUFFS_BASE__RESULT(int64_t) wuffs_base__result_i64;
typedef WUFFS_BASE__RESULT(uint64_t) wuffs_base__result_u64;

// --------

// wuffs_base__transform__output is the result of transforming from a src slice
// to a dst slice.
typedef struct wuffs_base__transform__output__struct {
  wuffs_base__status status;
  size_t num_dst;
  size_t num_src;
} wuffs_base__transform__output;

// --------

// FourCC constants. Four Character Codes are literally four ASCII characters
// (sometimes padded with ' ' spaces) that pack neatly into a signed or
// unsigned 32-bit integer. ASCII letters are conventionally upper case.
//
// They are often used to identify video codecs (e.g. "H265") and pixel formats
// (e.g. "YV12"). Wuffs uses them for that but also generally for naming
// various things: compression formats (e.g. "BZ2 "), image metadata (e.g.
// "EXIF"), file formats (e.g. "HTML"), etc.
//
// Wuffs' u32 values are big-endian ("JPEG" is 0x4A504547 not 0x4745504A) to
// preserve ordering: "JPEG" < "MP3 " and 0x4A504547 < 0x4D503320.

// Background Color.
#define WUFFS_BASE__FOURCC__BGCL 0x4247434C

// Bitmap.
#define WUFFS_BASE__FOURCC__BMP 0x424D5020

// Brotli.
#define WUFFS_BASE__FOURCC__BRTL 0x4252544C

// Bzip2.
#define WUFFS_BASE__FOURCC__BZ2 0x425A3220

// Concise Binary Object Representation.
#define WUFFS_BASE__FOURCC__CBOR 0x43424F52

// Primary Chromaticities and White Point.
#define WUFFS_BASE__FOURCC__CHRM 0x4348524D

// Cascading Style Sheets.
#define WUFFS_BASE__FOURCC__CSS 0x43535320

// Encapsulated PostScript.
#define WUFFS_BASE__FOURCC__EPS 0x45505320

// Exchangeable Image File Format.
#define WUFFS_BASE__FOURCC__EXIF 0x45584946

// Free Lossless Audio Codec.
#define WUFFS_BASE__FOURCC__FLAC 0x464C4143

// Gamma Correction.
#define WUFFS_BASE__FOURCC__GAMA 0x47414D41

// Graphics Interchange Format.
#define WUFFS_BASE__FOURCC__GIF 0x47494620

// GNU Zip.
#define WUFFS_BASE__FOURCC__GZ 0x475A2020

// High Efficiency Image File.
#define WUFFS_BASE__FOURCC__HEIF 0x48454946

// Hypertext Markup Language.
#define WUFFS_BASE__FOURCC__HTML 0x48544D4C

// International Color Consortium Profile.
#define WUFFS_BASE__FOURCC__ICCP 0x49434350

// Icon.
#define WUFFS_BASE__FOURCC__ICO 0x49434F20

// Icon Vector Graphics.
#define WUFFS_BASE__FOURCC__ICVG 0x49435647

// Initialization.
#define WUFFS_BASE__FOURCC__INI 0x494E4920

// Joint Photographic Experts Group.
#define WUFFS_BASE__FOURCC__JPEG 0x4A504547

// JavaScript.
#define WUFFS_BASE__FOURCC__JS 0x4A532020

// JavaScript Object Notation.
#define WUFFS_BASE__FOURCC__JSON 0x4A534F4E

// JSON With Commas and Comments.
#define WUFFS_BASE__FOURCC__JWCC 0x4A574343

// Key-Value Pair.
#define WUFFS_BASE__FOURCC__KVP 0x4B565020

// Key-Value Pair (Key).
#define WUFFS_BASE__FOURCC__KVPK 0x4B56504B

// Key-Value Pair (Value).
#define WUFFS_BASE__FOURCC__KVPV 0x4B565056

// Lempel–Ziv 4.
#define WUFFS_BASE__FOURCC__LZ4 0x4C5A3420

// Markdown.
#define WUFFS_BASE__FOURCC__MD 0x4D442020

// Modification Time.
#define WUFFS_BASE__FOURCC__MTIM 0x4D54494D

// MPEG-1 Audio Layer III.
#define WUFFS_BASE__FOURCC__MP3 0x4D503320

// Naive Image.
#define WUFFS_BASE__FOURCC__NIE 0x4E494520

// Offset (2-Dimensional).
#define WUFFS_BASE__FOURCC__OFS2 0x4F465332

// Open Type Format.
#define WUFFS_BASE__FOURCC__OTF 0x4F544620

// Portable Document Format.
#define WUFFS_BASE__FOURCC__PDF 0x50444620

// Physical Dimensions.
#define WUFFS_BASE__FOURCC__PHYD 0x50485944

// Portable Network Graphics.
#define WUFFS_BASE__FOURCC__PNG 0x504E4720

// Portable Anymap.
#define WUFFS_BASE__FOURCC__PNM 0x504E4D20

// PostScript.
#define WUFFS_BASE__FOURCC__PS 0x50532020

// Quite OK Image.
#define WUFFS_BASE__FOURCC__QOI 0x514F4920

// Random Access Compression.
#define WUFFS_BASE__FOURCC__RAC 0x52414320

// Raw.
#define WUFFS_BASE__FOURCC__RAW 0x52415720

// Resource Interchange File Format.
#define WUFFS_BASE__FOURCC__RIFF 0x52494646

// Riegeli Records.
#define WUFFS_BASE__FOURCC__RIGL 0x5249474C

// Snappy.
#define WUFFS_BASE__FOURCC__SNPY 0x534E5059

// Standard Red Green Blue (Rendering Intent).
#define WUFFS_BASE__FOURCC__SRGB 0x53524742

// Scalable Vector Graphics.
#define WUFFS_BASE__FOURCC__SVG 0x53564720

// Tape Archive.
#define WUFFS_BASE__FOURCC__TAR 0x54415220

// Text.
#define WUFFS_BASE__FOURCC__TEXT 0x54455854

// Truevision Advanced Raster Graphics Adapter.
#define WUFFS_BASE__FOURCC__TGA 0x54474120

// Tagged Image File Format.
#define WUFFS_BASE__FOURCC__TIFF 0x54494646

// Tom's Obvious Minimal Language.
#define WUFFS_BASE__FOURCC__TOML 0x544F4D4C

// Waveform.
#define WUFFS_BASE__FOURCC__WAVE 0x57415645

// Wireless Bitmap.
#define WUFFS_BASE__FOURCC__WBMP 0x57424D50

// Web Picture.
#define WUFFS_BASE__FOURCC__WEBP 0x57454250

// Web Open Font Format.
#define WUFFS_BASE__FOURCC__WOFF 0x574F4646

// Extensible Markup Language.
#define WUFFS_BASE__FOURCC__XML 0x584D4C20

// Extensible Metadata Platform.
#define WUFFS_BASE__FOURCC__XMP 0x584D5020

// Xz.
#define WUFFS_BASE__FOURCC__XZ 0x585A2020

// Zip.
#define WUFFS_BASE__FOURCC__ZIP 0x5A495020

// Zlib.
#define WUFFS_BASE__FOURCC__ZLIB 0x5A4C4942

// Zstandard.
#define WUFFS_BASE__FOURCC__ZSTD 0x5A535444

// --------

// Quirks.

#define WUFFS_BASE__QUIRK_IGNORE_CHECKSUM 1

// --------

// Flicks are a unit of time. One flick (frame-tick) is 1 / 705_600_000 of a
// second. See https://github.com/OculusVR/Flicks
typedef int64_t wuffs_base__flicks;

#define WUFFS_BASE__FLICKS_PER_SECOND ((uint64_t)705600000)
#define WUFFS_BASE__FLICKS_PER_MILLISECOND ((uint64_t)705600)

// ---------------- Numeric Types

// The helpers below are functions, instead of macros, because their arguments
// can be an expression that we shouldn't evaluate more than once.
//
// They are static, so that linking multiple wuffs .o files won't complain about
// duplicate function definitions.
//
// They are explicitly marked inline, even if modern compilers don't use the
// inline attribute to guide optimizations such as inlining, to avoid the
// -Wunused-function warning, and we like to compile with -Wall -Werror.

static inline int8_t  //
wuffs_base__i8__min(int8_t x, int8_t y) {
  return x < y ? x : y;
}

static inline int8_t  //
wuffs_base__i8__max(int8_t x, int8_t y) {
  return x > y ? x : y;
}

static inline int16_t  //
wuffs_base__i16__min(int16_t x, int16_t y) {
  return x < y ? x : y;
}

static inline int16_t  //
wuffs_base__i16__max(int16_t x, int16_t y) {
  return x > y ? x : y;
}

static inline int32_t  //
wuffs_base__i32__min(int32_t x, int32_t y) {
  return x < y ? x : y;
}

static inline int32_t  //
wuffs_base__i32__max(int32_t x, int32_t y) {
  return x > y ? x : y;
}

static inline int64_t  //
wuffs_base__i64__min(int64_t x, int64_t y) {
  return x < y ? x : y;
}

static inline int64_t  //
wuffs_base__i64__max(int64_t x, int64_t y) {
  return x > y ? x : y;
}

static inline uint8_t  //
wuffs_base__u8__min(uint8_t x, uint8_t y) {
  return x < y ? x : y;
}

static inline uint8_t  //
wuffs_base__u8__max(uint8_t x, uint8_t y) {
  return x > y ? x : y;
}

static inline uint16_t  //
wuffs_base__u16__min(uint16_t x, uint16_t y) {
  return x < y ? x : y;
}

static inline uint16_t  //
wuffs_base__u16__max(uint16_t x, uint16_t y) {
  return x > y ? x : y;
}

static inline uint32_t  //
wuffs_base__u32__min(uint32_t x, uint32_t y) {
  return x < y ? x : y;
}

static inline uint32_t  //
wuffs_base__u32__max(uint32_t x, uint32_t y) {
  return x > y ? x : y;
}

static inline uint64_t  //
wuffs_base__u64__min(uint64_t x, uint64_t y) {
  return x < y ? x : y;
}

static inline uint64_t  //
wuffs_base__u64__max(uint64_t x, uint64_t y) {
  return x > y ? x : y;
}

// --------

static inline uint8_t  //
wuffs_base__u8__rotate_left(uint8_t x, uint32_t n) {
  n &= 7;
  return ((uint8_t)(x << n)) | ((uint8_t)(x >> (8 - n)));
}

static inline uint8_t  //
wuffs_base__u8__rotate_right(uint8_t x, uint32_t n) {
  n &= 7;
  return ((uint8_t)(x >> n)) | ((uint8_t)(x << (8 - n)));
}

static inline uint16_t  //
wuffs_base__u16__rotate_left(uint16_t x, uint32_t n) {
  n &= 15;
  return ((uint16_t)(x << n)) | ((uint16_t)(x >> (16 - n)));
}

static inline uint16_t  //
wuffs_base__u16__rotate_right(uint16_t x, uint32_t n) {
  n &= 15;
  return ((uint16_t)(x >> n)) | ((uint16_t)(x << (16 - n)));
}

static inline uint32_t  //
wuffs_base__u32__rotate_left(uint32_t x, uint32_t n) {
  n &= 31;
  return ((uint32_t)(x << n)) | ((uint32_t)(x >> (32 - n)));
}

static inline uint32_t  //
wuffs_base__u32__rotate_right(uint32_t x, uint32_t n) {
  n &= 31;
  return ((uint32_t)(x >> n)) | ((uint32_t)(x << (32 - n)));
}

static inline uint64_t  //
wuffs_base__u64__rotate_left(uint64_t x, uint32_t n) {
  n &= 63;
  return ((uint64_t)(x << n)) | ((uint64_t)(x >> (64 - n)));
}

static inline uint64_t  //
wuffs_base__u64__rotate_right(uint64_t x, uint32_t n) {
  n &= 63;
  return ((uint64_t)(x >> n)) | ((uint64_t)(x << (64 - n)));
}

// --------

// Saturating arithmetic (sat_add, sat_sub) branchless bit-twiddling algorithms
// are per https://locklessinc.com/articles/sat_arithmetic/
//
// It is important that the underlying types are unsigned integers, as signed
// integer arithmetic overflow is undefined behavior in C.

static inline uint8_t  //
wuffs_base__u8__sat_add(uint8_t x, uint8_t y) {
  uint8_t res = (uint8_t)(x + y);
  res |= (uint8_t)(-(res < x));
  return res;
}

static inline uint8_t  //
wuffs_base__u8__sat_sub(uint8_t x, uint8_t y) {
  uint8_t res = (uint8_t)(x - y);
  res &= (uint8_t)(-(res <= x));
  return res;
}

static inline uint16_t  //
wuffs_base__u16__sat_add(uint16_t x, uint16_t y) {
  uint16_t res = (uint16_t)(x + y);
  res |= (uint16_t)(-(res < x));
  return res;
}

static inline uint16_t  //
wuffs_base__u16__sat_sub(uint16_t x, uint16_t y) {
  uint16_t res = (uint16_t)(x - y);
  res &= (uint16_t)(-(res <= x));
  return res;
}

static inline uint32_t  //
wuffs_base__u32__sat_add(uint32_t x, uint32_t y) {
  uint32_t res = (uint32_t)(x + y);
  res |= (uint32_t)(-(res < x));
  return res;
}

static inline uint32_t  //
wuffs_base__u32__sat_sub(uint32_t x, uint32_t y) {
  uint32_t res = (uint32_t)(x - y);
  res &= (uint32_t)(-(res <= x));
  return res;
}

static inline uint64_t  //
wuffs_base__u64__sat_add(uint64_t x, uint64_t y) {
  uint64_t res = (uint64_t)(x + y);
  res |= (uint64_t)(-(res < x));
  return res;
}

static inline uint64_t  //
wuffs_base__u64__sat_sub(uint64_t x, uint64_t y) {
  uint64_t res = (uint64_t)(x - y);
  res &= (uint64_t)(-(res <= x));
  return res;
}

// --------

typedef struct wuffs_base__multiply_u64__output__struct {
  uint64_t lo;
  uint64_t hi;
} wuffs_base__multiply_u64__output;

// wuffs_base__multiply_u64 returns x*y as a 128-bit value.
//
// The maximum inclusive output hi_lo is 0xFFFFFFFFFFFFFFFE_0000000000000001.
static inline wuffs_base__multiply_u64__output  //
wuffs_base__multiply_u64(uint64_t x, uint64_t y) {
#if defined(__SIZEOF_INT128__)
  __uint128_t z = ((__uint128_t)x) * ((__uint128_t)y);
  wuffs_base__multiply_u64__output o;
  o.lo = ((uint64_t)(z));
  o.hi = ((uint64_t)(z >> 64));
  return o;
#else
  // TODO: consider using the _mul128 intrinsic if defined(_MSC_VER).
  uint64_t x0 = x & 0xFFFFFFFF;
  uint64_t x1 = x >> 32;
  uint64_t y0 = y & 0xFFFFFFFF;
  uint64_t y1 = y >> 32;
  uint64_t w0 = x0 * y0;
  uint64_t t = (x1 * y0) + (w0 >> 32);
  uint64_t w1 = t & 0xFFFFFFFF;
  uint64_t w2 = t >> 32;
  w1 += x0 * y1;
  wuffs_base__multiply_u64__output o;
  o.lo = x * y;
  o.hi = (x1 * y1) + w2 + (w1 >> 32);
  return o;
#endif
}

// --------

// The "defined(__clang__)" isn't redundant. While vanilla clang defines
// __GNUC__, clang-cl (which mimics MSVC's cl.exe) does not.
#if (defined(__GNUC__) || defined(__clang__)) && (__SIZEOF_LONG__ == 8)

static inline uint32_t  //
wuffs_base__count_leading_zeroes_u64(uint64_t u) {
  return u ? ((uint32_t)(__builtin_clzl(u))) : 64u;
}

#else
// TODO: consider using the _BitScanReverse intrinsic if defined(_MSC_VER).

static inline uint32_t  //
wuffs_base__count_leading_zeroes_u64(uint64_t u) {
  if (u == 0) {
    return 64;
  }

  uint32_t n = 0;
  if ((u >> 32) == 0) {
    n |= 32;
    u <<= 32;
  }
  if ((u >> 48) == 0) {
    n |= 16;
    u <<= 16;
  }
  if ((u >> 56) == 0) {
    n |= 8;
    u <<= 8;
  }
  if ((u >> 60) == 0) {
    n |= 4;
    u <<= 4;
  }
  if ((u >> 62) == 0) {
    n |= 2;
    u <<= 2;
  }
  if ((u >> 63) == 0) {
    n |= 1;
    u <<= 1;
  }
  return n;
}

#endif  // (defined(__GNUC__) || defined(__clang__)) && (__SIZEOF_LONG__ == 8)

// --------

// Normally, the wuffs_base__peek_etc and wuffs_base__poke_etc implementations
// are both (1) correct regardless of CPU endianness and (2) very fast (e.g. an
// inlined wuffs_base__peek_u32le__no_bounds_check call, in an optimized clang
// or gcc build, is a single MOV instruction on x86_64).
//
// However, the endian-agnostic implementations are slow on Microsoft's C
// compiler (MSC). Alternative memcpy-based implementations restore speed, but
// they are only correct on little-endian CPU architectures. Defining
// WUFFS_BASE__USE_MEMCPY_LE_PEEK_POKE opts in to these implementations.
//
// https://godbolt.org/z/q4MfjzTPh
#if defined(_MSC_VER) && !defined(__clang__) && \
    (defined(_M_ARM64) || defined(_M_X64))
#define WUFFS_BASE__USE_MEMCPY_LE_PEEK_POKE
#endif

#define wuffs_base__peek_u8be__no_bounds_check \
  wuffs_base__peek_u8__no_bounds_check
#define wuffs_base__peek_u8le__no_bounds_check \
  wuffs_base__peek_u8__no_bounds_check

static inline uint8_t  //
wuffs_base__peek_u8__no_bounds_check(const uint8_t* p) {
  return p[0];
}

static inline uint16_t  //
wuffs_base__peek_u16be__no_bounds_check(const uint8_t* p) {
#if defined(WUFFS_BASE__USE_MEMCPY_LE_PEEK_POKE)
  uint16_t x;
  memcpy(&x, p, 2);
  return _byteswap_ushort(x);
#else
  return (uint16_t)(((uint16_t)(p[0]) << 8) | ((uint16_t)(p[1]) << 0));
#endif
}

static inline uint16_t  //
wuffs_base__peek_u16le__no_bounds_check(const uint8_t* p) {
#if defined(WUFFS_BASE__USE_MEMCPY_LE_PEEK_POKE)
  uint16_t x;
  memcpy(&x, p, 2);
  return x;
#else
  return (uint16_t)(((uint16_t)(p[0]) << 0) | ((uint16_t)(p[1]) << 8));
#endif
}

static inline uint32_t  //
wuffs_base__peek_u24be__no_bounds_check(const uint8_t* p) {
  return ((uint32_t)(p[0]) << 16) | ((uint32_t)(p[1]) << 8) |
         ((uint32_t)(p[2]) << 0);
}

static inline uint32_t  //
wuffs_base__peek_u24le__no_bounds_check(const uint8_t* p) {
  return ((uint32_t)(p[0]) << 0) | ((uint32_t)(p[1]) << 8) |
         ((uint32_t)(p[2]) << 16);
}

static inline uint32_t  //
wuffs_base__peek_u32be__no_bounds_check(const uint8_t* p) {
#if defined(WUFFS_BASE__USE_MEMCPY_LE_PEEK_POKE)
  uint32_t x;
  memcpy(&x, p, 4);
  return _byteswap_ulong(x);
#else
  return ((uint32_t)(p[0]) << 24) | ((uint32_t)(p[1]) << 16) |
         ((uint32_t)(p[2]) << 8) | ((uint32_t)(p[3]) << 0);
#endif
}

static inline uint32_t  //
wuffs_base__peek_u32le__no_bounds_check(const uint8_t* p) {
#if defined(WUFFS_BASE__USE_MEMCPY_LE_PEEK_POKE)
  uint32_t x;
  memcpy(&x, p, 4);
  return x;
#else
  return ((uint32_t)(p[0]) << 0) | ((uint32_t)(p[1]) << 8) |
         ((uint32_t)(p[2]) << 16) | ((uint32_t)(p[3]) << 24);
#endif
}

static inline uint64_t  //
wuffs_base__peek_u40be__no_bounds_check(const uint8_t* p) {
  return ((uint64_t)(p[0]) << 32) | ((uint64_t)(p[1]) << 24) |
         ((uint64_t)(p[2]) << 16) | ((uint64_t)(p[3]) << 8) |
         ((uint64_t)(p[4]) << 0);
}

static inline uint64_t  //
wuffs_base__peek_u40le__no_bounds_check(const uint8_t* p) {
  return ((uint64_t)(p[0]) << 0) | ((uint64_t)(p[1]) << 8) |
         ((uint64_t)(p[2]) << 16) | ((uint64_t)(p[3]) << 24) |
         ((uint64_t)(p[4]) << 32);
}

static inline uint64_t  //
wuffs_base__peek_u48be__no_bounds_check(const uint8_t* p) {
  return ((uint64_t)(p[0]) << 40) | ((uint64_t)(p[1]) << 32) |
         ((uint64_t)(p[2]) << 24) | ((uint64_t)(p[3]) << 16) |
         ((uint64_t)(p[4]) << 8) | ((uint64_t)(p[5]) << 0);
}

static inline uint64_t  //
wuffs_base__peek_u48le__no_bounds_check(const uint8_t* p) {
  return ((uint64_t)(p[0]) << 0) | ((uint64_t)(p[1]) << 8) |
         ((uint64_t)(p[2]) << 16) | ((uint64_t)(p[3]) << 24) |
         ((uint64_t)(p[4]) << 32) | ((uint64_t)(p[5]) << 40);
}

static inline uint64_t  //
wuffs_base__peek_u56be__no_bounds_check(const uint8_t* p) {
  return ((uint64_t)(p[0]) << 48) | ((uint64_t)(p[1]) << 40) |
         ((uint64_t)(p[2]) << 32) | ((uint64_t)(p[3]) << 24) |
         ((uint64_t)(p[4]) << 16) | ((uint64_t)(p[5]) << 8) |
         ((uint64_t)(p[6]) << 0);
}

static inline uint64_t  //
wuffs_base__peek_u56le__no_bounds_check(const uint8_t* p) {
  return ((uint64_t)(p[0]) << 0) | ((uint64_t)(p[1]) << 8) |
         ((uint64_t)(p[2]) << 16) | ((uint64_t)(p[3]) << 24) |
         ((uint64_t)(p[4]) << 32) | ((uint64_t)(p[5]) << 40) |
         ((uint64_t)(p[6]) << 48);
}

static inline uint64_t  //
wuffs_base__peek_u64be__no_bounds_check(const uint8_t* p) {
#if defined(WUFFS_BASE__USE_MEMCPY_LE_PEEK_POKE)
  uint64_t x;
  memcpy(&x, p, 8);
  return _byteswap_uint64(x);
#else
  return ((uint64_t)(p[0]) << 56) | ((uint64_t)(p[1]) << 48) |
         ((uint64_t)(p[2]) << 40) | ((uint64_t)(p[3]) << 32) |
         ((uint64_t)(p[4]) << 24) | ((uint64_t)(p[5]) << 16) |
         ((uint64_t)(p[6]) << 8) | ((uint64_t)(p[7]) << 0);
#endif
}

static inline uint64_t  //
wuffs_base__peek_u64le__no_bounds_check(const uint8_t* p) {
#if defined(WUFFS_BASE__USE_MEMCPY_LE_PEEK_POKE)
  uint64_t x;
  memcpy(&x, p, 8);
  return x;
#else
  return ((uint64_t)(p[0]) << 0) | ((uint64_t)(p[1]) << 8) |
         ((uint64_t)(p[2]) << 16) | ((uint64_t)(p[3]) << 24) |
         ((uint64_t)(p[4]) << 32) | ((uint64_t)(p[5]) << 40) |
         ((uint64_t)(p[6]) << 48) | ((uint64_t)(p[7]) << 56);
#endif
}

// --------

#define wuffs_base__poke_u8be__no_bounds_check \
  wuffs_base__poke_u8__no_bounds_check
#define wuffs_base__poke_u8le__no_bounds_check \
  wuffs_base__poke_u8__no_bounds_check

static inline void  //
wuffs_base__poke_u8__no_bounds_check(uint8_t* p, uint8_t x) {
  p[0] = x;
}

static inline void  //
wuffs_base__poke_u16be__no_bounds_check(uint8_t* p, uint16_t x) {
  p[0] = (uint8_t)(x >> 8);
  p[1] = (uint8_t)(x >> 0);
}

static inline void  //
wuffs_base__poke_u16le__no_bounds_check(uint8_t* p, uint16_t x) {
#if defined(WUFFS_BASE__USE_MEMCPY_LE_PEEK_POKE) || \
    (defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__))
  // This seems to perform better on gcc 10 (but not clang 9). Clang also
  // defines "__GNUC__".
  memcpy(p, &x, 2);
#else
  p[0] = (uint8_t)(x >> 0);
  p[1] = (uint8_t)(x >> 8);
#endif
}

static inline void  //
wuffs_base__poke_u24be__no_bounds_check(uint8_t* p, uint32_t x) {
  p[0] = (uint8_t)(x >> 16);
  p[1] = (uint8_t)(x >> 8);
  p[2] = (uint8_t)(x >> 0);
}

static inline void  //
wuffs_base__poke_u24le__no_bounds_check(uint8_t* p, uint32_t x) {
  p[0] = (uint8_t)(x >> 0);
  p[1] = (uint8_t)(x >> 8);
  p[2] = (uint8_t)(x >> 16);
}

static inline void  //
wuffs_base__poke_u32be__no_bounds_check(uint8_t* p, uint32_t x) {
  p[0] = (uint8_t)(x >> 24);
  p[1] = (uint8_t)(x >> 16);
  p[2] = (uint8_t)(x >> 8);
  p[3] = (uint8_t)(x >> 0);
}

static inline void  //
wuffs_base__poke_u32le__no_bounds_check(uint8_t* p, uint32_t x) {
#if defined(WUFFS_BASE__USE_MEMCPY_LE_PEEK_POKE) || \
    (defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__))
  // This seems to perform better on gcc 10 (but not clang 9). Clang also
  // defines "__GNUC__".
  memcpy(p, &x, 4);
#else
  p[0] = (uint8_t)(x >> 0);
  p[1] = (uint8_t)(x >> 8);
  p[2] = (uint8_t)(x >> 16);
  p[3] = (uint8_t)(x >> 24);
#endif
}

static inline void  //
wuffs_base__poke_u40be__no_bounds_check(uint8_t* p, uint64_t x) {
  p[0] = (uint8_t)(x >> 32);
  p[1] = (uint8_t)(x >> 24);
  p[2] = (uint8_t)(x >> 16);
  p[3] = (uint8_t)(x >> 8);
  p[4] = (uint8_t)(x >> 0);
}

static inline void  //
wuffs_base__poke_u40le__no_bounds_check(uint8_t* p, uint64_t x) {
  p[0] = (uint8_t)(x >> 0);
  p[1] = (uint8_t)(x >> 8);
  p[2] = (uint8_t)(x >> 16);
  p[3] = (uint8_t)(x >> 24);
  p[4] = (uint8_t)(x >> 32);
}

static inline void  //
wuffs_base__poke_u48be__no_bounds_check(uint8_t* p, uint64_t x) {
  p[0] = (uint8_t)(x >> 40);
  p[1] = (uint8_t)(x >> 32);
  p[2] = (uint8_t)(x >> 24);
  p[3] = (uint8_t)(x >> 16);
  p[4] = (uint8_t)(x >> 8);
  p[5] = (uint8_t)(x >> 0);
}

static inline void  //
wuffs_base__poke_u48le__no_bounds_check(uint8_t* p, uint64_t x) {
  p[0] = (uint8_t)(x >> 0);
  p[1] = (uint8_t)(x >> 8);
  p[2] = (uint8_t)(x >> 16);
  p[3] = (uint8_t)(x >> 24);
  p[4] = (uint8_t)(x >> 32);
  p[5] = (uint8_t)(x >> 40);
}

static inline void  //
wuffs_base__poke_u56be__no_bounds_check(uint8_t* p, uint64_t x) {
  p[0] = (uint8_t)(x >> 48);
  p[1] = (uint8_t)(x >> 40);
  p[2] = (uint8_t)(x >> 32);
  p[3] = (uint8_t)(x >> 24);
  p[4] = (uint8_t)(x >> 16);
  p[5] = (uint8_t)(x >> 8);
  p[6] = (uint8_t)(x >> 0);
}

static inline void  //
wuffs_base__poke_u56le__no_bounds_check(uint8_t* p, uint64_t x) {
  p[0] = (uint8_t)(x >> 0);
  p[1] = (uint8_t)(x >> 8);
  p[2] = (uint8_t)(x >> 16);
  p[3] = (uint8_t)(x >> 24);
  p[4] = (uint8_t)(x >> 32);
  p[5] = (uint8_t)(x >> 40);
  p[6] = (uint8_t)(x >> 48);
}

static inline void  //
wuffs_base__poke_u64be__no_bounds_check(uint8_t* p, uint64_t x) {
  p[0] = (uint8_t)(x >> 56);
  p[1] = (uint8_t)(x >> 48);
  p[2] = (uint8_t)(x >> 40);
  p[3] = (uint8_t)(x >> 32);
  p[4] = (uint8_t)(x >> 24);
  p[5] = (uint8_t)(x >> 16);
  p[6] = (uint8_t)(x >> 8);
  p[7] = (uint8_t)(x >> 0);
}

static inline void  //
wuffs_base__poke_u64le__no_bounds_check(uint8_t* p, uint64_t x) {
#if defined(WUFFS_BASE__USE_MEMCPY_LE_PEEK_POKE) || \
    (defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__))
  // This seems to perform better on gcc 10 (but not clang 9). Clang also
  // defines "__GNUC__".
  memcpy(p, &x, 8);
#else
  p[0] = (uint8_t)(x >> 0);
  p[1] = (uint8_t)(x >> 8);
  p[2] = (uint8_t)(x >> 16);
  p[3] = (uint8_t)(x >> 24);
  p[4] = (uint8_t)(x >> 32);
  p[5] = (uint8_t)(x >> 40);
  p[6] = (uint8_t)(x >> 48);
  p[7] = (uint8_t)(x >> 56);
#endif
}

// --------

// Load and Store functions are deprecated. Use Peek and Poke instead.

#define wuffs_base__load_u8__no_bounds_check \
  wuffs_base__peek_u8__no_bounds_check
#define wuffs_base__load_u16be__no_bounds_check \
  wuffs_base__peek_u16be__no_bounds_check
#define wuffs_base__load_u16le__no_bounds_check \
  wuffs_base__peek_u16le__no_bounds_check
#define wuffs_base__load_u24be__no_bounds_check \
  wuffs_base__peek_u24be__no_bounds_check
#define wuffs_base__load_u24le__no_bounds_check \
  wuffs_base__peek_u24le__no_bounds_check
#define wuffs_base__load_u32be__no_bounds_check \
  wuffs_base__peek_u32be__no_bounds_check
#define wuffs_base__load_u32le__no_bounds_check \
  wuffs_base__peek_u32le__no_bounds_check
#define wuffs_base__load_u40be__no_bounds_check \
  wuffs_base__peek_u40be__no_bounds_check
#define wuffs_base__load_u40le__no_bounds_check \
  wuffs_base__peek_u40le__no_bounds_check
#define wuffs_base__load_u48be__no_bounds_check \
  wuffs_base__peek_u48be__no_bounds_check
#define wuffs_base__load_u48le__no_bounds_check \
  wuffs_base__peek_u48le__no_bounds_check
#define wuffs_base__load_u56be__no_bounds_check \
  wuffs_base__peek_u56be__no_bounds_check
#define wuffs_base__load_u56le__no_bounds_check \
  wuffs_base__peek_u56le__no_bounds_check
#define wuffs_base__load_u64be__no_bounds_check \
  wuffs_base__peek_u64be__no_bounds_check
#define wuffs_base__load_u64le__no_bounds_check \
  wuffs_base__peek_u64le__no_bounds_check

#define wuffs_base__store_u8__no_bounds_check \
  wuffs_base__poke_u8__no_bounds_check
#define wuffs_base__store_u16be__no_bounds_check \
  wuffs_base__poke_u16be__no_bounds_check
#define wuffs_base__store_u16le__no_bounds_check \
  wuffs_base__poke_u16le__no_bounds_check
#define wuffs_base__store_u24be__no_bounds_check \
  wuffs_base__poke_u24be__no_bounds_check
#define wuffs_base__store_u24le__no_bounds_check \
  wuffs_base__poke_u24le__no_bounds_check
#define wuffs_base__store_u32be__no_bounds_check \
  wuffs_base__poke_u32be__no_bounds_check
#define wuffs_base__store_u32le__no_bounds_check \
  wuffs_base__poke_u32le__no_bounds_check
#define wuffs_base__store_u40be__no_bounds_check \
  wuffs_base__poke_u40be__no_bounds_check
#define wuffs_base__store_u40le__no_bounds_check \
  wuffs_base__poke_u40le__no_bounds_check
#define wuffs_base__store_u48be__no_bounds_check \
  wuffs_base__poke_u48be__no_bounds_check
#define wuffs_base__store_u48le__no_bounds_check \
  wuffs_base__poke_u48le__no_bounds_check
#define wuffs_base__store_u56be__no_bounds_check \
  wuffs_base__poke_u56be__no_bounds_check
#define wuffs_base__store_u56le__no_bounds_check \
  wuffs_base__poke_u56le__no_bounds_check
#define wuffs_base__store_u64be__no_bounds_check \
  wuffs_base__poke_u64be__no_bounds_check
#define wuffs_base__store_u64le__no_bounds_check \
  wuffs_base__poke_u64le__no_bounds_check

// ---------------- Slices and Tables

// WUFFS_BASE__SLICE is a 1-dimensional buffer.
//
// len measures a number of elements, not necessarily a size in bytes.
//
// A value with all fields NULL or zero is a valid, empty slice.
#define WUFFS_BASE__SLICE(T) \
  struct {                   \
    T* ptr;                  \
    size_t len;              \
  }

// WUFFS_BASE__TABLE is a 2-dimensional buffer.
//
// width, height and stride measure a number of elements, not necessarily a
// size in bytes.
//
// A value with all fields NULL or zero is a valid, empty table.
#define WUFFS_BASE__TABLE(T) \
  struct {                   \
    T* ptr;                  \
    size_t width;            \
    size_t height;           \
    size_t stride;           \
  }

typedef WUFFS_BASE__SLICE(uint8_t) wuffs_base__slice_u8;
typedef WUFFS_BASE__SLICE(uint16_t) wuffs_base__slice_u16;
typedef WUFFS_BASE__SLICE(uint32_t) wuffs_base__slice_u32;
typedef WUFFS_BASE__SLICE(uint64_t) wuffs_base__slice_u64;

typedef WUFFS_BASE__TABLE(uint8_t) wuffs_base__table_u8;
typedef WUFFS_BASE__TABLE(uint16_t) wuffs_base__table_u16;
typedef WUFFS_BASE__TABLE(uint32_t) wuffs_base__table_u32;
typedef WUFFS_BASE__TABLE(uint64_t) wuffs_base__table_u64;

static inline wuffs_base__slice_u8  //
wuffs_base__make_slice_u8(uint8_t* ptr, size_t len) {
  wuffs_base__slice_u8 ret;
  ret.ptr = ptr;
  ret.len = len;
  return ret;
}

static inline wuffs_base__slice_u16  //
wuffs_base__make_slice_u16(uint16_t* ptr, size_t len) {
  wuffs_base__slice_u16 ret;
  ret.ptr = ptr;
  ret.len = len;
  return ret;
}

static inline wuffs_base__slice_u32  //
wuffs_base__make_slice_u32(uint32_t* ptr, size_t len) {
  wuffs_base__slice_u32 ret;
  ret.ptr = ptr;
  ret.len = len;
  return ret;
}

static inline wuffs_base__slice_u64  //
wuffs_base__make_slice_u64(uint64_t* ptr, size_t len) {
  wuffs_base__slice_u64 ret;
  ret.ptr = ptr;
  ret.len = len;
  return ret;
}

static inline wuffs_base__slice_u8  //
wuffs_base__make_slice_u8_ij(uint8_t* ptr, size_t i, size_t j) {
  wuffs_base__slice_u8 ret;
  ret.ptr = ptr + i;
  ret.len = (j >= i) ? (j - i) : 0;
  return ret;
}

static inline wuffs_base__slice_u16  //
wuffs_base__make_slice_u16_ij(uint16_t* ptr, size_t i, size_t j) {
  wuffs_base__slice_u16 ret;
  ret.ptr = ptr + i;
  ret.len = (j >= i) ? (j - i) : 0;
  return ret;
}

static inline wuffs_base__slice_u32  //
wuffs_base__make_slice_u32_ij(uint32_t* ptr, size_t i, size_t j) {
  wuffs_base__slice_u32 ret;
  ret.ptr = ptr + i;
  ret.len = (j >= i) ? (j - i) : 0;
  return ret;
}

static inline wuffs_base__slice_u64  //
wuffs_base__make_slice_u64_ij(uint64_t* ptr, size_t i, size_t j) {
  wuffs_base__slice_u64 ret;
  ret.ptr = ptr + i;
  ret.len = (j >= i) ? (j - i) : 0;
  return ret;
}

static inline wuffs_base__slice_u8  //
wuffs_base__empty_slice_u8() {
  wuffs_base__slice_u8 ret;
  ret.ptr = NULL;
  ret.len = 0;
  return ret;
}

static inline wuffs_base__slice_u16  //
wuffs_base__empty_slice_u16() {
  wuffs_base__slice_u16 ret;
  ret.ptr = NULL;
  ret.len = 0;
  return ret;
}

static inline wuffs_base__slice_u32  //
wuffs_base__empty_slice_u32() {
  wuffs_base__slice_u32 ret;
  ret.ptr = NULL;
  ret.len = 0;
  return ret;
}

static inline wuffs_base__slice_u64  //
wuffs_base__empty_slice_u64() {
  wuffs_base__slice_u64 ret;
  ret.ptr = NULL;
  ret.len = 0;
  return ret;
}

static inline wuffs_base__table_u8  //
wuffs_base__make_table_u8(uint8_t* ptr,
                          size_t width,
                          size_t height,
                          size_t stride) {
  wuffs_base__table_u8 ret;
  ret.ptr = ptr;
  ret.width = width;
  ret.height = height;
  ret.stride = stride;
  return ret;
}

static inline wuffs_base__table_u16  //
wuffs_base__make_table_u16(uint16_t* ptr,
                           size_t width,
                           size_t height,
                           size_t stride) {
  wuffs_base__table_u16 ret;
  ret.ptr = ptr;
  ret.width = width;
  ret.height = height;
  ret.stride = stride;
  return ret;
}

static inline wuffs_base__table_u32  //
wuffs_base__make_table_u32(uint32_t* ptr,
                           size_t width,
                           size_t height,
                           size_t stride) {
  wuffs_base__table_u32 ret;
  ret.ptr = ptr;
  ret.width = width;
  ret.height = height;
  ret.stride = stride;
  return ret;
}

static inline wuffs_base__table_u64  //
wuffs_base__make_table_u64(uint64_t* ptr,
                           size_t width,
                           size_t height,
                           size_t stride) {
  wuffs_base__table_u64 ret;
  ret.ptr = ptr;
  ret.width = width;
  ret.height = height;
  ret.stride = stride;
  return ret;
}

static inline wuffs_base__table_u8  //
wuffs_base__empty_table_u8() {
  wuffs_base__table_u8 ret;
  ret.ptr = NULL;
  ret.width = 0;
  ret.height = 0;
  ret.stride = 0;
  return ret;
}

static inline wuffs_base__table_u16  //
wuffs_base__empty_table_u16() {
  wuffs_base__table_u16 ret;
  ret.ptr = NULL;
  ret.width = 0;
  ret.height = 0;
  ret.stride = 0;
  return ret;
}

static inline wuffs_base__table_u32  //
wuffs_base__empty_table_u32() {
  wuffs_base__table_u32 ret;
  ret.ptr = NULL;
  ret.width = 0;
  ret.height = 0;
  ret.stride = 0;
  return ret;
}

static inline wuffs_base__table_u64  //
wuffs_base__empty_table_u64() {
  wuffs_base__table_u64 ret;
  ret.ptr = NULL;
  ret.width = 0;
  ret.height = 0;
  ret.stride = 0;
  return ret;
}

static inline bool  //
wuffs_base__slice_u8__overlaps(wuffs_base__slice_u8 s, wuffs_base__slice_u8 t) {
  return ((s.ptr <= t.ptr) && (t.ptr < (s.ptr + s.len))) ||
         ((t.ptr <= s.ptr) && (s.ptr < (t.ptr + t.len)));
}

// wuffs_base__slice_u8__subslice_i returns s[i:].
//
// It returns an empty slice if i is out of bounds.
static inline wuffs_base__slice_u8  //
wuffs_base__slice_u8__subslice_i(wuffs_base__slice_u8 s, uint64_t i) {
  if ((i <= SIZE_MAX) && (i <= s.len)) {
    return wuffs_base__make_slice_u8(s.ptr + i, ((size_t)(s.len - i)));
  }
  return wuffs_base__make_slice_u8(NULL, 0);
}

// wuffs_base__slice_u8__subslice_j returns s[:j].
//
// It returns an empty slice if j is out of bounds.
static inline wuffs_base__slice_u8  //
wuffs_base__slice_u8__subslice_j(wuffs_base__slice_u8 s, uint64_t j) {
  if ((j <= SIZE_MAX) && (j <= s.len)) {
    return wuffs_base__make_slice_u8(s.ptr, ((size_t)j));
  }
  return wuffs_base__make_slice_u8(NULL, 0);
}

// wuffs_base__slice_u8__subslice_ij returns s[i:j].
//
// It returns an empty slice if i or j is out of bounds.
static inline wuffs_base__slice_u8  //
wuffs_base__slice_u8__subslice_ij(wuffs_base__slice_u8 s,
                                  uint64_t i,
                                  uint64_t j) {
  if ((i <= j) && (j <= SIZE_MAX) && (j <= s.len)) {
    return wuffs_base__make_slice_u8(s.ptr + i, ((size_t)(j - i)));
  }
  return wuffs_base__make_slice_u8(NULL, 0);
}

// wuffs_base__table_u8__subtable_ij returns t[ix:jx, iy:jy].
//
// It returns an empty table if i or j is out of bounds.
static inline wuffs_base__table_u8  //
wuffs_base__table_u8__subtable_ij(wuffs_base__table_u8 t,
                                  uint64_t ix,
                                  uint64_t iy,
                                  uint64_t jx,
                                  uint64_t jy) {
  if ((ix <= jx) && (jx <= SIZE_MAX) && (jx <= t.width) &&  //
      (iy <= jy) && (jy <= SIZE_MAX) && (jy <= t.height)) {
    return wuffs_base__make_table_u8(t.ptr + ix + (iy * t.stride),  //
                                     ((size_t)(jx - ix)),           //
                                     ((size_t)(jy - iy)),           //
                                     t.stride);                     //
  }
  return wuffs_base__make_table_u8(NULL, 0, 0, 0);
}

// wuffs_base__table__flattened_length returns the number of elements covered
// by the 1-dimensional span that backs a 2-dimensional table. This counts the
// elements inside the table and, when width != stride, the elements outside
// the table but between its rows.
//
// For example, consider a width 10, height 4, stride 10 table. Mark its first
// and last (inclusive) elements with 'a' and 'z'. This function returns 40.
//
//    a123456789
//    0123456789
//    0123456789
//    012345678z
//
// Now consider the sub-table of that from (2, 1) inclusive to (8, 4) exclusive.
//
//    a123456789
//    01iiiiiioo
//    ooiiiiiioo
//    ooiiiiii8z
//
// This function (called with width 6, height 3, stride 10) returns 26: 18 'i'
// inside elements plus 8 'o' outside elements. Note that 26 is less than a
// naive (height * stride = 30) computation. Indeed, advancing 29 elements from
// the first 'i' would venture past 'z', out of bounds of the original table.
//
// It does not check for overflow, but if the arguments come from a table that
// exists in memory and each element occupies a positive number of bytes then
// the result should be bounded by the amount of allocatable memory (which
// shouldn't overflow SIZE_MAX).
static inline size_t  //
wuffs_base__table__flattened_length(size_t width,
                                    size_t height,
                                    size_t stride) {
  if (height == 0) {
    return 0;
  }
  return ((height - 1) * stride) + width;
}

// ---------------- Magic Numbers

// wuffs_base__magic_number_guess_fourcc guesses the file format of some data,
// given its starting bytes (the prefix_data argument) and whether or not there
// may be further bytes (the prefix_closed argument; true means that
// prefix_data is the entire data).
//
// It returns a positive FourCC value on success.
//
// It returns zero if nothing matches its hard-coded list of 'magic numbers'.
//
// It returns a negative value if prefix_closed is false and a longer prefix is
// required for a conclusive result. For example, a single 'B' byte (without
// further data) is not enough to discriminate the BMP and BPG image file
// formats. Similarly, a single '\xFF' byte might be the start of JPEG data or
// it might be the start of some other binary data.
//
// It does not do a full validity check. Like any guess made from a short
// prefix of the data, it may return false positives. Data that starts with 99
// bytes of valid JPEG followed by corruption or truncation is an invalid JPEG
// image overall, but this function will still return WUFFS_BASE__FOURCC__JPEG.
//
// Another source of false positives is that some 'magic numbers' are valid
// ASCII data. A file starting with "GIF87a and GIF89a are the two versions of
// GIF" will match GIF's 'magic number' even if it's plain text, not an image.
//
// For modular builds that divide the base module into sub-modules, using this
// function requires the WUFFS_CONFIG__MODULE__BASE__MAGIC sub-module, not just
// WUFFS_CONFIG__MODULE__BASE__CORE.
WUFFS_BASE__MAYBE_STATIC int32_t  //
wuffs_base__magic_number_guess_fourcc(wuffs_base__slice_u8 prefix_data,
                                      bool prefix_closed);

// ---------------- Ranges and Rects

// See https://github.com/google/wuffs/blob/main/doc/note/ranges-and-rects.md

typedef struct wuffs_base__range_ii_u32__struct {
  uint32_t min_incl;
  uint32_t max_incl;

#ifdef __cplusplus
  inline bool is_empty() const;
  inline bool equals(wuffs_base__range_ii_u32__struct s) const;
  inline wuffs_base__range_ii_u32__struct intersect(
      wuffs_base__range_ii_u32__struct s) const;
  inline wuffs_base__range_ii_u32__struct unite(
      wuffs_base__range_ii_u32__struct s) const;
  inline bool contains(uint32_t x) const;
  inline bool contains_range(wuffs_base__range_ii_u32__struct s) const;
#endif  // __cplusplus

} wuffs_base__range_ii_u32;

static inline wuffs_base__range_ii_u32  //
wuffs_base__empty_range_ii_u32() {
  wuffs_base__range_ii_u32 ret;
  ret.min_incl = 0;
  ret.max_incl = 0;
  return ret;
}

static inline wuffs_base__range_ii_u32  //
wuffs_base__make_range_ii_u32(uint32_t min_incl, uint32_t max_incl) {
  wuffs_base__range_ii_u32 ret;
  ret.min_incl = min_incl;
  ret.max_incl = max_incl;
  return ret;
}

static inline bool  //
wuffs_base__range_ii_u32__is_empty(const wuffs_base__range_ii_u32* r) {
  return r->min_incl > r->max_incl;
}

static inline bool  //
wuffs_base__range_ii_u32__equals(const wuffs_base__range_ii_u32* r,
                                 wuffs_base__range_ii_u32 s) {
  return (r->min_incl == s.min_incl && r->max_incl == s.max_incl) ||
         (wuffs_base__range_ii_u32__is_empty(r) &&
          wuffs_base__range_ii_u32__is_empty(&s));
}

static inline wuffs_base__range_ii_u32  //
wuffs_base__range_ii_u32__intersect(const wuffs_base__range_ii_u32* r,
                                    wuffs_base__range_ii_u32 s) {
  wuffs_base__range_ii_u32 t;
  t.min_incl = wuffs_base__u32__max(r->min_incl, s.min_incl);
  t.max_incl = wuffs_base__u32__min(r->max_incl, s.max_incl);
  return t;
}

static inline wuffs_base__range_ii_u32  //
wuffs_base__range_ii_u32__unite(const wuffs_base__range_ii_u32* r,
                                wuffs_base__range_ii_u32 s) {
  if (wuffs_base__range_ii_u32__is_empty(r)) {
    return s;
  }
  if (wuffs_base__range_ii_u32__is_empty(&s)) {
    return *r;
  }
  wuffs_base__range_ii_u32 t;
  t.min_incl = wuffs_base__u32__min(r->min_incl, s.min_incl);
  t.max_incl = wuffs_base__u32__max(r->max_incl, s.max_incl);
  return t;
}

static inline bool  //
wuffs_base__range_ii_u32__contains(const wuffs_base__range_ii_u32* r,
                                   uint32_t x) {
  return (r->min_incl <= x) && (x <= r->max_incl);
}

static inline bool  //
wuffs_base__range_ii_u32__contains_range(const wuffs_base__range_ii_u32* r,
                                         wuffs_base__range_ii_u32 s) {
  return wuffs_base__range_ii_u32__equals(
      &s, wuffs_base__range_ii_u32__intersect(r, s));
}

#ifdef __cplusplus

inline bool  //
wuffs_base__range_ii_u32::is_empty() const {
  return wuffs_base__range_ii_u32__is_empty(this);
}

inline bool  //
wuffs_base__range_ii_u32::equals(wuffs_base__range_ii_u32 s) const {
  return wuffs_base__range_ii_u32__equals(this, s);
}

inline wuffs_base__range_ii_u32  //
wuffs_base__range_ii_u32::intersect(wuffs_base__range_ii_u32 s) const {
  return wuffs_base__range_ii_u32__intersect(this, s);
}

inline wuffs_base__range_ii_u32  //
wuffs_base__range_ii_u32::unite(wuffs_base__range_ii_u32 s) const {
  return wuffs_base__range_ii_u32__unite(this, s);
}

inline bool  //
wuffs_base__range_ii_u32::contains(uint32_t x) const {
  return wuffs_base__range_ii_u32__contains(this, x);
}

inline bool  //
wuffs_base__range_ii_u32::contains_range(wuffs_base__range_ii_u32 s) const {
  return wuffs_base__range_ii_u32__contains_range(this, s);
}

#endif  // __cplusplus

// --------

typedef struct wuffs_base__range_ie_u32__struct {
  uint32_t min_incl;
  uint32_t max_excl;

#ifdef __cplusplus
  inline bool is_empty() const;
  inline bool equals(wuffs_base__range_ie_u32__struct s) const;
  inline wuffs_base__range_ie_u32__struct intersect(
      wuffs_base__range_ie_u32__struct s) const;
  inline wuffs_base__range_ie_u32__struct unite(
      wuffs_base__range_ie_u32__struct s) const;
  inline bool contains(uint32_t x) const;
  inline bool contains_range(wuffs_base__range_ie_u32__struct s) const;
  inline uint32_t length() const;
#endif  // __cplusplus

} wuffs_base__range_ie_u32;

static inline wuffs_base__range_ie_u32  //
wuffs_base__empty_range_ie_u32() {
  wuffs_base__range_ie_u32 ret;
  ret.min_incl = 0;
  ret.max_excl = 0;
  return ret;
}

static inline wuffs_base__range_ie_u32  //
wuffs_base__make_range_ie_u32(uint32_t min_incl, uint32_t max_excl) {
  wuffs_base__range_ie_u32 ret;
  ret.min_incl = min_incl;
  ret.max_excl = max_excl;
  return ret;
}

static inline bool  //
wuffs_base__range_ie_u32__is_empty(const wuffs_base__range_ie_u32* r) {
  return r->min_incl >= r->max_excl;
}

static inline bool  //
wuffs_base__range_ie_u32__equals(const wuffs_base__range_ie_u32* r,
                                 wuffs_base__range_ie_u32 s) {
  return (r->min_incl == s.min_incl && r->max_excl == s.max_excl) ||
         (wuffs_base__range_ie_u32__is_empty(r) &&
          wuffs_base__range_ie_u32__is_empty(&s));
}

static inline wuffs_base__range_ie_u32  //
wuffs_base__range_ie_u32__intersect(const wuffs_base__range_ie_u32* r,
                                    wuffs_base__range_ie_u32 s) {
  wuffs_base__range_ie_u32 t;
  t.min_incl = wuffs_base__u32__max(r->min_incl, s.min_incl);
  t.max_excl = wuffs_base__u32__min(r->max_excl, s.max_excl);
  return t;
}

static inline wuffs_base__range_ie_u32  //
wuffs_base__range_ie_u32__unite(const wuffs_base__range_ie_u32* r,
                                wuffs_base__range_ie_u32 s) {
  if (wuffs_base__range_ie_u32__is_empty(r)) {
    return s;
  }
  if (wuffs_base__range_ie_u32__is_empty(&s)) {
    return *r;
  }
  wuffs_base__range_ie_u32 t;
  t.min_incl = wuffs_base__u32__min(r->min_incl, s.min_incl);
  t.max_excl = wuffs_base__u32__max(r->max_excl, s.max_excl);
  return t;
}

static inline bool  //
wuffs_base__range_ie_u32__contains(const wuffs_base__range_ie_u32* r,
                                   uint32_t x) {
  return (r->min_incl <= x) && (x < r->max_excl);
}

static inline bool  //
wuffs_base__range_ie_u32__contains_range(const wuffs_base__range_ie_u32* r,
                                         wuffs_base__range_ie_u32 s) {
  return wuffs_base__range_ie_u32__equals(
      &s, wuffs_base__range_ie_u32__intersect(r, s));
}

static inline uint32_t  //
wuffs_base__range_ie_u32__length(const wuffs_base__range_ie_u32* r) {
  return wuffs_base__u32__sat_sub(r->max_excl, r->min_incl);
}

#ifdef __cplusplus

inline bool  //
wuffs_base__range_ie_u32::is_empty() const {
  return wuffs_base__range_ie_u32__is_empty(this);
}

inline bool  //
wuffs_base__range_ie_u32::equals(wuffs_base__range_ie_u32 s) const {
  return wuffs_base__range_ie_u32__equals(this, s);
}

inline wuffs_base__range_ie_u32  //
wuffs_base__range_ie_u32::intersect(wuffs_base__range_ie_u32 s) const {
  return wuffs_base__range_ie_u32__intersect(this, s);
}

inline wuffs_base__range_ie_u32  //
wuffs_base__range_ie_u32::unite(wuffs_base__range_ie_u32 s) const {
  return wuffs_base__range_ie_u32__unite(this, s);
}

inline bool  //
wuffs_base__range_ie_u32::contains(uint32_t x) const {
  return wuffs_base__range_ie_u32__contains(this, x);
}

inline bool  //
wuffs_base__range_ie_u32::contains_range(wuffs_base__range_ie_u32 s) const {
  return wuffs_base__range_ie_u32__contains_range(this, s);
}

inline uint32_t  //
wuffs_base__range_ie_u32::length() const {
  return wuffs_base__range_ie_u32__length(this);
}

#endif  // __cplusplus

// --------

typedef struct wuffs_base__range_ii_u64__struct {
  uint64_t min_incl;
  uint64_t max_incl;

#ifdef __cplusplus
  inline bool is_empty() const;
  inline bool equals(wuffs_base__range_ii_u64__struct s) const;
  inline wuffs_base__range_ii_u64__struct intersect(
      wuffs_base__range_ii_u64__struct s) const;
  inline wuffs_base__range_ii_u64__struct unite(
      wuffs_base__range_ii_u64__struct s) const;
  inline bool contains(uint64_t x) const;
  inline bool contains_range(wuffs_base__range_ii_u64__struct s) const;
#endif  // __cplusplus

} wuffs_base__range_ii_u64;

static inline wuffs_base__range_ii_u64  //
wuffs_base__empty_range_ii_u64() {
  wuffs_base__range_ii_u64 ret;
  ret.min_incl = 0;
  ret.max_incl = 0;
  return ret;
}

static inline wuffs_base__range_ii_u64  //
wuffs_base__make_range_ii_u64(uint64_t min_incl, uint64_t max_incl) {
  wuffs_base__range_ii_u64 ret;
  ret.min_incl = min_incl;
  ret.max_incl = max_incl;
  return ret;
}

static inline bool  //
wuffs_base__range_ii_u64__is_empty(const wuffs_base__range_ii_u64* r) {
  return r->min_incl > r->max_incl;
}

static inline bool  //
wuffs_base__range_ii_u64__equals(const wuffs_base__range_ii_u64* r,
                                 wuffs_base__range_ii_u64 s) {
  return (r->min_incl == s.min_incl && r->max_incl == s.max_incl) ||
         (wuffs_base__range_ii_u64__is_empty(r) &&
          wuffs_base__range_ii_u64__is_empty(&s));
}

static inline wuffs_base__range_ii_u64  //
wuffs_base__range_ii_u64__intersect(const wuffs_base__range_ii_u64* r,
                                    wuffs_base__range_ii_u64 s) {
  wuffs_base__range_ii_u64 t;
  t.min_incl = wuffs_base__u64__max(r->min_incl, s.min_incl);
  t.max_incl = wuffs_base__u64__min(r->max_incl, s.max_incl);
  return t;
}

static inline wuffs_base__range_ii_u64  //
wuffs_base__range_ii_u64__unite(const wuffs_base__range_ii_u64* r,
                                wuffs_base__range_ii_u64 s) {
  if (wuffs_base__range_ii_u64__is_empty(r)) {
    return s;
  }
  if (wuffs_base__range_ii_u64__is_empty(&s)) {
    return *r;
  }
  wuffs_base__range_ii_u64 t;
  t.min_incl = wuffs_base__u64__min(r->min_incl, s.min_incl);
  t.max_incl = wuffs_base__u64__max(r->max_incl, s.max_incl);
  return t;
}

static inline bool  //
wuffs_base__range_ii_u64__contains(const wuffs_base__range_ii_u64* r,
                                   uint64_t x) {
  return (r->min_incl <= x) && (x <= r->max_incl);
}

static inline bool  //
wuffs_base__range_ii_u64__contains_range(const wuffs_base__range_ii_u64* r,
                                         wuffs_base__range_ii_u64 s) {
  return wuffs_base__range_ii_u64__equals(
      &s, wuffs_base__range_ii_u64__intersect(r, s));
}

#ifdef __cplusplus

inline bool  //
wuffs_base__range_ii_u64::is_empty() const {
  return wuffs_base__range_ii_u64__is_empty(this);
}

inline bool  //
wuffs_base__range_ii_u64::equals(wuffs_base__range_ii_u64 s) const {
  return wuffs_base__range_ii_u64__equals(this, s);
}

inline wuffs_base__range_ii_u64  //
wuffs_base__range_ii_u64::intersect(wuffs_base__range_ii_u64 s) const {
  return wuffs_base__range_ii_u64__intersect(this, s);
}

inline wuffs_base__range_ii_u64  //
wuffs_base__range_ii_u64::unite(wuffs_base__range_ii_u64 s) const {
  return wuffs_base__range_ii_u64__unite(this, s);
}

inline bool  //
wuffs_base__range_ii_u64::contains(uint64_t x) const {
  return wuffs_base__range_ii_u64__contains(this, x);
}

inline bool  //
wuffs_base__range_ii_u64::contains_range(wuffs_base__range_ii_u64 s) const {
  return wuffs_base__range_ii_u64__contains_range(this, s);
}

#endif  // __cplusplus

// --------

typedef struct wuffs_base__range_ie_u64__struct {
  uint64_t min_incl;
  uint64_t max_excl;

#ifdef __cplusplus
  inline bool is_empty() const;
  inline bool equals(wuffs_base__range_ie_u64__struct s) const;
  inline wuffs_base__range_ie_u64__struct intersect(
      wuffs_base__range_ie_u64__struct s) const;
  inline wuffs_base__range_ie_u64__struct unite(
      wuffs_base__range_ie_u64__struct s) const;
  inline bool contains(uint64_t x) const;
  inline bool contains_range(wuffs_base__range_ie_u64__struct s) const;
  inline uint64_t length() const;
#endif  // __cplusplus

} wuffs_base__range_ie_u64;

static inline wuffs_base__range_ie_u64  //
wuffs_base__empty_range_ie_u64() {
  wuffs_base__range_ie_u64 ret;
  ret.min_incl = 0;
  ret.max_excl = 0;
  return ret;
}

static inline wuffs_base__range_ie_u64  //
wuffs_base__make_range_ie_u64(uint64_t min_incl, uint64_t max_excl) {
  wuffs_base__range_ie_u64 ret;
  ret.min_incl = min_incl;
  ret.max_excl = max_excl;
  return ret;
}

static inline bool  //
wuffs_base__range_ie_u64__is_empty(const wuffs_base__range_ie_u64* r) {
  return r->min_incl >= r->max_excl;
}

static inline bool  //
wuffs_base__range_ie_u64__equals(const wuffs_base__range_ie_u64* r,
                                 wuffs_base__range_ie_u64 s) {
  return (r->min_incl == s.min_incl && r->max_excl == s.max_excl) ||
         (wuffs_base__range_ie_u64__is_empty(r) &&
          wuffs_base__range_ie_u64__is_empty(&s));
}

static inline wuffs_base__range_ie_u64  //
wuffs_base__range_ie_u64__intersect(const wuffs_base__range_ie_u64* r,
                                    wuffs_base__range_ie_u64 s) {
  wuffs_base__range_ie_u64 t;
  t.min_incl = wuffs_base__u64__max(r->min_incl, s.min_incl);
  t.max_excl = wuffs_base__u64__min(r->max_excl, s.max_excl);
  return t;
}

static inline wuffs_base__range_ie_u64  //
wuffs_base__range_ie_u64__unite(const wuffs_base__range_ie_u64* r,
                                wuffs_base__range_ie_u64 s) {
  if (wuffs_base__range_ie_u64__is_empty(r)) {
    return s;
  }
  if (wuffs_base__range_ie_u64__is_empty(&s)) {
    return *r;
  }
  wuffs_base__range_ie_u64 t;
  t.min_incl = wuffs_base__u64__min(r->min_incl, s.min_incl);
  t.max_excl = wuffs_base__u64__max(r->max_excl, s.max_excl);
  return t;
}

static inline bool  //
wuffs_base__range_ie_u64__contains(const wuffs_base__range_ie_u64* r,
                                   uint64_t x) {
  return (r->min_incl <= x) && (x < r->max_excl);
}

static inline bool  //
wuffs_base__range_ie_u64__contains_range(const wuffs_base__range_ie_u64* r,
                                         wuffs_base__range_ie_u64 s) {
  return wuffs_base__range_ie_u64__equals(
      &s, wuffs_base__range_ie_u64__intersect(r, s));
}

static inline uint64_t  //
wuffs_base__range_ie_u64__length(const wuffs_base__range_ie_u64* r) {
  return wuffs_base__u64__sat_sub(r->max_excl, r->min_incl);
}

#ifdef __cplusplus

inline bool  //
wuffs_base__range_ie_u64::is_empty() const {
  return wuffs_base__range_ie_u64__is_empty(this);
}

inline bool  //
wuffs_base__range_ie_u64::equals(wuffs_base__range_ie_u64 s) const {
  return wuffs_base__range_ie_u64__equals(this, s);
}

inline wuffs_base__range_ie_u64  //
wuffs_base__range_ie_u64::intersect(wuffs_base__range_ie_u64 s) const {
  return wuffs_base__range_ie_u64__intersect(this, s);
}

inline wuffs_base__range_ie_u64  //
wuffs_base__range_ie_u64::unite(wuffs_base__range_ie_u64 s) const {
  return wuffs_base__range_ie_u64__unite(this, s);
}

inline bool  //
wuffs_base__range_ie_u64::contains(uint64_t x) const {
  return wuffs_base__range_ie_u64__contains(this, x);
}

inline bool  //
wuffs_base__range_ie_u64::contains_range(wuffs_base__range_ie_u64 s) const {
  return wuffs_base__range_ie_u64__contains_range(this, s);
}

inline uint64_t  //
wuffs_base__range_ie_u64::length() const {
  return wuffs_base__range_ie_u64__length(this);
}

#endif  // __cplusplus

// --------

typedef struct wuffs_base__rect_ii_u32__struct {
  uint32_t min_incl_x;
  uint32_t min_incl_y;
  uint32_t max_incl_x;
  uint32_t max_incl_y;

#ifdef __cplusplus
  inline bool is_empty() const;
  inline bool equals(wuffs_base__rect_ii_u32__struct s) const;
  inline wuffs_base__rect_ii_u32__struct intersect(
      wuffs_base__rect_ii_u32__struct s) const;
  inline wuffs_base__rect_ii_u32__struct unite(
      wuffs_base__rect_ii_u32__struct s) const;
  inline bool contains(uint32_t x, uint32_t y) const;
  inline bool contains_rect(wuffs_base__rect_ii_u32__struct s) const;
#endif  // __cplusplus

} wuffs_base__rect_ii_u32;

static inline wuffs_base__rect_ii_u32  //
wuffs_base__empty_rect_ii_u32() {
  wuffs_base__rect_ii_u32 ret;
  ret.min_incl_x = 0;
  ret.min_incl_y = 0;
  ret.max_incl_x = 0;
  ret.max_incl_y = 0;
  return ret;
}

static inline wuffs_base__rect_ii_u32  //
wuffs_base__make_rect_ii_u32(uint32_t min_incl_x,
                             uint32_t min_incl_y,
                             uint32_t max_incl_x,
                             uint32_t max_incl_y) {
  wuffs_base__rect_ii_u32 ret;
  ret.min_incl_x = min_incl_x;
  ret.min_incl_y = min_incl_y;
  ret.max_incl_x = max_incl_x;
  ret.max_incl_y = max_incl_y;
  return ret;
}

static inline bool  //
wuffs_base__rect_ii_u32__is_empty(const wuffs_base__rect_ii_u32* r) {
  return (r->min_incl_x > r->max_incl_x) || (r->min_incl_y > r->max_incl_y);
}

static inline bool  //
wuffs_base__rect_ii_u32__equals(const wuffs_base__rect_ii_u32* r,
                                wuffs_base__rect_ii_u32 s) {
  return (r->min_incl_x == s.min_incl_x && r->min_incl_y == s.min_incl_y &&
          r->max_incl_x == s.max_incl_x && r->max_incl_y == s.max_incl_y) ||
         (wuffs_base__rect_ii_u32__is_empty(r) &&
          wuffs_base__rect_ii_u32__is_empty(&s));
}

static inline wuffs_base__rect_ii_u32  //
wuffs_base__rect_ii_u32__intersect(const wuffs_base__rect_ii_u32* r,
                                   wuffs_base__rect_ii_u32 s) {
  wuffs_base__rect_ii_u32 t;
  t.min_incl_x = wuffs_base__u32__max(r->min_incl_x, s.min_incl_x);
  t.min_incl_y = wuffs_base__u32__max(r->min_incl_y, s.min_incl_y);
  t.max_incl_x = wuffs_base__u32__min(r->max_incl_x, s.max_incl_x);
  t.max_incl_y = wuffs_base__u32__min(r->max_incl_y, s.max_incl_y);
  return t;
}

static inline wuffs_base__rect_ii_u32  //
wuffs_base__rect_ii_u32__unite(const wuffs_base__rect_ii_u32* r,
                               wuffs_base__rect_ii_u32 s) {
  if (wuffs_base__rect_ii_u32__is_empty(r)) {
    return s;
  }
  if (wuffs_base__rect_ii_u32__is_empty(&s)) {
    return *r;
  }
  wuffs_base__rect_ii_u32 t;
  t.min_incl_x = wuffs_base__u32__min(r->min_incl_x, s.min_incl_x);
  t.min_incl_y = wuffs_base__u32__min(r->min_incl_y, s.min_incl_y);
  t.max_incl_x = wuffs_base__u32__max(r->max_incl_x, s.max_incl_x);
  t.max_incl_y = wuffs_base__u32__max(r->max_incl_y, s.max_incl_y);
  return t;
}

static inline bool  //
wuffs_base__rect_ii_u32__contains(const wuffs_base__rect_ii_u32* r,
                                  uint32_t x,
                                  uint32_t y) {
  return (r->min_incl_x <= x) && (x <= r->max_incl_x) && (r->min_incl_y <= y) &&
         (y <= r->max_incl_y);
}

static inline bool  //
wuffs_base__rect_ii_u32__contains_rect(const wuffs_base__rect_ii_u32* r,
                                       wuffs_base__rect_ii_u32 s) {
  return wuffs_base__rect_ii_u32__equals(
      &s, wuffs_base__rect_ii_u32__intersect(r, s));
}

#ifdef __cplusplus

inline bool  //
wuffs_base__rect_ii_u32::is_empty() const {
  return wuffs_base__rect_ii_u32__is_empty(this);
}

inline bool  //
wuffs_base__rect_ii_u32::equals(wuffs_base__rect_ii_u32 s) const {
  return wuffs_base__rect_ii_u32__equals(this, s);
}

inline wuffs_base__rect_ii_u32  //
wuffs_base__rect_ii_u32::intersect(wuffs_base__rect_ii_u32 s) const {
  return wuffs_base__rect_ii_u32__intersect(this, s);
}

inline wuffs_base__rect_ii_u32  //
wuffs_base__rect_ii_u32::unite(wuffs_base__rect_ii_u32 s) const {
  return wuffs_base__rect_ii_u32__unite(this, s);
}

inline bool  //
wuffs_base__rect_ii_u32::contains(uint32_t x, uint32_t y) const {
  return wuffs_base__rect_ii_u32__contains(this, x, y);
}

inline bool  //
wuffs_base__rect_ii_u32::contains_rect(wuffs_base__rect_ii_u32 s) const {
  return wuffs_base__rect_ii_u32__contains_rect(this, s);
}

#endif  // __cplusplus

// --------

typedef struct wuffs_base__rect_ie_u32__struct {
  uint32_t min_incl_x;
  uint32_t min_incl_y;
  uint32_t max_excl_x;
  uint32_t max_excl_y;

#ifdef __cplusplus
  inline bool is_empty() const;
  inline bool equals(wuffs_base__rect_ie_u32__struct s) const;
  inline wuffs_base__rect_ie_u32__struct intersect(
      wuffs_base__rect_ie_u32__struct s) const;
  inline wuffs_base__rect_ie_u32__struct unite(
      wuffs_base__rect_ie_u32__struct s) const;
  inline bool contains(uint32_t x, uint32_t y) const;
  inline bool contains_rect(wuffs_base__rect_ie_u32__struct s) const;
  inline uint32_t width() const;
  inline uint32_t height() const;
#endif  // __cplusplus

} wuffs_base__rect_ie_u32;

static inline wuffs_base__rect_ie_u32  //
wuffs_base__empty_rect_ie_u32() {
  wuffs_base__rect_ie_u32 ret;
  ret.min_incl_x = 0;
  ret.min_incl_y = 0;
  ret.max_excl_x = 0;
  ret.max_excl_y = 0;
  return ret;
}

static inline wuffs_base__rect_ie_u32  //
wuffs_base__make_rect_ie_u32(uint32_t min_incl_x,
                             uint32_t min_incl_y,
                             uint32_t max_excl_x,
                             uint32_t max_excl_y) {
  wuffs_base__rect_ie_u32 ret;
  ret.min_incl_x = min_incl_x;
  ret.min_incl_y = min_incl_y;
  ret.max_excl_x = max_excl_x;
  ret.max_excl_y = max_excl_y;
  return ret;
}

static inline bool  //
wuffs_base__rect_ie_u32__is_empty(const wuffs_base__rect_ie_u32* r) {
  return (r->min_incl_x >= r->max_excl_x) || (r->min_incl_y >= r->max_excl_y);
}

static inline bool  //
wuffs_base__rect_ie_u32__equals(const wuffs_base__rect_ie_u32* r,
                                wuffs_base__rect_ie_u32 s) {
  return (r->min_incl_x == s.min_incl_x && r->min_incl_y == s.min_incl_y &&
          r->max_excl_x == s.max_excl_x && r->max_excl_y == s.max_excl_y) ||
         (wuffs_base__rect_ie_u32__is_empty(r) &&
          wuffs_base__rect_ie_u32__is_empty(&s));
}

static inline wuffs_base__rect_ie_u32  //
wuffs_base__rect_ie_u32__intersect(const wuffs_base__rect_ie_u32* r,
                                   wuffs_base__rect_ie_u32 s) {
  wuffs_base__rect_ie_u32 t;
  t.min_incl_x = wuffs_base__u32__max(r->min_incl_x, s.min_incl_x);
  t.min_incl_y = wuffs_base__u32__max(r->min_incl_y, s.min_incl_y);
  t.max_excl_x = wuffs_base__u32__min(r->max_excl_x, s.max_excl_x);
  t.max_excl_y = wuffs_base__u32__min(r->max_excl_y, s.max_excl_y);
  return t;
}

static inline wuffs_base__rect_ie_u32  //
wuffs_base__rect_ie_u32__unite(const wuffs_base__rect_ie_u32* r,
                               wuffs_base__rect_ie_u32 s) {
  if (wuffs_base__rect_ie_u32__is_empty(r)) {
    return s;
  }
  if (wuffs_base__rect_ie_u32__is_empty(&s)) {
    return *r;
  }
  wuffs_base__rect_ie_u32 t;
  t.min_incl_x = wuffs_base__u32__min(r->min_incl_x, s.min_incl_x);
  t.min_incl_y = wuffs_base__u32__min(r->min_incl_y, s.min_incl_y);
  t.max_excl_x = wuffs_base__u32__max(r->max_excl_x, s.max_excl_x);
  t.max_excl_y = wuffs_base__u32__max(r->max_excl_y, s.max_excl_y);
  return t;
}

static inline bool  //
wuffs_base__rect_ie_u32__contains(const wuffs_base__rect_ie_u32* r,
                                  uint32_t x,
                                  uint32_t y) {
  return (r->min_incl_x <= x) && (x < r->max_excl_x) && (r->min_incl_y <= y) &&
         (y < r->max_excl_y);
}

static inline bool  //
wuffs_base__rect_ie_u32__contains_rect(const wuffs_base__rect_ie_u32* r,
                                       wuffs_base__rect_ie_u32 s) {
  return wuffs_base__rect_ie_u32__equals(
      &s, wuffs_base__rect_ie_u32__intersect(r, s));
}

static inline uint32_t  //
wuffs_base__rect_ie_u32__width(const wuffs_base__rect_ie_u32* r) {
  return wuffs_base__u32__sat_sub(r->max_excl_x, r->min_incl_x);
}

static inline uint32_t  //
wuffs_base__rect_ie_u32__height(const wuffs_base__rect_ie_u32* r) {
  return wuffs_base__u32__sat_sub(r->max_excl_y, r->min_incl_y);
}

#ifdef __cplusplus

inline bool  //
wuffs_base__rect_ie_u32::is_empty() const {
  return wuffs_base__rect_ie_u32__is_empty(this);
}

inline bool  //
wuffs_base__rect_ie_u32::equals(wuffs_base__rect_ie_u32 s) const {
  return wuffs_base__rect_ie_u32__equals(this, s);
}

inline wuffs_base__rect_ie_u32  //
wuffs_base__rect_ie_u32::intersect(wuffs_base__rect_ie_u32 s) const {
  return wuffs_base__rect_ie_u32__intersect(this, s);
}

inline wuffs_base__rect_ie_u32  //
wuffs_base__rect_ie_u32::unite(wuffs_base__rect_ie_u32 s) const {
  return wuffs_base__rect_ie_u32__unite(this, s);
}

inline bool  //
wuffs_base__rect_ie_u32::contains(uint32_t x, uint32_t y) const {
  return wuffs_base__rect_ie_u32__contains(this, x, y);
}

inline bool  //
wuffs_base__rect_ie_u32::contains_rect(wuffs_base__rect_ie_u32 s) const {
  return wuffs_base__rect_ie_u32__contains_rect(this, s);
}

inline uint32_t  //
wuffs_base__rect_ie_u32::width() const {
  return wuffs_base__rect_ie_u32__width(this);
}

inline uint32_t  //
wuffs_base__rect_ie_u32::height() const {
  return wuffs_base__rect_ie_u32__height(this);
}

#endif  // __cplusplus

// ---------------- More Information

// wuffs_base__more_information holds additional fields, typically when a Wuffs
// method returns a [note status](/doc/note/statuses.md).
//
// The flavor field follows the base38 namespace
// convention](/doc/note/base38-and-fourcc.md). The other fields' semantics
// depends on the flavor.
typedef struct wuffs_base__more_information__struct {
  uint32_t flavor;
  uint32_t w;
  uint64_t x;
  uint64_t y;
  uint64_t z;

#ifdef __cplusplus
  inline void set(uint32_t flavor_arg,
                  uint32_t w_arg,
                  uint64_t x_arg,
                  uint64_t y_arg,
                  uint64_t z_arg);
  inline uint32_t io_redirect__fourcc() const;
  inline wuffs_base__range_ie_u64 io_redirect__range() const;
  inline uint64_t io_seek__position() const;
  inline uint32_t metadata__fourcc() const;
  // Deprecated: use metadata_raw_passthrough__range.
  inline wuffs_base__range_ie_u64 metadata__range() const;
  inline wuffs_base__range_ie_u64 metadata_raw_passthrough__range() const;
  inline int32_t metadata_parsed__chrm(uint32_t component) const;
  inline uint32_t metadata_parsed__gama() const;
  inline uint32_t metadata_parsed__srgb() const;
#endif  // __cplusplus

} wuffs_base__more_information;

#define WUFFS_BASE__MORE_INFORMATION__FLAVOR__IO_REDIRECT 1
#define WUFFS_BASE__MORE_INFORMATION__FLAVOR__IO_SEEK 2
// Deprecated: use
// WUFFS_BASE__MORE_INFORMATION__FLAVOR__METADATA_RAW_PASSTHROUGH.
#define WUFFS_BASE__MORE_INFORMATION__FLAVOR__METADATA 3
#define WUFFS_BASE__MORE_INFORMATION__FLAVOR__METADATA_RAW_PASSTHROUGH 3
#define WUFFS_BASE__MORE_INFORMATION__FLAVOR__METADATA_RAW_TRANSFORM 4
#define WUFFS_BASE__MORE_INFORMATION__FLAVOR__METADATA_PARSED 5

static inline wuffs_base__more_information  //
wuffs_base__empty_more_information() {
  wuffs_base__more_information ret;
  ret.flavor = 0;
  ret.w = 0;
  ret.x = 0;
  ret.y = 0;
  ret.z = 0;
  return ret;
}

static inline void  //
wuffs_base__more_information__set(wuffs_base__more_information* m,
                                  uint32_t flavor,
                                  uint32_t w,
                                  uint64_t x,
                                  uint64_t y,
                                  uint64_t z) {
  if (!m) {
    return;
  }
  m->flavor = flavor;
  m->w = w;
  m->x = x;
  m->y = y;
  m->z = z;
}

static inline uint32_t  //
wuffs_base__more_information__io_redirect__fourcc(
    const wuffs_base__more_information* m) {
  return m->w;
}

static inline wuffs_base__range_ie_u64  //
wuffs_base__more_information__io_redirect__range(
    const wuffs_base__more_information* m) {
  wuffs_base__range_ie_u64 ret;
  ret.min_incl = m->y;
  ret.max_excl = m->z;
  return ret;
}

static inline uint64_t  //
wuffs_base__more_information__io_seek__position(
    const wuffs_base__more_information* m) {
  return m->x;
}

static inline uint32_t  //
wuffs_base__more_information__metadata__fourcc(
    const wuffs_base__more_information* m) {
  return m->w;
}

// Deprecated: use
// wuffs_base__more_information__metadata_raw_passthrough__range.
static inline wuffs_base__range_ie_u64  //
wuffs_base__more_information__metadata__range(
    const wuffs_base__more_information* m) {
  wuffs_base__range_ie_u64 ret;
  ret.min_incl = m->y;
  ret.max_excl = m->z;
  return ret;
}

static inline wuffs_base__range_ie_u64  //
wuffs_base__more_information__metadata_raw_passthrough__range(
    const wuffs_base__more_information* m) {
  wuffs_base__range_ie_u64 ret;
  ret.min_incl = m->y;
  ret.max_excl = m->z;
  return ret;
}

#define WUFFS_BASE__MORE_INFORMATION__METADATA_PARSED__CHRM__WHITE_X 0
#define WUFFS_BASE__MORE_INFORMATION__METADATA_PARSED__CHRM__WHITE_Y 1
#define WUFFS_BASE__MORE_INFORMATION__METADATA_PARSED__CHRM__RED_X 2
#define WUFFS_BASE__MORE_INFORMATION__METADATA_PARSED__CHRM__RED_Y 3
#define WUFFS_BASE__MORE_INFORMATION__METADATA_PARSED__CHRM__GREEN_X 4
#define WUFFS_BASE__MORE_INFORMATION__METADATA_PARSED__CHRM__GREEN_Y 5
#define WUFFS_BASE__MORE_INFORMATION__METADATA_PARSED__CHRM__BLUE_X 6
#define WUFFS_BASE__MORE_INFORMATION__METADATA_PARSED__CHRM__BLUE_Y 7

// wuffs_base__more_information__metadata_parsed__chrm returns chromaticity
// values (scaled by 100000) like the PNG "cHRM" chunk. For example, the sRGB
// color space corresponds to:
//  - ETC__CHRM__WHITE_X 31270
//  - ETC__CHRM__WHITE_Y 32900
//  - ETC__CHRM__RED_X   64000
//  - ETC__CHRM__RED_Y   33000
//  - ETC__CHRM__GREEN_X 30000
//  - ETC__CHRM__GREEN_Y 60000
//  - ETC__CHRM__BLUE_X  15000
//  - ETC__CHRM__BLUE_Y   6000
//
// See
// https://ciechanow.ski/color-spaces/#chromaticity-and-white-point-coordinates
static inline int32_t  //
wuffs_base__more_information__metadata_parsed__chrm(
    const wuffs_base__more_information* m,
    uint32_t component) {
  // After the flavor and the w field (holding a FourCC), a
  // wuffs_base__more_information holds 24 bytes of data in three uint64_t
  // typed fields (x, y and z). We pack the eight chromaticity values (wx, wy,
  // rx, ..., by), basically int24_t values, into 24 bytes like this:
  //  -    LSB                 MSB
  //  - x: wx wx wx wy wy wy rx rx
  //  - y: rx ry ry ry gx gx gx gy
  //  - z: gy gy bx bx bx by by by
  uint32_t u = 0;
  switch (component & 7) {
    case 0:
      u = ((uint32_t)(m->x >> 0));
      break;
    case 1:
      u = ((uint32_t)(m->x >> 24));
      break;
    case 2:
      u = ((uint32_t)((m->x >> 48) | (m->y << 16)));
      break;
    case 3:
      u = ((uint32_t)(m->y >> 8));
      break;
    case 4:
      u = ((uint32_t)(m->y >> 32));
      break;
    case 5:
      u = ((uint32_t)((m->y >> 56) | (m->z << 8)));
      break;
    case 6:
      u = ((uint32_t)(m->z >> 16));
      break;
    case 7:
      u = ((uint32_t)(m->z >> 40));
      break;
  }
  // The left-right shifts sign-extend from 24-bit to 32-bit integers.
  return ((int32_t)(u << 8)) >> 8;
}

// wuffs_base__more_information__metadata_parsed__gama returns inverse gamma
// correction values (scaled by 100000) like the PNG "gAMA" chunk. For example,
// for gamma = 2.2, this returns 45455 (approximating 100000 / 2.2).
static inline uint32_t  //
wuffs_base__more_information__metadata_parsed__gama(
    const wuffs_base__more_information* m) {
  return ((uint32_t)(m->x));
}

#define WUFFS_BASE__SRGB_RENDERING_INTENT__PERCEPTUAL 0
#define WUFFS_BASE__SRGB_RENDERING_INTENT__RELATIVE_COLORIMETRIC 1
#define WUFFS_BASE__SRGB_RENDERING_INTENT__SATURATION 2
#define WUFFS_BASE__SRGB_RENDERING_INTENT__ABSOLUTE_COLORIMETRIC 3

// wuffs_base__more_information__metadata_parsed__srgb returns the sRGB
// rendering intent like the PNG "sRGB" chunk.
static inline uint32_t  //
wuffs_base__more_information__metadata_parsed__srgb(
    const wuffs_base__more_information* m) {
  return m->x & 3;
}

#ifdef __cplusplus

inline void  //
wuffs_base__more_information::set(uint32_t flavor_arg,
                                  uint32_t w_arg,
                                  uint64_t x_arg,
                                  uint64_t y_arg,
                                  uint64_t z_arg) {
  wuffs_base__more_information__set(this, flavor_arg, w_arg, x_arg, y_arg,
                                    z_arg);
}

inline uint32_t  //
wuffs_base__more_information::io_redirect__fourcc() const {
  return wuffs_base__more_information__io_redirect__fourcc(this);
}

inline wuffs_base__range_ie_u64  //
wuffs_base__more_information::io_redirect__range() const {
  return wuffs_base__more_information__io_redirect__range(this);
}

inline uint64_t  //
wuffs_base__more_information::io_seek__position() const {
  return wuffs_base__more_information__io_seek__position(this);
}

inline uint32_t  //
wuffs_base__more_information::metadata__fourcc() const {
  return wuffs_base__more_information__metadata__fourcc(this);
}

inline wuffs_base__range_ie_u64  //
wuffs_base__more_information::metadata__range() const {
  return wuffs_base__more_information__metadata__range(this);
}

inline wuffs_base__range_ie_u64  //
wuffs_base__more_information::metadata_raw_passthrough__range() const {
  return wuffs_base__more_information__metadata_raw_passthrough__range(this);
}

inline int32_t  //
wuffs_base__more_information::metadata_parsed__chrm(uint32_t component) const {
  return wuffs_base__more_information__metadata_parsed__chrm(this, component);
}

inline uint32_t  //
wuffs_base__more_information::metadata_parsed__gama() const {
  return wuffs_base__more_information__metadata_parsed__gama(this);
}

inline uint32_t  //
wuffs_base__more_information::metadata_parsed__srgb() const {
  return wuffs_base__more_information__metadata_parsed__srgb(this);
}

#endif  // __cplusplus

// ---------------- I/O
//
// See (/doc/note/io-input-output.md).

// wuffs_base__io_buffer_meta is the metadata for a wuffs_base__io_buffer's
// data.
typedef struct wuffs_base__io_buffer_meta__struct {
  size_t wi;     // Write index. Invariant: wi <= len.
  size_t ri;     // Read  index. Invariant: ri <= wi.
  uint64_t pos;  // Buffer position (relative to the start of stream).
  bool closed;   // No further writes are expected.
} wuffs_base__io_buffer_meta;

// wuffs_base__io_buffer is a 1-dimensional buffer (a pointer and length) plus
// additional metadata.
//
// A value with all fields zero is a valid, empty buffer.
typedef struct wuffs_base__io_buffer__struct {
  wuffs_base__slice_u8 data;
  wuffs_base__io_buffer_meta meta;

#ifdef __cplusplus
  inline bool is_valid() const;
  inline void compact();
  inline size_t reader_length() const;
  inline uint8_t* reader_pointer() const;
  inline uint64_t reader_position() const;
  inline wuffs_base__slice_u8 reader_slice() const;
  inline size_t writer_length() const;
  inline uint8_t* writer_pointer() const;
  inline uint64_t writer_position() const;
  inline wuffs_base__slice_u8 writer_slice() const;

  // Deprecated: use reader_position.
  inline uint64_t reader_io_position() const;
  // Deprecated: use writer_position.
  inline uint64_t writer_io_position() const;
#endif  // __cplusplus

} wuffs_base__io_buffer;

static inline wuffs_base__io_buffer  //
wuffs_base__make_io_buffer(wuffs_base__slice_u8 data,
                           wuffs_base__io_buffer_meta meta) {
  wuffs_base__io_buffer ret;
  ret.data = data;
  ret.meta = meta;
  return ret;
}

static inline wuffs_base__io_buffer_meta  //
wuffs_base__make_io_buffer_meta(size_t wi,
                                size_t ri,
                                uint64_t pos,
                                bool closed) {
  wuffs_base__io_buffer_meta ret;
  ret.wi = wi;
  ret.ri = ri;
  ret.pos = pos;
  ret.closed = closed;
  return ret;
}

static inline wuffs_base__io_buffer  //
wuffs_base__ptr_u8__reader(uint8_t* ptr, size_t len, bool closed) {
  wuffs_base__io_buffer ret;
  ret.data.ptr = ptr;
  ret.data.len = len;
  ret.meta.wi = len;
  ret.meta.ri = 0;
  ret.meta.pos = 0;
  ret.meta.closed = closed;
  return ret;
}

static inline wuffs_base__io_buffer  //
wuffs_base__ptr_u8__writer(uint8_t* ptr, size_t len) {
  wuffs_base__io_buffer ret;
  ret.data.ptr = ptr;
  ret.data.len = len;
  ret.meta.wi = 0;
  ret.meta.ri = 0;
  ret.meta.pos = 0;
  ret.meta.closed = false;
  return ret;
}

static inline wuffs_base__io_buffer  //
wuffs_base__slice_u8__reader(wuffs_base__slice_u8 s, bool closed) {
  wuffs_base__io_buffer ret;
  ret.data.ptr = s.ptr;
  ret.data.len = s.len;
  ret.meta.wi = s.len;
  ret.meta.ri = 0;
  ret.meta.pos = 0;
  ret.meta.closed = closed;
  return ret;
}

static inline wuffs_base__io_buffer  //
wuffs_base__slice_u8__writer(wuffs_base__slice_u8 s) {
  wuffs_base__io_buffer ret;
  ret.data.ptr = s.ptr;
  ret.data.len = s.len;
  ret.meta.wi = 0;
  ret.meta.ri = 0;
  ret.meta.pos = 0;
  ret.meta.closed = false;
  return ret;
}

static inline wuffs_base__io_buffer  //
wuffs_base__empty_io_buffer() {
  wuffs_base__io_buffer ret;
  ret.data.ptr = NULL;
  ret.data.len = 0;
  ret.meta.wi = 0;
  ret.meta.ri = 0;
  ret.meta.pos = 0;
  ret.meta.closed = false;
  return ret;
}

static inline wuffs_base__io_buffer_meta  //
wuffs_base__empty_io_buffer_meta() {
  wuffs_base__io_buffer_meta ret;
  ret.wi = 0;
  ret.ri = 0;
  ret.pos = 0;
  ret.closed = false;
  return ret;
}

static inline bool  //
wuffs_base__io_buffer__is_valid(const wuffs_base__io_buffer* buf) {
  if (buf) {
    if (buf->data.ptr) {
      return (buf->meta.ri <= buf->meta.wi) && (buf->meta.wi <= buf->data.len);
    } else {
      return (buf->meta.ri == 0) && (buf->meta.wi == 0) && (buf->data.len == 0);
    }
  }
  return false;
}

// wuffs_base__io_buffer__compact moves any written but unread bytes to the
// start of the buffer.
static inline void  //
wuffs_base__io_buffer__compact(wuffs_base__io_buffer* buf) {
  if (!buf || (buf->meta.ri == 0)) {
    return;
  }
  buf->meta.pos = wuffs_base__u64__sat_add(buf->meta.pos, buf->meta.ri);
  size_t n = buf->meta.wi - buf->meta.ri;
  if (n != 0) {
    memmove(buf->data.ptr, buf->data.ptr + buf->meta.ri, n);
  }
  buf->meta.wi = n;
  buf->meta.ri = 0;
}

// Deprecated. Use wuffs_base__io_buffer__reader_position.
static inline uint64_t  //
wuffs_base__io_buffer__reader_io_position(const wuffs_base__io_buffer* buf) {
  return buf ? wuffs_base__u64__sat_add(buf->meta.pos, buf->meta.ri) : 0;
}

static inline size_t  //
wuffs_base__io_buffer__reader_length(const wuffs_base__io_buffer* buf) {
  return buf ? buf->meta.wi - buf->meta.ri : 0;
}

static inline uint8_t*  //
wuffs_base__io_buffer__reader_pointer(const wuffs_base__io_buffer* buf) {
  return buf ? (buf->data.ptr + buf->meta.ri) : NULL;
}

static inline uint64_t  //
wuffs_base__io_buffer__reader_position(const wuffs_base__io_buffer* buf) {
  return buf ? wuffs_base__u64__sat_add(buf->meta.pos, buf->meta.ri) : 0;
}

static inline wuffs_base__slice_u8  //
wuffs_base__io_buffer__reader_slice(const wuffs_base__io_buffer* buf) {
  return buf ? wuffs_base__make_slice_u8(buf->data.ptr + buf->meta.ri,
                                         buf->meta.wi - buf->meta.ri)
             : wuffs_base__empty_slice_u8();
}

// Deprecated. Use wuffs_base__io_buffer__writer_position.
static inline uint64_t  //
wuffs_base__io_buffer__writer_io_position(const wuffs_base__io_buffer* buf) {
  return buf ? wuffs_base__u64__sat_add(buf->meta.pos, buf->meta.wi) : 0;
}

static inline size_t  //
wuffs_base__io_buffer__writer_length(const wuffs_base__io_buffer* buf) {
  return buf ? buf->data.len - buf->meta.wi : 0;
}

static inline uint8_t*  //
wuffs_base__io_buffer__writer_pointer(const wuffs_base__io_buffer* buf) {
  return buf ? (buf->data.ptr + buf->meta.wi) : NULL;
}

static inline uint64_t  //
wuffs_base__io_buffer__writer_position(const wuffs_base__io_buffer* buf) {
  return buf ? wuffs_base__u64__sat_add(buf->meta.pos, buf->meta.wi) : 0;
}

static inline wuffs_base__slice_u8  //
wuffs_base__io_buffer__writer_slice(const wuffs_base__io_buffer* buf) {
  return buf ? wuffs_base__make_slice_u8(buf->data.ptr + buf->meta.wi,
                                         buf->data.len - buf->meta.wi)
             : wuffs_base__empty_slice_u8();
}

#ifdef __cplusplus

inline bool  //
wuffs_base__io_buffer::is_valid() const {
  return wuffs_base__io_buffer__is_valid(this);
}

inline void  //
wuffs_base__io_buffer::compact() {
  wuffs_base__io_buffer__compact(this);
}

inline uint64_t  //
wuffs_base__io_buffer::reader_io_position() const {
  return wuffs_base__io_buffer__reader_io_position(this);
}

inline size_t  //
wuffs_base__io_buffer::reader_length() const {
  return wuffs_base__io_buffer__reader_length(this);
}

inline uint8_t*  //
wuffs_base__io_buffer::reader_pointer() const {
  return wuffs_base__io_buffer__reader_pointer(this);
}

inline uint64_t  //
wuffs_base__io_buffer::reader_position() const {
  return wuffs_base__io_buffer__reader_position(this);
}

inline wuffs_base__slice_u8  //
wuffs_base__io_buffer::reader_slice() const {
  return wuffs_base__io_buffer__reader_slice(this);
}

inline uint64_t  //
wuffs_base__io_buffer::writer_io_position() const {
  return wuffs_base__io_buffer__writer_io_position(this);
}

inline size_t  //
wuffs_base__io_buffer::writer_length() const {
  return wuffs_base__io_buffer__writer_length(this);
}

inline uint8_t*  //
wuffs_base__io_buffer::writer_pointer() const {
  return wuffs_base__io_buffer__writer_pointer(this);
}

inline uint64_t  //
wuffs_base__io_buffer::writer_position() const {
  return wuffs_base__io_buffer__writer_position(this);
}

inline wuffs_base__slice_u8  //
wuffs_base__io_buffer::writer_slice() const {
  return wuffs_base__io_buffer__writer_slice(this);
}

#endif  // __cplusplus

// ---------------- Tokens

// wuffs_base__token is an element of a byte stream's tokenization.
//
// See https://github.com/google/wuffs/blob/main/doc/note/tokens.md
typedef struct wuffs_base__token__struct {
  uint64_t repr;

#ifdef __cplusplus
  inline int64_t value() const;
  inline int64_t value_extension() const;
  inline int64_t value_major() const;
  inline int64_t value_base_category() const;
  inline uint64_t value_minor() const;
  inline uint64_t value_base_detail() const;
  inline int64_t value_base_detail__sign_extended() const;
  inline bool continued() const;
  inline uint64_t length() const;
#endif  // __cplusplus

} wuffs_base__token;

static inline wuffs_base__token  //
wuffs_base__make_token(uint64_t repr) {
  wuffs_base__token ret;
  ret.repr = repr;
  return ret;
}

// --------

#define WUFFS_BASE__TOKEN__LENGTH__MAX_INCL 0xFFFF

#define WUFFS_BASE__TOKEN__VALUE__SHIFT 17
#define WUFFS_BASE__TOKEN__VALUE_EXTENSION__SHIFT 17
#define WUFFS_BASE__TOKEN__VALUE_MAJOR__SHIFT 42
#define WUFFS_BASE__TOKEN__VALUE_MINOR__SHIFT 17
#define WUFFS_BASE__TOKEN__VALUE_BASE_CATEGORY__SHIFT 38
#define WUFFS_BASE__TOKEN__VALUE_BASE_DETAIL__SHIFT 17
#define WUFFS_BASE__TOKEN__CONTINUED__SHIFT 16
#define WUFFS_BASE__TOKEN__LENGTH__SHIFT 0

#define WUFFS_BASE__TOKEN__VALUE_EXTENSION__NUM_BITS 46

// --------

#define WUFFS_BASE__TOKEN__VBC__FILLER 0
#define WUFFS_BASE__TOKEN__VBC__STRUCTURE 1
#define WUFFS_BASE__TOKEN__VBC__STRING 2
#define WUFFS_BASE__TOKEN__VBC__UNICODE_CODE_POINT 3
#define WUFFS_BASE__TOKEN__VBC__LITERAL 4
#define WUFFS_BASE__TOKEN__VBC__NUMBER 5
#define WUFFS_BASE__TOKEN__VBC__INLINE_INTEGER_SIGNED 6
#define WUFFS_BASE__TOKEN__VBC__INLINE_INTEGER_UNSIGNED 7

// --------

#define WUFFS_BASE__TOKEN__VBD__FILLER__PUNCTUATION 0x00001
#define WUFFS_BASE__TOKEN__VBD__FILLER__COMMENT_BLOCK 0x00002
#define WUFFS_BASE__TOKEN__VBD__FILLER__COMMENT_LINE 0x00004

// COMMENT_ANY is a bit-wise or of COMMENT_BLOCK AND COMMENT_LINE.
#define WUFFS_BASE__TOKEN__VBD__FILLER__COMMENT_ANY 0x00006

// --------

#define WUFFS_BASE__TOKEN__VBD__STRUCTURE__PUSH 0x00001
#define WUFFS_BASE__TOKEN__VBD__STRUCTURE__POP 0x00002
#define WUFFS_BASE__TOKEN__VBD__STRUCTURE__FROM_NONE 0x00010
#define WUFFS_BASE__TOKEN__VBD__STRUCTURE__FROM_LIST 0x00020
#define WUFFS_BASE__TOKEN__VBD__STRUCTURE__FROM_DICT 0x00040
#define WUFFS_BASE__TOKEN__VBD__STRUCTURE__TO_NONE 0x01000
#define WUFFS_BASE__TOKEN__VBD__STRUCTURE__TO_LIST 0x02000
#define WUFFS_BASE__TOKEN__VBD__STRUCTURE__TO_DICT 0x04000

// --------

// DEFINITELY_FOO means that the destination bytes (and also the source bytes,
// for 1_DST_1_SRC_COPY) are in the FOO format. Definitely means that the lack
// of the bit means "maybe FOO". It does not necessarily mean "not FOO".
//
// CHAIN_ETC means that decoding the entire token chain forms a UTF-8 or ASCII
// string, not just this current token. CHAIN_ETC_UTF_8 therefore distinguishes
// Unicode (UTF-8) strings from byte strings. MUST means that the the token
// producer (e.g. parser) must verify this. SHOULD means that the token
// consumer (e.g. renderer) should verify this.
//
// When a CHAIN_ETC_UTF_8 bit is set, the parser must ensure that non-ASCII
// code points (with multi-byte UTF-8 encodings) do not straddle token
// boundaries. Checking UTF-8 validity can inspect each token separately.
//
// The lack of any particular bit is conservative: it is valid for all-ASCII
// strings, in a single- or multi-token chain, to have none of these bits set.
#define WUFFS_BASE__TOKEN__VBD__STRING__DEFINITELY_UTF_8 0x00001
#define WUFFS_BASE__TOKEN__VBD__STRING__CHAIN_MUST_BE_UTF_8 0x00002
#define WUFFS_BASE__TOKEN__VBD__STRING__CHAIN_SHOULD_BE_UTF_8 0x00004
#define WUFFS_BASE__TOKEN__VBD__STRING__DEFINITELY_ASCII 0x00010
#define WUFFS_BASE__TOKEN__VBD__STRING__CHAIN_MUST_BE_ASCII 0x00020
#define WUFFS_BASE__TOKEN__VBD__STRING__CHAIN_SHOULD_BE_ASCII 0x00040

// CONVERT_D_DST_S_SRC means that multiples of S source bytes (possibly padded)
// produces multiples of D destination bytes. For example,
// CONVERT_1_DST_4_SRC_BACKSLASH_X means a source like "\\x23\\x67\\xAB", where
// 12 src bytes encode 3 dst bytes.
//
// Post-processing may further transform those D destination bytes (e.g. treat
// "\\xFF" as the Unicode code point U+00FF instead of the byte 0xFF), but that
// is out of scope of this VBD's semantics.
//
// When src is the empty string, multiple conversion algorithms are applicable
// (so these bits are not necessarily mutually exclusive), all producing the
// same empty dst string.
#define WUFFS_BASE__TOKEN__VBD__STRING__CONVERT_0_DST_1_SRC_DROP 0x00100
#define WUFFS_BASE__TOKEN__VBD__STRING__CONVERT_1_DST_1_SRC_COPY 0x00200
#define WUFFS_BASE__TOKEN__VBD__STRING__CONVERT_1_DST_2_SRC_HEXADECIMAL 0x00400
#define WUFFS_BASE__TOKEN__VBD__STRING__CONVERT_1_DST_4_SRC_BACKSLASH_X 0x00800
#define WUFFS_BASE__TOKEN__VBD__STRING__CONVERT_3_DST_4_SRC_BASE_64_STD 0x01000
#define WUFFS_BASE__TOKEN__VBD__STRING__CONVERT_3_DST_4_SRC_BASE_64_URL 0x02000
#define WUFFS_BASE__TOKEN__VBD__STRING__CONVERT_4_DST_5_SRC_ASCII_85 0x04000
#define WUFFS_BASE__TOKEN__VBD__STRING__CONVERT_5_DST_8_SRC_BASE_32_HEX 0x08000
#define WUFFS_BASE__TOKEN__VBD__STRING__CONVERT_5_DST_8_SRC_BASE_32_STD 0x10000

// --------

#define WUFFS_BASE__TOKEN__VBD__LITERAL__UNDEFINED 0x00001
#define WUFFS_BASE__TOKEN__VBD__LITERAL__NULL 0x00002
#define WUFFS_BASE__TOKEN__VBD__LITERAL__FALSE 0x00004
#define WUFFS_BASE__TOKEN__VBD__LITERAL__TRUE 0x00008

// --------

// For a source string of "123" or "0x9A", it is valid for a tokenizer to
// return any combination of:
//  - WUFFS_BASE__TOKEN__VBD__NUMBER__CONTENT_FLOATING_POINT.
//  - WUFFS_BASE__TOKEN__VBD__NUMBER__CONTENT_INTEGER_SIGNED.
//  - WUFFS_BASE__TOKEN__VBD__NUMBER__CONTENT_INTEGER_UNSIGNED.
//
// For a source string of "+123" or "-0x9A", only the first two are valid.
//
// For a source string of "123.", only the first one is valid.
#define WUFFS_BASE__TOKEN__VBD__NUMBER__CONTENT_FLOATING_POINT 0x00001
#define WUFFS_BASE__TOKEN__VBD__NUMBER__CONTENT_INTEGER_SIGNED 0x00002
#define WUFFS_BASE__TOKEN__VBD__NUMBER__CONTENT_INTEGER_UNSIGNED 0x00004

#define WUFFS_BASE__TOKEN__VBD__NUMBER__CONTENT_NEG_INF 0x00010
#define WUFFS_BASE__TOKEN__VBD__NUMBER__CONTENT_POS_INF 0x00020
#define WUFFS_BASE__TOKEN__VBD__NUMBER__CONTENT_NEG_NAN 0x00040
#define WUFFS_BASE__TOKEN__VBD__NUMBER__CONTENT_POS_NAN 0x00080

// The number 300 might be represented as "\x01\x2C", "\x2C\x01\x00\x00" or
// "300", which are big-endian, little-endian or text. For binary formats, the
// token length (after adjusting for FORMAT_IGNORE_ETC) discriminates
// e.g. u16 little-endian vs u32 little-endian.
#define WUFFS_BASE__TOKEN__VBD__NUMBER__FORMAT_BINARY_BIG_ENDIAN 0x00100
#define WUFFS_BASE__TOKEN__VBD__NUMBER__FORMAT_BINARY_LITTLE_ENDIAN 0x00200
#define WUFFS_BASE__TOKEN__VBD__NUMBER__FORMAT_TEXT 0x00400

#define WUFFS_BASE__TOKEN__VBD__NUMBER__FORMAT_IGNORE_FIRST_BYTE 0x01000

// --------

// wuffs_base__token__value returns the token's high 46 bits, sign-extended. A
// negative value means an extended token, non-negative means a simple token.
static inline int64_t  //
wuffs_base__token__value(const wuffs_base__token* t) {
  return ((int64_t)(t->repr)) >> WUFFS_BASE__TOKEN__VALUE__SHIFT;
}

// wuffs_base__token__value_extension returns a negative value if the token was
// not an extended token.
static inline int64_t  //
wuffs_base__token__value_extension(const wuffs_base__token* t) {
  return (~(int64_t)(t->repr)) >> WUFFS_BASE__TOKEN__VALUE_EXTENSION__SHIFT;
}

// wuffs_base__token__value_major returns a negative value if the token was not
// a simple token.
static inline int64_t  //
wuffs_base__token__value_major(const wuffs_base__token* t) {
  return ((int64_t)(t->repr)) >> WUFFS_BASE__TOKEN__VALUE_MAJOR__SHIFT;
}

// wuffs_base__token__value_base_category returns a negative value if the token
// was not a simple token.
static inline int64_t  //
wuffs_base__token__value_base_category(const wuffs_base__token* t) {
  return ((int64_t)(t->repr)) >> WUFFS_BASE__TOKEN__VALUE_BASE_CATEGORY__SHIFT;
}

static inline uint64_t  //
wuffs_base__token__value_minor(const wuffs_base__token* t) {
  return (t->repr >> WUFFS_BASE__TOKEN__VALUE_MINOR__SHIFT) & 0x1FFFFFF;
}

static inline uint64_t  //
wuffs_base__token__value_base_detail(const wuffs_base__token* t) {
  return (t->repr >> WUFFS_BASE__TOKEN__VALUE_BASE_DETAIL__SHIFT) & 0x1FFFFF;
}

static inline int64_t  //
wuffs_base__token__value_base_detail__sign_extended(
    const wuffs_base__token* t) {
  // The VBD is 21 bits in the middle of t->repr. Left shift the high (64 - 21
  // - ETC__SHIFT) bits off, then right shift (sign-extending) back down.
  uint64_t u = t->repr << (43 - WUFFS_BASE__TOKEN__VALUE_BASE_DETAIL__SHIFT);
  return ((int64_t)u) >> 43;
}

static inline bool  //
wuffs_base__token__continued(const wuffs_base__token* t) {
  return t->repr & 0x10000;
}

static inline uint64_t  //
wuffs_base__token__length(const wuffs_base__token* t) {
  return (t->repr >> WUFFS_BASE__TOKEN__LENGTH__SHIFT) & 0xFFFF;
}

#ifdef __cplusplus

inline int64_t  //
wuffs_base__token::value() const {
  return wuffs_base__token__value(this);
}

inline int64_t  //
wuffs_base__token::value_extension() const {
  return wuffs_base__token__value_extension(this);
}

inline int64_t  //
wuffs_base__token::value_major() const {
  return wuffs_base__token__value_major(this);
}

inline int64_t  //
wuffs_base__token::value_base_category() const {
  return wuffs_base__token__value_base_category(this);
}

inline uint64_t  //
wuffs_base__token::value_minor() const {
  return wuffs_base__token__value_minor(this);
}

inline uint64_t  //
wuffs_base__token::value_base_detail() const {
  return wuffs_base__token__value_base_detail(this);
}

inline int64_t  //
wuffs_base__token::value_base_detail__sign_extended() const {
  return wuffs_base__token__value_base_detail__sign_extended(this);
}

inline bool  //
wuffs_base__token::continued() const {
  return wuffs_base__token__continued(this);
}

inline uint64_t  //
wuffs_base__token::length() const {
  return wuffs_base__token__length(this);
}

#endif  // __cplusplus

// --------

typedef WUFFS_BASE__SLICE(wuffs_base__token) wuffs_base__slice_token;

static inline wuffs_base__slice_token  //
wuffs_base__make_slice_token(wuffs_base__token* ptr, size_t len) {
  wuffs_base__slice_token ret;
  ret.ptr = ptr;
  ret.len = len;
  return ret;
}

static inline wuffs_base__slice_token  //
wuffs_base__empty_slice_token() {
  wuffs_base__slice_token ret;
  ret.ptr = NULL;
  ret.len = 0;
  return ret;
}

// --------

// wuffs_base__token_buffer_meta is the metadata for a
// wuffs_base__token_buffer's data.
typedef struct wuffs_base__token_buffer_meta__struct {
  size_t wi;     // Write index. Invariant: wi <= len.
  size_t ri;     // Read  index. Invariant: ri <= wi.
  uint64_t pos;  // Position of the buffer start relative to the stream start.
  bool closed;   // No further writes are expected.
} wuffs_base__token_buffer_meta;

// wuffs_base__token_buffer is a 1-dimensional buffer (a pointer and length)
// plus additional metadata.
//
// A value with all fields zero is a valid, empty buffer.
typedef struct wuffs_base__token_buffer__struct {
  wuffs_base__slice_token data;
  wuffs_base__token_buffer_meta meta;

#ifdef __cplusplus
  inline bool is_valid() const;
  inline void compact();
  inline uint64_t reader_length() const;
  inline wuffs_base__token* reader_pointer() const;
  inline wuffs_base__slice_token reader_slice() const;
  inline uint64_t reader_token_position() const;
  inline uint64_t writer_length() const;
  inline uint64_t writer_token_position() const;
  inline wuffs_base__token* writer_pointer() const;
  inline wuffs_base__slice_token writer_slice() const;
#endif  // __cplusplus

} wuffs_base__token_buffer;

static inline wuffs_base__token_buffer  //
wuffs_base__make_token_buffer(wuffs_base__slice_token data,
                              wuffs_base__token_buffer_meta meta) {
  wuffs_base__token_buffer ret;
  ret.data = data;
  ret.meta = meta;
  return ret;
}

static inline wuffs_base__token_buffer_meta  //
wuffs_base__make_token_buffer_meta(size_t wi,
                                   size_t ri,
                                   uint64_t pos,
                                   bool closed) {
  wuffs_base__token_buffer_meta ret;
  ret.wi = wi;
  ret.ri = ri;
  ret.pos = pos;
  ret.closed = closed;
  return ret;
}

static inline wuffs_base__token_buffer  //
wuffs_base__slice_token__reader(wuffs_base__slice_token s, bool closed) {
  wuffs_base__token_buffer ret;
  ret.data.ptr = s.ptr;
  ret.data.len = s.len;
  ret.meta.wi = s.len;
  ret.meta.ri = 0;
  ret.meta.pos = 0;
  ret.meta.closed = closed;
  return ret;
}

static inline wuffs_base__token_buffer  //
wuffs_base__slice_token__writer(wuffs_base__slice_token s) {
  wuffs_base__token_buffer ret;
  ret.data.ptr = s.ptr;
  ret.data.len = s.len;
  ret.meta.wi = 0;
  ret.meta.ri = 0;
  ret.meta.pos = 0;
  ret.meta.closed = false;
  return ret;
}

static inline wuffs_base__token_buffer  //
wuffs_base__empty_token_buffer() {
  wuffs_base__token_buffer ret;
  ret.data.ptr = NULL;
  ret.data.len = 0;
  ret.meta.wi = 0;
  ret.meta.ri = 0;
  ret.meta.pos = 0;
  ret.meta.closed = false;
  return ret;
}

static inline wuffs_base__token_buffer_meta  //
wuffs_base__empty_token_buffer_meta() {
  wuffs_base__token_buffer_meta ret;
  ret.wi = 0;
  ret.ri = 0;
  ret.pos = 0;
  ret.closed = false;
  return ret;
}

static inline bool  //
wuffs_base__token_buffer__is_valid(const wuffs_base__token_buffer* buf) {
  if (buf) {
    if (buf->data.ptr) {
      return (buf->meta.ri <= buf->meta.wi) && (buf->meta.wi <= buf->data.len);
    } else {
      return (buf->meta.ri == 0) && (buf->meta.wi == 0) && (buf->data.len == 0);
    }
  }
  return false;
}

// wuffs_base__token_buffer__compact moves any written but unread tokens to the
// start of the buffer.
static inline void  //
wuffs_base__token_buffer__compact(wuffs_base__token_buffer* buf) {
  if (!buf || (buf->meta.ri == 0)) {
    return;
  }
  buf->meta.pos = wuffs_base__u64__sat_add(buf->meta.pos, buf->meta.ri);
  size_t n = buf->meta.wi - buf->meta.ri;
  if (n != 0) {
    memmove(buf->data.ptr, buf->data.ptr + buf->meta.ri,
            n * sizeof(wuffs_base__token));
  }
  buf->meta.wi = n;
  buf->meta.ri = 0;
}

static inline uint64_t  //
wuffs_base__token_buffer__reader_length(const wuffs_base__token_buffer* buf) {
  return buf ? buf->meta.wi - buf->meta.ri : 0;
}

static inline wuffs_base__token*  //
wuffs_base__token_buffer__reader_pointer(const wuffs_base__token_buffer* buf) {
  return buf ? (buf->data.ptr + buf->meta.ri) : NULL;
}

static inline wuffs_base__slice_token  //
wuffs_base__token_buffer__reader_slice(const wuffs_base__token_buffer* buf) {
  return buf ? wuffs_base__make_slice_token(buf->data.ptr + buf->meta.ri,
                                            buf->meta.wi - buf->meta.ri)
             : wuffs_base__empty_slice_token();
}

static inline uint64_t  //
wuffs_base__token_buffer__reader_token_position(
    const wuffs_base__token_buffer* buf) {
  return buf ? wuffs_base__u64__sat_add(buf->meta.pos, buf->meta.ri) : 0;
}

static inline uint64_t  //
wuffs_base__token_buffer__writer_length(const wuffs_base__token_buffer* buf) {
  return buf ? buf->data.len - buf->meta.wi : 0;
}

static inline wuffs_base__token*  //
wuffs_base__token_buffer__writer_pointer(const wuffs_base__token_buffer* buf) {
  return buf ? (buf->data.ptr + buf->meta.wi) : NULL;
}

static inline wuffs_base__slice_token  //
wuffs_base__token_buffer__writer_slice(const wuffs_base__token_buffer* buf) {
  return buf ? wuffs_base__make_slice_token(buf->data.ptr + buf->meta.wi,
                                            buf->data.len - buf->meta.wi)
             : wuffs_base__empty_slice_token();
}

static inline uint64_t  //
wuffs_base__token_buffer__writer_token_position(
    const wuffs_base__token_buffer* buf) {
  return buf ? wuffs_base__u64__sat_add(buf->meta.pos, buf->meta.wi) : 0;
}

#ifdef __cplusplus

inline bool  //
wuffs_base__token_buffer::is_valid() const {
  return wuffs_base__token_buffer__is_valid(this);
}

inline void  //
wuffs_base__token_buffer::compact() {
  wuffs_base__token_buffer__compact(this);
}

inline uint64_t  //
wuffs_base__token_buffer::reader_length() const {
  return wuffs_base__token_buffer__reader_length(this);
}

inline wuffs_base__token*  //
wuffs_base__token_buffer::reader_pointer() const {
  return wuffs_base__token_buffer__reader_pointer(this);
}

inline wuffs_base__slice_token  //
wuffs_base__token_buffer::reader_slice() const {
  return wuffs_base__token_buffer__reader_slice(this);
}

inline uint64_t  //
wuffs_base__token_buffer::reader_token_position() const {
  return wuffs_base__token_buffer__reader_token_position(this);
}

inline uint64_t  //
wuffs_base__token_buffer::writer_length() const {
  return wuffs_base__token_buffer__writer_length(this);
}

inline wuffs_base__token*  //
wuffs_base__token_buffer::writer_pointer() const {
  return wuffs_base__token_buffer__writer_pointer(this);
}

inline wuffs_base__slice_token  //
wuffs_base__token_buffer::writer_slice() const {
  return wuffs_base__token_buffer__writer_slice(this);
}

inline uint64_t  //
wuffs_base__token_buffer::writer_token_position() const {
  return wuffs_base__token_buffer__writer_token_position(this);
}

#endif  // __cplusplus

// ---------------- Memory Allocation

// The memory allocation related functions in this section aren't used by Wuffs
// per se, but they may be helpful to the code that uses Wuffs.

// wuffs_base__malloc_slice_uxx wraps calling a malloc-like function, except
// that it takes a uint64_t number of elements instead of a size_t size in
// bytes, and it returns a slice (a pointer and a length) instead of just a
// pointer.
//
// You can pass the C stdlib's malloc as the malloc_func.
//
// It returns an empty slice (containing a NULL ptr field) if (num_uxx *
// sizeof(uintxx_t)) would overflow SIZE_MAX.

static inline wuffs_base__slice_u8  //
wuffs_base__malloc_slice_u8(void* (*malloc_func)(size_t), uint64_t num_u8) {
  if (malloc_func && (num_u8 <= (SIZE_MAX / sizeof(uint8_t)))) {
    void* p = (*malloc_func)((size_t)(num_u8 * sizeof(uint8_t)));
    if (p) {
      return wuffs_base__make_slice_u8((uint8_t*)(p), (size_t)num_u8);
    }
  }
  return wuffs_base__make_slice_u8(NULL, 0);
}

static inline wuffs_base__slice_u16  //
wuffs_base__malloc_slice_u16(void* (*malloc_func)(size_t), uint64_t num_u16) {
  if (malloc_func && (num_u16 <= (SIZE_MAX / sizeof(uint16_t)))) {
    void* p = (*malloc_func)((size_t)(num_u16 * sizeof(uint16_t)));
    if (p) {
      return wuffs_base__make_slice_u16((uint16_t*)(p), (size_t)num_u16);
    }
  }
  return wuffs_base__make_slice_u16(NULL, 0);
}

static inline wuffs_base__slice_u32  //
wuffs_base__malloc_slice_u32(void* (*malloc_func)(size_t), uint64_t num_u32) {
  if (malloc_func && (num_u32 <= (SIZE_MAX / sizeof(uint32_t)))) {
    void* p = (*malloc_func)((size_t)(num_u32 * sizeof(uint32_t)));
    if (p) {
      return wuffs_base__make_slice_u32((uint32_t*)(p), (size_t)num_u32);
    }
  }
  return wuffs_base__make_slice_u32(NULL, 0);
}

static inline wuffs_base__slice_u64  //
wuffs_base__malloc_slice_u64(void* (*malloc_func)(size_t), uint64_t num_u64) {
  if (malloc_func && (num_u64 <= (SIZE_MAX / sizeof(uint64_t)))) {
    void* p = (*malloc_func)((size_t)(num_u64 * sizeof(uint64_t)));
    if (p) {
      return wuffs_base__make_slice_u64((uint64_t*)(p), (size_t)num_u64);
    }
  }
  return wuffs_base__make_slice_u64(NULL, 0);
}

// ---------------- Images

// wuffs_base__color_u32_argb_premul is an 8 bit per channel premultiplied
// Alpha, Red, Green, Blue color, as a uint32_t value. Its value is always
// 0xAARRGGBB (Alpha most significant, Blue least), regardless of endianness.
typedef uint32_t wuffs_base__color_u32_argb_premul;

// wuffs_base__color_u32_argb_premul__is_valid returns whether c's Red, Green
// and Blue channels are all less than or equal to its Alpha channel. c uses
// premultiplied alpha, so 50% opaque 100% saturated red is 0x7F7F_0000 and a
// value like 0x7F80_0000 is invalid.
static inline bool  //
wuffs_base__color_u32_argb_premul__is_valid(
    wuffs_base__color_u32_argb_premul c) {
  uint32_t a = 0xFF & (c >> 24);
  uint32_t r = 0xFF & (c >> 16);
  uint32_t g = 0xFF & (c >> 8);
  uint32_t b = 0xFF & (c >> 0);
  return (a >= r) && (a >= g) && (a >= b);
}

static inline uint16_t  //
wuffs_base__color_u32_argb_premul__as__color_u16_rgb_565(
    wuffs_base__color_u32_argb_premul c) {
  uint32_t r5 = 0xF800 & (c >> 8);
  uint32_t g6 = 0x07E0 & (c >> 5);
  uint32_t b5 = 0x001F & (c >> 3);
  return (uint16_t)(r5 | g6 | b5);
}

static inline wuffs_base__color_u32_argb_premul  //
wuffs_base__color_u16_rgb_565__as__color_u32_argb_premul(uint16_t rgb_565) {
  uint32_t b5 = 0x1F & (rgb_565 >> 0);
  uint32_t b = (b5 << 3) | (b5 >> 2);
  uint32_t g6 = 0x3F & (rgb_565 >> 5);
  uint32_t g = (g6 << 2) | (g6 >> 4);
  uint32_t r5 = 0x1F & (rgb_565 >> 11);
  uint32_t r = (r5 << 3) | (r5 >> 2);
  return 0xFF000000 | (r << 16) | (g << 8) | (b << 0);
}

static inline uint8_t  //
wuffs_base__color_u32_argb_premul__as__color_u8_gray(
    wuffs_base__color_u32_argb_premul c) {
  // Work in 16-bit color.
  uint32_t cr = 0x101 * (0xFF & (c >> 16));
  uint32_t cg = 0x101 * (0xFF & (c >> 8));
  uint32_t cb = 0x101 * (0xFF & (c >> 0));

  // These coefficients (the fractions 0.299, 0.587 and 0.114) are the same
  // as those given by the JFIF specification.
  //
  // Note that 19595 + 38470 + 7471 equals 65536, also known as (1 << 16). We
  // shift by 24, not just by 16, because the return value is 8-bit color, not
  // 16-bit color.
  uint32_t weighted_average = (19595 * cr) + (38470 * cg) + (7471 * cb) + 32768;
  return (uint8_t)(weighted_average >> 24);
}

static inline uint16_t  //
wuffs_base__color_u32_argb_premul__as__color_u16_gray(
    wuffs_base__color_u32_argb_premul c) {
  // Work in 16-bit color.
  uint32_t cr = 0x101 * (0xFF & (c >> 16));
  uint32_t cg = 0x101 * (0xFF & (c >> 8));
  uint32_t cb = 0x101 * (0xFF & (c >> 0));

  // These coefficients (the fractions 0.299, 0.587 and 0.114) are the same
  // as those given by the JFIF specification.
  //
  // Note that 19595 + 38470 + 7471 equals 65536, also known as (1 << 16).
  uint32_t weighted_average = (19595 * cr) + (38470 * cg) + (7471 * cb) + 32768;
  return (uint16_t)(weighted_average >> 16);
}

// wuffs_base__color_u32_argb_nonpremul__as__color_u32_argb_premul converts
// from non-premultiplied alpha to premultiplied alpha.
static inline wuffs_base__color_u32_argb_premul  //
wuffs_base__color_u32_argb_nonpremul__as__color_u32_argb_premul(
    uint32_t argb_nonpremul) {
  // Multiplying by 0x101 (twice, once for alpha and once for color) converts
  // from 8-bit to 16-bit color. Shifting right by 8 undoes that.
  //
  // Working in the higher bit depth can produce slightly different (and
  // arguably slightly more accurate) results. For example, given 8-bit blue
  // and alpha of 0x80 and 0x81:
  //
  //  - ((0x80   * 0x81  ) / 0xFF  )      = 0x40        = 0x40
  //  - ((0x8080 * 0x8181) / 0xFFFF) >> 8 = 0x4101 >> 8 = 0x41
  uint32_t a = 0xFF & (argb_nonpremul >> 24);
  uint32_t a16 = a * (0x101 * 0x101);

  uint32_t r = 0xFF & (argb_nonpremul >> 16);
  r = ((r * a16) / 0xFFFF) >> 8;
  uint32_t g = 0xFF & (argb_nonpremul >> 8);
  g = ((g * a16) / 0xFFFF) >> 8;
  uint32_t b = 0xFF & (argb_nonpremul >> 0);
  b = ((b * a16) / 0xFFFF) >> 8;

  return (a << 24) | (r << 16) | (g << 8) | (b << 0);
}

// wuffs_base__color_u32_argb_premul__as__color_u32_argb_nonpremul converts
// from premultiplied alpha to non-premultiplied alpha.
static inline uint32_t  //
wuffs_base__color_u32_argb_premul__as__color_u32_argb_nonpremul(
    wuffs_base__color_u32_argb_premul c) {
  uint32_t a = 0xFF & (c >> 24);
  if (a == 0xFF) {
    return c;
  } else if (a == 0) {
    return 0;
  }
  uint32_t a16 = a * 0x101;

  uint32_t r = 0xFF & (c >> 16);
  r = ((r * (0x101 * 0xFFFF)) / a16) >> 8;
  uint32_t g = 0xFF & (c >> 8);
  g = ((g * (0x101 * 0xFFFF)) / a16) >> 8;
  uint32_t b = 0xFF & (c >> 0);
  b = ((b * (0x101 * 0xFFFF)) / a16) >> 8;

  return (a << 24) | (r << 16) | (g << 8) | (b << 0);
}

// wuffs_base__color_u64_argb_nonpremul__as__color_u32_argb_premul converts
// from 4x16LE non-premultiplied alpha to 4x8 premultiplied alpha.
static inline wuffs_base__color_u32_argb_premul  //
wuffs_base__color_u64_argb_nonpremul__as__color_u32_argb_premul(
    uint64_t argb_nonpremul) {
  uint32_t a16 = ((uint32_t)(0xFFFF & (argb_nonpremul >> 48)));

  uint32_t r16 = ((uint32_t)(0xFFFF & (argb_nonpremul >> 32)));
  r16 = (r16 * a16) / 0xFFFF;
  uint32_t g16 = ((uint32_t)(0xFFFF & (argb_nonpremul >> 16)));
  g16 = (g16 * a16) / 0xFFFF;
  uint32_t b16 = ((uint32_t)(0xFFFF & (argb_nonpremul >> 0)));
  b16 = (b16 * a16) / 0xFFFF;

  return ((a16 >> 8) << 24) | ((r16 >> 8) << 16) | ((g16 >> 8) << 8) |
         ((b16 >> 8) << 0);
}

// wuffs_base__color_u32_argb_premul__as__color_u64_argb_nonpremul converts
// from 4x8 premultiplied alpha to 4x16LE non-premultiplied alpha.
static inline uint64_t  //
wuffs_base__color_u32_argb_premul__as__color_u64_argb_nonpremul(
    wuffs_base__color_u32_argb_premul c) {
  uint32_t a = 0xFF & (c >> 24);
  if (a == 0xFF) {
    uint64_t r16 = 0x101 * (0xFF & (c >> 16));
    uint64_t g16 = 0x101 * (0xFF & (c >> 8));
    uint64_t b16 = 0x101 * (0xFF & (c >> 0));
    return 0xFFFF000000000000u | (r16 << 32) | (g16 << 16) | (b16 << 0);
  } else if (a == 0) {
    return 0;
  }
  uint64_t a16 = a * 0x101;

  uint64_t r = 0xFF & (c >> 16);
  uint64_t r16 = (r * (0x101 * 0xFFFF)) / a16;
  uint64_t g = 0xFF & (c >> 8);
  uint64_t g16 = (g * (0x101 * 0xFFFF)) / a16;
  uint64_t b = 0xFF & (c >> 0);
  uint64_t b16 = (b * (0x101 * 0xFFFF)) / a16;

  return (a16 << 48) | (r16 << 32) | (g16 << 16) | (b16 << 0);
}

static inline uint64_t  //
wuffs_base__color_u32__as__color_u64(uint32_t c) {
  uint64_t a16 = 0x101 * (0xFF & (c >> 24));
  uint64_t r16 = 0x101 * (0xFF & (c >> 16));
  uint64_t g16 = 0x101 * (0xFF & (c >> 8));
  uint64_t b16 = 0x101 * (0xFF & (c >> 0));
  return (a16 << 48) | (r16 << 32) | (g16 << 16) | (b16 << 0);
}

static inline uint32_t  //
wuffs_base__color_u64__as__color_u32(uint64_t c) {
  uint32_t a = ((uint32_t)(0xFF & (c >> 56)));
  uint32_t r = ((uint32_t)(0xFF & (c >> 40)));
  uint32_t g = ((uint32_t)(0xFF & (c >> 24)));
  uint32_t b = ((uint32_t)(0xFF & (c >> 8)));
  return (a << 24) | (r << 16) | (g << 8) | (b << 0);
}

// --------

typedef uint8_t wuffs_base__pixel_blend;

// wuffs_base__pixel_blend encodes how to blend source and destination pixels,
// accounting for transparency. It encompasses the Porter-Duff compositing
// operators as well as the other blending modes defined by PDF.
//
// TODO: implement the other modes.
#define WUFFS_BASE__PIXEL_BLEND__SRC ((wuffs_base__pixel_blend)0)
#define WUFFS_BASE__PIXEL_BLEND__SRC_OVER ((wuffs_base__pixel_blend)1)

// --------

// wuffs_base__pixel_alpha_transparency is a pixel format's alpha channel
// model. It is a property of the pixel format in general, not of a specific
// pixel. An RGBA pixel format (with alpha) can still have fully opaque pixels.
typedef uint32_t wuffs_base__pixel_alpha_transparency;

#define WUFFS_BASE__PIXEL_ALPHA_TRANSPARENCY__OPAQUE 0
#define WUFFS_BASE__PIXEL_ALPHA_TRANSPARENCY__NONPREMULTIPLIED_ALPHA 1
#define WUFFS_BASE__PIXEL_ALPHA_TRANSPARENCY__PREMULTIPLIED_ALPHA 2
#define WUFFS_BASE__PIXEL_ALPHA_TRANSPARENCY__BINARY_ALPHA 3

// Deprecated: use WUFFS_BASE__PIXEL_ALPHA_TRANSPARENCY__NONPREMULTIPLIED_ALPHA
// instead.
#define WUFFS_BASE__PIXEL_ALPHA_TRANSPARENCY__NON_PREMULTIPLIED_ALPHA 1

// --------

#define WUFFS_BASE__PIXEL_FORMAT__NUM_PLANES_MAX 4

#define WUFFS_BASE__PIXEL_FORMAT__INDEXED__INDEX_PLANE 0
#define WUFFS_BASE__PIXEL_FORMAT__INDEXED__COLOR_PLANE 3

// A palette is 256 entries × 4 bytes per entry (e.g. BGRA).
#define WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH 1024

// wuffs_base__pixel_format encodes the format of the bytes that constitute an
// image frame's pixel data.
//
// See https://github.com/google/wuffs/blob/main/doc/note/pixel-formats.md
//
// Do not manipulate its bits directly; they are private implementation
// details. Use methods such as wuffs_base__pixel_format__num_planes instead.
typedef struct wuffs_base__pixel_format__struct {
  uint32_t repr;

#ifdef __cplusplus
  inline bool is_valid() const;
  inline uint32_t bits_per_pixel() const;
  inline bool is_direct() const;
  inline bool is_indexed() const;
  inline bool is_interleaved() const;
  inline bool is_planar() const;
  inline uint32_t num_planes() const;
  inline wuffs_base__pixel_alpha_transparency transparency() const;
#endif  // __cplusplus

} wuffs_base__pixel_format;

static inline wuffs_base__pixel_format  //
wuffs_base__make_pixel_format(uint32_t repr) {
  wuffs_base__pixel_format f;
  f.repr = repr;
  return f;
}

// Common 8-bit-depth pixel formats. This list is not exhaustive; not all valid
// wuffs_base__pixel_format values are present.

#define WUFFS_BASE__PIXEL_FORMAT__INVALID 0x00000000

#define WUFFS_BASE__PIXEL_FORMAT__A 0x02000008

#define WUFFS_BASE__PIXEL_FORMAT__Y 0x20000008
#define WUFFS_BASE__PIXEL_FORMAT__Y_16LE 0x2000000B
#define WUFFS_BASE__PIXEL_FORMAT__Y_16BE 0x2010000B
#define WUFFS_BASE__PIXEL_FORMAT__YA_NONPREMUL 0x21000008
#define WUFFS_BASE__PIXEL_FORMAT__YA_PREMUL 0x22000008

#define WUFFS_BASE__PIXEL_FORMAT__YCBCR 0x40020888
#define WUFFS_BASE__PIXEL_FORMAT__YCBCRA_NONPREMUL 0x41038888
#define WUFFS_BASE__PIXEL_FORMAT__YCBCRK 0x50038888

#define WUFFS_BASE__PIXEL_FORMAT__YCOCG 0x60020888
#define WUFFS_BASE__PIXEL_FORMAT__YCOCGA_NONPREMUL 0x61038888
#define WUFFS_BASE__PIXEL_FORMAT__YCOCGK 0x70038888

#define WUFFS_BASE__PIXEL_FORMAT__INDEXED__BGRA_NONPREMUL 0x81040008
#define WUFFS_BASE__PIXEL_FORMAT__INDEXED__BGRA_PREMUL 0x82040008
#define WUFFS_BASE__PIXEL_FORMAT__INDEXED__BGRA_BINARY 0x83040008

#define WUFFS_BASE__PIXEL_FORMAT__BGR_565 0x80000565
#define WUFFS_BASE__PIXEL_FORMAT__BGR 0x80000888
#define WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL 0x81008888
#define WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL_4X16LE 0x8100BBBB
#define WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL 0x82008888
#define WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL_4X16LE 0x8200BBBB
#define WUFFS_BASE__PIXEL_FORMAT__BGRA_BINARY 0x83008888
#define WUFFS_BASE__PIXEL_FORMAT__BGRX 0x90008888

#define WUFFS_BASE__PIXEL_FORMAT__RGB 0xA0000888
#define WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL 0xA1008888
#define WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL_4X16LE 0xA100BBBB
#define WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL 0xA2008888
#define WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL_4X16LE 0xA200BBBB
#define WUFFS_BASE__PIXEL_FORMAT__RGBA_BINARY 0xA3008888
#define WUFFS_BASE__PIXEL_FORMAT__RGBX 0xB0008888

#define WUFFS_BASE__PIXEL_FORMAT__CMY 0xC0020888
#define WUFFS_BASE__PIXEL_FORMAT__CMYK 0xD0038888

extern const uint32_t wuffs_base__pixel_format__bits_per_channel[16];

static inline bool  //
wuffs_base__pixel_format__is_valid(const wuffs_base__pixel_format* f) {
  return f->repr != 0;
}

// wuffs_base__pixel_format__bits_per_pixel returns the number of bits per
// pixel for interleaved pixel formats, and returns 0 for planar pixel formats.
static inline uint32_t  //
wuffs_base__pixel_format__bits_per_pixel(const wuffs_base__pixel_format* f) {
  if (((f->repr >> 16) & 0x03) != 0) {
    return 0;
  }
  return wuffs_base__pixel_format__bits_per_channel[0x0F & (f->repr >> 0)] +
         wuffs_base__pixel_format__bits_per_channel[0x0F & (f->repr >> 4)] +
         wuffs_base__pixel_format__bits_per_channel[0x0F & (f->repr >> 8)] +
         wuffs_base__pixel_format__bits_per_channel[0x0F & (f->repr >> 12)];
}

static inline bool  //
wuffs_base__pixel_format__is_direct(const wuffs_base__pixel_format* f) {
  return ((f->repr >> 18) & 0x01) == 0;
}

static inline bool  //
wuffs_base__pixel_format__is_indexed(const wuffs_base__pixel_format* f) {
  return ((f->repr >> 18) & 0x01) != 0;
}

static inline bool  //
wuffs_base__pixel_format__is_interleaved(const wuffs_base__pixel_format* f) {
  return ((f->repr >> 16) & 0x03) == 0;
}

static inline bool  //
wuffs_base__pixel_format__is_planar(const wuffs_base__pixel_format* f) {
  return ((f->repr >> 16) & 0x03) != 0;
}

static inline uint32_t  //
wuffs_base__pixel_format__num_planes(const wuffs_base__pixel_format* f) {
  return ((f->repr >> 16) & 0x03) + 1;
}

static inline wuffs_base__pixel_alpha_transparency  //
wuffs_base__pixel_format__transparency(const wuffs_base__pixel_format* f) {
  return (wuffs_base__pixel_alpha_transparency)((f->repr >> 24) & 0x03);
}

#ifdef __cplusplus

inline bool  //
wuffs_base__pixel_format::is_valid() const {
  return wuffs_base__pixel_format__is_valid(this);
}

inline uint32_t  //
wuffs_base__pixel_format::bits_per_pixel() const {
  return wuffs_base__pixel_format__bits_per_pixel(this);
}

inline bool  //
wuffs_base__pixel_format::is_direct() const {
  return wuffs_base__pixel_format__is_direct(this);
}

inline bool  //
wuffs_base__pixel_format::is_indexed() const {
  return wuffs_base__pixel_format__is_indexed(this);
}

inline bool  //
wuffs_base__pixel_format::is_interleaved() const {
  return wuffs_base__pixel_format__is_interleaved(this);
}

inline bool  //
wuffs_base__pixel_format::is_planar() const {
  return wuffs_base__pixel_format__is_planar(this);
}

inline uint32_t  //
wuffs_base__pixel_format::num_planes() const {
  return wuffs_base__pixel_format__num_planes(this);
}

inline wuffs_base__pixel_alpha_transparency  //
wuffs_base__pixel_format::transparency() const {
  return wuffs_base__pixel_format__transparency(this);
}

#endif  // __cplusplus

// --------

// wuffs_base__pixel_subsampling encodes whether sample values cover one pixel
// or cover multiple pixels.
//
// See https://github.com/google/wuffs/blob/main/doc/note/pixel-subsampling.md
//
// Do not manipulate its bits directly; they are private implementation
// details. Use methods such as wuffs_base__pixel_subsampling__bias_x instead.
typedef struct wuffs_base__pixel_subsampling__struct {
  uint32_t repr;

#ifdef __cplusplus
  inline uint32_t bias_x(uint32_t plane) const;
  inline uint32_t denominator_x(uint32_t plane) const;
  inline uint32_t bias_y(uint32_t plane) const;
  inline uint32_t denominator_y(uint32_t plane) const;
#endif  // __cplusplus

} wuffs_base__pixel_subsampling;

static inline wuffs_base__pixel_subsampling  //
wuffs_base__make_pixel_subsampling(uint32_t repr) {
  wuffs_base__pixel_subsampling s;
  s.repr = repr;
  return s;
}

#define WUFFS_BASE__PIXEL_SUBSAMPLING__NONE 0x00000000

#define WUFFS_BASE__PIXEL_SUBSAMPLING__444 0x000000
#define WUFFS_BASE__PIXEL_SUBSAMPLING__440 0x010100
#define WUFFS_BASE__PIXEL_SUBSAMPLING__422 0x101000
#define WUFFS_BASE__PIXEL_SUBSAMPLING__420 0x111100
#define WUFFS_BASE__PIXEL_SUBSAMPLING__411 0x303000
#define WUFFS_BASE__PIXEL_SUBSAMPLING__410 0x313100

static inline uint32_t  //
wuffs_base__pixel_subsampling__bias_x(const wuffs_base__pixel_subsampling* s,
                                      uint32_t plane) {
  uint32_t shift = ((plane & 0x03) * 8) + 6;
  return (s->repr >> shift) & 0x03;
}

static inline uint32_t  //
wuffs_base__pixel_subsampling__denominator_x(
    const wuffs_base__pixel_subsampling* s,
    uint32_t plane) {
  uint32_t shift = ((plane & 0x03) * 8) + 4;
  return ((s->repr >> shift) & 0x03) + 1;
}

static inline uint32_t  //
wuffs_base__pixel_subsampling__bias_y(const wuffs_base__pixel_subsampling* s,
                                      uint32_t plane) {
  uint32_t shift = ((plane & 0x03) * 8) + 2;
  return (s->repr >> shift) & 0x03;
}

static inline uint32_t  //
wuffs_base__pixel_subsampling__denominator_y(
    const wuffs_base__pixel_subsampling* s,
    uint32_t plane) {
  uint32_t shift = ((plane & 0x03) * 8) + 0;
  return ((s->repr >> shift) & 0x03) + 1;
}

#ifdef __cplusplus

inline uint32_t  //
wuffs_base__pixel_subsampling::bias_x(uint32_t plane) const {
  return wuffs_base__pixel_subsampling__bias_x(this, plane);
}

inline uint32_t  //
wuffs_base__pixel_subsampling::denominator_x(uint32_t plane) const {
  return wuffs_base__pixel_subsampling__denominator_x(this, plane);
}

inline uint32_t  //
wuffs_base__pixel_subsampling::bias_y(uint32_t plane) const {
  return wuffs_base__pixel_subsampling__bias_y(this, plane);
}

inline uint32_t  //
wuffs_base__pixel_subsampling::denominator_y(uint32_t plane) const {
  return wuffs_base__pixel_subsampling__denominator_y(this, plane);
}

#endif  // __cplusplus

// --------

typedef struct wuffs_base__pixel_config__struct {
  // Do not access the private_impl's fields directly. There is no API/ABI
  // compatibility or safety guarantee if you do so.
  struct {
    wuffs_base__pixel_format pixfmt;
    wuffs_base__pixel_subsampling pixsub;
    uint32_t width;
    uint32_t height;
  } private_impl;

#ifdef __cplusplus
  inline void set(uint32_t pixfmt_repr,
                  uint32_t pixsub_repr,
                  uint32_t width,
                  uint32_t height);
  inline void invalidate();
  inline bool is_valid() const;
  inline wuffs_base__pixel_format pixel_format() const;
  inline wuffs_base__pixel_subsampling pixel_subsampling() const;
  inline wuffs_base__rect_ie_u32 bounds() const;
  inline uint32_t width() const;
  inline uint32_t height() const;
  inline uint64_t pixbuf_len() const;
#endif  // __cplusplus

} wuffs_base__pixel_config;

static inline wuffs_base__pixel_config  //
wuffs_base__null_pixel_config() {
  wuffs_base__pixel_config ret;
  ret.private_impl.pixfmt.repr = 0;
  ret.private_impl.pixsub.repr = 0;
  ret.private_impl.width = 0;
  ret.private_impl.height = 0;
  return ret;
}

// TODO: Should this function return bool? An error type?
static inline void  //
wuffs_base__pixel_config__set(wuffs_base__pixel_config* c,
                              uint32_t pixfmt_repr,
                              uint32_t pixsub_repr,
                              uint32_t width,
                              uint32_t height) {
  if (!c) {
    return;
  }
  if (pixfmt_repr) {
    uint64_t wh = ((uint64_t)width) * ((uint64_t)height);
    // TODO: handle things other than 1 byte per pixel.
    if (wh <= ((uint64_t)SIZE_MAX)) {
      c->private_impl.pixfmt.repr = pixfmt_repr;
      c->private_impl.pixsub.repr = pixsub_repr;
      c->private_impl.width = width;
      c->private_impl.height = height;
      return;
    }
  }

  c->private_impl.pixfmt.repr = 0;
  c->private_impl.pixsub.repr = 0;
  c->private_impl.width = 0;
  c->private_impl.height = 0;
}

static inline void  //
wuffs_base__pixel_config__invalidate(wuffs_base__pixel_config* c) {
  if (c) {
    c->private_impl.pixfmt.repr = 0;
    c->private_impl.pixsub.repr = 0;
    c->private_impl.width = 0;
    c->private_impl.height = 0;
  }
}

static inline bool  //
wuffs_base__pixel_config__is_valid(const wuffs_base__pixel_config* c) {
  return c && c->private_impl.pixfmt.repr;
}

static inline wuffs_base__pixel_format  //
wuffs_base__pixel_config__pixel_format(const wuffs_base__pixel_config* c) {
  return c ? c->private_impl.pixfmt : wuffs_base__make_pixel_format(0);
}

static inline wuffs_base__pixel_subsampling  //
wuffs_base__pixel_config__pixel_subsampling(const wuffs_base__pixel_config* c) {
  return c ? c->private_impl.pixsub : wuffs_base__make_pixel_subsampling(0);
}

static inline wuffs_base__rect_ie_u32  //
wuffs_base__pixel_config__bounds(const wuffs_base__pixel_config* c) {
  if (c) {
    wuffs_base__rect_ie_u32 ret;
    ret.min_incl_x = 0;
    ret.min_incl_y = 0;
    ret.max_excl_x = c->private_impl.width;
    ret.max_excl_y = c->private_impl.height;
    return ret;
  }

  wuffs_base__rect_ie_u32 ret;
  ret.min_incl_x = 0;
  ret.min_incl_y = 0;
  ret.max_excl_x = 0;
  ret.max_excl_y = 0;
  return ret;
}

static inline uint32_t  //
wuffs_base__pixel_config__width(const wuffs_base__pixel_config* c) {
  return c ? c->private_impl.width : 0;
}

static inline uint32_t  //
wuffs_base__pixel_config__height(const wuffs_base__pixel_config* c) {
  return c ? c->private_impl.height : 0;
}

// TODO: this is the right API for planar (not interleaved) pixbufs? Should it
// allow decoding into a color model different from the format's intrinsic one?
// For example, decoding a JPEG image straight to RGBA instead of to YCbCr?
static inline uint64_t  //
wuffs_base__pixel_config__pixbuf_len(const wuffs_base__pixel_config* c) {
  if (!c) {
    return 0;
  }
  if (wuffs_base__pixel_format__is_planar(&c->private_impl.pixfmt)) {
    // TODO: support planar pixel formats, concious of pixel subsampling.
    return 0;
  }
  uint32_t bits_per_pixel =
      wuffs_base__pixel_format__bits_per_pixel(&c->private_impl.pixfmt);
  if ((bits_per_pixel == 0) || ((bits_per_pixel % 8) != 0)) {
    // TODO: support fraction-of-byte pixels, e.g. 1 bit per pixel?
    return 0;
  }
  uint64_t bytes_per_pixel = bits_per_pixel / 8;

  uint64_t n =
      ((uint64_t)c->private_impl.width) * ((uint64_t)c->private_impl.height);
  if (n > (UINT64_MAX / bytes_per_pixel)) {
    return 0;
  }
  n *= bytes_per_pixel;

  if (wuffs_base__pixel_format__is_indexed(&c->private_impl.pixfmt)) {
    if (n >
        (UINT64_MAX - WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH)) {
      return 0;
    }
    n += WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH;
  }

  return n;
}

#ifdef __cplusplus

inline void  //
wuffs_base__pixel_config::set(uint32_t pixfmt_repr,
                              uint32_t pixsub_repr,
                              uint32_t width,
                              uint32_t height) {
  wuffs_base__pixel_config__set(this, pixfmt_repr, pixsub_repr, width, height);
}

inline void  //
wuffs_base__pixel_config::invalidate() {
  wuffs_base__pixel_config__invalidate(this);
}

inline bool  //
wuffs_base__pixel_config::is_valid() const {
  return wuffs_base__pixel_config__is_valid(this);
}

inline wuffs_base__pixel_format  //
wuffs_base__pixel_config::pixel_format() const {
  return wuffs_base__pixel_config__pixel_format(this);
}

inline wuffs_base__pixel_subsampling  //
wuffs_base__pixel_config::pixel_subsampling() const {
  return wuffs_base__pixel_config__pixel_subsampling(this);
}

inline wuffs_base__rect_ie_u32  //
wuffs_base__pixel_config::bounds() const {
  return wuffs_base__pixel_config__bounds(this);
}

inline uint32_t  //
wuffs_base__pixel_config::width() const {
  return wuffs_base__pixel_config__width(this);
}

inline uint32_t  //
wuffs_base__pixel_config::height() const {
  return wuffs_base__pixel_config__height(this);
}

inline uint64_t  //
wuffs_base__pixel_config::pixbuf_len() const {
  return wuffs_base__pixel_config__pixbuf_len(this);
}

#endif  // __cplusplus

// --------

typedef struct wuffs_base__image_config__struct {
  wuffs_base__pixel_config pixcfg;

  // Do not access the private_impl's fields directly. There is no API/ABI
  // compatibility or safety guarantee if you do so.
  struct {
    uint64_t first_frame_io_position;
    bool first_frame_is_opaque;
  } private_impl;

#ifdef __cplusplus
  inline void set(uint32_t pixfmt_repr,
                  uint32_t pixsub_repr,
                  uint32_t width,
                  uint32_t height,
                  uint64_t first_frame_io_position,
                  bool first_frame_is_opaque);
  inline void invalidate();
  inline bool is_valid() const;
  inline uint64_t first_frame_io_position() const;
  inline bool first_frame_is_opaque() const;
#endif  // __cplusplus

} wuffs_base__image_config;

static inline wuffs_base__image_config  //
wuffs_base__null_image_config() {
  wuffs_base__image_config ret;
  ret.pixcfg = wuffs_base__null_pixel_config();
  ret.private_impl.first_frame_io_position = 0;
  ret.private_impl.first_frame_is_opaque = false;
  return ret;
}

// TODO: Should this function return bool? An error type?
static inline void  //
wuffs_base__image_config__set(wuffs_base__image_config* c,
                              uint32_t pixfmt_repr,
                              uint32_t pixsub_repr,
                              uint32_t width,
                              uint32_t height,
                              uint64_t first_frame_io_position,
                              bool first_frame_is_opaque) {
  if (!c) {
    return;
  }
  if (pixfmt_repr) {
    c->pixcfg.private_impl.pixfmt.repr = pixfmt_repr;
    c->pixcfg.private_impl.pixsub.repr = pixsub_repr;
    c->pixcfg.private_impl.width = width;
    c->pixcfg.private_impl.height = height;
    c->private_impl.first_frame_io_position = first_frame_io_position;
    c->private_impl.first_frame_is_opaque = first_frame_is_opaque;
    return;
  }

  c->pixcfg.private_impl.pixfmt.repr = 0;
  c->pixcfg.private_impl.pixsub.repr = 0;
  c->pixcfg.private_impl.width = 0;
  c->pixcfg.private_impl.height = 0;
  c->private_impl.first_frame_io_position = 0;
  c->private_impl.first_frame_is_opaque = 0;
}

static inline void  //
wuffs_base__image_config__invalidate(wuffs_base__image_config* c) {
  if (c) {
    c->pixcfg.private_impl.pixfmt.repr = 0;
    c->pixcfg.private_impl.pixsub.repr = 0;
    c->pixcfg.private_impl.width = 0;
    c->pixcfg.private_impl.height = 0;
    c->private_impl.first_frame_io_position = 0;
    c->private_impl.first_frame_is_opaque = 0;
  }
}

static inline bool  //
wuffs_base__image_config__is_valid(const wuffs_base__image_config* c) {
  return c && wuffs_base__pixel_config__is_valid(&(c->pixcfg));
}

static inline uint64_t  //
wuffs_base__image_config__first_frame_io_position(
    const wuffs_base__image_config* c) {
  return c ? c->private_impl.first_frame_io_position : 0;
}

static inline bool  //
wuffs_base__image_config__first_frame_is_opaque(
    const wuffs_base__image_config* c) {
  return c ? c->private_impl.first_frame_is_opaque : false;
}

#ifdef __cplusplus

inline void  //
wuffs_base__image_config::set(uint32_t pixfmt_repr,
                              uint32_t pixsub_repr,
                              uint32_t width,
                              uint32_t height,
                              uint64_t first_frame_io_position,
                              bool first_frame_is_opaque) {
  wuffs_base__image_config__set(this, pixfmt_repr, pixsub_repr, width, height,
                                first_frame_io_position, first_frame_is_opaque);
}

inline void  //
wuffs_base__image_config::invalidate() {
  wuffs_base__image_config__invalidate(this);
}

inline bool  //
wuffs_base__image_config::is_valid() const {
  return wuffs_base__image_config__is_valid(this);
}

inline uint64_t  //
wuffs_base__image_config::first_frame_io_position() const {
  return wuffs_base__image_config__first_frame_io_position(this);
}

inline bool  //
wuffs_base__image_config::first_frame_is_opaque() const {
  return wuffs_base__image_config__first_frame_is_opaque(this);
}

#endif  // __cplusplus

// --------

// wuffs_base__animation_disposal encodes, for an animated image, how to
// dispose of a frame after displaying it:
//  - None means to draw the next frame on top of this one.
//  - Restore Background means to clear the frame's dirty rectangle to "the
//    background color" (in practice, this means transparent black) before
//    drawing the next frame.
//  - Restore Previous means to undo the current frame, so that the next frame
//    is drawn on top of the previous one.
typedef uint8_t wuffs_base__animation_disposal;

#define WUFFS_BASE__ANIMATION_DISPOSAL__NONE ((wuffs_base__animation_disposal)0)
#define WUFFS_BASE__ANIMATION_DISPOSAL__RESTORE_BACKGROUND \
  ((wuffs_base__animation_disposal)1)
#define WUFFS_BASE__ANIMATION_DISPOSAL__RESTORE_PREVIOUS \
  ((wuffs_base__animation_disposal)2)

// --------

typedef struct wuffs_base__frame_config__struct {
  // Do not access the private_impl's fields directly. There is no API/ABI
  // compatibility or safety guarantee if you do so.
  struct {
    wuffs_base__rect_ie_u32 bounds;
    wuffs_base__flicks duration;
    uint64_t index;
    uint64_t io_position;
    wuffs_base__animation_disposal disposal;
    bool opaque_within_bounds;
    bool overwrite_instead_of_blend;
    wuffs_base__color_u32_argb_premul background_color;
  } private_impl;

#ifdef __cplusplus
  inline void set(wuffs_base__rect_ie_u32 bounds,
                  wuffs_base__flicks duration,
                  uint64_t index,
                  uint64_t io_position,
                  wuffs_base__animation_disposal disposal,
                  bool opaque_within_bounds,
                  bool overwrite_instead_of_blend,
                  wuffs_base__color_u32_argb_premul background_color);
  inline wuffs_base__rect_ie_u32 bounds() const;
  inline uint32_t width() const;
  inline uint32_t height() const;
  inline wuffs_base__flicks duration() const;
  inline uint64_t index() const;
  inline uint64_t io_position() const;
  inline wuffs_base__animation_disposal disposal() const;
  inline bool opaque_within_bounds() const;
  inline bool overwrite_instead_of_blend() const;
  inline wuffs_base__color_u32_argb_premul background_color() const;
#endif  // __cplusplus

} wuffs_base__frame_config;

static inline wuffs_base__frame_config  //
wuffs_base__null_frame_config() {
  wuffs_base__frame_config ret;
  ret.private_impl.bounds = wuffs_base__make_rect_ie_u32(0, 0, 0, 0);
  ret.private_impl.duration = 0;
  ret.private_impl.index = 0;
  ret.private_impl.io_position = 0;
  ret.private_impl.disposal = 0;
  ret.private_impl.opaque_within_bounds = false;
  ret.private_impl.overwrite_instead_of_blend = false;
  return ret;
}

static inline void  //
wuffs_base__frame_config__set(
    wuffs_base__frame_config* c,
    wuffs_base__rect_ie_u32 bounds,
    wuffs_base__flicks duration,
    uint64_t index,
    uint64_t io_position,
    wuffs_base__animation_disposal disposal,
    bool opaque_within_bounds,
    bool overwrite_instead_of_blend,
    wuffs_base__color_u32_argb_premul background_color) {
  if (!c) {
    return;
  }

  c->private_impl.bounds = bounds;
  c->private_impl.duration = duration;
  c->private_impl.index = index;
  c->private_impl.io_position = io_position;
  c->private_impl.disposal = disposal;
  c->private_impl.opaque_within_bounds = opaque_within_bounds;
  c->private_impl.overwrite_instead_of_blend = overwrite_instead_of_blend;
  c->private_impl.background_color = background_color;
}

static inline wuffs_base__rect_ie_u32  //
wuffs_base__frame_config__bounds(const wuffs_base__frame_config* c) {
  if (c) {
    return c->private_impl.bounds;
  }

  wuffs_base__rect_ie_u32 ret;
  ret.min_incl_x = 0;
  ret.min_incl_y = 0;
  ret.max_excl_x = 0;
  ret.max_excl_y = 0;
  return ret;
}

static inline uint32_t  //
wuffs_base__frame_config__width(const wuffs_base__frame_config* c) {
  return c ? wuffs_base__rect_ie_u32__width(&c->private_impl.bounds) : 0;
}

static inline uint32_t  //
wuffs_base__frame_config__height(const wuffs_base__frame_config* c) {
  return c ? wuffs_base__rect_ie_u32__height(&c->private_impl.bounds) : 0;
}

// wuffs_base__frame_config__duration returns the amount of time to display
// this frame. Zero means to display forever - a still (non-animated) image.
static inline wuffs_base__flicks  //
wuffs_base__frame_config__duration(const wuffs_base__frame_config* c) {
  return c ? c->private_impl.duration : 0;
}

// wuffs_base__frame_config__index returns the index of this frame. The first
// frame in an image has index 0, the second frame has index 1, and so on.
static inline uint64_t  //
wuffs_base__frame_config__index(const wuffs_base__frame_config* c) {
  return c ? c->private_impl.index : 0;
}

// wuffs_base__frame_config__io_position returns the I/O stream position before
// the frame config.
static inline uint64_t  //
wuffs_base__frame_config__io_position(const wuffs_base__frame_config* c) {
  return c ? c->private_impl.io_position : 0;
}

// wuffs_base__frame_config__disposal returns, for an animated image, how to
// dispose of this frame after displaying it.
static inline wuffs_base__animation_disposal  //
wuffs_base__frame_config__disposal(const wuffs_base__frame_config* c) {
  return c ? c->private_impl.disposal : 0;
}

// wuffs_base__frame_config__opaque_within_bounds returns whether all pixels
// within the frame's bounds are fully opaque. It makes no claim about pixels
// outside the frame bounds but still inside the overall image. The two
// bounding rectangles can differ for animated images.
//
// Its semantics are conservative. It is valid for a fully opaque frame to have
// this value be false: a false negative.
//
// If true, drawing the frame with WUFFS_BASE__PIXEL_BLEND__SRC and
// WUFFS_BASE__PIXEL_BLEND__SRC_OVER should be equivalent, in terms of
// resultant pixels, but the former may be faster.
static inline bool  //
wuffs_base__frame_config__opaque_within_bounds(
    const wuffs_base__frame_config* c) {
  return c && c->private_impl.opaque_within_bounds;
}

// wuffs_base__frame_config__overwrite_instead_of_blend returns, for an
// animated image, whether to ignore the previous image state (within the frame
// bounds) when drawing this incremental frame. Equivalently, whether to use
// WUFFS_BASE__PIXEL_BLEND__SRC instead of WUFFS_BASE__PIXEL_BLEND__SRC_OVER.
//
// The WebP spec (https://developers.google.com/speed/webp/docs/riff_container)
// calls this the "Blending method" bit. WebP's "Do not blend" corresponds to
// Wuffs' "overwrite_instead_of_blend".
static inline bool  //
wuffs_base__frame_config__overwrite_instead_of_blend(
    const wuffs_base__frame_config* c) {
  return c && c->private_impl.overwrite_instead_of_blend;
}

static inline wuffs_base__color_u32_argb_premul  //
wuffs_base__frame_config__background_color(const wuffs_base__frame_config* c) {
  return c ? c->private_impl.background_color : 0;
}

#ifdef __cplusplus

inline void  //
wuffs_base__frame_config::set(
    wuffs_base__rect_ie_u32 bounds,
    wuffs_base__flicks duration,
    uint64_t index,
    uint64_t io_position,
    wuffs_base__animation_disposal disposal,
    bool opaque_within_bounds,
    bool overwrite_instead_of_blend,
    wuffs_base__color_u32_argb_premul background_color) {
  wuffs_base__frame_config__set(this, bounds, duration, index, io_position,
                                disposal, opaque_within_bounds,
                                overwrite_instead_of_blend, background_color);
}

inline wuffs_base__rect_ie_u32  //
wuffs_base__frame_config::bounds() const {
  return wuffs_base__frame_config__bounds(this);
}

inline uint32_t  //
wuffs_base__frame_config::width() const {
  return wuffs_base__frame_config__width(this);
}

inline uint32_t  //
wuffs_base__frame_config::height() const {
  return wuffs_base__frame_config__height(this);
}

inline wuffs_base__flicks  //
wuffs_base__frame_config::duration() const {
  return wuffs_base__frame_config__duration(this);
}

inline uint64_t  //
wuffs_base__frame_config::index() const {
  return wuffs_base__frame_config__index(this);
}

inline uint64_t  //
wuffs_base__frame_config::io_position() const {
  return wuffs_base__frame_config__io_position(this);
}

inline wuffs_base__animation_disposal  //
wuffs_base__frame_config::disposal() const {
  return wuffs_base__frame_config__disposal(this);
}

inline bool  //
wuffs_base__frame_config::opaque_within_bounds() const {
  return wuffs_base__frame_config__opaque_within_bounds(this);
}

inline bool  //
wuffs_base__frame_config::overwrite_instead_of_blend() const {
  return wuffs_base__frame_config__overwrite_instead_of_blend(this);
}

inline wuffs_base__color_u32_argb_premul  //
wuffs_base__frame_config::background_color() const {
  return wuffs_base__frame_config__background_color(this);
}

#endif  // __cplusplus

// --------

typedef struct wuffs_base__pixel_buffer__struct {
  wuffs_base__pixel_config pixcfg;

  // Do not access the private_impl's fields directly. There is no API/ABI
  // compatibility or safety guarantee if you do so.
  struct {
    wuffs_base__table_u8 planes[WUFFS_BASE__PIXEL_FORMAT__NUM_PLANES_MAX];
    // TODO: color spaces.
  } private_impl;

#ifdef __cplusplus
  inline wuffs_base__status set_interleaved(
      const wuffs_base__pixel_config* pixcfg,
      wuffs_base__table_u8 primary_memory,
      wuffs_base__slice_u8 palette_memory);
  inline wuffs_base__status set_from_slice(
      const wuffs_base__pixel_config* pixcfg,
      wuffs_base__slice_u8 pixbuf_memory);
  inline wuffs_base__status set_from_table(
      const wuffs_base__pixel_config* pixcfg,
      wuffs_base__table_u8 primary_memory);
  inline wuffs_base__slice_u8 palette();
  inline wuffs_base__slice_u8 palette_or_else(wuffs_base__slice_u8 fallback);
  inline wuffs_base__pixel_format pixel_format() const;
  inline wuffs_base__table_u8 plane(uint32_t p);
  inline wuffs_base__color_u32_argb_premul color_u32_at(uint32_t x,
                                                        uint32_t y) const;
  inline wuffs_base__status set_color_u32_at(
      uint32_t x,
      uint32_t y,
      wuffs_base__color_u32_argb_premul color);
  inline wuffs_base__status set_color_u32_fill_rect(
      wuffs_base__rect_ie_u32 rect,
      wuffs_base__color_u32_argb_premul color);
#endif  // __cplusplus

} wuffs_base__pixel_buffer;

static inline wuffs_base__pixel_buffer  //
wuffs_base__null_pixel_buffer() {
  wuffs_base__pixel_buffer ret;
  ret.pixcfg = wuffs_base__null_pixel_config();
  ret.private_impl.planes[0] = wuffs_base__empty_table_u8();
  ret.private_impl.planes[1] = wuffs_base__empty_table_u8();
  ret.private_impl.planes[2] = wuffs_base__empty_table_u8();
  ret.private_impl.planes[3] = wuffs_base__empty_table_u8();
  return ret;
}

static inline wuffs_base__status  //
wuffs_base__pixel_buffer__set_interleaved(
    wuffs_base__pixel_buffer* pb,
    const wuffs_base__pixel_config* pixcfg,
    wuffs_base__table_u8 primary_memory,
    wuffs_base__slice_u8 palette_memory) {
  if (!pb) {
    return wuffs_base__make_status(wuffs_base__error__bad_receiver);
  }
  memset(pb, 0, sizeof(*pb));
  if (!pixcfg ||
      wuffs_base__pixel_format__is_planar(&pixcfg->private_impl.pixfmt)) {
    return wuffs_base__make_status(wuffs_base__error__bad_argument);
  }
  if (wuffs_base__pixel_format__is_indexed(&pixcfg->private_impl.pixfmt) &&
      (palette_memory.len <
       WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH)) {
    return wuffs_base__make_status(
        wuffs_base__error__bad_argument_length_too_short);
  }
  uint32_t bits_per_pixel =
      wuffs_base__pixel_format__bits_per_pixel(&pixcfg->private_impl.pixfmt);
  if ((bits_per_pixel == 0) || ((bits_per_pixel % 8) != 0)) {
    // TODO: support fraction-of-byte pixels, e.g. 1 bit per pixel?
    return wuffs_base__make_status(wuffs_base__error__unsupported_option);
  }
  uint64_t bytes_per_pixel = bits_per_pixel / 8;

  uint64_t width_in_bytes =
      ((uint64_t)pixcfg->private_impl.width) * bytes_per_pixel;
  if ((width_in_bytes > primary_memory.width) ||
      (pixcfg->private_impl.height > primary_memory.height)) {
    return wuffs_base__make_status(wuffs_base__error__bad_argument);
  }

  pb->pixcfg = *pixcfg;
  pb->private_impl.planes[0] = primary_memory;
  if (wuffs_base__pixel_format__is_indexed(&pixcfg->private_impl.pixfmt)) {
    wuffs_base__table_u8* tab =
        &pb->private_impl
             .planes[WUFFS_BASE__PIXEL_FORMAT__INDEXED__COLOR_PLANE];
    tab->ptr = palette_memory.ptr;
    tab->width = WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH;
    tab->height = 1;
    tab->stride = WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH;
  }
  return wuffs_base__make_status(NULL);
}

static inline wuffs_base__status  //
wuffs_base__pixel_buffer__set_from_slice(wuffs_base__pixel_buffer* pb,
                                         const wuffs_base__pixel_config* pixcfg,
                                         wuffs_base__slice_u8 pixbuf_memory) {
  if (!pb) {
    return wuffs_base__make_status(wuffs_base__error__bad_receiver);
  }
  memset(pb, 0, sizeof(*pb));
  if (!pixcfg) {
    return wuffs_base__make_status(wuffs_base__error__bad_argument);
  }
  if (wuffs_base__pixel_format__is_planar(&pixcfg->private_impl.pixfmt)) {
    // TODO: support planar pixel formats, concious of pixel subsampling.
    return wuffs_base__make_status(wuffs_base__error__unsupported_option);
  }
  uint32_t bits_per_pixel =
      wuffs_base__pixel_format__bits_per_pixel(&pixcfg->private_impl.pixfmt);
  if ((bits_per_pixel == 0) || ((bits_per_pixel % 8) != 0)) {
    // TODO: support fraction-of-byte pixels, e.g. 1 bit per pixel?
    return wuffs_base__make_status(wuffs_base__error__unsupported_option);
  }
  uint64_t bytes_per_pixel = bits_per_pixel / 8;

  uint8_t* ptr = pixbuf_memory.ptr;
  uint64_t len = pixbuf_memory.len;
  if (wuffs_base__pixel_format__is_indexed(&pixcfg->private_impl.pixfmt)) {
    // Split a WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH byte
    // chunk (1024 bytes = 256 palette entries × 4 bytes per entry) from the
    // start of pixbuf_memory. We split from the start, not the end, so that
    // the both chunks' pointers have the same alignment as the original
    // pointer, up to an alignment of 1024.
    if (len < WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH) {
      return wuffs_base__make_status(
          wuffs_base__error__bad_argument_length_too_short);
    }
    wuffs_base__table_u8* tab =
        &pb->private_impl
             .planes[WUFFS_BASE__PIXEL_FORMAT__INDEXED__COLOR_PLANE];
    tab->ptr = ptr;
    tab->width = WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH;
    tab->height = 1;
    tab->stride = WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH;
    ptr += WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH;
    len -= WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH;
  }

  uint64_t wh = ((uint64_t)pixcfg->private_impl.width) *
                ((uint64_t)pixcfg->private_impl.height);
  size_t width = (size_t)(pixcfg->private_impl.width);
  if ((wh > (UINT64_MAX / bytes_per_pixel)) ||
      (width > (SIZE_MAX / bytes_per_pixel))) {
    return wuffs_base__make_status(wuffs_base__error__bad_argument);
  }
  wh *= bytes_per_pixel;
  width = ((size_t)(width * bytes_per_pixel));
  if (wh > len) {
    return wuffs_base__make_status(
        wuffs_base__error__bad_argument_length_too_short);
  }

  pb->pixcfg = *pixcfg;
  wuffs_base__table_u8* tab = &pb->private_impl.planes[0];
  tab->ptr = ptr;
  tab->width = width;
  tab->height = pixcfg->private_impl.height;
  tab->stride = width;
  return wuffs_base__make_status(NULL);
}

// Deprecated: does not handle indexed pixel configurations. Use
// wuffs_base__pixel_buffer__set_interleaved instead.
static inline wuffs_base__status  //
wuffs_base__pixel_buffer__set_from_table(wuffs_base__pixel_buffer* pb,
                                         const wuffs_base__pixel_config* pixcfg,
                                         wuffs_base__table_u8 primary_memory) {
  if (!pb) {
    return wuffs_base__make_status(wuffs_base__error__bad_receiver);
  }
  memset(pb, 0, sizeof(*pb));
  if (!pixcfg ||
      wuffs_base__pixel_format__is_indexed(&pixcfg->private_impl.pixfmt) ||
      wuffs_base__pixel_format__is_planar(&pixcfg->private_impl.pixfmt)) {
    return wuffs_base__make_status(wuffs_base__error__bad_argument);
  }
  uint32_t bits_per_pixel =
      wuffs_base__pixel_format__bits_per_pixel(&pixcfg->private_impl.pixfmt);
  if ((bits_per_pixel == 0) || ((bits_per_pixel % 8) != 0)) {
    // TODO: support fraction-of-byte pixels, e.g. 1 bit per pixel?
    return wuffs_base__make_status(wuffs_base__error__unsupported_option);
  }
  uint64_t bytes_per_pixel = bits_per_pixel / 8;

  uint64_t width_in_bytes =
      ((uint64_t)pixcfg->private_impl.width) * bytes_per_pixel;
  if ((width_in_bytes > primary_memory.width) ||
      (pixcfg->private_impl.height > primary_memory.height)) {
    return wuffs_base__make_status(wuffs_base__error__bad_argument);
  }

  pb->pixcfg = *pixcfg;
  pb->private_impl.planes[0] = primary_memory;
  return wuffs_base__make_status(NULL);
}

// wuffs_base__pixel_buffer__palette returns the palette color data. If
// non-empty, it will have length
// WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH.
static inline wuffs_base__slice_u8  //
wuffs_base__pixel_buffer__palette(wuffs_base__pixel_buffer* pb) {
  if (pb &&
      wuffs_base__pixel_format__is_indexed(&pb->pixcfg.private_impl.pixfmt)) {
    wuffs_base__table_u8* tab =
        &pb->private_impl
             .planes[WUFFS_BASE__PIXEL_FORMAT__INDEXED__COLOR_PLANE];
    if ((tab->width ==
         WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH) &&
        (tab->height == 1)) {
      return wuffs_base__make_slice_u8(
          tab->ptr, WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH);
    }
  }
  return wuffs_base__make_slice_u8(NULL, 0);
}

static inline wuffs_base__slice_u8  //
wuffs_base__pixel_buffer__palette_or_else(wuffs_base__pixel_buffer* pb,
                                          wuffs_base__slice_u8 fallback) {
  if (pb &&
      wuffs_base__pixel_format__is_indexed(&pb->pixcfg.private_impl.pixfmt)) {
    wuffs_base__table_u8* tab =
        &pb->private_impl
             .planes[WUFFS_BASE__PIXEL_FORMAT__INDEXED__COLOR_PLANE];
    if ((tab->width ==
         WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH) &&
        (tab->height == 1)) {
      return wuffs_base__make_slice_u8(
          tab->ptr, WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH);
    }
  }
  return fallback;
}

static inline wuffs_base__pixel_format  //
wuffs_base__pixel_buffer__pixel_format(const wuffs_base__pixel_buffer* pb) {
  if (pb) {
    return pb->pixcfg.private_impl.pixfmt;
  }
  return wuffs_base__make_pixel_format(WUFFS_BASE__PIXEL_FORMAT__INVALID);
}

static inline wuffs_base__table_u8  //
wuffs_base__pixel_buffer__plane(wuffs_base__pixel_buffer* pb, uint32_t p) {
  if (pb && (p < WUFFS_BASE__PIXEL_FORMAT__NUM_PLANES_MAX)) {
    return pb->private_impl.planes[p];
  }

  wuffs_base__table_u8 ret;
  ret.ptr = NULL;
  ret.width = 0;
  ret.height = 0;
  ret.stride = 0;
  return ret;
}

WUFFS_BASE__MAYBE_STATIC wuffs_base__color_u32_argb_premul  //
wuffs_base__pixel_buffer__color_u32_at(const wuffs_base__pixel_buffer* pb,
                                       uint32_t x,
                                       uint32_t y);

WUFFS_BASE__MAYBE_STATIC wuffs_base__status  //
wuffs_base__pixel_buffer__set_color_u32_at(
    wuffs_base__pixel_buffer* pb,
    uint32_t x,
    uint32_t y,
    wuffs_base__color_u32_argb_premul color);

WUFFS_BASE__MAYBE_STATIC wuffs_base__status  //
wuffs_base__pixel_buffer__set_color_u32_fill_rect(
    wuffs_base__pixel_buffer* pb,
    wuffs_base__rect_ie_u32 rect,
    wuffs_base__color_u32_argb_premul color);

#ifdef __cplusplus

inline wuffs_base__status  //
wuffs_base__pixel_buffer::set_interleaved(
    const wuffs_base__pixel_config* pixcfg_arg,
    wuffs_base__table_u8 primary_memory,
    wuffs_base__slice_u8 palette_memory) {
  return wuffs_base__pixel_buffer__set_interleaved(
      this, pixcfg_arg, primary_memory, palette_memory);
}

inline wuffs_base__status  //
wuffs_base__pixel_buffer::set_from_slice(
    const wuffs_base__pixel_config* pixcfg_arg,
    wuffs_base__slice_u8 pixbuf_memory) {
  return wuffs_base__pixel_buffer__set_from_slice(this, pixcfg_arg,
                                                  pixbuf_memory);
}

inline wuffs_base__status  //
wuffs_base__pixel_buffer::set_from_table(
    const wuffs_base__pixel_config* pixcfg_arg,
    wuffs_base__table_u8 primary_memory) {
  return wuffs_base__pixel_buffer__set_from_table(this, pixcfg_arg,
                                                  primary_memory);
}

inline wuffs_base__slice_u8  //
wuffs_base__pixel_buffer::palette() {
  return wuffs_base__pixel_buffer__palette(this);
}

inline wuffs_base__slice_u8  //
wuffs_base__pixel_buffer::palette_or_else(wuffs_base__slice_u8 fallback) {
  return wuffs_base__pixel_buffer__palette_or_else(this, fallback);
}

inline wuffs_base__pixel_format  //
wuffs_base__pixel_buffer::pixel_format() const {
  return wuffs_base__pixel_buffer__pixel_format(this);
}

inline wuffs_base__table_u8  //
wuffs_base__pixel_buffer::plane(uint32_t p) {
  return wuffs_base__pixel_buffer__plane(this, p);
}

inline wuffs_base__color_u32_argb_premul  //
wuffs_base__pixel_buffer::color_u32_at(uint32_t x, uint32_t y) const {
  return wuffs_base__pixel_buffer__color_u32_at(this, x, y);
}

WUFFS_BASE__MAYBE_STATIC wuffs_base__status  //
wuffs_base__pixel_buffer__set_color_u32_fill_rect(
    wuffs_base__pixel_buffer* pb,
    wuffs_base__rect_ie_u32 rect,
    wuffs_base__color_u32_argb_premul color);

inline wuffs_base__status  //
wuffs_base__pixel_buffer::set_color_u32_at(
    uint32_t x,
    uint32_t y,
    wuffs_base__color_u32_argb_premul color) {
  return wuffs_base__pixel_buffer__set_color_u32_at(this, x, y, color);
}

inline wuffs_base__status  //
wuffs_base__pixel_buffer::set_color_u32_fill_rect(
    wuffs_base__rect_ie_u32 rect,
    wuffs_base__color_u32_argb_premul color) {
  return wuffs_base__pixel_buffer__set_color_u32_fill_rect(this, rect, color);
}

#endif  // __cplusplus

// --------

typedef struct wuffs_base__decode_frame_options__struct {
  // Do not access the private_impl's fields directly. There is no API/ABI
  // compatibility or safety guarantee if you do so.
  struct {
    uint8_t TODO;
  } private_impl;

#ifdef __cplusplus
#endif  // __cplusplus

} wuffs_base__decode_frame_options;

#ifdef __cplusplus

#endif  // __cplusplus

// --------

// wuffs_base__pixel_palette__closest_element returns the index of the palette
// element that minimizes the sum of squared differences of the four ARGB
// channels, working in premultiplied alpha. Ties favor the smaller index.
//
// The palette_slice.len may equal (N*4), for N less than 256, which means that
// only the first N palette elements are considered. It returns 0 when N is 0.
//
// Applying this function on a per-pixel basis will not produce whole-of-image
// dithering.
WUFFS_BASE__MAYBE_STATIC uint8_t  //
wuffs_base__pixel_palette__closest_element(
    wuffs_base__slice_u8 palette_slice,
    wuffs_base__pixel_format palette_format,
    wuffs_base__color_u32_argb_premul c);

// --------

// TODO: should the func type take restrict pointers?
typedef uint64_t (*wuffs_base__pixel_swizzler__func)(uint8_t* dst_ptr,
                                                     size_t dst_len,
                                                     uint8_t* dst_palette_ptr,
                                                     size_t dst_palette_len,
                                                     const uint8_t* src_ptr,
                                                     size_t src_len);

typedef uint64_t (*wuffs_base__pixel_swizzler__transparent_black_func)(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    uint64_t num_pixels,
    uint32_t dst_pixfmt_bytes_per_pixel);

typedef struct wuffs_base__pixel_swizzler__struct {
  // Do not access the private_impl's fields directly. There is no API/ABI
  // compatibility or safety guarantee if you do so.
  struct {
    wuffs_base__pixel_swizzler__func func;
    wuffs_base__pixel_swizzler__transparent_black_func transparent_black_func;
    uint32_t dst_pixfmt_bytes_per_pixel;
    uint32_t src_pixfmt_bytes_per_pixel;
  } private_impl;

#ifdef __cplusplus
  inline wuffs_base__status prepare(wuffs_base__pixel_format dst_pixfmt,
                                    wuffs_base__slice_u8 dst_palette,
                                    wuffs_base__pixel_format src_pixfmt,
                                    wuffs_base__slice_u8 src_palette,
                                    wuffs_base__pixel_blend blend);
  inline uint64_t swizzle_interleaved_from_slice(
      wuffs_base__slice_u8 dst,
      wuffs_base__slice_u8 dst_palette,
      wuffs_base__slice_u8 src) const;
#endif  // __cplusplus

} wuffs_base__pixel_swizzler;

// wuffs_base__pixel_swizzler__prepare readies the pixel swizzler so that its
// other methods may be called.
//
// For modular builds that divide the base module into sub-modules, using this
// function requires the WUFFS_CONFIG__MODULE__BASE__PIXCONV sub-module, not
// just WUFFS_CONFIG__MODULE__BASE__CORE.
WUFFS_BASE__MAYBE_STATIC wuffs_base__status  //
wuffs_base__pixel_swizzler__prepare(wuffs_base__pixel_swizzler* p,
                                    wuffs_base__pixel_format dst_pixfmt,
                                    wuffs_base__slice_u8 dst_palette,
                                    wuffs_base__pixel_format src_pixfmt,
                                    wuffs_base__slice_u8 src_palette,
                                    wuffs_base__pixel_blend blend);

// wuffs_base__pixel_swizzler__swizzle_interleaved_from_slice converts pixels
// from a source format to a destination format.
//
// For modular builds that divide the base module into sub-modules, using this
// function requires the WUFFS_CONFIG__MODULE__BASE__PIXCONV sub-module, not
// just WUFFS_CONFIG__MODULE__BASE__CORE.
WUFFS_BASE__MAYBE_STATIC uint64_t  //
wuffs_base__pixel_swizzler__swizzle_interleaved_from_slice(
    const wuffs_base__pixel_swizzler* p,
    wuffs_base__slice_u8 dst,
    wuffs_base__slice_u8 dst_palette,
    wuffs_base__slice_u8 src);

#ifdef __cplusplus

inline wuffs_base__status  //
wuffs_base__pixel_swizzler::prepare(wuffs_base__pixel_format dst_pixfmt,
                                    wuffs_base__slice_u8 dst_palette,
                                    wuffs_base__pixel_format src_pixfmt,
                                    wuffs_base__slice_u8 src_palette,
                                    wuffs_base__pixel_blend blend) {
  return wuffs_base__pixel_swizzler__prepare(this, dst_pixfmt, dst_palette,
                                             src_pixfmt, src_palette, blend);
}

uint64_t  //
wuffs_base__pixel_swizzler::swizzle_interleaved_from_slice(
    wuffs_base__slice_u8 dst,
    wuffs_base__slice_u8 dst_palette,
    wuffs_base__slice_u8 src) const {
  return wuffs_base__pixel_swizzler__swizzle_interleaved_from_slice(
      this, dst, dst_palette, src);
}

#endif  // __cplusplus

// ---------------- String Conversions

// Options (bitwise or'ed together) for wuffs_base__parse_number_xxx
// functions. The XXX options apply to both integer and floating point. The FXX
// options apply only to floating point.

#define WUFFS_BASE__PARSE_NUMBER_XXX__DEFAULT_OPTIONS ((uint32_t)0x00000000)

// WUFFS_BASE__PARSE_NUMBER_XXX__ALLOW_MULTIPLE_LEADING_ZEROES means to accept
// inputs like "00", "0644" and "00.7". By default, they are rejected.
#define WUFFS_BASE__PARSE_NUMBER_XXX__ALLOW_MULTIPLE_LEADING_ZEROES \
  ((uint32_t)0x00000001)

// WUFFS_BASE__PARSE_NUMBER_XXX__ALLOW_UNDERSCORES means to accept inputs like
// "1__2" and "_3.141_592". By default, they are rejected.
#define WUFFS_BASE__PARSE_NUMBER_XXX__ALLOW_UNDERSCORES ((uint32_t)0x00000002)

// WUFFS_BASE__PARSE_NUMBER_FXX__DECIMAL_SEPARATOR_IS_A_COMMA means to accept
// "1,5" and not "1.5" as one-and-a-half.
//
// If the caller wants to accept either, it is responsible for canonicalizing
// the input before calling wuffs_base__parse_number_fxx. The caller also has
// more context on e.g. exactly how to treat something like "$1,234".
#define WUFFS_BASE__PARSE_NUMBER_FXX__DECIMAL_SEPARATOR_IS_A_COMMA \
  ((uint32_t)0x00000010)

// WUFFS_BASE__PARSE_NUMBER_FXX__REJECT_INF_AND_NAN means to reject inputs that
// would lead to infinite or Not-a-Number floating point values. By default,
// they are accepted.
//
// This affects the literal "inf" as input, but also affects inputs like
// "1e999" that would overflow double-precision floating point.
#define WUFFS_BASE__PARSE_NUMBER_FXX__REJECT_INF_AND_NAN ((uint32_t)0x00000020)

// --------

// Options (bitwise or'ed together) for wuffs_base__render_number_xxx
// functions. The XXX options apply to both integer and floating point. The FXX
// options apply only to floating point.

#define WUFFS_BASE__RENDER_NUMBER_XXX__DEFAULT_OPTIONS ((uint32_t)0x00000000)

// WUFFS_BASE__RENDER_NUMBER_XXX__ALIGN_RIGHT means to render to the right side
// (higher indexes) of the destination slice, leaving any untouched bytes on
// the left side (lower indexes). The default is vice versa: rendering on the
// left with slack on the right.
#define WUFFS_BASE__RENDER_NUMBER_XXX__ALIGN_RIGHT ((uint32_t)0x00000100)

// WUFFS_BASE__RENDER_NUMBER_XXX__LEADING_PLUS_SIGN means to render the leading
// "+" for non-negative numbers: "+0" and "+12.3" instead of "0" and "12.3".
#define WUFFS_BASE__RENDER_NUMBER_XXX__LEADING_PLUS_SIGN ((uint32_t)0x00000200)

// WUFFS_BASE__RENDER_NUMBER_FXX__DECIMAL_SEPARATOR_IS_A_COMMA means to render
// one-and-a-half as "1,5" instead of "1.5".
#define WUFFS_BASE__RENDER_NUMBER_FXX__DECIMAL_SEPARATOR_IS_A_COMMA \
  ((uint32_t)0x00001000)

// WUFFS_BASE__RENDER_NUMBER_FXX__EXPONENT_ETC means whether to never
// (EXPONENT_ABSENT, equivalent to printf's "%f") or to always
// (EXPONENT_PRESENT, equivalent to printf's "%e") render a floating point
// number as "1.23e+05" instead of "123000".
//
// Having both bits set is the same has having neither bit set, where the
// notation used depends on whether the exponent is sufficiently large: "0.5"
// is preferred over "5e-01" but "5e-09" is preferred over "0.000000005".
#define WUFFS_BASE__RENDER_NUMBER_FXX__EXPONENT_ABSENT ((uint32_t)0x00002000)
#define WUFFS_BASE__RENDER_NUMBER_FXX__EXPONENT_PRESENT ((uint32_t)0x00004000)

// WUFFS_BASE__RENDER_NUMBER_FXX__JUST_ENOUGH_PRECISION means to render the
// smallest number of digits so that parsing the resultant string will recover
// the same double-precision floating point number.
//
// For example, double-precision cannot distinguish between 0.3 and
// 0.299999999999999988897769753748434595763683319091796875, so when this bit
// is set, rendering the latter will produce "0.3" but rendering
// 0.3000000000000000444089209850062616169452667236328125 will produce
// "0.30000000000000004".
#define WUFFS_BASE__RENDER_NUMBER_FXX__JUST_ENOUGH_PRECISION \
  ((uint32_t)0x00008000)

// ---------------- IEEE 754 Floating Point

// wuffs_base__ieee_754_bit_representation__etc converts between a double
// precision numerical value and its IEEE 754 representations:
//  - 16-bit: 1 sign bit,  5 exponent bits, 10 explicit significand bits.
//  - 32-bit: 1 sign bit,  8 exponent bits, 23 explicit significand bits.
//  - 64-bit: 1 sign bit, 11 exponent bits, 52 explicit significand bits.
//
// For example, it converts between:
//  - +1.0 and 0x3C00, 0x3F80_0000 or 0x3FF0_0000_0000_0000.
//  - +5.5 and 0x4580, 0x40B0_0000 or 0x4016_0000_0000_0000.
//  - -inf and 0xFC00, 0xFF80_0000 or 0xFFF0_0000_0000_0000.
//
// Converting from f64 to shorter formats (f16 or f32, represented in C as
// uint16_t and uint32_t) may be lossy. Such functions have names that look
// like etc_truncate, as converting finite numbers produce equal or smaller
// (closer-to-zero) finite numbers. For example, 1048576.0 is a perfectly valid
// f64 number, but converting it to a f16 (with truncation) produces 65504.0,
// the largest finite f16 number. Truncating a f64-typed value d to f32 does
// not always produce the same result as the C-style cast ((float)d), as
// casting can convert from finite numbers to infinite ones.
//
// Converting infinities or NaNs produces infinities or NaNs and always report
// no loss, even though there a multiple NaN representations so that round-
// tripping a f64-typed NaN may produce a different 64 bits. Nonetheless, the
// etc_truncate functions preserve a NaN's "quiet vs signaling" bit.
//
// See https://en.wikipedia.org/wiki/Double-precision_floating-point_format

typedef struct wuffs_base__lossy_value_u16__struct {
  uint16_t value;
  bool lossy;
} wuffs_base__lossy_value_u16;

typedef struct wuffs_base__lossy_value_u32__struct {
  uint32_t value;
  bool lossy;
} wuffs_base__lossy_value_u32;

WUFFS_BASE__MAYBE_STATIC wuffs_base__lossy_value_u16  //
wuffs_base__ieee_754_bit_representation__from_f64_to_u16_truncate(double f);

WUFFS_BASE__MAYBE_STATIC wuffs_base__lossy_value_u32  //
wuffs_base__ieee_754_bit_representation__from_f64_to_u32_truncate(double f);

static inline uint64_t  //
wuffs_base__ieee_754_bit_representation__from_f64_to_u64(double f) {
  uint64_t u = 0;
  if (sizeof(uint64_t) == sizeof(double)) {
    memcpy(&u, &f, sizeof(uint64_t));
  }
  return u;
}

static inline double  //
wuffs_base__ieee_754_bit_representation__from_u16_to_f64(uint16_t u) {
  uint64_t v = ((uint64_t)(u & 0x8000)) << 48;

  do {
    uint64_t exp = (u >> 10) & 0x1F;
    uint64_t man = u & 0x3FF;
    if (exp == 0x1F) {  // Infinity or NaN.
      exp = 2047;
    } else if (exp != 0) {  // Normal.
      exp += 1008;          // 1008 = 1023 - 15, the difference in biases.
    } else if (man != 0) {  // Subnormal but non-zero.
      uint32_t clz = wuffs_base__count_leading_zeroes_u64(man);
      exp = 1062 - clz;  // 1062 = 1008 + 64 - 10.
      man = 0x3FF & (man << (clz - 53));
    } else {  // Zero.
      break;
    }
    v |= (exp << 52) | (man << 42);
  } while (0);

  double f = 0;
  if (sizeof(uint64_t) == sizeof(double)) {
    memcpy(&f, &v, sizeof(uint64_t));
  }
  return f;
}

static inline double  //
wuffs_base__ieee_754_bit_representation__from_u32_to_f64(uint32_t u) {
  float f = 0;
  if (sizeof(uint32_t) == sizeof(float)) {
    memcpy(&f, &u, sizeof(uint32_t));
  }
  return (double)f;
}

static inline double  //
wuffs_base__ieee_754_bit_representation__from_u64_to_f64(uint64_t u) {
  double f = 0;
  if (sizeof(uint64_t) == sizeof(double)) {
    memcpy(&f, &u, sizeof(uint64_t));
  }
  return f;
}

// ---------------- Parsing and Rendering Numbers

// wuffs_base__parse_number_f64 parses the floating point number in s. For
// example, if s contains the bytes "1.5" then it will return the double 1.5.
//
// It returns an error if s does not contain a floating point number.
//
// It does not necessarily return an error if the conversion is lossy, e.g. if
// s is "0.3", which double-precision floating point cannot represent exactly.
//
// Similarly, the returned value may be infinite (and no error returned) even
// if s was not "inf", when the input is nominally finite but sufficiently
// larger than DBL_MAX, about 1.8e+308.
//
// It is similar to the C standard library's strtod function, but:
//  - Errors are returned in-band (in a result type), not out-of-band (errno).
//  - It takes a slice (a pointer and length), not a NUL-terminated C string.
//  - It does not take an optional endptr argument. It does not allow a partial
//    parse: it returns an error unless all of s is consumed.
//  - It does not allow whitespace, leading or otherwise.
//  - It does not allow hexadecimal floating point numbers.
//  - It is not affected by i18n / l10n settings such as environment variables.
//
// The options argument can change these, but by default, it:
//  - Allows "inf", "+Infinity" and "-NAN", case insensitive. Similarly,
//    without an explicit opt-out, it would successfully parse "1e999" as
//    infinity, even though it overflows double-precision floating point.
//  - Rejects underscores. With an explicit opt-in, "_3.141_592" would
//    successfully parse as an approximation to π.
//  - Rejects unnecessary leading zeroes: "00", "0644" and "00.7".
//  - Uses a dot '1.5' instead of a comma '1,5' for the decimal separator.
//
// For modular builds that divide the base module into sub-modules, using this
// function requires the WUFFS_CONFIG__MODULE__BASE__FLOATCONV sub-module, not
// just WUFFS_CONFIG__MODULE__BASE__CORE.
WUFFS_BASE__MAYBE_STATIC wuffs_base__result_f64  //
wuffs_base__parse_number_f64(wuffs_base__slice_u8 s, uint32_t options);

// wuffs_base__parse_number_i64 parses the ASCII integer in s. For example, if
// s contains the bytes "-123" then it will return the int64_t -123.
//
// It returns an error if s does not contain an integer or if the integer
// within would overflow an int64_t.
//
// It is similar to wuffs_base__parse_number_u64 but it returns a signed
// integer, not an unsigned integer. It also allows a leading '+' or '-'.
//
// For modular builds that divide the base module into sub-modules, using this
// function requires the WUFFS_CONFIG__MODULE__BASE__INTCONV sub-module, not
// just WUFFS_CONFIG__MODULE__BASE__CORE.
WUFFS_BASE__MAYBE_STATIC wuffs_base__result_i64  //
wuffs_base__parse_number_i64(wuffs_base__slice_u8 s, uint32_t options);

// wuffs_base__parse_number_u64 parses the ASCII integer in s. For example, if
// s contains the bytes "123" then it will return the uint64_t 123.
//
// It returns an error if s does not contain an integer or if the integer
// within would overflow a uint64_t.
//
// It is similar to the C standard library's strtoull function, but:
//  - Errors are returned in-band (in a result type), not out-of-band (errno).
//  - It takes a slice (a pointer and length), not a NUL-terminated C string.
//  - It does not take an optional endptr argument. It does not allow a partial
//    parse: it returns an error unless all of s is consumed.
//  - It does not allow whitespace, leading or otherwise.
//  - It does not allow a leading '+' or '-'.
//  - It does not take a base argument (e.g. base 10 vs base 16). Instead, it
//    always accepts both decimal (e.g "1234", "0d5678") and hexadecimal (e.g.
//    "0x9aBC"). The caller is responsible for prior filtering of e.g. hex
//    numbers if they are unwanted. For example, Wuffs' JSON decoder will only
//    produce a wuffs_base__token for decimal numbers, not hexadecimal.
//  - It is not affected by i18n / l10n settings such as environment variables.
//
// The options argument can change these, but by default, it:
//  - Rejects underscores. With an explicit opt-in, "__0D_1_002" would
//    successfully parse as "one thousand and two". Underscores are still
//    rejected inside the optional 2-byte opening "0d" or "0X" that denotes
//    base-10 or base-16.
//  - Rejects unnecessary leading zeroes: "00" and "0644".
//
// For modular builds that divide the base module into sub-modules, using this
// function requires the WUFFS_CONFIG__MODULE__BASE__INTCONV sub-module, not
// just WUFFS_CONFIG__MODULE__BASE__CORE.
WUFFS_BASE__MAYBE_STATIC wuffs_base__result_u64  //
wuffs_base__parse_number_u64(wuffs_base__slice_u8 s, uint32_t options);

// --------

// WUFFS_BASE__I64__BYTE_LENGTH__MAX_INCL is the string length of
// "-9223372036854775808" and "+9223372036854775807", INT64_MIN and INT64_MAX.
#define WUFFS_BASE__I64__BYTE_LENGTH__MAX_INCL 20

// WUFFS_BASE__U64__BYTE_LENGTH__MAX_INCL is the string length of
// "+18446744073709551615", UINT64_MAX.
#define WUFFS_BASE__U64__BYTE_LENGTH__MAX_INCL 21

// wuffs_base__render_number_f64 writes the decimal encoding of x to dst and
// returns the number of bytes written. If dst is shorter than the entire
// encoding, it returns 0 (and no bytes are written).
//
// For those familiar with C's printf or Go's fmt.Printf functions:
//  - "%e" means the WUFFS_BASE__RENDER_NUMBER_FXX__EXPONENT_PRESENT option.
//  - "%f" means the WUFFS_BASE__RENDER_NUMBER_FXX__EXPONENT_ABSENT  option.
//  - "%g" means neither or both bits are set.
//
// The precision argument controls the number of digits rendered, excluding the
// exponent (the "e+05" in "1.23e+05"):
//  - for "%e" and "%f" it is the number of digits after the decimal separator,
//  - for "%g" it is the number of significant digits (and trailing zeroes are
//    removed).
//
// A precision of 6 gives similar output to printf's defaults.
//
// A precision greater than 4095 is equivalent to 4095.
//
// The precision argument is ignored when the
// WUFFS_BASE__RENDER_NUMBER_FXX__JUST_ENOUGH_PRECISION option is set. This is
// similar to Go's strconv.FormatFloat with a negative (i.e. non-sensical)
// precision, but there is no corresponding feature in C's printf.
//
// Extreme values of x will be rendered as "NaN", "Inf" (or "+Inf" if the
// WUFFS_BASE__RENDER_NUMBER_XXX__LEADING_PLUS_SIGN option is set) or "-Inf".
//
// For modular builds that divide the base module into sub-modules, using this
// function requires the WUFFS_CONFIG__MODULE__BASE__FLOATCONV sub-module, not
// just WUFFS_CONFIG__MODULE__BASE__CORE.
WUFFS_BASE__MAYBE_STATIC size_t  //
wuffs_base__render_number_f64(wuffs_base__slice_u8 dst,
                              double x,
                              uint32_t precision,
                              uint32_t options);

// wuffs_base__render_number_i64 writes the decimal encoding of x to dst and
// returns the number of bytes written. If dst is shorter than the entire
// encoding, it returns 0 (and no bytes are written).
//
// dst will never be too short if its length is at least 20, also known as
// WUFFS_BASE__I64__BYTE_LENGTH__MAX_INCL.
//
// For modular builds that divide the base module into sub-modules, using this
// function requires the WUFFS_CONFIG__MODULE__BASE__INTCONV sub-module, not
// just WUFFS_CONFIG__MODULE__BASE__CORE.
WUFFS_BASE__MAYBE_STATIC size_t  //
wuffs_base__render_number_i64(wuffs_base__slice_u8 dst,
                              int64_t x,
                              uint32_t options);

// wuffs_base__render_number_u64 writes the decimal encoding of x to dst and
// returns the number of bytes written. If dst is shorter than the entire
// encoding, it returns 0 (and no bytes are written).
//
// dst will never be too short if its length is at least 21, also known as
// WUFFS_BASE__U64__BYTE_LENGTH__MAX_INCL.
//
// For modular builds that divide the base module into sub-modules, using this
// function requires the WUFFS_CONFIG__MODULE__BASE__INTCONV sub-module, not
// just WUFFS_CONFIG__MODULE__BASE__CORE.
WUFFS_BASE__MAYBE_STATIC size_t  //
wuffs_base__render_number_u64(wuffs_base__slice_u8 dst,
                              uint64_t x,
                              uint32_t options);

// ---------------- Base-16

// Options (bitwise or'ed together) for wuffs_base__base_16__xxx functions.

#define WUFFS_BASE__BASE_16__DEFAULT_OPTIONS ((uint32_t)0x00000000)

// wuffs_base__base_16__decode2 converts "6A6b" to "jk", where e.g. 'j' is
// U+006A. There are 2 src bytes for every dst byte.
//
// It assumes that the src bytes are two hexadecimal digits (0-9, A-F, a-f),
// repeated. It may write nonsense bytes if not, although it will not read or
// write out of bounds.
//
// For modular builds that divide the base module into sub-modules, using this
// function requires the WUFFS_CONFIG__MODULE__BASE__INTCONV sub-module, not
// just WUFFS_CONFIG__MODULE__BASE__CORE.
WUFFS_BASE__MAYBE_STATIC wuffs_base__transform__output  //
wuffs_base__base_16__decode2(wuffs_base__slice_u8 dst,
                             wuffs_base__slice_u8 src,
                             bool src_closed,
                             uint32_t options);

// wuffs_base__base_16__decode4 converts both "\\x6A\\x6b" and "??6a??6B" to
// "jk", where e.g. 'j' is U+006A. There are 4 src bytes for every dst byte.
//
// It assumes that the src bytes are two ignored bytes and then two hexadecimal
// digits (0-9, A-F, a-f), repeated. It may write nonsense bytes if not,
// although it will not read or write out of bounds.
//
// For modular builds that divide the base module into sub-modules, using this
// function requires the WUFFS_CONFIG__MODULE__BASE__INTCONV sub-module, not
// just WUFFS_CONFIG__MODULE__BASE__CORE.
WUFFS_BASE__MAYBE_STATIC wuffs_base__transform__output  //
wuffs_base__base_16__decode4(wuffs_base__slice_u8 dst,
                             wuffs_base__slice_u8 src,
                             bool src_closed,
                             uint32_t options);

// wuffs_base__base_16__encode2 converts "jk" to "6A6B", where e.g. 'j' is
// U+006A. There are 2 dst bytes for every src byte.
//
// For modular builds that divide the base module into sub-modules, using this
// function requires the WUFFS_CONFIG__MODULE__BASE__INTCONV sub-module, not
// just WUFFS_CONFIG__MODULE__BASE__CORE.
WUFFS_BASE__MAYBE_STATIC wuffs_base__transform__output  //
wuffs_base__base_16__encode2(wuffs_base__slice_u8 dst,
                             wuffs_base__slice_u8 src,
                             bool src_closed,
                             uint32_t options);

// wuffs_base__base_16__encode4 converts "jk" to "\\x6A\\x6B", where e.g. 'j'
// is U+006A. There are 4 dst bytes for every src byte.
//
// For modular builds that divide the base module into sub-modules, using this
// function requires the WUFFS_CONFIG__MODULE__BASE__INTCONV sub-module, not
// just WUFFS_CONFIG__MODULE__BASE__CORE.
WUFFS_BASE__MAYBE_STATIC wuffs_base__transform__output  //
wuffs_base__base_16__encode2(wuffs_base__slice_u8 dst,
                             wuffs_base__slice_u8 src,
                             bool src_closed,
                             uint32_t options);

// ---------------- Base-64

// Options (bitwise or'ed together) for wuffs_base__base_64__xxx functions.

#define WUFFS_BASE__BASE_64__DEFAULT_OPTIONS ((uint32_t)0x00000000)

// WUFFS_BASE__BASE_64__DECODE_ALLOW_PADDING means that, when decoding base-64,
// the input may (but does not need to) be padded with '=' bytes so that the
// overall encoded length in bytes is a multiple of 4. A successful decoding
// will return a num_src that includes those padding bytes.
//
// Excess padding (e.g. three final '='s) will be rejected as bad data.
#define WUFFS_BASE__BASE_64__DECODE_ALLOW_PADDING ((uint32_t)0x00000001)

// WUFFS_BASE__BASE_64__ENCODE_EMIT_PADDING means that, when encoding base-64,
// the output will be padded with '=' bytes so that the overall encoded length
// in bytes is a multiple of 4.
#define WUFFS_BASE__BASE_64__ENCODE_EMIT_PADDING ((uint32_t)0x00000002)

// WUFFS_BASE__BASE_64__URL_ALPHABET means that, for base-64, the URL-friendly
// and file-name-friendly alphabet be used, as per RFC 4648 section 5. When
// this option bit is off, the standard alphabet from section 4 is used.
#define WUFFS_BASE__BASE_64__URL_ALPHABET ((uint32_t)0x00000100)

// wuffs_base__base_64__decode transforms base-64 encoded bytes from src to
// arbitrary bytes in dst.
//
// It will not permit line breaks or other whitespace in src. Filtering those
// out is the responsibility of the caller.
//
// For modular builds that divide the base module into sub-modules, using this
// function requires the WUFFS_CONFIG__MODULE__BASE__INTCONV sub-module, not
// just WUFFS_CONFIG__MODULE__BASE__CORE.
WUFFS_BASE__MAYBE_STATIC wuffs_base__transform__output  //
wuffs_base__base_64__decode(wuffs_base__slice_u8 dst,
                            wuffs_base__slice_u8 src,
                            bool src_closed,
                            uint32_t options);

// wuffs_base__base_64__encode transforms arbitrary bytes from src to base-64
// encoded bytes in dst.
//
// For modular builds that divide the base module into sub-modules, using this
// function requires the WUFFS_CONFIG__MODULE__BASE__INTCONV sub-module, not
// just WUFFS_CONFIG__MODULE__BASE__CORE.
WUFFS_BASE__MAYBE_STATIC wuffs_base__transform__output  //
wuffs_base__base_64__encode(wuffs_base__slice_u8 dst,
                            wuffs_base__slice_u8 src,
                            bool src_closed,
                            uint32_t options);

// ---------------- Unicode and UTF-8

#define WUFFS_BASE__UNICODE_CODE_POINT__MIN_INCL 0x00000000
#define WUFFS_BASE__UNICODE_CODE_POINT__MAX_INCL 0x0010FFFF

#define WUFFS_BASE__UNICODE_REPLACEMENT_CHARACTER 0x0000FFFD

#define WUFFS_BASE__UNICODE_SURROGATE__MIN_INCL 0x0000D800
#define WUFFS_BASE__UNICODE_SURROGATE__MAX_INCL 0x0000DFFF

#define WUFFS_BASE__ASCII__MIN_INCL 0x00
#define WUFFS_BASE__ASCII__MAX_INCL 0x7F

#define WUFFS_BASE__UTF_8__BYTE_LENGTH__MIN_INCL 1
#define WUFFS_BASE__UTF_8__BYTE_LENGTH__MAX_INCL 4

#define WUFFS_BASE__UTF_8__BYTE_LENGTH_1__CODE_POINT__MIN_INCL 0x00000000
#define WUFFS_BASE__UTF_8__BYTE_LENGTH_1__CODE_POINT__MAX_INCL 0x0000007F
#define WUFFS_BASE__UTF_8__BYTE_LENGTH_2__CODE_POINT__MIN_INCL 0x00000080
#define WUFFS_BASE__UTF_8__BYTE_LENGTH_2__CODE_POINT__MAX_INCL 0x000007FF
#define WUFFS_BASE__UTF_8__BYTE_LENGTH_3__CODE_POINT__MIN_INCL 0x00000800
#define WUFFS_BASE__UTF_8__BYTE_LENGTH_3__CODE_POINT__MAX_INCL 0x0000FFFF
#define WUFFS_BASE__UTF_8__BYTE_LENGTH_4__CODE_POINT__MIN_INCL 0x00010000
#define WUFFS_BASE__UTF_8__BYTE_LENGTH_4__CODE_POINT__MAX_INCL 0x0010FFFF

// --------

// wuffs_base__utf_8__next__output is the type returned by
// wuffs_base__utf_8__next.
typedef struct wuffs_base__utf_8__next__output__struct {
  uint32_t code_point;
  uint32_t byte_length;

#ifdef __cplusplus
  inline bool is_valid() const;
#endif  // __cplusplus

} wuffs_base__utf_8__next__output;

static inline wuffs_base__utf_8__next__output  //
wuffs_base__make_utf_8__next__output(uint32_t code_point,
                                     uint32_t byte_length) {
  wuffs_base__utf_8__next__output ret;
  ret.code_point = code_point;
  ret.byte_length = byte_length;
  return ret;
}

static inline bool  //
wuffs_base__utf_8__next__output__is_valid(
    const wuffs_base__utf_8__next__output* o) {
  if (o) {
    uint32_t cp = o->code_point;
    switch (o->byte_length) {
      case 1:
        return (cp <= 0x7F);
      case 2:
        return (0x080 <= cp) && (cp <= 0x7FF);
      case 3:
        // Avoid the 0xD800 ..= 0xDFFF surrogate range.
        return ((0x0800 <= cp) && (cp <= 0xD7FF)) ||
               ((0xE000 <= cp) && (cp <= 0xFFFF));
      case 4:
        return (0x00010000 <= cp) && (cp <= 0x0010FFFF);
    }
  }
  return false;
}

#ifdef __cplusplus

inline bool  //
wuffs_base__utf_8__next__output::is_valid() const {
  return wuffs_base__utf_8__next__output__is_valid(this);
}

#endif  // __cplusplus

// --------

// wuffs_base__utf_8__encode writes the UTF-8 encoding of code_point to s and
// returns the number of bytes written. If code_point is invalid, or if s is
// shorter than the entire encoding, it returns 0 (and no bytes are written).
//
// s will never be too short if its length is at least 4, also known as
// WUFFS_BASE__UTF_8__BYTE_LENGTH__MAX_INCL.
//
// For modular builds that divide the base module into sub-modules, using this
// function requires the WUFFS_CONFIG__MODULE__BASE__UTF8 sub-module, not just
// WUFFS_CONFIG__MODULE__BASE__CORE.
WUFFS_BASE__MAYBE_STATIC size_t  //
wuffs_base__utf_8__encode(wuffs_base__slice_u8 dst, uint32_t code_point);

// wuffs_base__utf_8__next returns the next UTF-8 code point (and that code
// point's byte length) at the start of the read-only slice (s_ptr, s_len).
//
// There are exactly two cases in which this function returns something where
// wuffs_base__utf_8__next__output__is_valid is false:
//  - If s is empty then it returns {.code_point=0, .byte_length=0}.
//  - If s is non-empty and starts with invalid UTF-8 then it returns
//    {.code_point=WUFFS_BASE__UNICODE_REPLACEMENT_CHARACTER, .byte_length=1}.
//
// Otherwise, it returns something where
// wuffs_base__utf_8__next__output__is_valid is true.
//
// In any case, it always returns an output that satisfies both of:
//  - (output.code_point  <= WUFFS_BASE__UNICODE_CODE_POINT__MAX_INCL).
//  - (output.byte_length <= s_len).
//
// If s is a sub-slice of a larger slice of valid UTF-8, but that sub-slice
// boundary occurs in the middle of a multi-byte UTF-8 encoding of a single
// code point, then this function may return something invalid. It is the
// caller's responsibility to split on or otherwise manage UTF-8 boundaries.
//
// For modular builds that divide the base module into sub-modules, using this
// function requires the WUFFS_CONFIG__MODULE__BASE__UTF8 sub-module, not just
// WUFFS_CONFIG__MODULE__BASE__CORE.
WUFFS_BASE__MAYBE_STATIC wuffs_base__utf_8__next__output  //
wuffs_base__utf_8__next(const uint8_t* s_ptr, size_t s_len);

// wuffs_base__utf_8__next_from_end is like wuffs_base__utf_8__next except that
// it looks at the end of (s_ptr, s_len) instead of the start.
//
// For modular builds that divide the base module into sub-modules, using this
// function requires the WUFFS_CONFIG__MODULE__BASE__UTF8 sub-module, not just
// WUFFS_CONFIG__MODULE__BASE__CORE.
WUFFS_BASE__MAYBE_STATIC wuffs_base__utf_8__next__output  //
wuffs_base__utf_8__next_from_end(const uint8_t* s_ptr, size_t s_len);

// wuffs_base__utf_8__longest_valid_prefix returns the largest n such that the
// sub-slice s[..n] is valid UTF-8, where s is the read-only slice (s_ptr,
// s_len).
//
// In particular, it returns s_len if and only if all of s is valid UTF-8.
//
// If s is a sub-slice of a larger slice of valid UTF-8, but that sub-slice
// boundary occurs in the middle of a multi-byte UTF-8 encoding of a single
// code point, then this function will return less than s_len. It is the
// caller's responsibility to split on or otherwise manage UTF-8 boundaries.
//
// For modular builds that divide the base module into sub-modules, using this
// function requires the WUFFS_CONFIG__MODULE__BASE__UTF8 sub-module, not just
// WUFFS_CONFIG__MODULE__BASE__CORE.
WUFFS_BASE__MAYBE_STATIC size_t  //
wuffs_base__utf_8__longest_valid_prefix(const uint8_t* s_ptr, size_t s_len);

// wuffs_base__ascii__longest_valid_prefix returns the largest n such that the
// sub-slice s[..n] is valid ASCII, where s is the read-only slice (s_ptr,
// s_len).
//
// In particular, it returns s_len if and only if all of s is valid ASCII.
// Equivalently, when none of the bytes in s have the 0x80 high bit set.
//
// For modular builds that divide the base module into sub-modules, using this
// function requires the WUFFS_CONFIG__MODULE__BASE__UTF8 sub-module, not just
// WUFFS_CONFIG__MODULE__BASE__CORE.
WUFFS_BASE__MAYBE_STATIC size_t  //
wuffs_base__ascii__longest_valid_prefix(const uint8_t* s_ptr, size_t s_len);

// ---------------- Interface Declarations.

// For modular builds that divide the base module into sub-modules, using these
// functions require the WUFFS_CONFIG__MODULE__BASE__INTERFACES sub-module, not
// just WUFFS_CONFIG__MODULE__BASE__CORE.

// --------

extern const char wuffs_base__hasher_u32__vtable_name[];

typedef struct wuffs_base__hasher_u32__func_ptrs__struct {
  wuffs_base__empty_struct (*set_quirk_enabled)(
    void* self,
    uint32_t a_quirk,
    bool a_enabled);
  uint32_t (*update_u32)(
    void* self,
    wuffs_base__slice_u8 a_x);
} wuffs_base__hasher_u32__func_ptrs;

typedef struct wuffs_base__hasher_u32__struct wuffs_base__hasher_u32;

WUFFS_BASE__MAYBE_STATIC wuffs_base__empty_struct
wuffs_base__hasher_u32__set_quirk_enabled(
    wuffs_base__hasher_u32* self,
    uint32_t a_quirk,
    bool a_enabled);

WUFFS_BASE__MAYBE_STATIC uint32_t
wuffs_base__hasher_u32__update_u32(
    wuffs_base__hasher_u32* self,
    wuffs_base__slice_u8 a_x);

#if defined(__cplusplus) || defined(WUFFS_IMPLEMENTATION)

struct wuffs_base__hasher_u32__struct {
  struct {
    uint32_t magic;
    uint32_t active_coroutine;
    wuffs_base__vtable first_vtable;
  } private_impl;

#ifdef __cplusplus
#if defined(WUFFS_BASE__HAVE_UNIQUE_PTR)
  using unique_ptr = std::unique_ptr<wuffs_base__hasher_u32, decltype(&free)>;
#endif

  inline wuffs_base__empty_struct
  set_quirk_enabled(
      uint32_t a_quirk,
      bool a_enabled) {
    return wuffs_base__hasher_u32__set_quirk_enabled(
        this, a_quirk, a_enabled);
  }

  inline uint32_t
  update_u32(
      wuffs_base__slice_u8 a_x) {
    return wuffs_base__hasher_u32__update_u32(
        this, a_x);
  }

#endif  // __cplusplus
};  // struct wuffs_base__hasher_u32__struct

#endif  // defined(__cplusplus) || defined(WUFFS_IMPLEMENTATION)

// --------

extern const char wuffs_base__image_decoder__vtable_name[];

typedef struct wuffs_base__image_decoder__func_ptrs__struct {
  wuffs_base__status (*decode_frame)(
    void* self,
    wuffs_base__pixel_buffer* a_dst,
    wuffs_base__io_buffer* a_src,
    wuffs_base__pixel_blend a_blend,
    wuffs_base__slice_u8 a_workbuf,
    wuffs_base__decode_frame_options* a_opts);
  wuffs_base__status (*decode_frame_config)(
    void* self,
    wuffs_base__frame_config* a_dst,
    wuffs_base__io_buffer* a_src);
  wuffs_base__status (*decode_image_config)(
    void* self,
    wuffs_base__image_config* a_dst,
    wuffs_base__io_buffer* a_src);
  wuffs_base__rect_ie_u32 (*frame_dirty_rect)(
    const void* self);
  uint32_t (*num_animation_loops)(
    const void* self);
  uint64_t (*num_decoded_frame_configs)(
    const void* self);
  uint64_t (*num_decoded_frames)(
    const void* self);
  wuffs_base__status (*restart_frame)(
    void* self,
    uint64_t a_index,
    uint64_t a_io_position);
  wuffs_base__empty_struct (*set_quirk_enabled)(
    void* self,
    uint32_t a_quirk,
    bool a_enabled);
  wuffs_base__empty_struct (*set_report_metadata)(
    void* self,
    uint32_t a_fourcc,
    bool a_report);
  wuffs_base__status (*tell_me_more)(
    void* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__more_information* a_minfo,
    wuffs_base__io_buffer* a_src);
  wuffs_base__range_ii_u64 (*workbuf_len)(
    const void* self);
} wuffs_base__image_decoder__func_ptrs;

typedef struct wuffs_base__image_decoder__struct wuffs_base__image_decoder;

WUFFS_BASE__MAYBE_STATIC wuffs_base__status
wuffs_base__image_decoder__decode_frame(
    wuffs_base__image_decoder* self,
    wuffs_base__pixel_buffer* a_dst,
    wuffs_base__io_buffer* a_src,
    wuffs_base__pixel_blend a_blend,
    wuffs_base__slice_u8 a_workbuf,
    wuffs_base__decode_frame_options* a_opts);

WUFFS_BASE__MAYBE_STATIC wuffs_base__status
wuffs_base__image_decoder__decode_frame_config(
    wuffs_base__image_decoder* self,
    wuffs_base__frame_config* a_dst,
    wuffs_base__io_buffer* a_src);

WUFFS_BASE__MAYBE_STATIC wuffs_base__status
wuffs_base__image_decoder__decode_image_config(
    wuffs_base__image_decoder* self,
    wuffs_base__image_config* a_dst,
    wuffs_base__io_buffer* a_src);

WUFFS_BASE__MAYBE_STATIC wuffs_base__rect_ie_u32
wuffs_base__image_decoder__frame_dirty_rect(
    const wuffs_base__image_decoder* self);

WUFFS_BASE__MAYBE_STATIC uint32_t
wuffs_base__image_decoder__num_animation_loops(
    const wuffs_base__image_decoder* self);

WUFFS_BASE__MAYBE_STATIC uint64_t
wuffs_base__image_decoder__num_decoded_frame_configs(
    const wuffs_base__image_decoder* self);

WUFFS_BASE__MAYBE_STATIC uint64_t
wuffs_base__image_decoder__num_decoded_frames(
    const wuffs_base__image_decoder* self);

WUFFS_BASE__MAYBE_STATIC wuffs_base__status
wuffs_base__image_decoder__restart_frame(
    wuffs_base__image_decoder* self,
    uint64_t a_index,
    uint64_t a_io_position);

WUFFS_BASE__MAYBE_STATIC wuffs_base__empty_struct
wuffs_base__image_decoder__set_quirk_enabled(
    wuffs_base__image_decoder* self,
    uint32_t a_quirk,
    bool a_enabled);

WUFFS_BASE__MAYBE_STATIC wuffs_base__empty_struct
wuffs_base__image_decoder__set_report_metadata(
    wuffs_base__image_decoder* self,
    uint32_t a_fourcc,
    bool a_report);

WUFFS_BASE__MAYBE_STATIC wuffs_base__status
wuffs_base__image_decoder__tell_me_more(
    wuffs_base__image_decoder* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__more_information* a_minfo,
    wuffs_base__io_buffer* a_src);

WUFFS_BASE__MAYBE_STATIC wuffs_base__range_ii_u64
wuffs_base__image_decoder__workbuf_len(
    const wuffs_base__image_decoder* self);

#if defined(__cplusplus) || defined(WUFFS_IMPLEMENTATION)

struct wuffs_base__image_decoder__struct {
  struct {
    uint32_t magic;
    uint32_t active_coroutine;
    wuffs_base__vtable first_vtable;
  } private_impl;

#ifdef __cplusplus
#if defined(WUFFS_BASE__HAVE_UNIQUE_PTR)
  using unique_ptr = std::unique_ptr<wuffs_base__image_decoder, decltype(&free)>;
#endif

  inline wuffs_base__status
  decode_frame(
      wuffs_base__pixel_buffer* a_dst,
      wuffs_base__io_buffer* a_src,
      wuffs_base__pixel_blend a_blend,
      wuffs_base__slice_u8 a_workbuf,
      wuffs_base__decode_frame_options* a_opts) {
    return wuffs_base__image_decoder__decode_frame(
        this, a_dst, a_src, a_blend, a_workbuf, a_opts);
  }

  inline wuffs_base__status
  decode_frame_config(
      wuffs_base__frame_config* a_dst,
      wuffs_base__io_buffer* a_src) {
    return wuffs_base__image_decoder__decode_frame_config(
        this, a_dst, a_src);
  }

  inline wuffs_base__status
  decode_image_config(
      wuffs_base__image_config* a_dst,
      wuffs_base__io_buffer* a_src) {
    return wuffs_base__image_decoder__decode_image_config(
        this, a_dst, a_src);
  }

  inline wuffs_base__rect_ie_u32
  frame_dirty_rect() const {
    return wuffs_base__image_decoder__frame_dirty_rect(this);
  }

  inline uint32_t
  num_animation_loops() const {
    return wuffs_base__image_decoder__num_animation_loops(this);
  }

  inline uint64_t
  num_decoded_frame_configs() const {
    return wuffs_base__image_decoder__num_decoded_frame_configs(this);
  }

  inline uint64_t
  num_decoded_frames() const {
    return wuffs_base__image_decoder__num_decoded_frames(this);
  }

  inline wuffs_base__status
  restart_frame(
      uint64_t a_index,
      uint64_t a_io_position) {
    return wuffs_base__image_decoder__restart_frame(
        this, a_index, a_io_position);
  }

  inline wuffs_base__empty_struct
  set_quirk_enabled(
      uint32_t a_quirk,
      bool a_enabled) {
    return wuffs_base__image_decoder__set_quirk_enabled(
        this, a_quirk, a_enabled);
  }

  inline wuffs_base__empty_struct
  set_report_metadata(
      uint32_t a_fourcc,
      bool a_report) {
    return wuffs_base__image_decoder__set_report_metadata(
        this, a_fourcc, a_report);
  }

  inline wuffs_base__status
  tell_me_more(
      wuffs_base__io_buffer* a_dst,
      wuffs_base__more_information* a_minfo,
      wuffs_base__io_buffer* a_src) {
    return wuffs_base__image_decoder__tell_me_more(
        this, a_dst, a_minfo, a_src);
  }

  inline wuffs_base__range_ii_u64
  workbuf_len() const {
    return wuffs_base__image_decoder__workbuf_len(this);
  }

#endif  // __cplusplus
};  // struct wuffs_base__image_decoder__struct

#endif  // defined(__cplusplus) || defined(WUFFS_IMPLEMENTATION)

// --------

extern const char wuffs_base__io_transformer__vtable_name[];

typedef struct wuffs_base__io_transformer__func_ptrs__struct {
  wuffs_base__empty_struct (*set_quirk_enabled)(
    void* self,
    uint32_t a_quirk,
    bool a_enabled);
  wuffs_base__status (*transform_io)(
    void* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__io_buffer* a_src,
    wuffs_base__slice_u8 a_workbuf);
  wuffs_base__range_ii_u64 (*workbuf_len)(
    const void* self);
} wuffs_base__io_transformer__func_ptrs;

typedef struct wuffs_base__io_transformer__struct wuffs_base__io_transformer;

WUFFS_BASE__MAYBE_STATIC wuffs_base__empty_struct
wuffs_base__io_transformer__set_quirk_enabled(
    wuffs_base__io_transformer* self,
    uint32_t a_quirk,
    bool a_enabled);

WUFFS_BASE__MAYBE_STATIC wuffs_base__status
wuffs_base__io_transformer__transform_io(
    wuffs_base__io_transformer* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__io_buffer* a_src,
    wuffs_base__slice_u8 a_workbuf);

WUFFS_BASE__MAYBE_STATIC wuffs_base__range_ii_u64
wuffs_base__io_transformer__workbuf_len(
    const wuffs_base__io_transformer* self);

#if defined(__cplusplus) || defined(WUFFS_IMPLEMENTATION)

struct wuffs_base__io_transformer__struct {
  struct {
    uint32_t magic;
    uint32_t active_coroutine;
    wuffs_base__vtable first_vtable;
  } private_impl;

#ifdef __cplusplus
#if defined(WUFFS_BASE__HAVE_UNIQUE_PTR)
  using unique_ptr = std::unique_ptr<wuffs_base__io_transformer, decltype(&free)>;
#endif

  inline wuffs_base__empty_struct
  set_quirk_enabled(
      uint32_t a_quirk,
      bool a_enabled) {
    return wuffs_base__io_transformer__set_quirk_enabled(
        this, a_quirk, a_enabled);
  }

  inline wuffs_base__status
  transform_io(
      wuffs_base__io_buffer* a_dst,
      wuffs_base__io_buffer* a_src,
      wuffs_base__slice_u8 a_workbuf) {
    return wuffs_base__io_transformer__transform_io(
        this, a_dst, a_src, a_workbuf);
  }

  inline wuffs_base__range_ii_u64
  workbuf_len() const {
    return wuffs_base__io_transformer__workbuf_len(this);
  }

#endif  // __cplusplus
};  // struct wuffs_base__io_transformer__struct

#endif  // defined(__cplusplus) || defined(WUFFS_IMPLEMENTATION)

// --------

extern const char wuffs_base__token_decoder__vtable_name[];

typedef struct wuffs_base__token_decoder__func_ptrs__struct {
  wuffs_base__status (*decode_tokens)(
    void* self,
    wuffs_base__token_buffer* a_dst,
    wuffs_base__io_buffer* a_src,
    wuffs_base__slice_u8 a_workbuf);
  wuffs_base__empty_struct (*set_quirk_enabled)(
    void* self,
    uint32_t a_quirk,
    bool a_enabled);
  wuffs_base__range_ii_u64 (*workbuf_len)(
    const void* self);
} wuffs_base__token_decoder__func_ptrs;

typedef struct wuffs_base__token_decoder__struct wuffs_base__token_decoder;

WUFFS_BASE__MAYBE_STATIC wuffs_base__status
wuffs_base__token_decoder__decode_tokens(
    wuffs_base__token_decoder* self,
    wuffs_base__token_buffer* a_dst,
    wuffs_base__io_buffer* a_src,
    wuffs_base__slice_u8 a_workbuf);

WUFFS_BASE__MAYBE_STATIC wuffs_base__empty_struct
wuffs_base__token_decoder__set_quirk_enabled(
    wuffs_base__token_decoder* self,
    uint32_t a_quirk,
    bool a_enabled);

WUFFS_BASE__MAYBE_STATIC wuffs_base__range_ii_u64
wuffs_base__token_decoder__workbuf_len(
    const wuffs_base__token_decoder* self);

#if defined(__cplusplus) || defined(WUFFS_IMPLEMENTATION)

struct wuffs_base__token_decoder__struct {
  struct {
    uint32_t magic;
    uint32_t active_coroutine;
    wuffs_base__vtable first_vtable;
  } private_impl;

#ifdef __cplusplus
#if defined(WUFFS_BASE__HAVE_UNIQUE_PTR)
  using unique_ptr = std::unique_ptr<wuffs_base__token_decoder, decltype(&free)>;
#endif

  inline wuffs_base__status
  decode_tokens(
      wuffs_base__token_buffer* a_dst,
      wuffs_base__io_buffer* a_src,
      wuffs_base__slice_u8 a_workbuf) {
    return wuffs_base__token_decoder__decode_tokens(
        this, a_dst, a_src, a_workbuf);
  }

  inline wuffs_base__empty_struct
  set_quirk_enabled(
      uint32_t a_quirk,
      bool a_enabled) {
    return wuffs_base__token_decoder__set_quirk_enabled(
        this, a_quirk, a_enabled);
  }

  inline wuffs_base__range_ii_u64
  workbuf_len() const {
    return wuffs_base__token_decoder__workbuf_len(this);
  }

#endif  // __cplusplus
};  // struct wuffs_base__token_decoder__struct

#endif  // defined(__cplusplus) || defined(WUFFS_IMPLEMENTATION)

// ----------------

#ifdef __cplusplus
}  // extern "C"
#endif

#if !defined(WUFFS_CONFIG__MODULES) || defined(WUFFS_CONFIG__MODULE__ADLER32) || defined(WUFFS_NONMONOLITHIC)

// ---------------- Status Codes

// ---------------- Public Consts

// ---------------- Struct Declarations

typedef struct wuffs_adler32__hasher__struct wuffs_adler32__hasher;

#ifdef __cplusplus
extern "C" {
#endif

// ---------------- Public Initializer Prototypes

// For any given "wuffs_foo__bar* self", "wuffs_foo__bar__initialize(self,
// etc)" should be called before any other "wuffs_foo__bar__xxx(self, etc)".
//
// Pass sizeof(*self) and WUFFS_VERSION for sizeof_star_self and wuffs_version.
// Pass 0 (or some combination of WUFFS_INITIALIZE__XXX) for options.

wuffs_base__status WUFFS_BASE__WARN_UNUSED_RESULT
wuffs_adler32__hasher__initialize(
    wuffs_adler32__hasher* self,
    size_t sizeof_star_self,
    uint64_t wuffs_version,
    uint32_t options);

size_t
sizeof__wuffs_adler32__hasher();

// ---------------- Allocs

// These functions allocate and initialize Wuffs structs. They return NULL if
// memory allocation fails. If they return non-NULL, there is no need to call
// wuffs_foo__bar__initialize, but the caller is responsible for eventually
// calling free on the returned pointer. That pointer is effectively a C++
// std::unique_ptr<T, decltype(&free)>.

wuffs_adler32__hasher*
wuffs_adler32__hasher__alloc();

static inline wuffs_base__hasher_u32*
wuffs_adler32__hasher__alloc_as__wuffs_base__hasher_u32() {
  return (wuffs_base__hasher_u32*)(wuffs_adler32__hasher__alloc());
}

// ---------------- Upcasts

static inline wuffs_base__hasher_u32*
wuffs_adler32__hasher__upcast_as__wuffs_base__hasher_u32(
    wuffs_adler32__hasher* p) {
  return (wuffs_base__hasher_u32*)p;
}

// ---------------- Public Function Prototypes

WUFFS_BASE__MAYBE_STATIC wuffs_base__empty_struct
wuffs_adler32__hasher__set_quirk_enabled(
    wuffs_adler32__hasher* self,
    uint32_t a_quirk,
    bool a_enabled);

WUFFS_BASE__MAYBE_STATIC uint32_t
wuffs_adler32__hasher__update_u32(
    wuffs_adler32__hasher* self,
    wuffs_base__slice_u8 a_x);

#ifdef __cplusplus
}  // extern "C"
#endif

// ---------------- Struct Definitions

// These structs' fields, and the sizeof them, are private implementation
// details that aren't guaranteed to be stable across Wuffs versions.
//
// See https://en.wikipedia.org/wiki/Opaque_pointer#C

#if defined(__cplusplus) || defined(WUFFS_IMPLEMENTATION)

struct wuffs_adler32__hasher__struct {
  // Do not access the private_impl's or private_data's fields directly. There
  // is no API/ABI compatibility or safety guarantee if you do so. Instead, use
  // the wuffs_foo__bar__baz functions.
  //
  // It is a struct, not a struct*, so that the outermost wuffs_foo__bar struct
  // can be stack allocated when WUFFS_IMPLEMENTATION is defined.

  struct {
    uint32_t magic;
    uint32_t active_coroutine;
    wuffs_base__vtable vtable_for__wuffs_base__hasher_u32;
    wuffs_base__vtable null_vtable;

    uint32_t f_state;
    bool f_started;

    wuffs_base__empty_struct (*choosy_up)(
        wuffs_adler32__hasher* self,
        wuffs_base__slice_u8 a_x);
  } private_impl;

#ifdef __cplusplus
#if defined(WUFFS_BASE__HAVE_UNIQUE_PTR)
  using unique_ptr = std::unique_ptr<wuffs_adler32__hasher, decltype(&free)>;

  // On failure, the alloc_etc functions return nullptr. They don't throw.

  static inline unique_ptr
  alloc() {
    return unique_ptr(wuffs_adler32__hasher__alloc(), &free);
  }

  static inline wuffs_base__hasher_u32::unique_ptr
  alloc_as__wuffs_base__hasher_u32() {
    return wuffs_base__hasher_u32::unique_ptr(
        wuffs_adler32__hasher__alloc_as__wuffs_base__hasher_u32(), &free);
  }
#endif  // defined(WUFFS_BASE__HAVE_UNIQUE_PTR)

#if defined(WUFFS_BASE__HAVE_EQ_DELETE) && !defined(WUFFS_IMPLEMENTATION)
  // Disallow constructing or copying an object via standard C++ mechanisms,
  // e.g. the "new" operator, as this struct is intentionally opaque. Its total
  // size and field layout is not part of the public, stable, memory-safe API.
  // Use malloc or memcpy and the sizeof__wuffs_foo__bar function instead, and
  // call wuffs_foo__bar__baz methods (which all take a "this"-like pointer as
  // their first argument) rather than tweaking bar.private_impl.qux fields.
  //
  // In C, we can just leave wuffs_foo__bar as an incomplete type (unless
  // WUFFS_IMPLEMENTATION is #define'd). In C++, we define a complete type in
  // order to provide convenience methods. These forward on "this", so that you
  // can write "bar->baz(etc)" instead of "wuffs_foo__bar__baz(bar, etc)".
  wuffs_adler32__hasher__struct() = delete;
  wuffs_adler32__hasher__struct(const wuffs_adler32__hasher__struct&) = delete;
  wuffs_adler32__hasher__struct& operator=(
      const wuffs_adler32__hasher__struct&) = delete;
#endif  // defined(WUFFS_BASE__HAVE_EQ_DELETE) && !defined(WUFFS_IMPLEMENTATION)

#if !defined(WUFFS_IMPLEMENTATION)
  // As above, the size of the struct is not part of the public API, and unless
  // WUFFS_IMPLEMENTATION is #define'd, this struct type T should be heap
  // allocated, not stack allocated. Its size is not intended to be known at
  // compile time, but it is unfortunately divulged as a side effect of
  // defining C++ convenience methods. Use "sizeof__T()", calling the function,
  // instead of "sizeof T", invoking the operator. To make the two values
  // different, so that passing the latter will be rejected by the initialize
  // function, we add an arbitrary amount of dead weight.
  uint8_t dead_weight[123000000];  // 123 MB.
#endif  // !defined(WUFFS_IMPLEMENTATION)

  inline wuffs_base__status WUFFS_BASE__WARN_UNUSED_RESULT
  initialize(
      size_t sizeof_star_self,
      uint64_t wuffs_version,
      uint32_t options) {
    return wuffs_adler32__hasher__initialize(
        this, sizeof_star_self, wuffs_version, options);
  }

  inline wuffs_base__hasher_u32*
  upcast_as__wuffs_base__hasher_u32() {
    return (wuffs_base__hasher_u32*)this;
  }

  inline wuffs_base__empty_struct
  set_quirk_enabled(
      uint32_t a_quirk,
      bool a_enabled) {
    return wuffs_adler32__hasher__set_quirk_enabled(this, a_quirk, a_enabled);
  }

  inline uint32_t
  update_u32(
      wuffs_base__slice_u8 a_x) {
    return wuffs_adler32__hasher__update_u32(this, a_x);
  }

#endif  // __cplusplus
};  // struct wuffs_adler32__hasher__struct

#endif  // defined(__cplusplus) || defined(WUFFS_IMPLEMENTATION)

#endif  // !defined(WUFFS_CONFIG__MODULES) || defined(WUFFS_CONFIG__MODULE__ADLER32) || defined(WUFFS_NONMONOLITHIC)


#if !defined(WUFFS_CONFIG__MODULES) || defined(WUFFS_CONFIG__MODULE__CRC32) || defined(WUFFS_NONMONOLITHIC)

// ---------------- Status Codes

// ---------------- Public Consts

// ---------------- Struct Declarations

typedef struct wuffs_crc32__ieee_hasher__struct wuffs_crc32__ieee_hasher;

#ifdef __cplusplus
extern "C" {
#endif

// ---------------- Public Initializer Prototypes

// For any given "wuffs_foo__bar* self", "wuffs_foo__bar__initialize(self,
// etc)" should be called before any other "wuffs_foo__bar__xxx(self, etc)".
//
// Pass sizeof(*self) and WUFFS_VERSION for sizeof_star_self and wuffs_version.
// Pass 0 (or some combination of WUFFS_INITIALIZE__XXX) for options.

wuffs_base__status WUFFS_BASE__WARN_UNUSED_RESULT
wuffs_crc32__ieee_hasher__initialize(
    wuffs_crc32__ieee_hasher* self,
    size_t sizeof_star_self,
    uint64_t wuffs_version,
    uint32_t options);

size_t
sizeof__wuffs_crc32__ieee_hasher();

// ---------------- Allocs

// These functions allocate and initialize Wuffs structs. They return NULL if
// memory allocation fails. If they return non-NULL, there is no need to call
// wuffs_foo__bar__initialize, but the caller is responsible for eventually
// calling free on the returned pointer. That pointer is effectively a C++
// std::unique_ptr<T, decltype(&free)>.

wuffs_crc32__ieee_hasher*
wuffs_crc32__ieee_hasher__alloc();

static inline wuffs_base__hasher_u32*
wuffs_crc32__ieee_hasher__alloc_as__wuffs_base__hasher_u32() {
  return (wuffs_base__hasher_u32*)(wuffs_crc32__ieee_hasher__alloc());
}

// ---------------- Upcasts

static inline wuffs_base__hasher_u32*
wuffs_crc32__ieee_hasher__upcast_as__wuffs_base__hasher_u32(
    wuffs_crc32__ieee_hasher* p) {
  return (wuffs_base__hasher_u32*)p;
}

// ---------------- Public Function Prototypes

WUFFS_BASE__MAYBE_STATIC wuffs_base__empty_struct
wuffs_crc32__ieee_hasher__set_quirk_enabled(
    wuffs_crc32__ieee_hasher* self,
    uint32_t a_quirk,
    bool a_enabled);

WUFFS_BASE__MAYBE_STATIC uint32_t
wuffs_crc32__ieee_hasher__update_u32(
    wuffs_crc32__ieee_hasher* self,
    wuffs_base__slice_u8 a_x);

#ifdef __cplusplus
}  // extern "C"
#endif

// ---------------- Struct Definitions

// These structs' fields, and the sizeof them, are private implementation
// details that aren't guaranteed to be stable across Wuffs versions.
//
// See https://en.wikipedia.org/wiki/Opaque_pointer#C

#if defined(__cplusplus) || defined(WUFFS_IMPLEMENTATION)

struct wuffs_crc32__ieee_hasher__struct {
  // Do not access the private_impl's or private_data's fields directly. There
  // is no API/ABI compatibility or safety guarantee if you do so. Instead, use
  // the wuffs_foo__bar__baz functions.
  //
  // It is a struct, not a struct*, so that the outermost wuffs_foo__bar struct
  // can be stack allocated when WUFFS_IMPLEMENTATION is defined.

  struct {
    uint32_t magic;
    uint32_t active_coroutine;
    wuffs_base__vtable vtable_for__wuffs_base__hasher_u32;
    wuffs_base__vtable null_vtable;

    uint32_t f_state;

    wuffs_base__empty_struct (*choosy_up)(
        wuffs_crc32__ieee_hasher* self,
        wuffs_base__slice_u8 a_x);
  } private_impl;

#ifdef __cplusplus
#if defined(WUFFS_BASE__HAVE_UNIQUE_PTR)
  using unique_ptr = std::unique_ptr<wuffs_crc32__ieee_hasher, decltype(&free)>;

  // On failure, the alloc_etc functions return nullptr. They don't throw.

  static inline unique_ptr
  alloc() {
    return unique_ptr(wuffs_crc32__ieee_hasher__alloc(), &free);
  }

  static inline wuffs_base__hasher_u32::unique_ptr
  alloc_as__wuffs_base__hasher_u32() {
    return wuffs_base__hasher_u32::unique_ptr(
        wuffs_crc32__ieee_hasher__alloc_as__wuffs_base__hasher_u32(), &free);
  }
#endif  // defined(WUFFS_BASE__HAVE_UNIQUE_PTR)

#if defined(WUFFS_BASE__HAVE_EQ_DELETE) && !defined(WUFFS_IMPLEMENTATION)
  // Disallow constructing or copying an object via standard C++ mechanisms,
  // e.g. the "new" operator, as this struct is intentionally opaque. Its total
  // size and field layout is not part of the public, stable, memory-safe API.
  // Use malloc or memcpy and the sizeof__wuffs_foo__bar function instead, and
  // call wuffs_foo__bar__baz methods (which all take a "this"-like pointer as
  // their first argument) rather than tweaking bar.private_impl.qux fields.
  //
  // In C, we can just leave wuffs_foo__bar as an incomplete type (unless
  // WUFFS_IMPLEMENTATION is #define'd). In C++, we define a complete type in
  // order to provide convenience methods. These forward on "this", so that you
  // can write "bar->baz(etc)" instead of "wuffs_foo__bar__baz(bar, etc)".
  wuffs_crc32__ieee_hasher__struct() = delete;
  wuffs_crc32__ieee_hasher__struct(const wuffs_crc32__ieee_hasher__struct&) = delete;
  wuffs_crc32__ieee_hasher__struct& operator=(
      const wuffs_crc32__ieee_hasher__struct&) = delete;
#endif  // defined(WUFFS_BASE__HAVE_EQ_DELETE) && !defined(WUFFS_IMPLEMENTATION)

#if !defined(WUFFS_IMPLEMENTATION)
  // As above, the size of the struct is not part of the public API, and unless
  // WUFFS_IMPLEMENTATION is #define'd, this struct type T should be heap
  // allocated, not stack allocated. Its size is not intended to be known at
  // compile time, but it is unfortunately divulged as a side effect of
  // defining C++ convenience methods. Use "sizeof__T()", calling the function,
  // instead of "sizeof T", invoking the operator. To make the two values
  // different, so that passing the latter will be rejected by the initialize
  // function, we add an arbitrary amount of dead weight.
  uint8_t dead_weight[123000000];  // 123 MB.
#endif  // !defined(WUFFS_IMPLEMENTATION)

  inline wuffs_base__status WUFFS_BASE__WARN_UNUSED_RESULT
  initialize(
      size_t sizeof_star_self,
      uint64_t wuffs_version,
      uint32_t options) {
    return wuffs_crc32__ieee_hasher__initialize(
        this, sizeof_star_self, wuffs_version, options);
  }

  inline wuffs_base__hasher_u32*
  upcast_as__wuffs_base__hasher_u32() {
    return (wuffs_base__hasher_u32*)this;
  }

  inline wuffs_base__empty_struct
  set_quirk_enabled(
      uint32_t a_quirk,
      bool a_enabled) {
    return wuffs_crc32__ieee_hasher__set_quirk_enabled(this, a_quirk, a_enabled);
  }

  inline uint32_t
  update_u32(
      wuffs_base__slice_u8 a_x) {
    return wuffs_crc32__ieee_hasher__update_u32(this, a_x);
  }

#endif  // __cplusplus
};  // struct wuffs_crc32__ieee_hasher__struct

#endif  // defined(__cplusplus) || defined(WUFFS_IMPLEMENTATION)

#endif  // !defined(WUFFS_CONFIG__MODULES) || defined(WUFFS_CONFIG__MODULE__CRC32) || defined(WUFFS_NONMONOLITHIC)

#if !defined(WUFFS_CONFIG__MODULES) || defined(WUFFS_CONFIG__MODULE__DEFLATE) || defined(WUFFS_NONMONOLITHIC)

// ---------------- Status Codes

extern const char wuffs_deflate__error__bad_huffman_code_over_subscribed[];
extern const char wuffs_deflate__error__bad_huffman_code_under_subscribed[];
extern const char wuffs_deflate__error__bad_huffman_code_length_count[];
extern const char wuffs_deflate__error__bad_huffman_code_length_repetition[];
extern const char wuffs_deflate__error__bad_huffman_code[];
extern const char wuffs_deflate__error__bad_huffman_minimum_code_length[];
extern const char wuffs_deflate__error__bad_block[];
extern const char wuffs_deflate__error__bad_distance[];
extern const char wuffs_deflate__error__bad_distance_code_count[];
extern const char wuffs_deflate__error__bad_literal_length_code_count[];
extern const char wuffs_deflate__error__inconsistent_stored_block_length[];
extern const char wuffs_deflate__error__missing_end_of_block_code[];
extern const char wuffs_deflate__error__no_huffman_codes[];
extern const char wuffs_deflate__error__truncated_input[];

// ---------------- Public Consts

#define WUFFS_DEFLATE__DECODER_WORKBUF_LEN_MAX_INCL_WORST_CASE 1

// ---------------- Struct Declarations

typedef struct wuffs_deflate__decoder__struct wuffs_deflate__decoder;

#ifdef __cplusplus
extern "C" {
#endif

// ---------------- Public Initializer Prototypes

// For any given "wuffs_foo__bar* self", "wuffs_foo__bar__initialize(self,
// etc)" should be called before any other "wuffs_foo__bar__xxx(self, etc)".
//
// Pass sizeof(*self) and WUFFS_VERSION for sizeof_star_self and wuffs_version.
// Pass 0 (or some combination of WUFFS_INITIALIZE__XXX) for options.

wuffs_base__status WUFFS_BASE__WARN_UNUSED_RESULT
wuffs_deflate__decoder__initialize(
    wuffs_deflate__decoder* self,
    size_t sizeof_star_self,
    uint64_t wuffs_version,
    uint32_t options);

size_t
sizeof__wuffs_deflate__decoder();

// ---------------- Allocs

// These functions allocate and initialize Wuffs structs. They return NULL if
// memory allocation fails. If they return non-NULL, there is no need to call
// wuffs_foo__bar__initialize, but the caller is responsible for eventually
// calling free on the returned pointer. That pointer is effectively a C++
// std::unique_ptr<T, decltype(&free)>.

wuffs_deflate__decoder*
wuffs_deflate__decoder__alloc();

static inline wuffs_base__io_transformer*
wuffs_deflate__decoder__alloc_as__wuffs_base__io_transformer() {
  return (wuffs_base__io_transformer*)(wuffs_deflate__decoder__alloc());
}

// ---------------- Upcasts

static inline wuffs_base__io_transformer*
wuffs_deflate__decoder__upcast_as__wuffs_base__io_transformer(
    wuffs_deflate__decoder* p) {
  return (wuffs_base__io_transformer*)p;
}

// ---------------- Public Function Prototypes

WUFFS_BASE__MAYBE_STATIC wuffs_base__empty_struct
wuffs_deflate__decoder__add_history(
    wuffs_deflate__decoder* self,
    wuffs_base__slice_u8 a_hist);

WUFFS_BASE__MAYBE_STATIC wuffs_base__empty_struct
wuffs_deflate__decoder__set_quirk_enabled(
    wuffs_deflate__decoder* self,
    uint32_t a_quirk,
    bool a_enabled);

WUFFS_BASE__MAYBE_STATIC wuffs_base__range_ii_u64
wuffs_deflate__decoder__workbuf_len(
    const wuffs_deflate__decoder* self);

WUFFS_BASE__MAYBE_STATIC wuffs_base__status
wuffs_deflate__decoder__transform_io(
    wuffs_deflate__decoder* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__io_buffer* a_src,
    wuffs_base__slice_u8 a_workbuf);

#ifdef __cplusplus
}  // extern "C"
#endif

// ---------------- Struct Definitions

// These structs' fields, and the sizeof them, are private implementation
// details that aren't guaranteed to be stable across Wuffs versions.
//
// See https://en.wikipedia.org/wiki/Opaque_pointer#C

#if defined(__cplusplus) || defined(WUFFS_IMPLEMENTATION)

struct wuffs_deflate__decoder__struct {
  // Do not access the private_impl's or private_data's fields directly. There
  // is no API/ABI compatibility or safety guarantee if you do so. Instead, use
  // the wuffs_foo__bar__baz functions.
  //
  // It is a struct, not a struct*, so that the outermost wuffs_foo__bar struct
  // can be stack allocated when WUFFS_IMPLEMENTATION is defined.

  struct {
    uint32_t magic;
    uint32_t active_coroutine;
    wuffs_base__vtable vtable_for__wuffs_base__io_transformer;
    wuffs_base__vtable null_vtable;

    uint32_t f_bits;
    uint32_t f_n_bits;
    uint64_t f_transformed_history_count;
    uint32_t f_history_index;
    uint32_t f_n_huffs_bits[2];
    bool f_end_of_block;

    uint32_t p_transform_io[1];
    uint32_t p_do_transform_io[1];
    uint32_t p_decode_blocks[1];
    uint32_t p_decode_uncompressed[1];
    uint32_t p_init_dynamic_huffman[1];
    wuffs_base__status (*choosy_decode_huffman_fast64)(
        wuffs_deflate__decoder* self,
        wuffs_base__io_buffer* a_dst,
        wuffs_base__io_buffer* a_src);
    uint32_t p_decode_huffman_slow[1];
  } private_impl;

  struct {
    uint32_t f_huffs[2][1024];
    uint8_t f_history[33025];
    uint8_t f_code_lengths[320];

    struct {
      uint32_t v_final;
    } s_decode_blocks[1];
    struct {
      uint32_t v_length;
      uint64_t scratch;
    } s_decode_uncompressed[1];
    struct {
      uint32_t v_bits;
      uint32_t v_n_bits;
      uint32_t v_n_lit;
      uint32_t v_n_dist;
      uint32_t v_n_clen;
      uint32_t v_i;
      uint32_t v_mask;
      uint32_t v_n_extra_bits;
      uint8_t v_rep_symbol;
      uint32_t v_rep_count;
    } s_init_dynamic_huffman[1];
    struct {
      uint32_t v_bits;
      uint32_t v_n_bits;
      uint32_t v_table_entry_n_bits;
      uint32_t v_lmask;
      uint32_t v_dmask;
      uint32_t v_redir_top;
      uint32_t v_redir_mask;
      uint32_t v_length;
      uint32_t v_dist_minus_1;
      uint64_t scratch;
    } s_decode_huffman_slow[1];
  } private_data;

#ifdef __cplusplus
#if defined(WUFFS_BASE__HAVE_UNIQUE_PTR)
  using unique_ptr = std::unique_ptr<wuffs_deflate__decoder, decltype(&free)>;

  // On failure, the alloc_etc functions return nullptr. They don't throw.

  static inline unique_ptr
  alloc() {
    return unique_ptr(wuffs_deflate__decoder__alloc(), &free);
  }

  static inline wuffs_base__io_transformer::unique_ptr
  alloc_as__wuffs_base__io_transformer() {
    return wuffs_base__io_transformer::unique_ptr(
        wuffs_deflate__decoder__alloc_as__wuffs_base__io_transformer(), &free);
  }
#endif  // defined(WUFFS_BASE__HAVE_UNIQUE_PTR)

#if defined(WUFFS_BASE__HAVE_EQ_DELETE) && !defined(WUFFS_IMPLEMENTATION)
  // Disallow constructing or copying an object via standard C++ mechanisms,
  // e.g. the "new" operator, as this struct is intentionally opaque. Its total
  // size and field layout is not part of the public, stable, memory-safe API.
  // Use malloc or memcpy and the sizeof__wuffs_foo__bar function instead, and
  // call wuffs_foo__bar__baz methods (which all take a "this"-like pointer as
  // their first argument) rather than tweaking bar.private_impl.qux fields.
  //
  // In C, we can just leave wuffs_foo__bar as an incomplete type (unless
  // WUFFS_IMPLEMENTATION is #define'd). In C++, we define a complete type in
  // order to provide convenience methods. These forward on "this", so that you
  // can write "bar->baz(etc)" instead of "wuffs_foo__bar__baz(bar, etc)".
  wuffs_deflate__decoder__struct() = delete;
  wuffs_deflate__decoder__struct(const wuffs_deflate__decoder__struct&) = delete;
  wuffs_deflate__decoder__struct& operator=(
      const wuffs_deflate__decoder__struct&) = delete;
#endif  // defined(WUFFS_BASE__HAVE_EQ_DELETE) && !defined(WUFFS_IMPLEMENTATION)

#if !defined(WUFFS_IMPLEMENTATION)
  // As above, the size of the struct is not part of the public API, and unless
  // WUFFS_IMPLEMENTATION is #define'd, this struct type T should be heap
  // allocated, not stack allocated. Its size is not intended to be known at
  // compile time, but it is unfortunately divulged as a side effect of
  // defining C++ convenience methods. Use "sizeof__T()", calling the function,
  // instead of "sizeof T", invoking the operator. To make the two values
  // different, so that passing the latter will be rejected by the initialize
  // function, we add an arbitrary amount of dead weight.
  uint8_t dead_weight[123000000];  // 123 MB.
#endif  // !defined(WUFFS_IMPLEMENTATION)

  inline wuffs_base__status WUFFS_BASE__WARN_UNUSED_RESULT
  initialize(
      size_t sizeof_star_self,
      uint64_t wuffs_version,
      uint32_t options) {
    return wuffs_deflate__decoder__initialize(
        this, sizeof_star_self, wuffs_version, options);
  }

  inline wuffs_base__io_transformer*
  upcast_as__wuffs_base__io_transformer() {
    return (wuffs_base__io_transformer*)this;
  }

  inline wuffs_base__empty_struct
  add_history(
      wuffs_base__slice_u8 a_hist) {
    return wuffs_deflate__decoder__add_history(this, a_hist);
  }

  inline wuffs_base__empty_struct
  set_quirk_enabled(
      uint32_t a_quirk,
      bool a_enabled) {
    return wuffs_deflate__decoder__set_quirk_enabled(this, a_quirk, a_enabled);
  }

  inline wuffs_base__range_ii_u64
  workbuf_len() const {
    return wuffs_deflate__decoder__workbuf_len(this);
  }

  inline wuffs_base__status
  transform_io(
      wuffs_base__io_buffer* a_dst,
      wuffs_base__io_buffer* a_src,
      wuffs_base__slice_u8 a_workbuf) {
    return wuffs_deflate__decoder__transform_io(this, a_dst, a_src, a_workbuf);
  }

#endif  // __cplusplus
};  // struct wuffs_deflate__decoder__struct

#endif  // defined(__cplusplus) || defined(WUFFS_IMPLEMENTATION)

#endif  // !defined(WUFFS_CONFIG__MODULES) || defined(WUFFS_CONFIG__MODULE__DEFLATE) || defined(WUFFS_NONMONOLITHIC)

#if !defined(WUFFS_CONFIG__MODULES) || defined(WUFFS_CONFIG__MODULE__ZLIB) || defined(WUFFS_NONMONOLITHIC)

// ---------------- Status Codes

extern const char wuffs_zlib__note__dictionary_required[];
extern const char wuffs_zlib__error__bad_checksum[];
extern const char wuffs_zlib__error__bad_compression_method[];
extern const char wuffs_zlib__error__bad_compression_window_size[];
extern const char wuffs_zlib__error__bad_parity_check[];
extern const char wuffs_zlib__error__incorrect_dictionary[];
extern const char wuffs_zlib__error__truncated_input[];

// ---------------- Public Consts

#define WUFFS_ZLIB__QUIRK_JUST_RAW_DEFLATE 2113790976

#define WUFFS_ZLIB__DECODER_WORKBUF_LEN_MAX_INCL_WORST_CASE 1

// ---------------- Struct Declarations

typedef struct wuffs_zlib__decoder__struct wuffs_zlib__decoder;

#ifdef __cplusplus
extern "C" {
#endif

// ---------------- Public Initializer Prototypes

// For any given "wuffs_foo__bar* self", "wuffs_foo__bar__initialize(self,
// etc)" should be called before any other "wuffs_foo__bar__xxx(self, etc)".
//
// Pass sizeof(*self) and WUFFS_VERSION for sizeof_star_self and wuffs_version.
// Pass 0 (or some combination of WUFFS_INITIALIZE__XXX) for options.

wuffs_base__status WUFFS_BASE__WARN_UNUSED_RESULT
wuffs_zlib__decoder__initialize(
    wuffs_zlib__decoder* self,
    size_t sizeof_star_self,
    uint64_t wuffs_version,
    uint32_t options);

size_t
sizeof__wuffs_zlib__decoder();

// ---------------- Allocs

// These functions allocate and initialize Wuffs structs. They return NULL if
// memory allocation fails. If they return non-NULL, there is no need to call
// wuffs_foo__bar__initialize, but the caller is responsible for eventually
// calling free on the returned pointer. That pointer is effectively a C++
// std::unique_ptr<T, decltype(&free)>.

wuffs_zlib__decoder*
wuffs_zlib__decoder__alloc();

static inline wuffs_base__io_transformer*
wuffs_zlib__decoder__alloc_as__wuffs_base__io_transformer() {
  return (wuffs_base__io_transformer*)(wuffs_zlib__decoder__alloc());
}

// ---------------- Upcasts

static inline wuffs_base__io_transformer*
wuffs_zlib__decoder__upcast_as__wuffs_base__io_transformer(
    wuffs_zlib__decoder* p) {
  return (wuffs_base__io_transformer*)p;
}

// ---------------- Public Function Prototypes

WUFFS_BASE__MAYBE_STATIC uint32_t
wuffs_zlib__decoder__dictionary_id(
    const wuffs_zlib__decoder* self);

WUFFS_BASE__MAYBE_STATIC wuffs_base__empty_struct
wuffs_zlib__decoder__add_dictionary(
    wuffs_zlib__decoder* self,
    wuffs_base__slice_u8 a_dict);

WUFFS_BASE__MAYBE_STATIC wuffs_base__empty_struct
wuffs_zlib__decoder__set_quirk_enabled(
    wuffs_zlib__decoder* self,
    uint32_t a_quirk,
    bool a_enabled);

WUFFS_BASE__MAYBE_STATIC wuffs_base__range_ii_u64
wuffs_zlib__decoder__workbuf_len(
    const wuffs_zlib__decoder* self);

WUFFS_BASE__MAYBE_STATIC wuffs_base__status
wuffs_zlib__decoder__transform_io(
    wuffs_zlib__decoder* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__io_buffer* a_src,
    wuffs_base__slice_u8 a_workbuf);

#ifdef __cplusplus
}  // extern "C"
#endif

// ---------------- Struct Definitions

// These structs' fields, and the sizeof them, are private implementation
// details that aren't guaranteed to be stable across Wuffs versions.
//
// See https://en.wikipedia.org/wiki/Opaque_pointer#C

#if defined(__cplusplus) || defined(WUFFS_IMPLEMENTATION)

struct wuffs_zlib__decoder__struct {
  // Do not access the private_impl's or private_data's fields directly. There
  // is no API/ABI compatibility or safety guarantee if you do so. Instead, use
  // the wuffs_foo__bar__baz functions.
  //
  // It is a struct, not a struct*, so that the outermost wuffs_foo__bar struct
  // can be stack allocated when WUFFS_IMPLEMENTATION is defined.

  struct {
    uint32_t magic;
    uint32_t active_coroutine;
    wuffs_base__vtable vtable_for__wuffs_base__io_transformer;
    wuffs_base__vtable null_vtable;

    bool f_bad_call_sequence;
    bool f_header_complete;
    bool f_got_dictionary;
    bool f_want_dictionary;
    bool f_quirks[1];
    bool f_ignore_checksum;
    uint32_t f_dict_id_got;
    uint32_t f_dict_id_want;

    uint32_t p_transform_io[1];
    uint32_t p_do_transform_io[1];
  } private_impl;

  struct {
    wuffs_adler32__hasher f_checksum;
    wuffs_adler32__hasher f_dict_id_hasher;
    wuffs_deflate__decoder f_flate;

    struct {
      uint32_t v_checksum_got;
      uint64_t scratch;
    } s_do_transform_io[1];
  } private_data;

#ifdef __cplusplus
#if defined(WUFFS_BASE__HAVE_UNIQUE_PTR)
  using unique_ptr = std::unique_ptr<wuffs_zlib__decoder, decltype(&free)>;

  // On failure, the alloc_etc functions return nullptr. They don't throw.

  static inline unique_ptr
  alloc() {
    return unique_ptr(wuffs_zlib__decoder__alloc(), &free);
  }

  static inline wuffs_base__io_transformer::unique_ptr
  alloc_as__wuffs_base__io_transformer() {
    return wuffs_base__io_transformer::unique_ptr(
        wuffs_zlib__decoder__alloc_as__wuffs_base__io_transformer(), &free);
  }
#endif  // defined(WUFFS_BASE__HAVE_UNIQUE_PTR)

#if defined(WUFFS_BASE__HAVE_EQ_DELETE) && !defined(WUFFS_IMPLEMENTATION)
  // Disallow constructing or copying an object via standard C++ mechanisms,
  // e.g. the "new" operator, as this struct is intentionally opaque. Its total
  // size and field layout is not part of the public, stable, memory-safe API.
  // Use malloc or memcpy and the sizeof__wuffs_foo__bar function instead, and
  // call wuffs_foo__bar__baz methods (which all take a "this"-like pointer as
  // their first argument) rather than tweaking bar.private_impl.qux fields.
  //
  // In C, we can just leave wuffs_foo__bar as an incomplete type (unless
  // WUFFS_IMPLEMENTATION is #define'd). In C++, we define a complete type in
  // order to provide convenience methods. These forward on "this", so that you
  // can write "bar->baz(etc)" instead of "wuffs_foo__bar__baz(bar, etc)".
  wuffs_zlib__decoder__struct() = delete;
  wuffs_zlib__decoder__struct(const wuffs_zlib__decoder__struct&) = delete;
  wuffs_zlib__decoder__struct& operator=(
      const wuffs_zlib__decoder__struct&) = delete;
#endif  // defined(WUFFS_BASE__HAVE_EQ_DELETE) && !defined(WUFFS_IMPLEMENTATION)

#if !defined(WUFFS_IMPLEMENTATION)
  // As above, the size of the struct is not part of the public API, and unless
  // WUFFS_IMPLEMENTATION is #define'd, this struct type T should be heap
  // allocated, not stack allocated. Its size is not intended to be known at
  // compile time, but it is unfortunately divulged as a side effect of
  // defining C++ convenience methods. Use "sizeof__T()", calling the function,
  // instead of "sizeof T", invoking the operator. To make the two values
  // different, so that passing the latter will be rejected by the initialize
  // function, we add an arbitrary amount of dead weight.
  uint8_t dead_weight[123000000];  // 123 MB.
#endif  // !defined(WUFFS_IMPLEMENTATION)

  inline wuffs_base__status WUFFS_BASE__WARN_UNUSED_RESULT
  initialize(
      size_t sizeof_star_self,
      uint64_t wuffs_version,
      uint32_t options) {
    return wuffs_zlib__decoder__initialize(
        this, sizeof_star_self, wuffs_version, options);
  }

  inline wuffs_base__io_transformer*
  upcast_as__wuffs_base__io_transformer() {
    return (wuffs_base__io_transformer*)this;
  }

  inline uint32_t
  dictionary_id() const {
    return wuffs_zlib__decoder__dictionary_id(this);
  }

  inline wuffs_base__empty_struct
  add_dictionary(
      wuffs_base__slice_u8 a_dict) {
    return wuffs_zlib__decoder__add_dictionary(this, a_dict);
  }

  inline wuffs_base__empty_struct
  set_quirk_enabled(
      uint32_t a_quirk,
      bool a_enabled) {
    return wuffs_zlib__decoder__set_quirk_enabled(this, a_quirk, a_enabled);
  }

  inline wuffs_base__range_ii_u64
  workbuf_len() const {
    return wuffs_zlib__decoder__workbuf_len(this);
  }

  inline wuffs_base__status
  transform_io(
      wuffs_base__io_buffer* a_dst,
      wuffs_base__io_buffer* a_src,
      wuffs_base__slice_u8 a_workbuf) {
    return wuffs_zlib__decoder__transform_io(this, a_dst, a_src, a_workbuf);
  }

#endif  // __cplusplus
};  // struct wuffs_zlib__decoder__struct

#endif  // defined(__cplusplus) || defined(WUFFS_IMPLEMENTATION)

#endif  // !defined(WUFFS_CONFIG__MODULES) || defined(WUFFS_CONFIG__MODULE__ZLIB) || defined(WUFFS_NONMONOLITHIC)


// ‼ WUFFS C HEADER ENDS HERE.
#ifdef WUFFS_IMPLEMENTATION

#ifdef __cplusplus
extern "C" {
#endif

// ---------------- Fundamentals

// WUFFS_BASE__MAGIC is a magic number to check that initializers are called.
// It's not foolproof, given C doesn't automatically zero memory before use,
// but it should catch 99.99% of cases.
//
// Its (non-zero) value is arbitrary, based on md5sum("wuffs").
#define WUFFS_BASE__MAGIC ((uint32_t)0x3CCB6C71)

// WUFFS_BASE__DISABLED is a magic number to indicate that a non-recoverable
// error was previously encountered.
//
// Its (non-zero) value is arbitrary, based on md5sum("disabled").
#define WUFFS_BASE__DISABLED ((uint32_t)0x075AE3D2)

// Use switch cases for coroutine suspension points, similar to the technique
// in https://www.chiark.greenend.org.uk/~sgtatham/coroutines.html
//
// The implicit fallthrough is intentional.
//
// We use trivial macros instead of an explicit assignment and case statement
// so that clang-format doesn't get confused by the unusual "case"s.
#define WUFFS_BASE__COROUTINE_SUSPENSION_POINT_0 case 0:;
#define WUFFS_BASE__COROUTINE_SUSPENSION_POINT(n) \
  coro_susp_point = n;                            \
  case n:;

#define WUFFS_BASE__COROUTINE_SUSPENSION_POINT_MAYBE_SUSPEND(n) \
  if (!status.repr) {                                           \
    goto ok;                                                    \
  } else if (*status.repr != '$') {                             \
    goto exit;                                                  \
  }                                                             \
  coro_susp_point = n;                                          \
  goto suspend;                                                 \
  case n:;

// The "defined(__clang__)" isn't redundant. While vanilla clang defines
// __GNUC__, clang-cl (which mimics MSVC's cl.exe) does not.
#if defined(__GNUC__) || defined(__clang__)
#define WUFFS_BASE__LIKELY(expr) (__builtin_expect(!!(expr), 1))
#define WUFFS_BASE__UNLIKELY(expr) (__builtin_expect(!!(expr), 0))
#else
#define WUFFS_BASE__LIKELY(expr) (expr)
#define WUFFS_BASE__UNLIKELY(expr) (expr)
#endif

// --------

static inline wuffs_base__empty_struct  //
wuffs_base__ignore_status(wuffs_base__status z) {
  return wuffs_base__make_empty_struct();
}

static inline wuffs_base__status  //
wuffs_base__status__ensure_not_a_suspension(wuffs_base__status z) {
  if (z.repr && (*z.repr == '$')) {
    z.repr = wuffs_base__error__cannot_return_a_suspension;
  }
  return z;
}

// --------

// wuffs_base__iterate_total_advance returns the exclusive pointer-offset at
// which iteration should stop. The overall slice has length total_len, each
// iteration's sub-slice has length iter_len and are placed iter_advance apart.
//
// The iter_advance may not be larger than iter_len. The iter_advance may be
// smaller than iter_len, in which case the sub-slices will overlap.
//
// The return value r satisfies ((0 <= r) && (r <= total_len)).
//
// For example, if total_len = 15, iter_len = 5 and iter_advance = 3, there are
// four iterations at offsets 0, 3, 6 and 9. This function returns 12.
//
// 0123456789012345
// [....]
//    [....]
//       [....]
//          [....]
//             $
// 0123456789012345
//
// For example, if total_len = 15, iter_len = 5 and iter_advance = 5, there are
// three iterations at offsets 0, 5 and 10. This function returns 15.
//
// 0123456789012345
// [....]
//      [....]
//           [....]
//                $
// 0123456789012345
static inline size_t  //
wuffs_base__iterate_total_advance(size_t total_len,
                                  size_t iter_len,
                                  size_t iter_advance) {
  if (total_len >= iter_len) {
    size_t n = total_len - iter_len;
    return ((n / iter_advance) * iter_advance) + iter_advance;
  }
  return 0;
}

// ---------------- Numeric Types

extern const uint8_t wuffs_base__low_bits_mask__u8[8];
extern const uint16_t wuffs_base__low_bits_mask__u16[16];
extern const uint32_t wuffs_base__low_bits_mask__u32[32];
extern const uint64_t wuffs_base__low_bits_mask__u64[64];

#define WUFFS_BASE__LOW_BITS_MASK__U8(n) (wuffs_base__low_bits_mask__u8[n])
#define WUFFS_BASE__LOW_BITS_MASK__U16(n) (wuffs_base__low_bits_mask__u16[n])
#define WUFFS_BASE__LOW_BITS_MASK__U32(n) (wuffs_base__low_bits_mask__u32[n])
#define WUFFS_BASE__LOW_BITS_MASK__U64(n) (wuffs_base__low_bits_mask__u64[n])

// --------

static inline void  //
wuffs_base__u8__sat_add_indirect(uint8_t* x, uint8_t y) {
  *x = wuffs_base__u8__sat_add(*x, y);
}

static inline void  //
wuffs_base__u8__sat_sub_indirect(uint8_t* x, uint8_t y) {
  *x = wuffs_base__u8__sat_sub(*x, y);
}

static inline void  //
wuffs_base__u16__sat_add_indirect(uint16_t* x, uint16_t y) {
  *x = wuffs_base__u16__sat_add(*x, y);
}

static inline void  //
wuffs_base__u16__sat_sub_indirect(uint16_t* x, uint16_t y) {
  *x = wuffs_base__u16__sat_sub(*x, y);
}

static inline void  //
wuffs_base__u32__sat_add_indirect(uint32_t* x, uint32_t y) {
  *x = wuffs_base__u32__sat_add(*x, y);
}

static inline void  //
wuffs_base__u32__sat_sub_indirect(uint32_t* x, uint32_t y) {
  *x = wuffs_base__u32__sat_sub(*x, y);
}

static inline void  //
wuffs_base__u64__sat_add_indirect(uint64_t* x, uint64_t y) {
  *x = wuffs_base__u64__sat_add(*x, y);
}

static inline void  //
wuffs_base__u64__sat_sub_indirect(uint64_t* x, uint64_t y) {
  *x = wuffs_base__u64__sat_sub(*x, y);
}

// ---------------- Slices and Tables

// wuffs_base__slice_u8__prefix returns up to the first up_to bytes of s.
static inline wuffs_base__slice_u8  //
wuffs_base__slice_u8__prefix(wuffs_base__slice_u8 s, uint64_t up_to) {
  if (((uint64_t)(s.len)) > up_to) {
    s.len = ((size_t)up_to);
  }
  return s;
}

// wuffs_base__slice_u8__suffix returns up to the last up_to bytes of s.
static inline wuffs_base__slice_u8  //
wuffs_base__slice_u8__suffix(wuffs_base__slice_u8 s, uint64_t up_to) {
  if (((uint64_t)(s.len)) > up_to) {
    s.ptr += ((uint64_t)(s.len)) - up_to;
    s.len = ((size_t)up_to);
  }
  return s;
}

// wuffs_base__slice_u8__copy_from_slice calls memmove(dst.ptr, src.ptr, len)
// where len is the minimum of dst.len and src.len.
//
// Passing a wuffs_base__slice_u8 with all fields NULL or zero (a valid, empty
// slice) is valid and results in a no-op.
static inline uint64_t  //
wuffs_base__slice_u8__copy_from_slice(wuffs_base__slice_u8 dst,
                                      wuffs_base__slice_u8 src) {
  size_t len = dst.len < src.len ? dst.len : src.len;
  if (len > 0) {
    memmove(dst.ptr, src.ptr, len);
  }
  return len;
}

// --------

static inline wuffs_base__slice_u8  //
wuffs_base__table_u8__row_u32(wuffs_base__table_u8 t, uint32_t y) {
  if (y < t.height) {
    return wuffs_base__make_slice_u8(t.ptr + (t.stride * y), t.width);
  }
  return wuffs_base__make_slice_u8(NULL, 0);
}

// ---------------- Slices and Tables (Utility)

#define wuffs_base__utility__empty_slice_u8 wuffs_base__empty_slice_u8

// ---------------- Ranges and Rects

static inline uint32_t  //
wuffs_base__range_ii_u32__get_min_incl(const wuffs_base__range_ii_u32* r) {
  return r->min_incl;
}

static inline uint32_t  //
wuffs_base__range_ii_u32__get_max_incl(const wuffs_base__range_ii_u32* r) {
  return r->max_incl;
}

static inline uint32_t  //
wuffs_base__range_ie_u32__get_min_incl(const wuffs_base__range_ie_u32* r) {
  return r->min_incl;
}

static inline uint32_t  //
wuffs_base__range_ie_u32__get_max_excl(const wuffs_base__range_ie_u32* r) {
  return r->max_excl;
}

static inline uint64_t  //
wuffs_base__range_ii_u64__get_min_incl(const wuffs_base__range_ii_u64* r) {
  return r->min_incl;
}

static inline uint64_t  //
wuffs_base__range_ii_u64__get_max_incl(const wuffs_base__range_ii_u64* r) {
  return r->max_incl;
}

static inline uint64_t  //
wuffs_base__range_ie_u64__get_min_incl(const wuffs_base__range_ie_u64* r) {
  return r->min_incl;
}

static inline uint64_t  //
wuffs_base__range_ie_u64__get_max_excl(const wuffs_base__range_ie_u64* r) {
  return r->max_excl;
}

// ---------------- Ranges and Rects (Utility)

#define wuffs_base__utility__empty_range_ii_u32 wuffs_base__empty_range_ii_u32
#define wuffs_base__utility__empty_range_ie_u32 wuffs_base__empty_range_ie_u32
#define wuffs_base__utility__empty_range_ii_u64 wuffs_base__empty_range_ii_u64
#define wuffs_base__utility__empty_range_ie_u64 wuffs_base__empty_range_ie_u64
#define wuffs_base__utility__empty_rect_ii_u32 wuffs_base__empty_rect_ii_u32
#define wuffs_base__utility__empty_rect_ie_u32 wuffs_base__empty_rect_ie_u32
#define wuffs_base__utility__make_range_ii_u32 wuffs_base__make_range_ii_u32
#define wuffs_base__utility__make_range_ie_u32 wuffs_base__make_range_ie_u32
#define wuffs_base__utility__make_range_ii_u64 wuffs_base__make_range_ii_u64
#define wuffs_base__utility__make_range_ie_u64 wuffs_base__make_range_ie_u64
#define wuffs_base__utility__make_rect_ii_u32 wuffs_base__make_rect_ii_u32
#define wuffs_base__utility__make_rect_ie_u32 wuffs_base__make_rect_ie_u32

// ---------------- I/O

static inline uint64_t  //
wuffs_base__io__count_since(uint64_t mark, uint64_t index) {
  if (index >= mark) {
    return index - mark;
  }
  return 0;
}

// TODO: drop the "const" in "const uint8_t* ptr". Some though required about
// the base.io_reader.since method returning a mutable "slice base.u8".
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#endif
static inline wuffs_base__slice_u8  //
wuffs_base__io__since(uint64_t mark, uint64_t index, const uint8_t* ptr) {
  if (index >= mark) {
    return wuffs_base__make_slice_u8(((uint8_t*)ptr) + mark,
                                     ((size_t)(index - mark)));
  }
  return wuffs_base__make_slice_u8(NULL, 0);
}
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

// --------

static inline void  //
wuffs_base__io_reader__limit(const uint8_t** ptr_io2_r,
                             const uint8_t* iop_r,
                             uint64_t limit) {
  if (((uint64_t)(*ptr_io2_r - iop_r)) > limit) {
    *ptr_io2_r = iop_r + limit;
  }
}

static inline uint32_t  //
wuffs_base__io_reader__limited_copy_u32_to_slice(const uint8_t** ptr_iop_r,
                                                 const uint8_t* io2_r,
                                                 uint32_t length,
                                                 wuffs_base__slice_u8 dst) {
  const uint8_t* iop_r = *ptr_iop_r;
  size_t n = dst.len;
  if (n > length) {
    n = length;
  }
  if (n > ((size_t)(io2_r - iop_r))) {
    n = (size_t)(io2_r - iop_r);
  }
  if (n > 0) {
    memmove(dst.ptr, iop_r, n);
    *ptr_iop_r += n;
  }
  return (uint32_t)(n);
}

// wuffs_base__io_reader__match7 returns whether the io_reader's upcoming bytes
// start with the given prefix (up to 7 bytes long). It is peek-like, not
// read-like, in that there are no side-effects.
//
// The low 3 bits of a hold the prefix length, n.
//
// The high 56 bits of a hold the prefix itself, in little-endian order. The
// first prefix byte is in bits 8..=15, the second prefix byte is in bits
// 16..=23, etc. The high (8 * (7 - n)) bits are ignored.
//
// There are three possible return values:
//  - 0 means success.
//  - 1 means inconclusive, equivalent to "$short read".
//  - 2 means failure.
static inline uint32_t  //
wuffs_base__io_reader__match7(const uint8_t* iop_r,
                              const uint8_t* io2_r,
                              wuffs_base__io_buffer* r,
                              uint64_t a) {
  uint32_t n = a & 7;
  a >>= 8;
  if ((io2_r - iop_r) >= 8) {
    uint64_t x = wuffs_base__peek_u64le__no_bounds_check(iop_r);
    uint32_t shift = 8 * (8 - n);
    return ((a << shift) == (x << shift)) ? 0 : 2;
  }
  for (; n > 0; n--) {
    if (iop_r >= io2_r) {
      return (r && r->meta.closed) ? 2 : 1;
    } else if (*iop_r != ((uint8_t)(a))) {
      return 2;
    }
    iop_r++;
    a >>= 8;
  }
  return 0;
}

static inline wuffs_base__io_buffer*  //
wuffs_base__io_reader__set(wuffs_base__io_buffer* b,
                           const uint8_t** ptr_iop_r,
                           const uint8_t** ptr_io0_r,
                           const uint8_t** ptr_io1_r,
                           const uint8_t** ptr_io2_r,
                           wuffs_base__slice_u8 data,
                           uint64_t history_position) {
  b->data = data;
  b->meta.wi = data.len;
  b->meta.ri = 0;
  b->meta.pos = history_position;
  b->meta.closed = false;

  *ptr_iop_r = data.ptr;
  *ptr_io0_r = data.ptr;
  *ptr_io1_r = data.ptr;
  *ptr_io2_r = data.ptr + data.len;

  return b;
}

// --------

static inline uint64_t  //
wuffs_base__io_writer__copy_from_slice(uint8_t** ptr_iop_w,
                                       uint8_t* io2_w,
                                       wuffs_base__slice_u8 src) {
  uint8_t* iop_w = *ptr_iop_w;
  size_t n = src.len;
  if (n > ((size_t)(io2_w - iop_w))) {
    n = (size_t)(io2_w - iop_w);
  }
  if (n > 0) {
    memmove(iop_w, src.ptr, n);
    *ptr_iop_w += n;
  }
  return (uint64_t)(n);
}

static inline void  //
wuffs_base__io_writer__limit(uint8_t** ptr_io2_w,
                             uint8_t* iop_w,
                             uint64_t limit) {
  if (((uint64_t)(*ptr_io2_w - iop_w)) > limit) {
    *ptr_io2_w = iop_w + limit;
  }
}

static inline uint32_t  //
wuffs_base__io_writer__limited_copy_u32_from_history(uint8_t** ptr_iop_w,
                                                     uint8_t* io0_w,
                                                     uint8_t* io2_w,
                                                     uint32_t length,
                                                     uint32_t distance) {
  if (!distance) {
    return 0;
  }
  uint8_t* p = *ptr_iop_w;
  if ((size_t)(p - io0_w) < (size_t)(distance)) {
    return 0;
  }
  uint8_t* q = p - distance;
  size_t n = (size_t)(io2_w - p);
  if ((size_t)(length) > n) {
    length = (uint32_t)(n);
  } else {
    n = (size_t)(length);
  }
  // TODO: unrolling by 3 seems best for the std/deflate benchmarks, but that
  // is mostly because 3 is the minimum length for the deflate format. This
  // function implementation shouldn't overfit to that one format. Perhaps the
  // limited_copy_u32_from_history Wuffs method should also take an unroll hint
  // argument, and the cgen can look if that argument is the constant
  // expression '3'.
  //
  // See also wuffs_base__io_writer__limited_copy_u32_from_history_fast below.
  for (; n >= 3; n -= 3) {
    *p++ = *q++;
    *p++ = *q++;
    *p++ = *q++;
  }
  for (; n; n--) {
    *p++ = *q++;
  }
  *ptr_iop_w = p;
  return length;
}

// wuffs_base__io_writer__limited_copy_u32_from_history_fast is like the
// wuffs_base__io_writer__limited_copy_u32_from_history function above, but has
// stronger pre-conditions.
//
// The caller needs to prove that:
//  - length   <= (io2_w      - *ptr_iop_w)
//  - distance >= 1
//  - distance <= (*ptr_iop_w - io0_w)
static inline uint32_t  //
wuffs_base__io_writer__limited_copy_u32_from_history_fast(uint8_t** ptr_iop_w,
                                                          uint8_t* io0_w,
                                                          uint8_t* io2_w,
                                                          uint32_t length,
                                                          uint32_t distance) {
  uint8_t* p = *ptr_iop_w;
  uint8_t* q = p - distance;
  uint32_t n = length;
  for (; n >= 3; n -= 3) {
    *p++ = *q++;
    *p++ = *q++;
    *p++ = *q++;
  }
  for (; n; n--) {
    *p++ = *q++;
  }
  *ptr_iop_w = p;
  return length;
}

// wuffs_base__io_writer__limited_copy_u32_from_history_8_byte_chunks_distance_1_fast
// copies the previous byte (the one immediately before *ptr_iop_w), copying 8
// byte chunks at a time. Each chunk contains 8 repetitions of the same byte.
//
// In terms of number of bytes copied, length is rounded up to a multiple of 8.
// As a special case, a zero length rounds up to 8 (even though 0 is already a
// multiple of 8), since there is always at least one 8 byte chunk copied.
//
// In terms of advancing *ptr_iop_w, length is not rounded up.
//
// The caller needs to prove that:
//  - (length + 8) <= (io2_w      - *ptr_iop_w)
//  - distance     == 1
//  - distance     <= (*ptr_iop_w - io0_w)
static inline uint32_t  //
wuffs_base__io_writer__limited_copy_u32_from_history_8_byte_chunks_distance_1_fast(
    uint8_t** ptr_iop_w,
    uint8_t* io0_w,
    uint8_t* io2_w,
    uint32_t length,
    uint32_t distance) {
  uint8_t* p = *ptr_iop_w;
  uint64_t x = p[-1];
  x |= x << 8;
  x |= x << 16;
  x |= x << 32;
  uint32_t n = length;
  while (1) {
    wuffs_base__poke_u64le__no_bounds_check(p, x);
    if (n <= 8) {
      p += n;
      break;
    }
    p += 8;
    n -= 8;
  }
  *ptr_iop_w = p;
  return length;
}

// wuffs_base__io_writer__limited_copy_u32_from_history_8_byte_chunks_fast is
// like the wuffs_base__io_writer__limited_copy_u32_from_history_fast function
// above, but copies 8 byte chunks at a time.
//
// In terms of number of bytes copied, length is rounded up to a multiple of 8.
// As a special case, a zero length rounds up to 8 (even though 0 is already a
// multiple of 8), since there is always at least one 8 byte chunk copied.
//
// In terms of advancing *ptr_iop_w, length is not rounded up.
//
// The caller needs to prove that:
//  - (length + 8) <= (io2_w      - *ptr_iop_w)
//  - distance     >= 8
//  - distance     <= (*ptr_iop_w - io0_w)
static inline uint32_t  //
wuffs_base__io_writer__limited_copy_u32_from_history_8_byte_chunks_fast(
    uint8_t** ptr_iop_w,
    uint8_t* io0_w,
    uint8_t* io2_w,
    uint32_t length,
    uint32_t distance) {
  uint8_t* p = *ptr_iop_w;
  uint8_t* q = p - distance;
  uint32_t n = length;
  while (1) {
    memcpy(p, q, 8);
    if (n <= 8) {
      p += n;
      break;
    }
    p += 8;
    q += 8;
    n -= 8;
  }
  *ptr_iop_w = p;
  return length;
}

static inline uint32_t  //
wuffs_base__io_writer__limited_copy_u32_from_reader(uint8_t** ptr_iop_w,
                                                    uint8_t* io2_w,
                                                    uint32_t length,
                                                    const uint8_t** ptr_iop_r,
                                                    const uint8_t* io2_r) {
  uint8_t* iop_w = *ptr_iop_w;
  size_t n = length;
  if (n > ((size_t)(io2_w - iop_w))) {
    n = (size_t)(io2_w - iop_w);
  }
  const uint8_t* iop_r = *ptr_iop_r;
  if (n > ((size_t)(io2_r - iop_r))) {
    n = (size_t)(io2_r - iop_r);
  }
  if (n > 0) {
    memmove(iop_w, iop_r, n);
    *ptr_iop_w += n;
    *ptr_iop_r += n;
  }
  return (uint32_t)(n);
}

static inline uint32_t  //
wuffs_base__io_writer__limited_copy_u32_from_slice(uint8_t** ptr_iop_w,
                                                   uint8_t* io2_w,
                                                   uint32_t length,
                                                   wuffs_base__slice_u8 src) {
  uint8_t* iop_w = *ptr_iop_w;
  size_t n = src.len;
  if (n > length) {
    n = length;
  }
  if (n > ((size_t)(io2_w - iop_w))) {
    n = (size_t)(io2_w - iop_w);
  }
  if (n > 0) {
    memmove(iop_w, src.ptr, n);
    *ptr_iop_w += n;
  }
  return (uint32_t)(n);
}

static inline wuffs_base__io_buffer*  //
wuffs_base__io_writer__set(wuffs_base__io_buffer* b,
                           uint8_t** ptr_iop_w,
                           uint8_t** ptr_io0_w,
                           uint8_t** ptr_io1_w,
                           uint8_t** ptr_io2_w,
                           wuffs_base__slice_u8 data,
                           uint64_t history_position) {
  b->data = data;
  b->meta.wi = 0;
  b->meta.ri = 0;
  b->meta.pos = history_position;
  b->meta.closed = false;

  *ptr_iop_w = data.ptr;
  *ptr_io0_w = data.ptr;
  *ptr_io1_w = data.ptr;
  *ptr_io2_w = data.ptr + data.len;

  return b;
}

// ---------------- I/O (Utility)

#define wuffs_base__utility__empty_io_reader wuffs_base__empty_io_reader
#define wuffs_base__utility__empty_io_writer wuffs_base__empty_io_writer

// ---------------- Tokens

// ---------------- Tokens (Utility)

// ---------------- Memory Allocation

// ---------------- Images

WUFFS_BASE__MAYBE_STATIC uint64_t  //
wuffs_base__pixel_swizzler__limited_swizzle_u32_interleaved_from_reader(
    const wuffs_base__pixel_swizzler* p,
    uint32_t up_to_num_pixels,
    wuffs_base__slice_u8 dst,
    wuffs_base__slice_u8 dst_palette,
    const uint8_t** ptr_iop_r,
    const uint8_t* io2_r);

WUFFS_BASE__MAYBE_STATIC uint64_t  //
wuffs_base__pixel_swizzler__swizzle_interleaved_from_reader(
    const wuffs_base__pixel_swizzler* p,
    wuffs_base__slice_u8 dst,
    wuffs_base__slice_u8 dst_palette,
    const uint8_t** ptr_iop_r,
    const uint8_t* io2_r);

WUFFS_BASE__MAYBE_STATIC uint64_t  //
wuffs_base__pixel_swizzler__swizzle_interleaved_transparent_black(
    const wuffs_base__pixel_swizzler* p,
    wuffs_base__slice_u8 dst,
    wuffs_base__slice_u8 dst_palette,
    uint64_t num_pixels);

// ---------------- Images (Utility)

#define wuffs_base__utility__make_pixel_format wuffs_base__make_pixel_format

// ---------------- String Conversions

// ---------------- Unicode and UTF-8

// ----------------

#if !defined(WUFFS_CONFIG__MODULES) || defined(WUFFS_CONFIG__MODULE__BASE) || \
    defined(WUFFS_CONFIG__MODULE__BASE__CORE)

const uint8_t wuffs_base__low_bits_mask__u8[8] = {
    0x00, 0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3F, 0x7F,
};

const uint16_t wuffs_base__low_bits_mask__u16[16] = {
    0x0000, 0x0001, 0x0003, 0x0007, 0x000F, 0x001F, 0x003F, 0x007F,
    0x00FF, 0x01FF, 0x03FF, 0x07FF, 0x0FFF, 0x1FFF, 0x3FFF, 0x7FFF,
};

const uint32_t wuffs_base__low_bits_mask__u32[32] = {
    0x00000000, 0x00000001, 0x00000003, 0x00000007, 0x0000000F, 0x0000001F,
    0x0000003F, 0x0000007F, 0x000000FF, 0x000001FF, 0x000003FF, 0x000007FF,
    0x00000FFF, 0x00001FFF, 0x00003FFF, 0x00007FFF, 0x0000FFFF, 0x0001FFFF,
    0x0003FFFF, 0x0007FFFF, 0x000FFFFF, 0x001FFFFF, 0x003FFFFF, 0x007FFFFF,
    0x00FFFFFF, 0x01FFFFFF, 0x03FFFFFF, 0x07FFFFFF, 0x0FFFFFFF, 0x1FFFFFFF,
    0x3FFFFFFF, 0x7FFFFFFF,
};

const uint64_t wuffs_base__low_bits_mask__u64[64] = {
    0x0000000000000000, 0x0000000000000001, 0x0000000000000003,
    0x0000000000000007, 0x000000000000000F, 0x000000000000001F,
    0x000000000000003F, 0x000000000000007F, 0x00000000000000FF,
    0x00000000000001FF, 0x00000000000003FF, 0x00000000000007FF,
    0x0000000000000FFF, 0x0000000000001FFF, 0x0000000000003FFF,
    0x0000000000007FFF, 0x000000000000FFFF, 0x000000000001FFFF,
    0x000000000003FFFF, 0x000000000007FFFF, 0x00000000000FFFFF,
    0x00000000001FFFFF, 0x00000000003FFFFF, 0x00000000007FFFFF,
    0x0000000000FFFFFF, 0x0000000001FFFFFF, 0x0000000003FFFFFF,
    0x0000000007FFFFFF, 0x000000000FFFFFFF, 0x000000001FFFFFFF,
    0x000000003FFFFFFF, 0x000000007FFFFFFF, 0x00000000FFFFFFFF,
    0x00000001FFFFFFFF, 0x00000003FFFFFFFF, 0x00000007FFFFFFFF,
    0x0000000FFFFFFFFF, 0x0000001FFFFFFFFF, 0x0000003FFFFFFFFF,
    0x0000007FFFFFFFFF, 0x000000FFFFFFFFFF, 0x000001FFFFFFFFFF,
    0x000003FFFFFFFFFF, 0x000007FFFFFFFFFF, 0x00000FFFFFFFFFFF,
    0x00001FFFFFFFFFFF, 0x00003FFFFFFFFFFF, 0x00007FFFFFFFFFFF,
    0x0000FFFFFFFFFFFF, 0x0001FFFFFFFFFFFF, 0x0003FFFFFFFFFFFF,
    0x0007FFFFFFFFFFFF, 0x000FFFFFFFFFFFFF, 0x001FFFFFFFFFFFFF,
    0x003FFFFFFFFFFFFF, 0x007FFFFFFFFFFFFF, 0x00FFFFFFFFFFFFFF,
    0x01FFFFFFFFFFFFFF, 0x03FFFFFFFFFFFFFF, 0x07FFFFFFFFFFFFFF,
    0x0FFFFFFFFFFFFFFF, 0x1FFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF,
    0x7FFFFFFFFFFFFFFF,
};

const uint32_t wuffs_base__pixel_format__bits_per_channel[16] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x0A, 0x0C, 0x10, 0x18, 0x20, 0x30, 0x40,
};

const char wuffs_base__note__i_o_redirect[] = "@base: I/O redirect";
const char wuffs_base__note__end_of_data[] = "@base: end of data";
const char wuffs_base__note__metadata_reported[] = "@base: metadata reported";
const char wuffs_base__suspension__even_more_information[] = "$base: even more information";
const char wuffs_base__suspension__mispositioned_read[] = "$base: mispositioned read";
const char wuffs_base__suspension__mispositioned_write[] = "$base: mispositioned write";
const char wuffs_base__suspension__short_read[] = "$base: short read";
const char wuffs_base__suspension__short_write[] = "$base: short write";
const char wuffs_base__error__bad_i_o_position[] = "#base: bad I/O position";
const char wuffs_base__error__bad_argument_length_too_short[] = "#base: bad argument (length too short)";
const char wuffs_base__error__bad_argument[] = "#base: bad argument";
const char wuffs_base__error__bad_call_sequence[] = "#base: bad call sequence";
const char wuffs_base__error__bad_data[] = "#base: bad data";
const char wuffs_base__error__bad_receiver[] = "#base: bad receiver";
const char wuffs_base__error__bad_restart[] = "#base: bad restart";
const char wuffs_base__error__bad_sizeof_receiver[] = "#base: bad sizeof receiver";
const char wuffs_base__error__bad_vtable[] = "#base: bad vtable";
const char wuffs_base__error__bad_workbuf_length[] = "#base: bad workbuf length";
const char wuffs_base__error__bad_wuffs_version[] = "#base: bad wuffs version";
const char wuffs_base__error__cannot_return_a_suspension[] = "#base: cannot return a suspension";
const char wuffs_base__error__disabled_by_previous_error[] = "#base: disabled by previous error";
const char wuffs_base__error__initialize_falsely_claimed_already_zeroed[] = "#base: initialize falsely claimed already zeroed";
const char wuffs_base__error__initialize_not_called[] = "#base: initialize not called";
const char wuffs_base__error__interleaved_coroutine_calls[] = "#base: interleaved coroutine calls";
const char wuffs_base__error__no_more_information[] = "#base: no more information";
const char wuffs_base__error__not_enough_data[] = "#base: not enough data";
const char wuffs_base__error__out_of_bounds[] = "#base: out of bounds";
const char wuffs_base__error__unsupported_method[] = "#base: unsupported method";
const char wuffs_base__error__unsupported_option[] = "#base: unsupported option";
const char wuffs_base__error__unsupported_pixel_swizzler_option[] = "#base: unsupported pixel swizzler option";
const char wuffs_base__error__too_much_data[] = "#base: too much data";

const char wuffs_base__hasher_u32__vtable_name[] = "{vtable}wuffs_base__hasher_u32";
const char wuffs_base__image_decoder__vtable_name[] = "{vtable}wuffs_base__image_decoder";
const char wuffs_base__io_transformer__vtable_name[] = "{vtable}wuffs_base__io_transformer";
const char wuffs_base__token_decoder__vtable_name[] = "{vtable}wuffs_base__token_decoder";

#endif  // !defined(WUFFS_CONFIG__MODULES) ||
        // defined(WUFFS_CONFIG__MODULE__BASE)  ||
        // defined(WUFFS_CONFIG__MODULE__BASE__CORE)

#if !defined(WUFFS_CONFIG__MODULES) || defined(WUFFS_CONFIG__MODULE__BASE) || \
    defined(WUFFS_CONFIG__MODULE__BASE__INTERFACES)

// ---------------- Interface Definitions.

WUFFS_BASE__MAYBE_STATIC wuffs_base__empty_struct
wuffs_base__hasher_u32__set_quirk_enabled(
    wuffs_base__hasher_u32* self,
    uint32_t a_quirk,
    bool a_enabled) {
  if (!self) {
    return wuffs_base__make_empty_struct();
  }
  if (self->private_impl.magic != WUFFS_BASE__MAGIC) {
    return wuffs_base__make_empty_struct();
  }

  const wuffs_base__vtable* v = &self->private_impl.first_vtable;
  int i;
  for (i = 0; i < 63; i++) {
    if (v->vtable_name == wuffs_base__hasher_u32__vtable_name) {
      const wuffs_base__hasher_u32__func_ptrs* func_ptrs =
          (const wuffs_base__hasher_u32__func_ptrs*)(v->function_pointers);
      return (*func_ptrs->set_quirk_enabled)(self, a_quirk, a_enabled);
    } else if (v->vtable_name == NULL) {
      break;
    }
    v++;
  }

  return wuffs_base__make_empty_struct();
}

WUFFS_BASE__MAYBE_STATIC uint32_t
wuffs_base__hasher_u32__update_u32(
    wuffs_base__hasher_u32* self,
    wuffs_base__slice_u8 a_x) {
  if (!self) {
    return 0;
  }
  if (self->private_impl.magic != WUFFS_BASE__MAGIC) {
    return 0;
  }

  const wuffs_base__vtable* v = &self->private_impl.first_vtable;
  int i;
  for (i = 0; i < 63; i++) {
    if (v->vtable_name == wuffs_base__hasher_u32__vtable_name) {
      const wuffs_base__hasher_u32__func_ptrs* func_ptrs =
          (const wuffs_base__hasher_u32__func_ptrs*)(v->function_pointers);
      return (*func_ptrs->update_u32)(self, a_x);
    } else if (v->vtable_name == NULL) {
      break;
    }
    v++;
  }

  return 0;
}

// --------

WUFFS_BASE__MAYBE_STATIC wuffs_base__status
wuffs_base__image_decoder__decode_frame(
    wuffs_base__image_decoder* self,
    wuffs_base__pixel_buffer* a_dst,
    wuffs_base__io_buffer* a_src,
    wuffs_base__pixel_blend a_blend,
    wuffs_base__slice_u8 a_workbuf,
    wuffs_base__decode_frame_options* a_opts) {
  if (!self) {
    return wuffs_base__make_status(wuffs_base__error__bad_receiver);
  }
  if (self->private_impl.magic != WUFFS_BASE__MAGIC) {
    return wuffs_base__make_status(
        (self->private_impl.magic == WUFFS_BASE__DISABLED)
            ? wuffs_base__error__disabled_by_previous_error
            : wuffs_base__error__initialize_not_called);
  }

  const wuffs_base__vtable* v = &self->private_impl.first_vtable;
  int i;
  for (i = 0; i < 63; i++) {
    if (v->vtable_name == wuffs_base__image_decoder__vtable_name) {
      const wuffs_base__image_decoder__func_ptrs* func_ptrs =
          (const wuffs_base__image_decoder__func_ptrs*)(v->function_pointers);
      return (*func_ptrs->decode_frame)(self, a_dst, a_src, a_blend, a_workbuf, a_opts);
    } else if (v->vtable_name == NULL) {
      break;
    }
    v++;
  }

  return wuffs_base__make_status(wuffs_base__error__bad_vtable);
}

WUFFS_BASE__MAYBE_STATIC wuffs_base__status
wuffs_base__image_decoder__decode_frame_config(
    wuffs_base__image_decoder* self,
    wuffs_base__frame_config* a_dst,
    wuffs_base__io_buffer* a_src) {
  if (!self) {
    return wuffs_base__make_status(wuffs_base__error__bad_receiver);
  }
  if (self->private_impl.magic != WUFFS_BASE__MAGIC) {
    return wuffs_base__make_status(
        (self->private_impl.magic == WUFFS_BASE__DISABLED)
            ? wuffs_base__error__disabled_by_previous_error
            : wuffs_base__error__initialize_not_called);
  }

  const wuffs_base__vtable* v = &self->private_impl.first_vtable;
  int i;
  for (i = 0; i < 63; i++) {
    if (v->vtable_name == wuffs_base__image_decoder__vtable_name) {
      const wuffs_base__image_decoder__func_ptrs* func_ptrs =
          (const wuffs_base__image_decoder__func_ptrs*)(v->function_pointers);
      return (*func_ptrs->decode_frame_config)(self, a_dst, a_src);
    } else if (v->vtable_name == NULL) {
      break;
    }
    v++;
  }

  return wuffs_base__make_status(wuffs_base__error__bad_vtable);
}

WUFFS_BASE__MAYBE_STATIC wuffs_base__status
wuffs_base__image_decoder__decode_image_config(
    wuffs_base__image_decoder* self,
    wuffs_base__image_config* a_dst,
    wuffs_base__io_buffer* a_src) {
  if (!self) {
    return wuffs_base__make_status(wuffs_base__error__bad_receiver);
  }
  if (self->private_impl.magic != WUFFS_BASE__MAGIC) {
    return wuffs_base__make_status(
        (self->private_impl.magic == WUFFS_BASE__DISABLED)
            ? wuffs_base__error__disabled_by_previous_error
            : wuffs_base__error__initialize_not_called);
  }

  const wuffs_base__vtable* v = &self->private_impl.first_vtable;
  int i;
  for (i = 0; i < 63; i++) {
    if (v->vtable_name == wuffs_base__image_decoder__vtable_name) {
      const wuffs_base__image_decoder__func_ptrs* func_ptrs =
          (const wuffs_base__image_decoder__func_ptrs*)(v->function_pointers);
      return (*func_ptrs->decode_image_config)(self, a_dst, a_src);
    } else if (v->vtable_name == NULL) {
      break;
    }
    v++;
  }

  return wuffs_base__make_status(wuffs_base__error__bad_vtable);
}

WUFFS_BASE__MAYBE_STATIC wuffs_base__rect_ie_u32
wuffs_base__image_decoder__frame_dirty_rect(
    const wuffs_base__image_decoder* self) {
  if (!self) {
    return wuffs_base__utility__empty_rect_ie_u32();
  }
  if ((self->private_impl.magic != WUFFS_BASE__MAGIC) &&
      (self->private_impl.magic != WUFFS_BASE__DISABLED)) {
    return wuffs_base__utility__empty_rect_ie_u32();
  }

  const wuffs_base__vtable* v = &self->private_impl.first_vtable;
  int i;
  for (i = 0; i < 63; i++) {
    if (v->vtable_name == wuffs_base__image_decoder__vtable_name) {
      const wuffs_base__image_decoder__func_ptrs* func_ptrs =
          (const wuffs_base__image_decoder__func_ptrs*)(v->function_pointers);
      return (*func_ptrs->frame_dirty_rect)(self);
    } else if (v->vtable_name == NULL) {
      break;
    }
    v++;
  }

  return wuffs_base__utility__empty_rect_ie_u32();
}

WUFFS_BASE__MAYBE_STATIC uint32_t
wuffs_base__image_decoder__num_animation_loops(
    const wuffs_base__image_decoder* self) {
  if (!self) {
    return 0;
  }
  if ((self->private_impl.magic != WUFFS_BASE__MAGIC) &&
      (self->private_impl.magic != WUFFS_BASE__DISABLED)) {
    return 0;
  }

  const wuffs_base__vtable* v = &self->private_impl.first_vtable;
  int i;
  for (i = 0; i < 63; i++) {
    if (v->vtable_name == wuffs_base__image_decoder__vtable_name) {
      const wuffs_base__image_decoder__func_ptrs* func_ptrs =
          (const wuffs_base__image_decoder__func_ptrs*)(v->function_pointers);
      return (*func_ptrs->num_animation_loops)(self);
    } else if (v->vtable_name == NULL) {
      break;
    }
    v++;
  }

  return 0;
}

WUFFS_BASE__MAYBE_STATIC uint64_t
wuffs_base__image_decoder__num_decoded_frame_configs(
    const wuffs_base__image_decoder* self) {
  if (!self) {
    return 0;
  }
  if ((self->private_impl.magic != WUFFS_BASE__MAGIC) &&
      (self->private_impl.magic != WUFFS_BASE__DISABLED)) {
    return 0;
  }

  const wuffs_base__vtable* v = &self->private_impl.first_vtable;
  int i;
  for (i = 0; i < 63; i++) {
    if (v->vtable_name == wuffs_base__image_decoder__vtable_name) {
      const wuffs_base__image_decoder__func_ptrs* func_ptrs =
          (const wuffs_base__image_decoder__func_ptrs*)(v->function_pointers);
      return (*func_ptrs->num_decoded_frame_configs)(self);
    } else if (v->vtable_name == NULL) {
      break;
    }
    v++;
  }

  return 0;
}

WUFFS_BASE__MAYBE_STATIC uint64_t
wuffs_base__image_decoder__num_decoded_frames(
    const wuffs_base__image_decoder* self) {
  if (!self) {
    return 0;
  }
  if ((self->private_impl.magic != WUFFS_BASE__MAGIC) &&
      (self->private_impl.magic != WUFFS_BASE__DISABLED)) {
    return 0;
  }

  const wuffs_base__vtable* v = &self->private_impl.first_vtable;
  int i;
  for (i = 0; i < 63; i++) {
    if (v->vtable_name == wuffs_base__image_decoder__vtable_name) {
      const wuffs_base__image_decoder__func_ptrs* func_ptrs =
          (const wuffs_base__image_decoder__func_ptrs*)(v->function_pointers);
      return (*func_ptrs->num_decoded_frames)(self);
    } else if (v->vtable_name == NULL) {
      break;
    }
    v++;
  }

  return 0;
}

WUFFS_BASE__MAYBE_STATIC wuffs_base__status
wuffs_base__image_decoder__restart_frame(
    wuffs_base__image_decoder* self,
    uint64_t a_index,
    uint64_t a_io_position) {
  if (!self) {
    return wuffs_base__make_status(wuffs_base__error__bad_receiver);
  }
  if (self->private_impl.magic != WUFFS_BASE__MAGIC) {
    return wuffs_base__make_status(
        (self->private_impl.magic == WUFFS_BASE__DISABLED)
            ? wuffs_base__error__disabled_by_previous_error
            : wuffs_base__error__initialize_not_called);
  }

  const wuffs_base__vtable* v = &self->private_impl.first_vtable;
  int i;
  for (i = 0; i < 63; i++) {
    if (v->vtable_name == wuffs_base__image_decoder__vtable_name) {
      const wuffs_base__image_decoder__func_ptrs* func_ptrs =
          (const wuffs_base__image_decoder__func_ptrs*)(v->function_pointers);
      return (*func_ptrs->restart_frame)(self, a_index, a_io_position);
    } else if (v->vtable_name == NULL) {
      break;
    }
    v++;
  }

  return wuffs_base__make_status(wuffs_base__error__bad_vtable);
}

WUFFS_BASE__MAYBE_STATIC wuffs_base__empty_struct
wuffs_base__image_decoder__set_quirk_enabled(
    wuffs_base__image_decoder* self,
    uint32_t a_quirk,
    bool a_enabled) {
  if (!self) {
    return wuffs_base__make_empty_struct();
  }
  if (self->private_impl.magic != WUFFS_BASE__MAGIC) {
    return wuffs_base__make_empty_struct();
  }

  const wuffs_base__vtable* v = &self->private_impl.first_vtable;
  int i;
  for (i = 0; i < 63; i++) {
    if (v->vtable_name == wuffs_base__image_decoder__vtable_name) {
      const wuffs_base__image_decoder__func_ptrs* func_ptrs =
          (const wuffs_base__image_decoder__func_ptrs*)(v->function_pointers);
      return (*func_ptrs->set_quirk_enabled)(self, a_quirk, a_enabled);
    } else if (v->vtable_name == NULL) {
      break;
    }
    v++;
  }

  return wuffs_base__make_empty_struct();
}

WUFFS_BASE__MAYBE_STATIC wuffs_base__empty_struct
wuffs_base__image_decoder__set_report_metadata(
    wuffs_base__image_decoder* self,
    uint32_t a_fourcc,
    bool a_report) {
  if (!self) {
    return wuffs_base__make_empty_struct();
  }
  if (self->private_impl.magic != WUFFS_BASE__MAGIC) {
    return wuffs_base__make_empty_struct();
  }

  const wuffs_base__vtable* v = &self->private_impl.first_vtable;
  int i;
  for (i = 0; i < 63; i++) {
    if (v->vtable_name == wuffs_base__image_decoder__vtable_name) {
      const wuffs_base__image_decoder__func_ptrs* func_ptrs =
          (const wuffs_base__image_decoder__func_ptrs*)(v->function_pointers);
      return (*func_ptrs->set_report_metadata)(self, a_fourcc, a_report);
    } else if (v->vtable_name == NULL) {
      break;
    }
    v++;
  }

  return wuffs_base__make_empty_struct();
}

WUFFS_BASE__MAYBE_STATIC wuffs_base__status
wuffs_base__image_decoder__tell_me_more(
    wuffs_base__image_decoder* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__more_information* a_minfo,
    wuffs_base__io_buffer* a_src) {
  if (!self) {
    return wuffs_base__make_status(wuffs_base__error__bad_receiver);
  }
  if (self->private_impl.magic != WUFFS_BASE__MAGIC) {
    return wuffs_base__make_status(
        (self->private_impl.magic == WUFFS_BASE__DISABLED)
            ? wuffs_base__error__disabled_by_previous_error
            : wuffs_base__error__initialize_not_called);
  }

  const wuffs_base__vtable* v = &self->private_impl.first_vtable;
  int i;
  for (i = 0; i < 63; i++) {
    if (v->vtable_name == wuffs_base__image_decoder__vtable_name) {
      const wuffs_base__image_decoder__func_ptrs* func_ptrs =
          (const wuffs_base__image_decoder__func_ptrs*)(v->function_pointers);
      return (*func_ptrs->tell_me_more)(self, a_dst, a_minfo, a_src);
    } else if (v->vtable_name == NULL) {
      break;
    }
    v++;
  }

  return wuffs_base__make_status(wuffs_base__error__bad_vtable);
}

WUFFS_BASE__MAYBE_STATIC wuffs_base__range_ii_u64
wuffs_base__image_decoder__workbuf_len(
    const wuffs_base__image_decoder* self) {
  if (!self) {
    return wuffs_base__utility__empty_range_ii_u64();
  }
  if ((self->private_impl.magic != WUFFS_BASE__MAGIC) &&
      (self->private_impl.magic != WUFFS_BASE__DISABLED)) {
    return wuffs_base__utility__empty_range_ii_u64();
  }

  const wuffs_base__vtable* v = &self->private_impl.first_vtable;
  int i;
  for (i = 0; i < 63; i++) {
    if (v->vtable_name == wuffs_base__image_decoder__vtable_name) {
      const wuffs_base__image_decoder__func_ptrs* func_ptrs =
          (const wuffs_base__image_decoder__func_ptrs*)(v->function_pointers);
      return (*func_ptrs->workbuf_len)(self);
    } else if (v->vtable_name == NULL) {
      break;
    }
    v++;
  }

  return wuffs_base__utility__empty_range_ii_u64();
}

// --------

WUFFS_BASE__MAYBE_STATIC wuffs_base__empty_struct
wuffs_base__io_transformer__set_quirk_enabled(
    wuffs_base__io_transformer* self,
    uint32_t a_quirk,
    bool a_enabled) {
  if (!self) {
    return wuffs_base__make_empty_struct();
  }
  if (self->private_impl.magic != WUFFS_BASE__MAGIC) {
    return wuffs_base__make_empty_struct();
  }

  const wuffs_base__vtable* v = &self->private_impl.first_vtable;
  int i;
  for (i = 0; i < 63; i++) {
    if (v->vtable_name == wuffs_base__io_transformer__vtable_name) {
      const wuffs_base__io_transformer__func_ptrs* func_ptrs =
          (const wuffs_base__io_transformer__func_ptrs*)(v->function_pointers);
      return (*func_ptrs->set_quirk_enabled)(self, a_quirk, a_enabled);
    } else if (v->vtable_name == NULL) {
      break;
    }
    v++;
  }

  return wuffs_base__make_empty_struct();
}

WUFFS_BASE__MAYBE_STATIC wuffs_base__status
wuffs_base__io_transformer__transform_io(
    wuffs_base__io_transformer* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__io_buffer* a_src,
    wuffs_base__slice_u8 a_workbuf) {
  if (!self) {
    return wuffs_base__make_status(wuffs_base__error__bad_receiver);
  }
  if (self->private_impl.magic != WUFFS_BASE__MAGIC) {
    return wuffs_base__make_status(
        (self->private_impl.magic == WUFFS_BASE__DISABLED)
            ? wuffs_base__error__disabled_by_previous_error
            : wuffs_base__error__initialize_not_called);
  }

  const wuffs_base__vtable* v = &self->private_impl.first_vtable;
  int i;
  for (i = 0; i < 63; i++) {
    if (v->vtable_name == wuffs_base__io_transformer__vtable_name) {
      const wuffs_base__io_transformer__func_ptrs* func_ptrs =
          (const wuffs_base__io_transformer__func_ptrs*)(v->function_pointers);
      return (*func_ptrs->transform_io)(self, a_dst, a_src, a_workbuf);
    } else if (v->vtable_name == NULL) {
      break;
    }
    v++;
  }

  return wuffs_base__make_status(wuffs_base__error__bad_vtable);
}

WUFFS_BASE__MAYBE_STATIC wuffs_base__range_ii_u64
wuffs_base__io_transformer__workbuf_len(
    const wuffs_base__io_transformer* self) {
  if (!self) {
    return wuffs_base__utility__empty_range_ii_u64();
  }
  if ((self->private_impl.magic != WUFFS_BASE__MAGIC) &&
      (self->private_impl.magic != WUFFS_BASE__DISABLED)) {
    return wuffs_base__utility__empty_range_ii_u64();
  }

  const wuffs_base__vtable* v = &self->private_impl.first_vtable;
  int i;
  for (i = 0; i < 63; i++) {
    if (v->vtable_name == wuffs_base__io_transformer__vtable_name) {
      const wuffs_base__io_transformer__func_ptrs* func_ptrs =
          (const wuffs_base__io_transformer__func_ptrs*)(v->function_pointers);
      return (*func_ptrs->workbuf_len)(self);
    } else if (v->vtable_name == NULL) {
      break;
    }
    v++;
  }

  return wuffs_base__utility__empty_range_ii_u64();
}

// --------

WUFFS_BASE__MAYBE_STATIC wuffs_base__status
wuffs_base__token_decoder__decode_tokens(
    wuffs_base__token_decoder* self,
    wuffs_base__token_buffer* a_dst,
    wuffs_base__io_buffer* a_src,
    wuffs_base__slice_u8 a_workbuf) {
  if (!self) {
    return wuffs_base__make_status(wuffs_base__error__bad_receiver);
  }
  if (self->private_impl.magic != WUFFS_BASE__MAGIC) {
    return wuffs_base__make_status(
        (self->private_impl.magic == WUFFS_BASE__DISABLED)
            ? wuffs_base__error__disabled_by_previous_error
            : wuffs_base__error__initialize_not_called);
  }

  const wuffs_base__vtable* v = &self->private_impl.first_vtable;
  int i;
  for (i = 0; i < 63; i++) {
    if (v->vtable_name == wuffs_base__token_decoder__vtable_name) {
      const wuffs_base__token_decoder__func_ptrs* func_ptrs =
          (const wuffs_base__token_decoder__func_ptrs*)(v->function_pointers);
      return (*func_ptrs->decode_tokens)(self, a_dst, a_src, a_workbuf);
    } else if (v->vtable_name == NULL) {
      break;
    }
    v++;
  }

  return wuffs_base__make_status(wuffs_base__error__bad_vtable);
}

WUFFS_BASE__MAYBE_STATIC wuffs_base__empty_struct
wuffs_base__token_decoder__set_quirk_enabled(
    wuffs_base__token_decoder* self,
    uint32_t a_quirk,
    bool a_enabled) {
  if (!self) {
    return wuffs_base__make_empty_struct();
  }
  if (self->private_impl.magic != WUFFS_BASE__MAGIC) {
    return wuffs_base__make_empty_struct();
  }

  const wuffs_base__vtable* v = &self->private_impl.first_vtable;
  int i;
  for (i = 0; i < 63; i++) {
    if (v->vtable_name == wuffs_base__token_decoder__vtable_name) {
      const wuffs_base__token_decoder__func_ptrs* func_ptrs =
          (const wuffs_base__token_decoder__func_ptrs*)(v->function_pointers);
      return (*func_ptrs->set_quirk_enabled)(self, a_quirk, a_enabled);
    } else if (v->vtable_name == NULL) {
      break;
    }
    v++;
  }

  return wuffs_base__make_empty_struct();
}

WUFFS_BASE__MAYBE_STATIC wuffs_base__range_ii_u64
wuffs_base__token_decoder__workbuf_len(
    const wuffs_base__token_decoder* self) {
  if (!self) {
    return wuffs_base__utility__empty_range_ii_u64();
  }
  if ((self->private_impl.magic != WUFFS_BASE__MAGIC) &&
      (self->private_impl.magic != WUFFS_BASE__DISABLED)) {
    return wuffs_base__utility__empty_range_ii_u64();
  }

  const wuffs_base__vtable* v = &self->private_impl.first_vtable;
  int i;
  for (i = 0; i < 63; i++) {
    if (v->vtable_name == wuffs_base__token_decoder__vtable_name) {
      const wuffs_base__token_decoder__func_ptrs* func_ptrs =
          (const wuffs_base__token_decoder__func_ptrs*)(v->function_pointers);
      return (*func_ptrs->workbuf_len)(self);
    } else if (v->vtable_name == NULL) {
      break;
    }
    v++;
  }

  return wuffs_base__utility__empty_range_ii_u64();
}

#endif  // !defined(WUFFS_CONFIG__MODULES) ||
        // defined(WUFFS_CONFIG__MODULE__BASE) ||
        // defined(WUFFS_CONFIG__MODULE__BASE__INTERFACES)

#if !defined(WUFFS_CONFIG__MODULES) || defined(WUFFS_CONFIG__MODULE__BASE) || \
    defined(WUFFS_CONFIG__MODULE__BASE__FLOATCONV)

// ---------------- IEEE 754 Floating Point

// The etc__hpd_left_shift and etc__powers_of_5 tables were printed by
// script/print-hpd-left-shift.go. That script has an optional -comments flag,
// whose output is not copied here, which prints further detail.
//
// These tables are used in
// wuffs_base__private_implementation__high_prec_dec__lshift_num_new_digits.

// wuffs_base__private_implementation__hpd_left_shift[i] encodes the number of
// new digits created after multiplying a positive integer by (1 << i): the
// additional length in the decimal representation. For example, shifting "234"
// by 3 (equivalent to multiplying by 8) will produce "1872". Going from a
// 3-length string to a 4-length string means that 1 new digit was added (and
// existing digits may have changed).
//
// Shifting by i can add either N or N-1 new digits, depending on whether the
// original positive integer compares >= or < to the i'th power of 5 (as 10
// equals 2 * 5). Comparison is lexicographic, not numerical.
//
// For example, shifting by 4 (i.e. multiplying by 16) can add 1 or 2 new
// digits, depending on a lexicographic comparison to (5 ** 4), i.e. "625":
//  - ("1"      << 4) is "16",       which adds 1 new digit.
//  - ("5678"   << 4) is "90848",    which adds 1 new digit.
//  - ("624"    << 4) is "9984",     which adds 1 new digit.
//  - ("62498"  << 4) is "999968",   which adds 1 new digit.
//  - ("625"    << 4) is "10000",    which adds 2 new digits.
//  - ("625001" << 4) is "10000016", which adds 2 new digits.
//  - ("7008"   << 4) is "112128",   which adds 2 new digits.
//  - ("99"     << 4) is "1584",     which adds 2 new digits.
//
// Thus, when i is 4, N is 2 and (5 ** i) is "625". This etc__hpd_left_shift
// array encodes this as:
//  - etc__hpd_left_shift[4] is 0x1006 = (2 << 11) | 0x0006.
//  - etc__hpd_left_shift[5] is 0x1009 = (? << 11) | 0x0009.
// where the ? isn't relevant for i == 4.
//
// The high 5 bits of etc__hpd_left_shift[i] is N, the higher of the two
// possible number of new digits. The low 11 bits are an offset into the
// etc__powers_of_5 array (of length 0x051C, so offsets fit in 11 bits). When i
// is 4, its offset and the next one is 6 and 9, and etc__powers_of_5[6 .. 9]
// is the string "\x06\x02\x05", so the relevant power of 5 is "625".
//
// Thanks to Ken Thompson for the original idea.
static const uint16_t wuffs_base__private_implementation__hpd_left_shift[65] = {
    0x0000, 0x0800, 0x0801, 0x0803, 0x1006, 0x1009, 0x100D, 0x1812, 0x1817,
    0x181D, 0x2024, 0x202B, 0x2033, 0x203C, 0x2846, 0x2850, 0x285B, 0x3067,
    0x3073, 0x3080, 0x388E, 0x389C, 0x38AB, 0x38BB, 0x40CC, 0x40DD, 0x40EF,
    0x4902, 0x4915, 0x4929, 0x513E, 0x5153, 0x5169, 0x5180, 0x5998, 0x59B0,
    0x59C9, 0x61E3, 0x61FD, 0x6218, 0x6A34, 0x6A50, 0x6A6D, 0x6A8B, 0x72AA,
    0x72C9, 0x72E9, 0x7B0A, 0x7B2B, 0x7B4D, 0x8370, 0x8393, 0x83B7, 0x83DC,
    0x8C02, 0x8C28, 0x8C4F, 0x9477, 0x949F, 0x94C8, 0x9CF2, 0x051C, 0x051C,
    0x051C, 0x051C,
};

// wuffs_base__private_implementation__powers_of_5 contains the powers of 5,
// concatenated together: "5", "25", "125", "625", "3125", etc.
static const uint8_t wuffs_base__private_implementation__powers_of_5[0x051C] = {
    5, 2, 5, 1, 2, 5, 6, 2, 5, 3, 1, 2, 5, 1, 5, 6, 2, 5, 7, 8, 1, 2, 5, 3, 9,
    0, 6, 2, 5, 1, 9, 5, 3, 1, 2, 5, 9, 7, 6, 5, 6, 2, 5, 4, 8, 8, 2, 8, 1, 2,
    5, 2, 4, 4, 1, 4, 0, 6, 2, 5, 1, 2, 2, 0, 7, 0, 3, 1, 2, 5, 6, 1, 0, 3, 5,
    1, 5, 6, 2, 5, 3, 0, 5, 1, 7, 5, 7, 8, 1, 2, 5, 1, 5, 2, 5, 8, 7, 8, 9, 0,
    6, 2, 5, 7, 6, 2, 9, 3, 9, 4, 5, 3, 1, 2, 5, 3, 8, 1, 4, 6, 9, 7, 2, 6, 5,
    6, 2, 5, 1, 9, 0, 7, 3, 4, 8, 6, 3, 2, 8, 1, 2, 5, 9, 5, 3, 6, 7, 4, 3, 1,
    6, 4, 0, 6, 2, 5, 4, 7, 6, 8, 3, 7, 1, 5, 8, 2, 0, 3, 1, 2, 5, 2, 3, 8, 4,
    1, 8, 5, 7, 9, 1, 0, 1, 5, 6, 2, 5, 1, 1, 9, 2, 0, 9, 2, 8, 9, 5, 5, 0, 7,
    8, 1, 2, 5, 5, 9, 6, 0, 4, 6, 4, 4, 7, 7, 5, 3, 9, 0, 6, 2, 5, 2, 9, 8, 0,
    2, 3, 2, 2, 3, 8, 7, 6, 9, 5, 3, 1, 2, 5, 1, 4, 9, 0, 1, 1, 6, 1, 1, 9, 3,
    8, 4, 7, 6, 5, 6, 2, 5, 7, 4, 5, 0, 5, 8, 0, 5, 9, 6, 9, 2, 3, 8, 2, 8, 1,
    2, 5, 3, 7, 2, 5, 2, 9, 0, 2, 9, 8, 4, 6, 1, 9, 1, 4, 0, 6, 2, 5, 1, 8, 6,
    2, 6, 4, 5, 1, 4, 9, 2, 3, 0, 9, 5, 7, 0, 3, 1, 2, 5, 9, 3, 1, 3, 2, 2, 5,
    7, 4, 6, 1, 5, 4, 7, 8, 5, 1, 5, 6, 2, 5, 4, 6, 5, 6, 6, 1, 2, 8, 7, 3, 0,
    7, 7, 3, 9, 2, 5, 7, 8, 1, 2, 5, 2, 3, 2, 8, 3, 0, 6, 4, 3, 6, 5, 3, 8, 6,
    9, 6, 2, 8, 9, 0, 6, 2, 5, 1, 1, 6, 4, 1, 5, 3, 2, 1, 8, 2, 6, 9, 3, 4, 8,
    1, 4, 4, 5, 3, 1, 2, 5, 5, 8, 2, 0, 7, 6, 6, 0, 9, 1, 3, 4, 6, 7, 4, 0, 7,
    2, 2, 6, 5, 6, 2, 5, 2, 9, 1, 0, 3, 8, 3, 0, 4, 5, 6, 7, 3, 3, 7, 0, 3, 6,
    1, 3, 2, 8, 1, 2, 5, 1, 4, 5, 5, 1, 9, 1, 5, 2, 2, 8, 3, 6, 6, 8, 5, 1, 8,
    0, 6, 6, 4, 0, 6, 2, 5, 7, 2, 7, 5, 9, 5, 7, 6, 1, 4, 1, 8, 3, 4, 2, 5, 9,
    0, 3, 3, 2, 0, 3, 1, 2, 5, 3, 6, 3, 7, 9, 7, 8, 8, 0, 7, 0, 9, 1, 7, 1, 2,
    9, 5, 1, 6, 6, 0, 1, 5, 6, 2, 5, 1, 8, 1, 8, 9, 8, 9, 4, 0, 3, 5, 4, 5, 8,
    5, 6, 4, 7, 5, 8, 3, 0, 0, 7, 8, 1, 2, 5, 9, 0, 9, 4, 9, 4, 7, 0, 1, 7, 7,
    2, 9, 2, 8, 2, 3, 7, 9, 1, 5, 0, 3, 9, 0, 6, 2, 5, 4, 5, 4, 7, 4, 7, 3, 5,
    0, 8, 8, 6, 4, 6, 4, 1, 1, 8, 9, 5, 7, 5, 1, 9, 5, 3, 1, 2, 5, 2, 2, 7, 3,
    7, 3, 6, 7, 5, 4, 4, 3, 2, 3, 2, 0, 5, 9, 4, 7, 8, 7, 5, 9, 7, 6, 5, 6, 2,
    5, 1, 1, 3, 6, 8, 6, 8, 3, 7, 7, 2, 1, 6, 1, 6, 0, 2, 9, 7, 3, 9, 3, 7, 9,
    8, 8, 2, 8, 1, 2, 5, 5, 6, 8, 4, 3, 4, 1, 8, 8, 6, 0, 8, 0, 8, 0, 1, 4, 8,
    6, 9, 6, 8, 9, 9, 4, 1, 4, 0, 6, 2, 5, 2, 8, 4, 2, 1, 7, 0, 9, 4, 3, 0, 4,
    0, 4, 0, 0, 7, 4, 3, 4, 8, 4, 4, 9, 7, 0, 7, 0, 3, 1, 2, 5, 1, 4, 2, 1, 0,
    8, 5, 4, 7, 1, 5, 2, 0, 2, 0, 0, 3, 7, 1, 7, 4, 2, 2, 4, 8, 5, 3, 5, 1, 5,
    6, 2, 5, 7, 1, 0, 5, 4, 2, 7, 3, 5, 7, 6, 0, 1, 0, 0, 1, 8, 5, 8, 7, 1, 1,
    2, 4, 2, 6, 7, 5, 7, 8, 1, 2, 5, 3, 5, 5, 2, 7, 1, 3, 6, 7, 8, 8, 0, 0, 5,
    0, 0, 9, 2, 9, 3, 5, 5, 6, 2, 1, 3, 3, 7, 8, 9, 0, 6, 2, 5, 1, 7, 7, 6, 3,
    5, 6, 8, 3, 9, 4, 0, 0, 2, 5, 0, 4, 6, 4, 6, 7, 7, 8, 1, 0, 6, 6, 8, 9, 4,
    5, 3, 1, 2, 5, 8, 8, 8, 1, 7, 8, 4, 1, 9, 7, 0, 0, 1, 2, 5, 2, 3, 2, 3, 3,
    8, 9, 0, 5, 3, 3, 4, 4, 7, 2, 6, 5, 6, 2, 5, 4, 4, 4, 0, 8, 9, 2, 0, 9, 8,
    5, 0, 0, 6, 2, 6, 1, 6, 1, 6, 9, 4, 5, 2, 6, 6, 7, 2, 3, 6, 3, 2, 8, 1, 2,
    5, 2, 2, 2, 0, 4, 4, 6, 0, 4, 9, 2, 5, 0, 3, 1, 3, 0, 8, 0, 8, 4, 7, 2, 6,
    3, 3, 3, 6, 1, 8, 1, 6, 4, 0, 6, 2, 5, 1, 1, 1, 0, 2, 2, 3, 0, 2, 4, 6, 2,
    5, 1, 5, 6, 5, 4, 0, 4, 2, 3, 6, 3, 1, 6, 6, 8, 0, 9, 0, 8, 2, 0, 3, 1, 2,
    5, 5, 5, 5, 1, 1, 1, 5, 1, 2, 3, 1, 2, 5, 7, 8, 2, 7, 0, 2, 1, 1, 8, 1, 5,
    8, 3, 4, 0, 4, 5, 4, 1, 0, 1, 5, 6, 2, 5, 2, 7, 7, 5, 5, 5, 7, 5, 6, 1, 5,
    6, 2, 8, 9, 1, 3, 5, 1, 0, 5, 9, 0, 7, 9, 1, 7, 0, 2, 2, 7, 0, 5, 0, 7, 8,
    1, 2, 5, 1, 3, 8, 7, 7, 7, 8, 7, 8, 0, 7, 8, 1, 4, 4, 5, 6, 7, 5, 5, 2, 9,
    5, 3, 9, 5, 8, 5, 1, 1, 3, 5, 2, 5, 3, 9, 0, 6, 2, 5, 6, 9, 3, 8, 8, 9, 3,
    9, 0, 3, 9, 0, 7, 2, 2, 8, 3, 7, 7, 6, 4, 7, 6, 9, 7, 9, 2, 5, 5, 6, 7, 6,
    2, 6, 9, 5, 3, 1, 2, 5, 3, 4, 6, 9, 4, 4, 6, 9, 5, 1, 9, 5, 3, 6, 1, 4, 1,
    8, 8, 8, 2, 3, 8, 4, 8, 9, 6, 2, 7, 8, 3, 8, 1, 3, 4, 7, 6, 5, 6, 2, 5, 1,
    7, 3, 4, 7, 2, 3, 4, 7, 5, 9, 7, 6, 8, 0, 7, 0, 9, 4, 4, 1, 1, 9, 2, 4, 4,
    8, 1, 3, 9, 1, 9, 0, 6, 7, 3, 8, 2, 8, 1, 2, 5, 8, 6, 7, 3, 6, 1, 7, 3, 7,
    9, 8, 8, 4, 0, 3, 5, 4, 7, 2, 0, 5, 9, 6, 2, 2, 4, 0, 6, 9, 5, 9, 5, 3, 3,
    6, 9, 1, 4, 0, 6, 2, 5,
};

// --------

// wuffs_base__private_implementation__powers_of_10 contains truncated
// approximations to the powers of 10, ranging from 1e-307 to 1e+288 inclusive,
// as 596 pairs of uint64_t values (a 128-bit mantissa).
//
// There's also an implicit third column (implied by a linear formula involving
// the base-10 exponent) that is the base-2 exponent, biased by a magic
// constant. That constant (1214 or 0x04BE) equals 1023 + 191. 1023 is the bias
// for IEEE 754 double-precision floating point. 191 is ((3 * 64) - 1) and
// wuffs_base__private_implementation__parse_number_f64_eisel_lemire works with
// multiples-of-64-bit mantissas.
//
// For example, the third row holds the approximation to 1e-305:
//   0xE0B62E29_29ABA83C_331ACDAB_FE94DE87 * (2 ** (0x0049 - 0x04BE))
//
// Similarly, 1e+4 is approximated by:
//   0x9C400000_00000000_00000000_00000000 * (2 ** (0x044C - 0x04BE))
//
// Similarly, 1e+68 is approximated by:
//   0xED63A231_D4C4FB27_4CA7AAA8_63EE4BDD * (2 ** (0x0520 - 0x04BE))
//
// This table was generated by by script/print-mpb-powers-of-10.go
static const uint64_t wuffs_base__private_implementation__powers_of_10[596][2] =
    {
        {0xA5D3B6D479F8E056, 0x8FD0C16206306BAB},  // 1e-307
        {0x8F48A4899877186C, 0xB3C4F1BA87BC8696},  // 1e-306
        {0x331ACDABFE94DE87, 0xE0B62E2929ABA83C},  // 1e-305
        {0x9FF0C08B7F1D0B14, 0x8C71DCD9BA0B4925},  // 1e-304
        {0x07ECF0AE5EE44DD9, 0xAF8E5410288E1B6F},  // 1e-303
        {0xC9E82CD9F69D6150, 0xDB71E91432B1A24A},  // 1e-302
        {0xBE311C083A225CD2, 0x892731AC9FAF056E},  // 1e-301
        {0x6DBD630A48AAF406, 0xAB70FE17C79AC6CA},  // 1e-300
        {0x092CBBCCDAD5B108, 0xD64D3D9DB981787D},  // 1e-299
        {0x25BBF56008C58EA5, 0x85F0468293F0EB4E},  // 1e-298
        {0xAF2AF2B80AF6F24E, 0xA76C582338ED2621},  // 1e-297
        {0x1AF5AF660DB4AEE1, 0xD1476E2C07286FAA},  // 1e-296
        {0x50D98D9FC890ED4D, 0x82CCA4DB847945CA},  // 1e-295
        {0xE50FF107BAB528A0, 0xA37FCE126597973C},  // 1e-294
        {0x1E53ED49A96272C8, 0xCC5FC196FEFD7D0C},  // 1e-293
        {0x25E8E89C13BB0F7A, 0xFF77B1FCBEBCDC4F},  // 1e-292
        {0x77B191618C54E9AC, 0x9FAACF3DF73609B1},  // 1e-291
        {0xD59DF5B9EF6A2417, 0xC795830D75038C1D},  // 1e-290
        {0x4B0573286B44AD1D, 0xF97AE3D0D2446F25},  // 1e-289
        {0x4EE367F9430AEC32, 0x9BECCE62836AC577},  // 1e-288
        {0x229C41F793CDA73F, 0xC2E801FB244576D5},  // 1e-287
        {0x6B43527578C1110F, 0xF3A20279ED56D48A},  // 1e-286
        {0x830A13896B78AAA9, 0x9845418C345644D6},  // 1e-285
        {0x23CC986BC656D553, 0xBE5691EF416BD60C},  // 1e-284
        {0x2CBFBE86B7EC8AA8, 0xEDEC366B11C6CB8F},  // 1e-283
        {0x7BF7D71432F3D6A9, 0x94B3A202EB1C3F39},  // 1e-282
        {0xDAF5CCD93FB0CC53, 0xB9E08A83A5E34F07},  // 1e-281
        {0xD1B3400F8F9CFF68, 0xE858AD248F5C22C9},  // 1e-280
        {0x23100809B9C21FA1, 0x91376C36D99995BE},  // 1e-279
        {0xABD40A0C2832A78A, 0xB58547448FFFFB2D},  // 1e-278
        {0x16C90C8F323F516C, 0xE2E69915B3FFF9F9},  // 1e-277
        {0xAE3DA7D97F6792E3, 0x8DD01FAD907FFC3B},  // 1e-276
        {0x99CD11CFDF41779C, 0xB1442798F49FFB4A},  // 1e-275
        {0x40405643D711D583, 0xDD95317F31C7FA1D},  // 1e-274
        {0x482835EA666B2572, 0x8A7D3EEF7F1CFC52},  // 1e-273
        {0xDA3243650005EECF, 0xAD1C8EAB5EE43B66},  // 1e-272
        {0x90BED43E40076A82, 0xD863B256369D4A40},  // 1e-271
        {0x5A7744A6E804A291, 0x873E4F75E2224E68},  // 1e-270
        {0x711515D0A205CB36, 0xA90DE3535AAAE202},  // 1e-269
        {0x0D5A5B44CA873E03, 0xD3515C2831559A83},  // 1e-268
        {0xE858790AFE9486C2, 0x8412D9991ED58091},  // 1e-267
        {0x626E974DBE39A872, 0xA5178FFF668AE0B6},  // 1e-266
        {0xFB0A3D212DC8128F, 0xCE5D73FF402D98E3},  // 1e-265
        {0x7CE66634BC9D0B99, 0x80FA687F881C7F8E},  // 1e-264
        {0x1C1FFFC1EBC44E80, 0xA139029F6A239F72},  // 1e-263
        {0xA327FFB266B56220, 0xC987434744AC874E},  // 1e-262
        {0x4BF1FF9F0062BAA8, 0xFBE9141915D7A922},  // 1e-261
        {0x6F773FC3603DB4A9, 0x9D71AC8FADA6C9B5},  // 1e-260
        {0xCB550FB4384D21D3, 0xC4CE17B399107C22},  // 1e-259
        {0x7E2A53A146606A48, 0xF6019DA07F549B2B},  // 1e-258
        {0x2EDA7444CBFC426D, 0x99C102844F94E0FB},  // 1e-257
        {0xFA911155FEFB5308, 0xC0314325637A1939},  // 1e-256
        {0x793555AB7EBA27CA, 0xF03D93EEBC589F88},  // 1e-255
        {0x4BC1558B2F3458DE, 0x96267C7535B763B5},  // 1e-254
        {0x9EB1AAEDFB016F16, 0xBBB01B9283253CA2},  // 1e-253
        {0x465E15A979C1CADC, 0xEA9C227723EE8BCB},  // 1e-252
        {0x0BFACD89EC191EC9, 0x92A1958A7675175F},  // 1e-251
        {0xCEF980EC671F667B, 0xB749FAED14125D36},  // 1e-250
        {0x82B7E12780E7401A, 0xE51C79A85916F484},  // 1e-249
        {0xD1B2ECB8B0908810, 0x8F31CC0937AE58D2},  // 1e-248
        {0x861FA7E6DCB4AA15, 0xB2FE3F0B8599EF07},  // 1e-247
        {0x67A791E093E1D49A, 0xDFBDCECE67006AC9},  // 1e-246
        {0xE0C8BB2C5C6D24E0, 0x8BD6A141006042BD},  // 1e-245
        {0x58FAE9F773886E18, 0xAECC49914078536D},  // 1e-244
        {0xAF39A475506A899E, 0xDA7F5BF590966848},  // 1e-243
        {0x6D8406C952429603, 0x888F99797A5E012D},  // 1e-242
        {0xC8E5087BA6D33B83, 0xAAB37FD7D8F58178},  // 1e-241
        {0xFB1E4A9A90880A64, 0xD5605FCDCF32E1D6},  // 1e-240
        {0x5CF2EEA09A55067F, 0x855C3BE0A17FCD26},  // 1e-239
        {0xF42FAA48C0EA481E, 0xA6B34AD8C9DFC06F},  // 1e-238
        {0xF13B94DAF124DA26, 0xD0601D8EFC57B08B},  // 1e-237
        {0x76C53D08D6B70858, 0x823C12795DB6CE57},  // 1e-236
        {0x54768C4B0C64CA6E, 0xA2CB1717B52481ED},  // 1e-235
        {0xA9942F5DCF7DFD09, 0xCB7DDCDDA26DA268},  // 1e-234
        {0xD3F93B35435D7C4C, 0xFE5D54150B090B02},  // 1e-233
        {0xC47BC5014A1A6DAF, 0x9EFA548D26E5A6E1},  // 1e-232
        {0x359AB6419CA1091B, 0xC6B8E9B0709F109A},  // 1e-231
        {0xC30163D203C94B62, 0xF867241C8CC6D4C0},  // 1e-230
        {0x79E0DE63425DCF1D, 0x9B407691D7FC44F8},  // 1e-229
        {0x985915FC12F542E4, 0xC21094364DFB5636},  // 1e-228
        {0x3E6F5B7B17B2939D, 0xF294B943E17A2BC4},  // 1e-227
        {0xA705992CEECF9C42, 0x979CF3CA6CEC5B5A},  // 1e-226
        {0x50C6FF782A838353, 0xBD8430BD08277231},  // 1e-225
        {0xA4F8BF5635246428, 0xECE53CEC4A314EBD},  // 1e-224
        {0x871B7795E136BE99, 0x940F4613AE5ED136},  // 1e-223
        {0x28E2557B59846E3F, 0xB913179899F68584},  // 1e-222
        {0x331AEADA2FE589CF, 0xE757DD7EC07426E5},  // 1e-221
        {0x3FF0D2C85DEF7621, 0x9096EA6F3848984F},  // 1e-220
        {0x0FED077A756B53A9, 0xB4BCA50B065ABE63},  // 1e-219
        {0xD3E8495912C62894, 0xE1EBCE4DC7F16DFB},  // 1e-218
        {0x64712DD7ABBBD95C, 0x8D3360F09CF6E4BD},  // 1e-217
        {0xBD8D794D96AACFB3, 0xB080392CC4349DEC},  // 1e-216
        {0xECF0D7A0FC5583A0, 0xDCA04777F541C567},  // 1e-215
        {0xF41686C49DB57244, 0x89E42CAAF9491B60},  // 1e-214
        {0x311C2875C522CED5, 0xAC5D37D5B79B6239},  // 1e-213
        {0x7D633293366B828B, 0xD77485CB25823AC7},  // 1e-212
        {0xAE5DFF9C02033197, 0x86A8D39EF77164BC},  // 1e-211
        {0xD9F57F830283FDFC, 0xA8530886B54DBDEB},  // 1e-210
        {0xD072DF63C324FD7B, 0xD267CAA862A12D66},  // 1e-209
        {0x4247CB9E59F71E6D, 0x8380DEA93DA4BC60},  // 1e-208
        {0x52D9BE85F074E608, 0xA46116538D0DEB78},  // 1e-207
        {0x67902E276C921F8B, 0xCD795BE870516656},  // 1e-206
        {0x00BA1CD8A3DB53B6, 0x806BD9714632DFF6},  // 1e-205
        {0x80E8A40ECCD228A4, 0xA086CFCD97BF97F3},  // 1e-204
        {0x6122CD128006B2CD, 0xC8A883C0FDAF7DF0},  // 1e-203
        {0x796B805720085F81, 0xFAD2A4B13D1B5D6C},  // 1e-202
        {0xCBE3303674053BB0, 0x9CC3A6EEC6311A63},  // 1e-201
        {0xBEDBFC4411068A9C, 0xC3F490AA77BD60FC},  // 1e-200
        {0xEE92FB5515482D44, 0xF4F1B4D515ACB93B},  // 1e-199
        {0x751BDD152D4D1C4A, 0x991711052D8BF3C5},  // 1e-198
        {0xD262D45A78A0635D, 0xBF5CD54678EEF0B6},  // 1e-197
        {0x86FB897116C87C34, 0xEF340A98172AACE4},  // 1e-196
        {0xD45D35E6AE3D4DA0, 0x9580869F0E7AAC0E},  // 1e-195
        {0x8974836059CCA109, 0xBAE0A846D2195712},  // 1e-194
        {0x2BD1A438703FC94B, 0xE998D258869FACD7},  // 1e-193
        {0x7B6306A34627DDCF, 0x91FF83775423CC06},  // 1e-192
        {0x1A3BC84C17B1D542, 0xB67F6455292CBF08},  // 1e-191
        {0x20CABA5F1D9E4A93, 0xE41F3D6A7377EECA},  // 1e-190
        {0x547EB47B7282EE9C, 0x8E938662882AF53E},  // 1e-189
        {0xE99E619A4F23AA43, 0xB23867FB2A35B28D},  // 1e-188
        {0x6405FA00E2EC94D4, 0xDEC681F9F4C31F31},  // 1e-187
        {0xDE83BC408DD3DD04, 0x8B3C113C38F9F37E},  // 1e-186
        {0x9624AB50B148D445, 0xAE0B158B4738705E},  // 1e-185
        {0x3BADD624DD9B0957, 0xD98DDAEE19068C76},  // 1e-184
        {0xE54CA5D70A80E5D6, 0x87F8A8D4CFA417C9},  // 1e-183
        {0x5E9FCF4CCD211F4C, 0xA9F6D30A038D1DBC},  // 1e-182
        {0x7647C3200069671F, 0xD47487CC8470652B},  // 1e-181
        {0x29ECD9F40041E073, 0x84C8D4DFD2C63F3B},  // 1e-180
        {0xF468107100525890, 0xA5FB0A17C777CF09},  // 1e-179
        {0x7182148D4066EEB4, 0xCF79CC9DB955C2CC},  // 1e-178
        {0xC6F14CD848405530, 0x81AC1FE293D599BF},  // 1e-177
        {0xB8ADA00E5A506A7C, 0xA21727DB38CB002F},  // 1e-176
        {0xA6D90811F0E4851C, 0xCA9CF1D206FDC03B},  // 1e-175
        {0x908F4A166D1DA663, 0xFD442E4688BD304A},  // 1e-174
        {0x9A598E4E043287FE, 0x9E4A9CEC15763E2E},  // 1e-173
        {0x40EFF1E1853F29FD, 0xC5DD44271AD3CDBA},  // 1e-172
        {0xD12BEE59E68EF47C, 0xF7549530E188C128},  // 1e-171
        {0x82BB74F8301958CE, 0x9A94DD3E8CF578B9},  // 1e-170
        {0xE36A52363C1FAF01, 0xC13A148E3032D6E7},  // 1e-169
        {0xDC44E6C3CB279AC1, 0xF18899B1BC3F8CA1},  // 1e-168
        {0x29AB103A5EF8C0B9, 0x96F5600F15A7B7E5},  // 1e-167
        {0x7415D448F6B6F0E7, 0xBCB2B812DB11A5DE},  // 1e-166
        {0x111B495B3464AD21, 0xEBDF661791D60F56},  // 1e-165
        {0xCAB10DD900BEEC34, 0x936B9FCEBB25C995},  // 1e-164
        {0x3D5D514F40EEA742, 0xB84687C269EF3BFB},  // 1e-163
        {0x0CB4A5A3112A5112, 0xE65829B3046B0AFA},  // 1e-162
        {0x47F0E785EABA72AB, 0x8FF71A0FE2C2E6DC},  // 1e-161
        {0x59ED216765690F56, 0xB3F4E093DB73A093},  // 1e-160
        {0x306869C13EC3532C, 0xE0F218B8D25088B8},  // 1e-159
        {0x1E414218C73A13FB, 0x8C974F7383725573},  // 1e-158
        {0xE5D1929EF90898FA, 0xAFBD2350644EEACF},  // 1e-157
        {0xDF45F746B74ABF39, 0xDBAC6C247D62A583},  // 1e-156
        {0x6B8BBA8C328EB783, 0x894BC396CE5DA772},  // 1e-155
        {0x066EA92F3F326564, 0xAB9EB47C81F5114F},  // 1e-154
        {0xC80A537B0EFEFEBD, 0xD686619BA27255A2},  // 1e-153
        {0xBD06742CE95F5F36, 0x8613FD0145877585},  // 1e-152
        {0x2C48113823B73704, 0xA798FC4196E952E7},  // 1e-151
        {0xF75A15862CA504C5, 0xD17F3B51FCA3A7A0},  // 1e-150
        {0x9A984D73DBE722FB, 0x82EF85133DE648C4},  // 1e-149
        {0xC13E60D0D2E0EBBA, 0xA3AB66580D5FDAF5},  // 1e-148
        {0x318DF905079926A8, 0xCC963FEE10B7D1B3},  // 1e-147
        {0xFDF17746497F7052, 0xFFBBCFE994E5C61F},  // 1e-146
        {0xFEB6EA8BEDEFA633, 0x9FD561F1FD0F9BD3},  // 1e-145
        {0xFE64A52EE96B8FC0, 0xC7CABA6E7C5382C8},  // 1e-144
        {0x3DFDCE7AA3C673B0, 0xF9BD690A1B68637B},  // 1e-143
        {0x06BEA10CA65C084E, 0x9C1661A651213E2D},  // 1e-142
        {0x486E494FCFF30A62, 0xC31BFA0FE5698DB8},  // 1e-141
        {0x5A89DBA3C3EFCCFA, 0xF3E2F893DEC3F126},  // 1e-140
        {0xF89629465A75E01C, 0x986DDB5C6B3A76B7},  // 1e-139
        {0xF6BBB397F1135823, 0xBE89523386091465},  // 1e-138
        {0x746AA07DED582E2C, 0xEE2BA6C0678B597F},  // 1e-137
        {0xA8C2A44EB4571CDC, 0x94DB483840B717EF},  // 1e-136
        {0x92F34D62616CE413, 0xBA121A4650E4DDEB},  // 1e-135
        {0x77B020BAF9C81D17, 0xE896A0D7E51E1566},  // 1e-134
        {0x0ACE1474DC1D122E, 0x915E2486EF32CD60},  // 1e-133
        {0x0D819992132456BA, 0xB5B5ADA8AAFF80B8},  // 1e-132
        {0x10E1FFF697ED6C69, 0xE3231912D5BF60E6},  // 1e-131
        {0xCA8D3FFA1EF463C1, 0x8DF5EFABC5979C8F},  // 1e-130
        {0xBD308FF8A6B17CB2, 0xB1736B96B6FD83B3},  // 1e-129
        {0xAC7CB3F6D05DDBDE, 0xDDD0467C64BCE4A0},  // 1e-128
        {0x6BCDF07A423AA96B, 0x8AA22C0DBEF60EE4},  // 1e-127
        {0x86C16C98D2C953C6, 0xAD4AB7112EB3929D},  // 1e-126
        {0xE871C7BF077BA8B7, 0xD89D64D57A607744},  // 1e-125
        {0x11471CD764AD4972, 0x87625F056C7C4A8B},  // 1e-124
        {0xD598E40D3DD89BCF, 0xA93AF6C6C79B5D2D},  // 1e-123
        {0x4AFF1D108D4EC2C3, 0xD389B47879823479},  // 1e-122
        {0xCEDF722A585139BA, 0x843610CB4BF160CB},  // 1e-121
        {0xC2974EB4EE658828, 0xA54394FE1EEDB8FE},  // 1e-120
        {0x733D226229FEEA32, 0xCE947A3DA6A9273E},  // 1e-119
        {0x0806357D5A3F525F, 0x811CCC668829B887},  // 1e-118
        {0xCA07C2DCB0CF26F7, 0xA163FF802A3426A8},  // 1e-117
        {0xFC89B393DD02F0B5, 0xC9BCFF6034C13052},  // 1e-116
        {0xBBAC2078D443ACE2, 0xFC2C3F3841F17C67},  // 1e-115
        {0xD54B944B84AA4C0D, 0x9D9BA7832936EDC0},  // 1e-114
        {0x0A9E795E65D4DF11, 0xC5029163F384A931},  // 1e-113
        {0x4D4617B5FF4A16D5, 0xF64335BCF065D37D},  // 1e-112
        {0x504BCED1BF8E4E45, 0x99EA0196163FA42E},  // 1e-111
        {0xE45EC2862F71E1D6, 0xC06481FB9BCF8D39},  // 1e-110
        {0x5D767327BB4E5A4C, 0xF07DA27A82C37088},  // 1e-109
        {0x3A6A07F8D510F86F, 0x964E858C91BA2655},  // 1e-108
        {0x890489F70A55368B, 0xBBE226EFB628AFEA},  // 1e-107
        {0x2B45AC74CCEA842E, 0xEADAB0ABA3B2DBE5},  // 1e-106
        {0x3B0B8BC90012929D, 0x92C8AE6B464FC96F},  // 1e-105
        {0x09CE6EBB40173744, 0xB77ADA0617E3BBCB},  // 1e-104
        {0xCC420A6A101D0515, 0xE55990879DDCAABD},  // 1e-103
        {0x9FA946824A12232D, 0x8F57FA54C2A9EAB6},  // 1e-102
        {0x47939822DC96ABF9, 0xB32DF8E9F3546564},  // 1e-101
        {0x59787E2B93BC56F7, 0xDFF9772470297EBD},  // 1e-100
        {0x57EB4EDB3C55B65A, 0x8BFBEA76C619EF36},  // 1e-99
        {0xEDE622920B6B23F1, 0xAEFAE51477A06B03},  // 1e-98
        {0xE95FAB368E45ECED, 0xDAB99E59958885C4},  // 1e-97
        {0x11DBCB0218EBB414, 0x88B402F7FD75539B},  // 1e-96
        {0xD652BDC29F26A119, 0xAAE103B5FCD2A881},  // 1e-95
        {0x4BE76D3346F0495F, 0xD59944A37C0752A2},  // 1e-94
        {0x6F70A4400C562DDB, 0x857FCAE62D8493A5},  // 1e-93
        {0xCB4CCD500F6BB952, 0xA6DFBD9FB8E5B88E},  // 1e-92
        {0x7E2000A41346A7A7, 0xD097AD07A71F26B2},  // 1e-91
        {0x8ED400668C0C28C8, 0x825ECC24C873782F},  // 1e-90
        {0x728900802F0F32FA, 0xA2F67F2DFA90563B},  // 1e-89
        {0x4F2B40A03AD2FFB9, 0xCBB41EF979346BCA},  // 1e-88
        {0xE2F610C84987BFA8, 0xFEA126B7D78186BC},  // 1e-87
        {0x0DD9CA7D2DF4D7C9, 0x9F24B832E6B0F436},  // 1e-86
        {0x91503D1C79720DBB, 0xC6EDE63FA05D3143},  // 1e-85
        {0x75A44C6397CE912A, 0xF8A95FCF88747D94},  // 1e-84
        {0xC986AFBE3EE11ABA, 0x9B69DBE1B548CE7C},  // 1e-83
        {0xFBE85BADCE996168, 0xC24452DA229B021B},  // 1e-82
        {0xFAE27299423FB9C3, 0xF2D56790AB41C2A2},  // 1e-81
        {0xDCCD879FC967D41A, 0x97C560BA6B0919A5},  // 1e-80
        {0x5400E987BBC1C920, 0xBDB6B8E905CB600F},  // 1e-79
        {0x290123E9AAB23B68, 0xED246723473E3813},  // 1e-78
        {0xF9A0B6720AAF6521, 0x9436C0760C86E30B},  // 1e-77
        {0xF808E40E8D5B3E69, 0xB94470938FA89BCE},  // 1e-76
        {0xB60B1D1230B20E04, 0xE7958CB87392C2C2},  // 1e-75
        {0xB1C6F22B5E6F48C2, 0x90BD77F3483BB9B9},  // 1e-74
        {0x1E38AEB6360B1AF3, 0xB4ECD5F01A4AA828},  // 1e-73
        {0x25C6DA63C38DE1B0, 0xE2280B6C20DD5232},  // 1e-72
        {0x579C487E5A38AD0E, 0x8D590723948A535F},  // 1e-71
        {0x2D835A9DF0C6D851, 0xB0AF48EC79ACE837},  // 1e-70
        {0xF8E431456CF88E65, 0xDCDB1B2798182244},  // 1e-69
        {0x1B8E9ECB641B58FF, 0x8A08F0F8BF0F156B},  // 1e-68
        {0xE272467E3D222F3F, 0xAC8B2D36EED2DAC5},  // 1e-67
        {0x5B0ED81DCC6ABB0F, 0xD7ADF884AA879177},  // 1e-66
        {0x98E947129FC2B4E9, 0x86CCBB52EA94BAEA},  // 1e-65
        {0x3F2398D747B36224, 0xA87FEA27A539E9A5},  // 1e-64
        {0x8EEC7F0D19A03AAD, 0xD29FE4B18E88640E},  // 1e-63
        {0x1953CF68300424AC, 0x83A3EEEEF9153E89},  // 1e-62
        {0x5FA8C3423C052DD7, 0xA48CEAAAB75A8E2B},  // 1e-61
        {0x3792F412CB06794D, 0xCDB02555653131B6},  // 1e-60
        {0xE2BBD88BBEE40BD0, 0x808E17555F3EBF11},  // 1e-59
        {0x5B6ACEAEAE9D0EC4, 0xA0B19D2AB70E6ED6},  // 1e-58
        {0xF245825A5A445275, 0xC8DE047564D20A8B},  // 1e-57
        {0xEED6E2F0F0D56712, 0xFB158592BE068D2E},  // 1e-56
        {0x55464DD69685606B, 0x9CED737BB6C4183D},  // 1e-55
        {0xAA97E14C3C26B886, 0xC428D05AA4751E4C},  // 1e-54
        {0xD53DD99F4B3066A8, 0xF53304714D9265DF},  // 1e-53
        {0xE546A8038EFE4029, 0x993FE2C6D07B7FAB},  // 1e-52
        {0xDE98520472BDD033, 0xBF8FDB78849A5F96},  // 1e-51
        {0x963E66858F6D4440, 0xEF73D256A5C0F77C},  // 1e-50
        {0xDDE7001379A44AA8, 0x95A8637627989AAD},  // 1e-49
        {0x5560C018580D5D52, 0xBB127C53B17EC159},  // 1e-48
        {0xAAB8F01E6E10B4A6, 0xE9D71B689DDE71AF},  // 1e-47
        {0xCAB3961304CA70E8, 0x9226712162AB070D},  // 1e-46
        {0x3D607B97C5FD0D22, 0xB6B00D69BB55C8D1},  // 1e-45
        {0x8CB89A7DB77C506A, 0xE45C10C42A2B3B05},  // 1e-44
        {0x77F3608E92ADB242, 0x8EB98A7A9A5B04E3},  // 1e-43
        {0x55F038B237591ED3, 0xB267ED1940F1C61C},  // 1e-42
        {0x6B6C46DEC52F6688, 0xDF01E85F912E37A3},  // 1e-41
        {0x2323AC4B3B3DA015, 0x8B61313BBABCE2C6},  // 1e-40
        {0xABEC975E0A0D081A, 0xAE397D8AA96C1B77},  // 1e-39
        {0x96E7BD358C904A21, 0xD9C7DCED53C72255},  // 1e-38
        {0x7E50D64177DA2E54, 0x881CEA14545C7575},  // 1e-37
        {0xDDE50BD1D5D0B9E9, 0xAA242499697392D2},  // 1e-36
        {0x955E4EC64B44E864, 0xD4AD2DBFC3D07787},  // 1e-35
        {0xBD5AF13BEF0B113E, 0x84EC3C97DA624AB4},  // 1e-34
        {0xECB1AD8AEACDD58E, 0xA6274BBDD0FADD61},  // 1e-33
        {0x67DE18EDA5814AF2, 0xCFB11EAD453994BA},  // 1e-32
        {0x80EACF948770CED7, 0x81CEB32C4B43FCF4},  // 1e-31
        {0xA1258379A94D028D, 0xA2425FF75E14FC31},  // 1e-30
        {0x096EE45813A04330, 0xCAD2F7F5359A3B3E},  // 1e-29
        {0x8BCA9D6E188853FC, 0xFD87B5F28300CA0D},  // 1e-28
        {0x775EA264CF55347D, 0x9E74D1B791E07E48},  // 1e-27
        {0x95364AFE032A819D, 0xC612062576589DDA},  // 1e-26
        {0x3A83DDBD83F52204, 0xF79687AED3EEC551},  // 1e-25
        {0xC4926A9672793542, 0x9ABE14CD44753B52},  // 1e-24
        {0x75B7053C0F178293, 0xC16D9A0095928A27},  // 1e-23
        {0x5324C68B12DD6338, 0xF1C90080BAF72CB1},  // 1e-22
        {0xD3F6FC16EBCA5E03, 0x971DA05074DA7BEE},  // 1e-21
        {0x88F4BB1CA6BCF584, 0xBCE5086492111AEA},  // 1e-20
        {0x2B31E9E3D06C32E5, 0xEC1E4A7DB69561A5},  // 1e-19
        {0x3AFF322E62439FCF, 0x9392EE8E921D5D07},  // 1e-18
        {0x09BEFEB9FAD487C2, 0xB877AA3236A4B449},  // 1e-17
        {0x4C2EBE687989A9B3, 0xE69594BEC44DE15B},  // 1e-16
        {0x0F9D37014BF60A10, 0x901D7CF73AB0ACD9},  // 1e-15
        {0x538484C19EF38C94, 0xB424DC35095CD80F},  // 1e-14
        {0x2865A5F206B06FB9, 0xE12E13424BB40E13},  // 1e-13
        {0xF93F87B7442E45D3, 0x8CBCCC096F5088CB},  // 1e-12
        {0xF78F69A51539D748, 0xAFEBFF0BCB24AAFE},  // 1e-11
        {0xB573440E5A884D1B, 0xDBE6FECEBDEDD5BE},  // 1e-10
        {0x31680A88F8953030, 0x89705F4136B4A597},  // 1e-9
        {0xFDC20D2B36BA7C3D, 0xABCC77118461CEFC},  // 1e-8
        {0x3D32907604691B4C, 0xD6BF94D5E57A42BC},  // 1e-7
        {0xA63F9A49C2C1B10F, 0x8637BD05AF6C69B5},  // 1e-6
        {0x0FCF80DC33721D53, 0xA7C5AC471B478423},  // 1e-5
        {0xD3C36113404EA4A8, 0xD1B71758E219652B},  // 1e-4
        {0x645A1CAC083126E9, 0x83126E978D4FDF3B},  // 1e-3
        {0x3D70A3D70A3D70A3, 0xA3D70A3D70A3D70A},  // 1e-2
        {0xCCCCCCCCCCCCCCCC, 0xCCCCCCCCCCCCCCCC},  // 1e-1
        {0x0000000000000000, 0x8000000000000000},  // 1e0
        {0x0000000000000000, 0xA000000000000000},  // 1e1
        {0x0000000000000000, 0xC800000000000000},  // 1e2
        {0x0000000000000000, 0xFA00000000000000},  // 1e3
        {0x0000000000000000, 0x9C40000000000000},  // 1e4
        {0x0000000000000000, 0xC350000000000000},  // 1e5
        {0x0000000000000000, 0xF424000000000000},  // 1e6
        {0x0000000000000000, 0x9896800000000000},  // 1e7
        {0x0000000000000000, 0xBEBC200000000000},  // 1e8
        {0x0000000000000000, 0xEE6B280000000000},  // 1e9
        {0x0000000000000000, 0x9502F90000000000},  // 1e10
        {0x0000000000000000, 0xBA43B74000000000},  // 1e11
        {0x0000000000000000, 0xE8D4A51000000000},  // 1e12
        {0x0000000000000000, 0x9184E72A00000000},  // 1e13
        {0x0000000000000000, 0xB5E620F480000000},  // 1e14
        {0x0000000000000000, 0xE35FA931A0000000},  // 1e15
        {0x0000000000000000, 0x8E1BC9BF04000000},  // 1e16
        {0x0000000000000000, 0xB1A2BC2EC5000000},  // 1e17
        {0x0000000000000000, 0xDE0B6B3A76400000},  // 1e18
        {0x0000000000000000, 0x8AC7230489E80000},  // 1e19
        {0x0000000000000000, 0xAD78EBC5AC620000},  // 1e20
        {0x0000000000000000, 0xD8D726B7177A8000},  // 1e21
        {0x0000000000000000, 0x878678326EAC9000},  // 1e22
        {0x0000000000000000, 0xA968163F0A57B400},  // 1e23
        {0x0000000000000000, 0xD3C21BCECCEDA100},  // 1e24
        {0x0000000000000000, 0x84595161401484A0},  // 1e25
        {0x0000000000000000, 0xA56FA5B99019A5C8},  // 1e26
        {0x0000000000000000, 0xCECB8F27F4200F3A},  // 1e27
        {0x4000000000000000, 0x813F3978F8940984},  // 1e28
        {0x5000000000000000, 0xA18F07D736B90BE5},  // 1e29
        {0xA400000000000000, 0xC9F2C9CD04674EDE},  // 1e30
        {0x4D00000000000000, 0xFC6F7C4045812296},  // 1e31
        {0xF020000000000000, 0x9DC5ADA82B70B59D},  // 1e32
        {0x6C28000000000000, 0xC5371912364CE305},  // 1e33
        {0xC732000000000000, 0xF684DF56C3E01BC6},  // 1e34
        {0x3C7F400000000000, 0x9A130B963A6C115C},  // 1e35
        {0x4B9F100000000000, 0xC097CE7BC90715B3},  // 1e36
        {0x1E86D40000000000, 0xF0BDC21ABB48DB20},  // 1e37
        {0x1314448000000000, 0x96769950B50D88F4},  // 1e38
        {0x17D955A000000000, 0xBC143FA4E250EB31},  // 1e39
        {0x5DCFAB0800000000, 0xEB194F8E1AE525FD},  // 1e40
        {0x5AA1CAE500000000, 0x92EFD1B8D0CF37BE},  // 1e41
        {0xF14A3D9E40000000, 0xB7ABC627050305AD},  // 1e42
        {0x6D9CCD05D0000000, 0xE596B7B0C643C719},  // 1e43
        {0xE4820023A2000000, 0x8F7E32CE7BEA5C6F},  // 1e44
        {0xDDA2802C8A800000, 0xB35DBF821AE4F38B},  // 1e45
        {0xD50B2037AD200000, 0xE0352F62A19E306E},  // 1e46
        {0x4526F422CC340000, 0x8C213D9DA502DE45},  // 1e47
        {0x9670B12B7F410000, 0xAF298D050E4395D6},  // 1e48
        {0x3C0CDD765F114000, 0xDAF3F04651D47B4C},  // 1e49
        {0xA5880A69FB6AC800, 0x88D8762BF324CD0F},  // 1e50
        {0x8EEA0D047A457A00, 0xAB0E93B6EFEE0053},  // 1e51
        {0x72A4904598D6D880, 0xD5D238A4ABE98068},  // 1e52
        {0x47A6DA2B7F864750, 0x85A36366EB71F041},  // 1e53
        {0x999090B65F67D924, 0xA70C3C40A64E6C51},  // 1e54
        {0xFFF4B4E3F741CF6D, 0xD0CF4B50CFE20765},  // 1e55
        {0xBFF8F10E7A8921A4, 0x82818F1281ED449F},  // 1e56
        {0xAFF72D52192B6A0D, 0xA321F2D7226895C7},  // 1e57
        {0x9BF4F8A69F764490, 0xCBEA6F8CEB02BB39},  // 1e58
        {0x02F236D04753D5B4, 0xFEE50B7025C36A08},  // 1e59
        {0x01D762422C946590, 0x9F4F2726179A2245},  // 1e60
        {0x424D3AD2B7B97EF5, 0xC722F0EF9D80AAD6},  // 1e61
        {0xD2E0898765A7DEB2, 0xF8EBAD2B84E0D58B},  // 1e62
        {0x63CC55F49F88EB2F, 0x9B934C3B330C8577},  // 1e63
        {0x3CBF6B71C76B25FB, 0xC2781F49FFCFA6D5},  // 1e64
        {0x8BEF464E3945EF7A, 0xF316271C7FC3908A},  // 1e65
        {0x97758BF0E3CBB5AC, 0x97EDD871CFDA3A56},  // 1e66
        {0x3D52EEED1CBEA317, 0xBDE94E8E43D0C8EC},  // 1e67
        {0x4CA7AAA863EE4BDD, 0xED63A231D4C4FB27},  // 1e68
        {0x8FE8CAA93E74EF6A, 0x945E455F24FB1CF8},  // 1e69
        {0xB3E2FD538E122B44, 0xB975D6B6EE39E436},  // 1e70
        {0x60DBBCA87196B616, 0xE7D34C64A9C85D44},  // 1e71
        {0xBC8955E946FE31CD, 0x90E40FBEEA1D3A4A},  // 1e72
        {0x6BABAB6398BDBE41, 0xB51D13AEA4A488DD},  // 1e73
        {0xC696963C7EED2DD1, 0xE264589A4DCDAB14},  // 1e74
        {0xFC1E1DE5CF543CA2, 0x8D7EB76070A08AEC},  // 1e75
        {0x3B25A55F43294BCB, 0xB0DE65388CC8ADA8},  // 1e76
        {0x49EF0EB713F39EBE, 0xDD15FE86AFFAD912},  // 1e77
        {0x6E3569326C784337, 0x8A2DBF142DFCC7AB},  // 1e78
        {0x49C2C37F07965404, 0xACB92ED9397BF996},  // 1e79
        {0xDC33745EC97BE906, 0xD7E77A8F87DAF7FB},  // 1e80
        {0x69A028BB3DED71A3, 0x86F0AC99B4E8DAFD},  // 1e81
        {0xC40832EA0D68CE0C, 0xA8ACD7C0222311BC},  // 1e82
        {0xF50A3FA490C30190, 0xD2D80DB02AABD62B},  // 1e83
        {0x792667C6DA79E0FA, 0x83C7088E1AAB65DB},  // 1e84
        {0x577001B891185938, 0xA4B8CAB1A1563F52},  // 1e85
        {0xED4C0226B55E6F86, 0xCDE6FD5E09ABCF26},  // 1e86
        {0x544F8158315B05B4, 0x80B05E5AC60B6178},  // 1e87
        {0x696361AE3DB1C721, 0xA0DC75F1778E39D6},  // 1e88
        {0x03BC3A19CD1E38E9, 0xC913936DD571C84C},  // 1e89
        {0x04AB48A04065C723, 0xFB5878494ACE3A5F},  // 1e90
        {0x62EB0D64283F9C76, 0x9D174B2DCEC0E47B},  // 1e91
        {0x3BA5D0BD324F8394, 0xC45D1DF942711D9A},  // 1e92
        {0xCA8F44EC7EE36479, 0xF5746577930D6500},  // 1e93
        {0x7E998B13CF4E1ECB, 0x9968BF6ABBE85F20},  // 1e94
        {0x9E3FEDD8C321A67E, 0xBFC2EF456AE276E8},  // 1e95
        {0xC5CFE94EF3EA101E, 0xEFB3AB16C59B14A2},  // 1e96
        {0xBBA1F1D158724A12, 0x95D04AEE3B80ECE5},  // 1e97
        {0x2A8A6E45AE8EDC97, 0xBB445DA9CA61281F},  // 1e98
        {0xF52D09D71A3293BD, 0xEA1575143CF97226},  // 1e99
        {0x593C2626705F9C56, 0x924D692CA61BE758},  // 1e100
        {0x6F8B2FB00C77836C, 0xB6E0C377CFA2E12E},  // 1e101
        {0x0B6DFB9C0F956447, 0xE498F455C38B997A},  // 1e102
        {0x4724BD4189BD5EAC, 0x8EDF98B59A373FEC},  // 1e103
        {0x58EDEC91EC2CB657, 0xB2977EE300C50FE7},  // 1e104
        {0x2F2967B66737E3ED, 0xDF3D5E9BC0F653E1},  // 1e105
        {0xBD79E0D20082EE74, 0x8B865B215899F46C},  // 1e106
        {0xECD8590680A3AA11, 0xAE67F1E9AEC07187},  // 1e107
        {0xE80E6F4820CC9495, 0xDA01EE641A708DE9},  // 1e108
        {0x3109058D147FDCDD, 0x884134FE908658B2},  // 1e109
        {0xBD4B46F0599FD415, 0xAA51823E34A7EEDE},  // 1e110
        {0x6C9E18AC7007C91A, 0xD4E5E2CDC1D1EA96},  // 1e111
        {0x03E2CF6BC604DDB0, 0x850FADC09923329E},  // 1e112
        {0x84DB8346B786151C, 0xA6539930BF6BFF45},  // 1e113
        {0xE612641865679A63, 0xCFE87F7CEF46FF16},  // 1e114
        {0x4FCB7E8F3F60C07E, 0x81F14FAE158C5F6E},  // 1e115
        {0xE3BE5E330F38F09D, 0xA26DA3999AEF7749},  // 1e116
        {0x5CADF5BFD3072CC5, 0xCB090C8001AB551C},  // 1e117
        {0x73D9732FC7C8F7F6, 0xFDCB4FA002162A63},  // 1e118
        {0x2867E7FDDCDD9AFA, 0x9E9F11C4014DDA7E},  // 1e119
        {0xB281E1FD541501B8, 0xC646D63501A1511D},  // 1e120
        {0x1F225A7CA91A4226, 0xF7D88BC24209A565},  // 1e121
        {0x3375788DE9B06958, 0x9AE757596946075F},  // 1e122
        {0x0052D6B1641C83AE, 0xC1A12D2FC3978937},  // 1e123
        {0xC0678C5DBD23A49A, 0xF209787BB47D6B84},  // 1e124
        {0xF840B7BA963646E0, 0x9745EB4D50CE6332},  // 1e125
        {0xB650E5A93BC3D898, 0xBD176620A501FBFF},  // 1e126
        {0xA3E51F138AB4CEBE, 0xEC5D3FA8CE427AFF},  // 1e127
        {0xC66F336C36B10137, 0x93BA47C980E98CDF},  // 1e128
        {0xB80B0047445D4184, 0xB8A8D9BBE123F017},  // 1e129
        {0xA60DC059157491E5, 0xE6D3102AD96CEC1D},  // 1e130
        {0x87C89837AD68DB2F, 0x9043EA1AC7E41392},  // 1e131
        {0x29BABE4598C311FB, 0xB454E4A179DD1877},  // 1e132
        {0xF4296DD6FEF3D67A, 0xE16A1DC9D8545E94},  // 1e133
        {0x1899E4A65F58660C, 0x8CE2529E2734BB1D},  // 1e134
        {0x5EC05DCFF72E7F8F, 0xB01AE745B101E9E4},  // 1e135
        {0x76707543F4FA1F73, 0xDC21A1171D42645D},  // 1e136
        {0x6A06494A791C53A8, 0x899504AE72497EBA},  // 1e137
        {0x0487DB9D17636892, 0xABFA45DA0EDBDE69},  // 1e138
        {0x45A9D2845D3C42B6, 0xD6F8D7509292D603},  // 1e139
        {0x0B8A2392BA45A9B2, 0x865B86925B9BC5C2},  // 1e140
        {0x8E6CAC7768D7141E, 0xA7F26836F282B732},  // 1e141
        {0x3207D795430CD926, 0xD1EF0244AF2364FF},  // 1e142
        {0x7F44E6BD49E807B8, 0x8335616AED761F1F},  // 1e143
        {0x5F16206C9C6209A6, 0xA402B9C5A8D3A6E7},  // 1e144
        {0x36DBA887C37A8C0F, 0xCD036837130890A1},  // 1e145
        {0xC2494954DA2C9789, 0x802221226BE55A64},  // 1e146
        {0xF2DB9BAA10B7BD6C, 0xA02AA96B06DEB0FD},  // 1e147
        {0x6F92829494E5ACC7, 0xC83553C5C8965D3D},  // 1e148
        {0xCB772339BA1F17F9, 0xFA42A8B73ABBF48C},  // 1e149
        {0xFF2A760414536EFB, 0x9C69A97284B578D7},  // 1e150
        {0xFEF5138519684ABA, 0xC38413CF25E2D70D},  // 1e151
        {0x7EB258665FC25D69, 0xF46518C2EF5B8CD1},  // 1e152
        {0xEF2F773FFBD97A61, 0x98BF2F79D5993802},  // 1e153
        {0xAAFB550FFACFD8FA, 0xBEEEFB584AFF8603},  // 1e154
        {0x95BA2A53F983CF38, 0xEEAABA2E5DBF6784},  // 1e155
        {0xDD945A747BF26183, 0x952AB45CFA97A0B2},  // 1e156
        {0x94F971119AEEF9E4, 0xBA756174393D88DF},  // 1e157
        {0x7A37CD5601AAB85D, 0xE912B9D1478CEB17},  // 1e158
        {0xAC62E055C10AB33A, 0x91ABB422CCB812EE},  // 1e159
        {0x577B986B314D6009, 0xB616A12B7FE617AA},  // 1e160
        {0xED5A7E85FDA0B80B, 0xE39C49765FDF9D94},  // 1e161
        {0x14588F13BE847307, 0x8E41ADE9FBEBC27D},  // 1e162
        {0x596EB2D8AE258FC8, 0xB1D219647AE6B31C},  // 1e163
        {0x6FCA5F8ED9AEF3BB, 0xDE469FBD99A05FE3},  // 1e164
        {0x25DE7BB9480D5854, 0x8AEC23D680043BEE},  // 1e165
        {0xAF561AA79A10AE6A, 0xADA72CCC20054AE9},  // 1e166
        {0x1B2BA1518094DA04, 0xD910F7FF28069DA4},  // 1e167
        {0x90FB44D2F05D0842, 0x87AA9AFF79042286},  // 1e168
        {0x353A1607AC744A53, 0xA99541BF57452B28},  // 1e169
        {0x42889B8997915CE8, 0xD3FA922F2D1675F2},  // 1e170
        {0x69956135FEBADA11, 0x847C9B5D7C2E09B7},  // 1e171
        {0x43FAB9837E699095, 0xA59BC234DB398C25},  // 1e172
        {0x94F967E45E03F4BB, 0xCF02B2C21207EF2E},  // 1e173
        {0x1D1BE0EEBAC278F5, 0x8161AFB94B44F57D},  // 1e174
        {0x6462D92A69731732, 0xA1BA1BA79E1632DC},  // 1e175
        {0x7D7B8F7503CFDCFE, 0xCA28A291859BBF93},  // 1e176
        {0x5CDA735244C3D43E, 0xFCB2CB35E702AF78},  // 1e177
        {0x3A0888136AFA64A7, 0x9DEFBF01B061ADAB},  // 1e178
        {0x088AAA1845B8FDD0, 0xC56BAEC21C7A1916},  // 1e179
        {0x8AAD549E57273D45, 0xF6C69A72A3989F5B},  // 1e180
        {0x36AC54E2F678864B, 0x9A3C2087A63F6399},  // 1e181
        {0x84576A1BB416A7DD, 0xC0CB28A98FCF3C7F},  // 1e182
        {0x656D44A2A11C51D5, 0xF0FDF2D3F3C30B9F},  // 1e183
        {0x9F644AE5A4B1B325, 0x969EB7C47859E743},  // 1e184
        {0x873D5D9F0DDE1FEE, 0xBC4665B596706114},  // 1e185
        {0xA90CB506D155A7EA, 0xEB57FF22FC0C7959},  // 1e186
        {0x09A7F12442D588F2, 0x9316FF75DD87CBD8},  // 1e187
        {0x0C11ED6D538AEB2F, 0xB7DCBF5354E9BECE},  // 1e188
        {0x8F1668C8A86DA5FA, 0xE5D3EF282A242E81},  // 1e189
        {0xF96E017D694487BC, 0x8FA475791A569D10},  // 1e190
        {0x37C981DCC395A9AC, 0xB38D92D760EC4455},  // 1e191
        {0x85BBE253F47B1417, 0xE070F78D3927556A},  // 1e192
        {0x93956D7478CCEC8E, 0x8C469AB843B89562},  // 1e193
        {0x387AC8D1970027B2, 0xAF58416654A6BABB},  // 1e194
        {0x06997B05FCC0319E, 0xDB2E51BFE9D0696A},  // 1e195
        {0x441FECE3BDF81F03, 0x88FCF317F22241E2},  // 1e196
        {0xD527E81CAD7626C3, 0xAB3C2FDDEEAAD25A},  // 1e197
        {0x8A71E223D8D3B074, 0xD60B3BD56A5586F1},  // 1e198
        {0xF6872D5667844E49, 0x85C7056562757456},  // 1e199
        {0xB428F8AC016561DB, 0xA738C6BEBB12D16C},  // 1e200
        {0xE13336D701BEBA52, 0xD106F86E69D785C7},  // 1e201
        {0xECC0024661173473, 0x82A45B450226B39C},  // 1e202
        {0x27F002D7F95D0190, 0xA34D721642B06084},  // 1e203
        {0x31EC038DF7B441F4, 0xCC20CE9BD35C78A5},  // 1e204
        {0x7E67047175A15271, 0xFF290242C83396CE},  // 1e205
        {0x0F0062C6E984D386, 0x9F79A169BD203E41},  // 1e206
        {0x52C07B78A3E60868, 0xC75809C42C684DD1},  // 1e207
        {0xA7709A56CCDF8A82, 0xF92E0C3537826145},  // 1e208
        {0x88A66076400BB691, 0x9BBCC7A142B17CCB},  // 1e209
        {0x6ACFF893D00EA435, 0xC2ABF989935DDBFE},  // 1e210
        {0x0583F6B8C4124D43, 0xF356F7EBF83552FE},  // 1e211
        {0xC3727A337A8B704A, 0x98165AF37B2153DE},  // 1e212
        {0x744F18C0592E4C5C, 0xBE1BF1B059E9A8D6},  // 1e213
        {0x1162DEF06F79DF73, 0xEDA2EE1C7064130C},  // 1e214
        {0x8ADDCB5645AC2BA8, 0x9485D4D1C63E8BE7},  // 1e215
        {0x6D953E2BD7173692, 0xB9A74A0637CE2EE1},  // 1e216
        {0xC8FA8DB6CCDD0437, 0xE8111C87C5C1BA99},  // 1e217
        {0x1D9C9892400A22A2, 0x910AB1D4DB9914A0},  // 1e218
        {0x2503BEB6D00CAB4B, 0xB54D5E4A127F59C8},  // 1e219
        {0x2E44AE64840FD61D, 0xE2A0B5DC971F303A},  // 1e220
        {0x5CEAECFED289E5D2, 0x8DA471A9DE737E24},  // 1e221
        {0x7425A83E872C5F47, 0xB10D8E1456105DAD},  // 1e222
        {0xD12F124E28F77719, 0xDD50F1996B947518},  // 1e223
        {0x82BD6B70D99AAA6F, 0x8A5296FFE33CC92F},  // 1e224
        {0x636CC64D1001550B, 0xACE73CBFDC0BFB7B},  // 1e225
        {0x3C47F7E05401AA4E, 0xD8210BEFD30EFA5A},  // 1e226
        {0x65ACFAEC34810A71, 0x8714A775E3E95C78},  // 1e227
        {0x7F1839A741A14D0D, 0xA8D9D1535CE3B396},  // 1e228
        {0x1EDE48111209A050, 0xD31045A8341CA07C},  // 1e229
        {0x934AED0AAB460432, 0x83EA2B892091E44D},  // 1e230
        {0xF81DA84D5617853F, 0xA4E4B66B68B65D60},  // 1e231
        {0x36251260AB9D668E, 0xCE1DE40642E3F4B9},  // 1e232
        {0xC1D72B7C6B426019, 0x80D2AE83E9CE78F3},  // 1e233
        {0xB24CF65B8612F81F, 0xA1075A24E4421730},  // 1e234
        {0xDEE033F26797B627, 0xC94930AE1D529CFC},  // 1e235
        {0x169840EF017DA3B1, 0xFB9B7CD9A4A7443C},  // 1e236
        {0x8E1F289560EE864E, 0x9D412E0806E88AA5},  // 1e237
        {0xF1A6F2BAB92A27E2, 0xC491798A08A2AD4E},  // 1e238
        {0xAE10AF696774B1DB, 0xF5B5D7EC8ACB58A2},  // 1e239
        {0xACCA6DA1E0A8EF29, 0x9991A6F3D6BF1765},  // 1e240
        {0x17FD090A58D32AF3, 0xBFF610B0CC6EDD3F},  // 1e241
        {0xDDFC4B4CEF07F5B0, 0xEFF394DCFF8A948E},  // 1e242
        {0x4ABDAF101564F98E, 0x95F83D0A1FB69CD9},  // 1e243
        {0x9D6D1AD41ABE37F1, 0xBB764C4CA7A4440F},  // 1e244
        {0x84C86189216DC5ED, 0xEA53DF5FD18D5513},  // 1e245
        {0x32FD3CF5B4E49BB4, 0x92746B9BE2F8552C},  // 1e246
        {0x3FBC8C33221DC2A1, 0xB7118682DBB66A77},  // 1e247
        {0x0FABAF3FEAA5334A, 0xE4D5E82392A40515},  // 1e248
        {0x29CB4D87F2A7400E, 0x8F05B1163BA6832D},  // 1e249
        {0x743E20E9EF511012, 0xB2C71D5BCA9023F8},  // 1e250
        {0x914DA9246B255416, 0xDF78E4B2BD342CF6},  // 1e251
        {0x1AD089B6C2F7548E, 0x8BAB8EEFB6409C1A},  // 1e252
        {0xA184AC2473B529B1, 0xAE9672ABA3D0C320},  // 1e253
        {0xC9E5D72D90A2741E, 0xDA3C0F568CC4F3E8},  // 1e254
        {0x7E2FA67C7A658892, 0x8865899617FB1871},  // 1e255
        {0xDDBB901B98FEEAB7, 0xAA7EEBFB9DF9DE8D},  // 1e256
        {0x552A74227F3EA565, 0xD51EA6FA85785631},  // 1e257
        {0xD53A88958F87275F, 0x8533285C936B35DE},  // 1e258
        {0x8A892ABAF368F137, 0xA67FF273B8460356},  // 1e259
        {0x2D2B7569B0432D85, 0xD01FEF10A657842C},  // 1e260
        {0x9C3B29620E29FC73, 0x8213F56A67F6B29B},  // 1e261
        {0x8349F3BA91B47B8F, 0xA298F2C501F45F42},  // 1e262
        {0x241C70A936219A73, 0xCB3F2F7642717713},  // 1e263
        {0xED238CD383AA0110, 0xFE0EFB53D30DD4D7},  // 1e264
        {0xF4363804324A40AA, 0x9EC95D1463E8A506},  // 1e265
        {0xB143C6053EDCD0D5, 0xC67BB4597CE2CE48},  // 1e266
        {0xDD94B7868E94050A, 0xF81AA16FDC1B81DA},  // 1e267
        {0xCA7CF2B4191C8326, 0x9B10A4E5E9913128},  // 1e268
        {0xFD1C2F611F63A3F0, 0xC1D4CE1F63F57D72},  // 1e269
        {0xBC633B39673C8CEC, 0xF24A01A73CF2DCCF},  // 1e270
        {0xD5BE0503E085D813, 0x976E41088617CA01},  // 1e271
        {0x4B2D8644D8A74E18, 0xBD49D14AA79DBC82},  // 1e272
        {0xDDF8E7D60ED1219E, 0xEC9C459D51852BA2},  // 1e273
        {0xCABB90E5C942B503, 0x93E1AB8252F33B45},  // 1e274
        {0x3D6A751F3B936243, 0xB8DA1662E7B00A17},  // 1e275
        {0x0CC512670A783AD4, 0xE7109BFBA19C0C9D},  // 1e276
        {0x27FB2B80668B24C5, 0x906A617D450187E2},  // 1e277
        {0xB1F9F660802DEDF6, 0xB484F9DC9641E9DA},  // 1e278
        {0x5E7873F8A0396973, 0xE1A63853BBD26451},  // 1e279
        {0xDB0B487B6423E1E8, 0x8D07E33455637EB2},  // 1e280
        {0x91CE1A9A3D2CDA62, 0xB049DC016ABC5E5F},  // 1e281
        {0x7641A140CC7810FB, 0xDC5C5301C56B75F7},  // 1e282
        {0xA9E904C87FCB0A9D, 0x89B9B3E11B6329BA},  // 1e283
        {0x546345FA9FBDCD44, 0xAC2820D9623BF429},  // 1e284
        {0xA97C177947AD4095, 0xD732290FBACAF133},  // 1e285
        {0x49ED8EABCCCC485D, 0x867F59A9D4BED6C0},  // 1e286
        {0x5C68F256BFFF5A74, 0xA81F301449EE8C70},  // 1e287
        {0x73832EEC6FFF3111, 0xD226FC195C6A2F8C},  // 1e288
};

// wuffs_base__private_implementation__f64_powers_of_10 holds powers of 10 that
// can be exactly represented by a float64 (what C calls a double).
static const double wuffs_base__private_implementation__f64_powers_of_10[23] = {
    1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,  1e10, 1e11,
    1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
};

// ---------------- IEEE 754 Floating Point

WUFFS_BASE__MAYBE_STATIC wuffs_base__lossy_value_u16  //
wuffs_base__ieee_754_bit_representation__from_f64_to_u16_truncate(double f) {
  uint64_t u = 0;
  if (sizeof(uint64_t) == sizeof(double)) {
    memcpy(&u, &f, sizeof(uint64_t));
  }
  uint16_t neg = ((uint16_t)((u >> 63) << 15));
  u &= 0x7FFFFFFFFFFFFFFF;
  uint64_t exp = u >> 52;
  uint64_t man = u & 0x000FFFFFFFFFFFFF;

  if (exp == 0x7FF) {
    if (man == 0) {  // Infinity.
      wuffs_base__lossy_value_u16 ret;
      ret.value = neg | 0x7C00;
      ret.lossy = false;
      return ret;
    }
    // NaN. Shift the 52 mantissa bits to 10 mantissa bits, keeping the most
    // significant mantissa bit (quiet vs signaling NaNs). Also set the low 9
    // bits of ret.value so that the 10-bit mantissa is non-zero.
    wuffs_base__lossy_value_u16 ret;
    ret.value = neg | 0x7DFF | ((uint16_t)(man >> 42));
    ret.lossy = false;
    return ret;

  } else if (exp > 0x40E) {  // Truncate to the largest finite f16.
    wuffs_base__lossy_value_u16 ret;
    ret.value = neg | 0x7BFF;
    ret.lossy = true;
    return ret;

  } else if (exp <= 0x3E6) {  // Truncate to zero.
    wuffs_base__lossy_value_u16 ret;
    ret.value = neg;
    ret.lossy = (u != 0);
    return ret;

  } else if (exp <= 0x3F0) {  // Normal f64, subnormal f16.
    // Convert from a 53-bit mantissa (after realizing the implicit bit) to a
    // 10-bit mantissa and then adjust for the exponent.
    man |= 0x0010000000000000;
    uint32_t shift = ((uint32_t)(1051 - exp));  // 1051 = 0x3F0 + 53 - 10.
    uint64_t shifted_man = man >> shift;
    wuffs_base__lossy_value_u16 ret;
    ret.value = neg | ((uint16_t)shifted_man);
    ret.lossy = (shifted_man << shift) != man;
    return ret;
  }

  // Normal f64, normal f16.

  // Re-bias from 1023 to 15 and shift above f16's 10 mantissa bits.
  exp = (exp - 1008) << 10;  // 1008 = 1023 - 15 = 0x3FF - 0xF.

  // Convert from a 52-bit mantissa (excluding the implicit bit) to a 10-bit
  // mantissa (again excluding the implicit bit). We lose some information if
  // any of the bottom 42 bits are non-zero.
  wuffs_base__lossy_value_u16 ret;
  ret.value = neg | ((uint16_t)exp) | ((uint16_t)(man >> 42));
  ret.lossy = (man << 22) != 0;
  return ret;
}

WUFFS_BASE__MAYBE_STATIC wuffs_base__lossy_value_u32  //
wuffs_base__ieee_754_bit_representation__from_f64_to_u32_truncate(double f) {
  uint64_t u = 0;
  if (sizeof(uint64_t) == sizeof(double)) {
    memcpy(&u, &f, sizeof(uint64_t));
  }
  uint32_t neg = ((uint32_t)(u >> 63)) << 31;
  u &= 0x7FFFFFFFFFFFFFFF;
  uint64_t exp = u >> 52;
  uint64_t man = u & 0x000FFFFFFFFFFFFF;

  if (exp == 0x7FF) {
    if (man == 0) {  // Infinity.
      wuffs_base__lossy_value_u32 ret;
      ret.value = neg | 0x7F800000;
      ret.lossy = false;
      return ret;
    }
    // NaN. Shift the 52 mantissa bits to 23 mantissa bits, keeping the most
    // significant mantissa bit (quiet vs signaling NaNs). Also set the low 22
    // bits of ret.value so that the 23-bit mantissa is non-zero.
    wuffs_base__lossy_value_u32 ret;
    ret.value = neg | 0x7FBFFFFF | ((uint32_t)(man >> 29));
    ret.lossy = false;
    return ret;

  } else if (exp > 0x47E) {  // Truncate to the largest finite f32.
    wuffs_base__lossy_value_u32 ret;
    ret.value = neg | 0x7F7FFFFF;
    ret.lossy = true;
    return ret;

  } else if (exp <= 0x369) {  // Truncate to zero.
    wuffs_base__lossy_value_u32 ret;
    ret.value = neg;
    ret.lossy = (u != 0);
    return ret;

  } else if (exp <= 0x380) {  // Normal f64, subnormal f32.
    // Convert from a 53-bit mantissa (after realizing the implicit bit) to a
    // 23-bit mantissa and then adjust for the exponent.
    man |= 0x0010000000000000;
    uint32_t shift = ((uint32_t)(926 - exp));  // 926 = 0x380 + 53 - 23.
    uint64_t shifted_man = man >> shift;
    wuffs_base__lossy_value_u32 ret;
    ret.value = neg | ((uint32_t)shifted_man);
    ret.lossy = (shifted_man << shift) != man;
    return ret;
  }

  // Normal f64, normal f32.

  // Re-bias from 1023 to 127 and shift above f32's 23 mantissa bits.
  exp = (exp - 896) << 23;  // 896 = 1023 - 127 = 0x3FF - 0x7F.

  // Convert from a 52-bit mantissa (excluding the implicit bit) to a 23-bit
  // mantissa (again excluding the implicit bit). We lose some information if
  // any of the bottom 29 bits are non-zero.
  wuffs_base__lossy_value_u32 ret;
  ret.value = neg | ((uint32_t)exp) | ((uint32_t)(man >> 29));
  ret.lossy = (man << 35) != 0;
  return ret;
}

// --------

#define WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__DECIMAL_POINT__RANGE 2047
#define WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__DIGITS_PRECISION 800

// WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__SHIFT__MAX_INCL is the largest N
// such that ((10 << N) < (1 << 64)).
#define WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__SHIFT__MAX_INCL 60

// wuffs_base__private_implementation__high_prec_dec (abbreviated as HPD) is a
// fixed precision floating point decimal number, augmented with ±infinity
// values, but it cannot represent NaN (Not a Number).
//
// "High precision" means that the mantissa holds 800 decimal digits. 800 is
// WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__DIGITS_PRECISION.
//
// An HPD isn't for general purpose arithmetic, only for conversions to and
// from IEEE 754 double-precision floating point, where the largest and
// smallest positive, finite values are approximately 1.8e+308 and 4.9e-324.
// HPD exponents above +2047 mean infinity, below -2047 mean zero. The ±2047
// bounds are further away from zero than ±(324 + 800), where 800 and 2047 is
// WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__DIGITS_PRECISION and
// WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__DECIMAL_POINT__RANGE.
//
// digits[.. num_digits] are the number's digits in big-endian order. The
// uint8_t values are in the range [0 ..= 9], not ['0' ..= '9'], where e.g. '7'
// is the ASCII value 0x37.
//
// decimal_point is the index (within digits) of the decimal point. It may be
// negative or be larger than num_digits, in which case the explicit digits are
// padded with implicit zeroes.
//
// For example, if num_digits is 3 and digits is "\x07\x08\x09":
//  - A decimal_point of -2 means ".00789"
//  - A decimal_point of -1 means ".0789"
//  - A decimal_point of +0 means ".789"
//  - A decimal_point of +1 means "7.89"
//  - A decimal_point of +2 means "78.9"
//  - A decimal_point of +3 means "789."
//  - A decimal_point of +4 means "7890."
//  - A decimal_point of +5 means "78900."
//
// As above, a decimal_point higher than +2047 means that the overall value is
// infinity, lower than -2047 means zero.
//
// negative is a sign bit. An HPD can distinguish positive and negative zero.
//
// truncated is whether there are more than
// WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__DIGITS_PRECISION digits, and at
// least one of those extra digits are non-zero. The existence of long-tail
// digits can affect rounding.
//
// The "all fields are zero" value is valid, and represents the number +0.
typedef struct wuffs_base__private_implementation__high_prec_dec__struct {
  uint32_t num_digits;
  int32_t decimal_point;
  bool negative;
  bool truncated;
  uint8_t digits[WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__DIGITS_PRECISION];
} wuffs_base__private_implementation__high_prec_dec;

// wuffs_base__private_implementation__high_prec_dec__trim trims trailing
// zeroes from the h->digits[.. h->num_digits] slice. They have no benefit,
// since we explicitly track h->decimal_point.
//
// Preconditions:
//  - h is non-NULL.
static inline void  //
wuffs_base__private_implementation__high_prec_dec__trim(
    wuffs_base__private_implementation__high_prec_dec* h) {
  while ((h->num_digits > 0) && (h->digits[h->num_digits - 1] == 0)) {
    h->num_digits--;
  }
}

// wuffs_base__private_implementation__high_prec_dec__assign sets h to
// represent the number x.
//
// Preconditions:
//  - h is non-NULL.
static void  //
wuffs_base__private_implementation__high_prec_dec__assign(
    wuffs_base__private_implementation__high_prec_dec* h,
    uint64_t x,
    bool negative) {
  uint32_t n = 0;

  // Set h->digits.
  if (x > 0) {
    // Calculate the digits, working right-to-left. After we determine n (how
    // many digits there are), copy from buf to h->digits.
    //
    // UINT64_MAX, 18446744073709551615, is 20 digits long. It can be faster to
    // copy a constant number of bytes than a variable number (20 instead of
    // n). Make buf large enough (and start writing to it from the middle) so
    // that can we always copy 20 bytes: the slice buf[(20-n) .. (40-n)].
    uint8_t buf[40] = {0};
    uint8_t* ptr = &buf[20];
    do {
      uint64_t remaining = x / 10;
      x -= remaining * 10;
      ptr--;
      *ptr = (uint8_t)x;
      n++;
      x = remaining;
    } while (x > 0);
    memcpy(h->digits, ptr, 20);
  }

  // Set h's other fields.
  h->num_digits = n;
  h->decimal_point = (int32_t)n;
  h->negative = negative;
  h->truncated = false;
  wuffs_base__private_implementation__high_prec_dec__trim(h);
}

static wuffs_base__status  //
wuffs_base__private_implementation__high_prec_dec__parse(
    wuffs_base__private_implementation__high_prec_dec* h,
    wuffs_base__slice_u8 s,
    uint32_t options) {
  if (!h) {
    return wuffs_base__make_status(wuffs_base__error__bad_receiver);
  }
  h->num_digits = 0;
  h->decimal_point = 0;
  h->negative = false;
  h->truncated = false;

  uint8_t* p = s.ptr;
  uint8_t* q = s.ptr + s.len;

  if (options & WUFFS_BASE__PARSE_NUMBER_XXX__ALLOW_UNDERSCORES) {
    for (;; p++) {
      if (p >= q) {
        return wuffs_base__make_status(wuffs_base__error__bad_argument);
      } else if (*p != '_') {
        break;
      }
    }
  }

  // Parse sign.
  do {
    if (*p == '+') {
      p++;
    } else if (*p == '-') {
      h->negative = true;
      p++;
    } else {
      break;
    }
    if (options & WUFFS_BASE__PARSE_NUMBER_XXX__ALLOW_UNDERSCORES) {
      for (;; p++) {
        if (p >= q) {
          return wuffs_base__make_status(wuffs_base__error__bad_argument);
        } else if (*p != '_') {
          break;
        }
      }
    }
  } while (0);

  // Parse digits, up to (and including) a '.', 'E' or 'e'. Examples for each
  // limb in this if-else chain:
  //  - "0.789"
  //  - "1002.789"
  //  - ".789"
  //  - Other (invalid input).
  uint32_t nd = 0;
  int32_t dp = 0;
  bool no_digits_before_separator = false;
  if (('0' == *p) &&
      !(options &
        WUFFS_BASE__PARSE_NUMBER_XXX__ALLOW_MULTIPLE_LEADING_ZEROES)) {
    p++;
    for (;; p++) {
      if (p >= q) {
        goto after_all;
      } else if (*p ==
                 ((options &
                   WUFFS_BASE__PARSE_NUMBER_FXX__DECIMAL_SEPARATOR_IS_A_COMMA)
                      ? ','
                      : '.')) {
        p++;
        goto after_sep;
      } else if ((*p == 'E') || (*p == 'e')) {
        p++;
        goto after_exp;
      } else if ((*p != '_') ||
                 !(options & WUFFS_BASE__PARSE_NUMBER_XXX__ALLOW_UNDERSCORES)) {
        return wuffs_base__make_status(wuffs_base__error__bad_argument);
      }
    }

  } else if (('0' <= *p) && (*p <= '9')) {
    if (*p == '0') {
      for (; (p < q) && (*p == '0'); p++) {
      }
    } else {
      h->digits[nd++] = (uint8_t)(*p - '0');
      dp = (int32_t)nd;
      p++;
    }

    for (;; p++) {
      if (p >= q) {
        goto after_all;
      } else if (('0' <= *p) && (*p <= '9')) {
        if (nd < WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__DIGITS_PRECISION) {
          h->digits[nd++] = (uint8_t)(*p - '0');
          dp = (int32_t)nd;
        } else if ('0' != *p) {
          // Long-tail non-zeroes set the truncated bit.
          h->truncated = true;
        }
      } else if (*p ==
                 ((options &
                   WUFFS_BASE__PARSE_NUMBER_FXX__DECIMAL_SEPARATOR_IS_A_COMMA)
                      ? ','
                      : '.')) {
        p++;
        goto after_sep;
      } else if ((*p == 'E') || (*p == 'e')) {
        p++;
        goto after_exp;
      } else if ((*p != '_') ||
                 !(options & WUFFS_BASE__PARSE_NUMBER_XXX__ALLOW_UNDERSCORES)) {
        return wuffs_base__make_status(wuffs_base__error__bad_argument);
      }
    }

  } else if (*p == ((options &
                     WUFFS_BASE__PARSE_NUMBER_FXX__DECIMAL_SEPARATOR_IS_A_COMMA)
                        ? ','
                        : '.')) {
    p++;
    no_digits_before_separator = true;

  } else {
    return wuffs_base__make_status(wuffs_base__error__bad_argument);
  }

after_sep:
  for (;; p++) {
    if (p >= q) {
      goto after_all;
    } else if ('0' == *p) {
      if (nd == 0) {
        // Track leading zeroes implicitly.
        dp--;
      } else if (nd <
                 WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__DIGITS_PRECISION) {
        h->digits[nd++] = (uint8_t)(*p - '0');
      }
    } else if (('0' < *p) && (*p <= '9')) {
      if (nd < WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__DIGITS_PRECISION) {
        h->digits[nd++] = (uint8_t)(*p - '0');
      } else {
        // Long-tail non-zeroes set the truncated bit.
        h->truncated = true;
      }
    } else if ((*p == 'E') || (*p == 'e')) {
      p++;
      goto after_exp;
    } else if ((*p != '_') ||
               !(options & WUFFS_BASE__PARSE_NUMBER_XXX__ALLOW_UNDERSCORES)) {
      return wuffs_base__make_status(wuffs_base__error__bad_argument);
    }
  }

after_exp:
  do {
    if (options & WUFFS_BASE__PARSE_NUMBER_XXX__ALLOW_UNDERSCORES) {
      for (;; p++) {
        if (p >= q) {
          return wuffs_base__make_status(wuffs_base__error__bad_argument);
        } else if (*p != '_') {
          break;
        }
      }
    }

    int32_t exp_sign = +1;
    if (*p == '+') {
      p++;
    } else if (*p == '-') {
      exp_sign = -1;
      p++;
    }

    int32_t exp = 0;
    const int32_t exp_large =
        WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__DECIMAL_POINT__RANGE +
        WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__DIGITS_PRECISION;
    bool saw_exp_digits = false;
    for (; p < q; p++) {
      if ((*p == '_') &&
          (options & WUFFS_BASE__PARSE_NUMBER_XXX__ALLOW_UNDERSCORES)) {
        // No-op.
      } else if (('0' <= *p) && (*p <= '9')) {
        saw_exp_digits = true;
        if (exp < exp_large) {
          exp = (10 * exp) + ((int32_t)(*p - '0'));
        }
      } else {
        break;
      }
    }
    if (!saw_exp_digits) {
      return wuffs_base__make_status(wuffs_base__error__bad_argument);
    }
    dp += exp_sign * exp;
  } while (0);

after_all:
  if (p != q) {
    return wuffs_base__make_status(wuffs_base__error__bad_argument);
  }
  h->num_digits = nd;
  if (nd == 0) {
    if (no_digits_before_separator) {
      return wuffs_base__make_status(wuffs_base__error__bad_argument);
    }
    h->decimal_point = 0;
  } else if (dp <
             -WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__DECIMAL_POINT__RANGE) {
    h->decimal_point =
        -WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__DECIMAL_POINT__RANGE - 1;
  } else if (dp >
             +WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__DECIMAL_POINT__RANGE) {
    h->decimal_point =
        +WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__DECIMAL_POINT__RANGE + 1;
  } else {
    h->decimal_point = dp;
  }
  wuffs_base__private_implementation__high_prec_dec__trim(h);
  return wuffs_base__make_status(NULL);
}

// --------

// wuffs_base__private_implementation__high_prec_dec__lshift_num_new_digits
// returns the number of additional decimal digits when left-shifting by shift.
//
// See below for preconditions.
static uint32_t  //
wuffs_base__private_implementation__high_prec_dec__lshift_num_new_digits(
    wuffs_base__private_implementation__high_prec_dec* h,
    uint32_t shift) {
  // Masking with 0x3F should be unnecessary (assuming the preconditions) but
  // it's cheap and ensures that we don't overflow the
  // wuffs_base__private_implementation__hpd_left_shift array.
  shift &= 63;

  uint32_t x_a = wuffs_base__private_implementation__hpd_left_shift[shift];
  uint32_t x_b = wuffs_base__private_implementation__hpd_left_shift[shift + 1];
  uint32_t num_new_digits = x_a >> 11;
  uint32_t pow5_a = 0x7FF & x_a;
  uint32_t pow5_b = 0x7FF & x_b;

  const uint8_t* pow5 =
      &wuffs_base__private_implementation__powers_of_5[pow5_a];
  uint32_t i = 0;
  uint32_t n = pow5_b - pow5_a;
  for (; i < n; i++) {
    if (i >= h->num_digits) {
      return num_new_digits - 1;
    } else if (h->digits[i] == pow5[i]) {
      continue;
    } else if (h->digits[i] < pow5[i]) {
      return num_new_digits - 1;
    } else {
      return num_new_digits;
    }
  }
  return num_new_digits;
}

// --------

// wuffs_base__private_implementation__high_prec_dec__rounded_integer returns
// the integral (non-fractional) part of h, provided that it is 18 or fewer
// decimal digits. For 19 or more digits, it returns UINT64_MAX. Note that:
//  - (1 << 53) is    9007199254740992, which has 16 decimal digits.
//  - (1 << 56) is   72057594037927936, which has 17 decimal digits.
//  - (1 << 59) is  576460752303423488, which has 18 decimal digits.
//  - (1 << 63) is 9223372036854775808, which has 19 decimal digits.
// and that IEEE 754 double precision has 52 mantissa bits.
//
// That integral part is rounded-to-even: rounding 7.5 or 8.5 both give 8.
//
// h's negative bit is ignored: rounding -8.6 returns 9.
//
// See below for preconditions.
static uint64_t  //
wuffs_base__private_implementation__high_prec_dec__rounded_integer(
    wuffs_base__private_implementation__high_prec_dec* h) {
  if ((h->num_digits == 0) || (h->decimal_point < 0)) {
    return 0;
  } else if (h->decimal_point > 18) {
    return UINT64_MAX;
  }

  uint32_t dp = (uint32_t)(h->decimal_point);
  uint64_t n = 0;
  uint32_t i = 0;
  for (; i < dp; i++) {
    n = (10 * n) + ((i < h->num_digits) ? h->digits[i] : 0);
  }

  bool round_up = false;
  if (dp < h->num_digits) {
    round_up = h->digits[dp] >= 5;
    if ((h->digits[dp] == 5) && (dp + 1 == h->num_digits)) {
      // We are exactly halfway. If we're truncated, round up, otherwise round
      // to even.
      round_up = h->truncated ||  //
                 ((dp > 0) && (1 & h->digits[dp - 1]));
    }
  }
  if (round_up) {
    n++;
  }

  return n;
}

// wuffs_base__private_implementation__high_prec_dec__small_xshift shifts h's
// number (where 'x' is 'l' or 'r' for left or right) by a small shift value.
//
// Preconditions:
//  - h is non-NULL.
//  - h->decimal_point is "not extreme".
//  - shift is non-zero.
//  - shift is "a small shift".
//
// "Not extreme" means within
// ±WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__DECIMAL_POINT__RANGE.
//
// "A small shift" means not more than
// WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__SHIFT__MAX_INCL.
//
// wuffs_base__private_implementation__high_prec_dec__rounded_integer and
// wuffs_base__private_implementation__high_prec_dec__lshift_num_new_digits
// have the same preconditions.
//
// wuffs_base__private_implementation__high_prec_dec__lshift keeps the first
// two preconditions but not the last two. Its shift argument is signed and
// does not need to be "small": zero is a no-op, positive means left shift and
// negative means right shift.

static void  //
wuffs_base__private_implementation__high_prec_dec__small_lshift(
    wuffs_base__private_implementation__high_prec_dec* h,
    uint32_t shift) {
  if (h->num_digits == 0) {
    return;
  }
  uint32_t num_new_digits =
      wuffs_base__private_implementation__high_prec_dec__lshift_num_new_digits(
          h, shift);
  uint32_t rx = h->num_digits - 1;                   // Read  index.
  uint32_t wx = h->num_digits - 1 + num_new_digits;  // Write index.
  uint64_t n = 0;

  // Repeat: pick up a digit, put down a digit, right to left.
  while (((int32_t)rx) >= 0) {
    n += ((uint64_t)(h->digits[rx])) << shift;
    uint64_t quo = n / 10;
    uint64_t rem = n - (10 * quo);
    if (wx < WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__DIGITS_PRECISION) {
      h->digits[wx] = (uint8_t)rem;
    } else if (rem > 0) {
      h->truncated = true;
    }
    n = quo;
    wx--;
    rx--;
  }

  // Put down leading digits, right to left.
  while (n > 0) {
    uint64_t quo = n / 10;
    uint64_t rem = n - (10 * quo);
    if (wx < WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__DIGITS_PRECISION) {
      h->digits[wx] = (uint8_t)rem;
    } else if (rem > 0) {
      h->truncated = true;
    }
    n = quo;
    wx--;
  }

  // Finish.
  h->num_digits += num_new_digits;
  if (h->num_digits >
      WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__DIGITS_PRECISION) {
    h->num_digits = WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__DIGITS_PRECISION;
  }
  h->decimal_point += (int32_t)num_new_digits;
  wuffs_base__private_implementation__high_prec_dec__trim(h);
}

static void  //
wuffs_base__private_implementation__high_prec_dec__small_rshift(
    wuffs_base__private_implementation__high_prec_dec* h,
    uint32_t shift) {
  uint32_t rx = 0;  // Read  index.
  uint32_t wx = 0;  // Write index.
  uint64_t n = 0;

  // Pick up enough leading digits to cover the first shift.
  while ((n >> shift) == 0) {
    if (rx < h->num_digits) {
      // Read a digit.
      n = (10 * n) + h->digits[rx++];
    } else if (n == 0) {
      // h's number used to be zero and remains zero.
      return;
    } else {
      // Read sufficient implicit trailing zeroes.
      while ((n >> shift) == 0) {
        n = 10 * n;
        rx++;
      }
      break;
    }
  }
  h->decimal_point -= ((int32_t)(rx - 1));
  if (h->decimal_point <
      -WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__DECIMAL_POINT__RANGE) {
    // After the shift, h's number is effectively zero.
    h->num_digits = 0;
    h->decimal_point = 0;
    h->truncated = false;
    return;
  }

  // Repeat: pick up a digit, put down a digit, left to right.
  uint64_t mask = (((uint64_t)(1)) << shift) - 1;
  while (rx < h->num_digits) {
    uint8_t new_digit = ((uint8_t)(n >> shift));
    n = (10 * (n & mask)) + h->digits[rx++];
    h->digits[wx++] = new_digit;
  }

  // Put down trailing digits, left to right.
  while (n > 0) {
    uint8_t new_digit = ((uint8_t)(n >> shift));
    n = 10 * (n & mask);
    if (wx < WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__DIGITS_PRECISION) {
      h->digits[wx++] = new_digit;
    } else if (new_digit > 0) {
      h->truncated = true;
    }
  }

  // Finish.
  h->num_digits = wx;
  wuffs_base__private_implementation__high_prec_dec__trim(h);
}

static void  //
wuffs_base__private_implementation__high_prec_dec__lshift(
    wuffs_base__private_implementation__high_prec_dec* h,
    int32_t shift) {
  if (shift > 0) {
    while (shift > +WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__SHIFT__MAX_INCL) {
      wuffs_base__private_implementation__high_prec_dec__small_lshift(
          h, WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__SHIFT__MAX_INCL);
      shift -= WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__SHIFT__MAX_INCL;
    }
    wuffs_base__private_implementation__high_prec_dec__small_lshift(
        h, ((uint32_t)(+shift)));
  } else if (shift < 0) {
    while (shift < -WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__SHIFT__MAX_INCL) {
      wuffs_base__private_implementation__high_prec_dec__small_rshift(
          h, WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__SHIFT__MAX_INCL);
      shift += WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__SHIFT__MAX_INCL;
    }
    wuffs_base__private_implementation__high_prec_dec__small_rshift(
        h, ((uint32_t)(-shift)));
  }
}

// --------

// wuffs_base__private_implementation__high_prec_dec__round_etc rounds h's
// number. For those functions that take an n argument, rounding produces at
// most n digits (which is not necessarily at most n decimal places). Negative
// n values are ignored, as well as any n greater than or equal to h's number
// of digits. The etc__round_just_enough function implicitly chooses an n to
// implement WUFFS_BASE__RENDER_NUMBER_FXX__JUST_ENOUGH_PRECISION.
//
// Preconditions:
//  - h is non-NULL.
//  - h->decimal_point is "not extreme".
//
// "Not extreme" means within
// ±WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__DECIMAL_POINT__RANGE.

static void  //
wuffs_base__private_implementation__high_prec_dec__round_down(
    wuffs_base__private_implementation__high_prec_dec* h,
    int32_t n) {
  if ((n < 0) || (h->num_digits <= (uint32_t)n)) {
    return;
  }
  h->num_digits = (uint32_t)(n);
  wuffs_base__private_implementation__high_prec_dec__trim(h);
}

static void  //
wuffs_base__private_implementation__high_prec_dec__round_up(
    wuffs_base__private_implementation__high_prec_dec* h,
    int32_t n) {
  if ((n < 0) || (h->num_digits <= (uint32_t)n)) {
    return;
  }

  for (n--; n >= 0; n--) {
    if (h->digits[n] < 9) {
      h->digits[n]++;
      h->num_digits = (uint32_t)(n + 1);
      return;
    }
  }

  // The number is all 9s. Change to a single 1 and adjust the decimal point.
  h->digits[0] = 1;
  h->num_digits = 1;
  h->decimal_point++;
}

static void  //
wuffs_base__private_implementation__high_prec_dec__round_nearest(
    wuffs_base__private_implementation__high_prec_dec* h,
    int32_t n) {
  if ((n < 0) || (h->num_digits <= (uint32_t)n)) {
    return;
  }
  bool up = h->digits[n] >= 5;
  if ((h->digits[n] == 5) && ((n + 1) == ((int32_t)(h->num_digits)))) {
    up = h->truncated ||  //
         ((n > 0) && ((h->digits[n - 1] & 1) != 0));
  }

  if (up) {
    wuffs_base__private_implementation__high_prec_dec__round_up(h, n);
  } else {
    wuffs_base__private_implementation__high_prec_dec__round_down(h, n);
  }
}

static void  //
wuffs_base__private_implementation__high_prec_dec__round_just_enough(
    wuffs_base__private_implementation__high_prec_dec* h,
    int32_t exp2,
    uint64_t mantissa) {
  // The magic numbers 52 and 53 in this function are because IEEE 754 double
  // precision has 52 mantissa bits.
  //
  // Let f be the floating point number represented by exp2 and mantissa (and
  // also the number in h): the number (mantissa * (2 ** (exp2 - 52))).
  //
  // If f is zero or a small integer, we can return early.
  if ((mantissa == 0) ||
      ((exp2 < 53) && (h->decimal_point >= ((int32_t)(h->num_digits))))) {
    return;
  }

  // The smallest normal f has an exp2 of -1022 and a mantissa of (1 << 52).
  // Subnormal numbers have the same exp2 but a smaller mantissa.
  static const int32_t min_incl_normal_exp2 = -1022;
  static const uint64_t min_incl_normal_mantissa = 0x0010000000000000ul;

  // Compute lower and upper bounds such that any number between them (possibly
  // inclusive) will round to f. First, the lower bound. Our number f is:
  //   ((mantissa + 0)         * (2 ** (  exp2 - 52)))
  //
  // The next lowest floating point number is:
  //   ((mantissa - 1)         * (2 ** (  exp2 - 52)))
  // unless (mantissa - 1) drops the (1 << 52) bit and exp2 is not the
  // min_incl_normal_exp2. Either way, call it:
  //   ((l_mantissa)           * (2 ** (l_exp2 - 52)))
  //
  // The lower bound is halfway between them (noting that 52 became 53):
  //   (((2 * l_mantissa) + 1) * (2 ** (l_exp2 - 53)))
  int32_t l_exp2 = exp2;
  uint64_t l_mantissa = mantissa - 1;
  if ((exp2 > min_incl_normal_exp2) && (mantissa <= min_incl_normal_mantissa)) {
    l_exp2 = exp2 - 1;
    l_mantissa = (2 * mantissa) - 1;
  }
  wuffs_base__private_implementation__high_prec_dec lower;
  wuffs_base__private_implementation__high_prec_dec__assign(
      &lower, (2 * l_mantissa) + 1, false);
  wuffs_base__private_implementation__high_prec_dec__lshift(&lower,
                                                            l_exp2 - 53);

  // Next, the upper bound. Our number f is:
  //   ((mantissa + 0)       * (2 ** (exp2 - 52)))
  //
  // The next highest floating point number is:
  //   ((mantissa + 1)       * (2 ** (exp2 - 52)))
  //
  // The upper bound is halfway between them (noting that 52 became 53):
  //   (((2 * mantissa) + 1) * (2 ** (exp2 - 53)))
  wuffs_base__private_implementation__high_prec_dec upper;
  wuffs_base__private_implementation__high_prec_dec__assign(
      &upper, (2 * mantissa) + 1, false);
  wuffs_base__private_implementation__high_prec_dec__lshift(&upper, exp2 - 53);

  // The lower and upper bounds are possible outputs only if the original
  // mantissa is even, so that IEEE round-to-even would round to the original
  // mantissa and not its neighbors.
  bool inclusive = (mantissa & 1) == 0;

  // As we walk the digits, we want to know whether rounding up would fall
  // within the upper bound. This is tracked by upper_delta:
  //  - When -1, the digits of h and upper are the same so far.
  //  - When +0, we saw a difference of 1 between h and upper on a previous
  //    digit and subsequently only 9s for h and 0s for upper. Thus, rounding
  //    up may fall outside of the bound if !inclusive.
  //  - When +1, the difference is greater than 1 and we know that rounding up
  //    falls within the bound.
  //
  // This is a state machine with three states. The numerical value for each
  // state (-1, +0 or +1) isn't important, other than their order.
  int upper_delta = -1;

  // We can now figure out the shortest number of digits required. Walk the
  // digits until h has distinguished itself from lower or upper.
  //
  // The zi and zd variables are indexes and digits, for z in l (lower), h (the
  // number) and u (upper).
  //
  // The lower, h and upper numbers may have their decimal points at different
  // places. In this case, upper is the longest, so we iterate ui starting from
  // 0 and iterate li and hi starting from either 0 or -1.
  int32_t ui = 0;
  for (;; ui++) {
    // Calculate hd, the middle number's digit.
    int32_t hi = ui - upper.decimal_point + h->decimal_point;
    if (hi >= ((int32_t)(h->num_digits))) {
      break;
    }
    uint8_t hd = (((uint32_t)hi) < h->num_digits) ? h->digits[hi] : 0;

    // Calculate ld, the lower bound's digit.
    int32_t li = ui - upper.decimal_point + lower.decimal_point;
    uint8_t ld = (((uint32_t)li) < lower.num_digits) ? lower.digits[li] : 0;

    // We can round down (truncate) if lower has a different digit than h or if
    // lower is inclusive and is exactly the result of rounding down (i.e. we
    // have reached the final digit of lower).
    bool can_round_down =
        (ld != hd) ||  //
        (inclusive && ((li + 1) == ((int32_t)(lower.num_digits))));

    // Calculate ud, the upper bound's digit, and update upper_delta.
    uint8_t ud = (((uint32_t)ui) < upper.num_digits) ? upper.digits[ui] : 0;
    if (upper_delta < 0) {
      if ((hd + 1) < ud) {
        // For example:
        // h     = 12345???
        // upper = 12347???
        upper_delta = +1;
      } else if (hd != ud) {
        // For example:
        // h     = 12345???
        // upper = 12346???
        upper_delta = +0;
      }
    } else if (upper_delta == 0) {
      if ((hd != 9) || (ud != 0)) {
        // For example:
        // h     = 1234598?
        // upper = 1234600?
        upper_delta = +1;
      }
    }

    // We can round up if upper has a different digit than h and either upper
    // is inclusive or upper is bigger than the result of rounding up.
    bool can_round_up =
        (upper_delta > 0) ||    //
        ((upper_delta == 0) &&  //
         (inclusive || ((ui + 1) < ((int32_t)(upper.num_digits)))));

    // If we can round either way, round to nearest. If we can round only one
    // way, do it. If we can't round, continue the loop.
    if (can_round_down) {
      if (can_round_up) {
        wuffs_base__private_implementation__high_prec_dec__round_nearest(
            h, hi + 1);
        return;
      } else {
        wuffs_base__private_implementation__high_prec_dec__round_down(h,
                                                                      hi + 1);
        return;
      }
    } else {
      if (can_round_up) {
        wuffs_base__private_implementation__high_prec_dec__round_up(h, hi + 1);
        return;
      }
    }
  }
}

// --------

// wuffs_base__private_implementation__parse_number_f64_eisel_lemire produces
// the IEEE 754 double-precision value for an exact mantissa and base-10
// exponent. For example:
//  - when parsing "12345.678e+02", man is 12345678 and exp10 is -1.
//  - when parsing "-12", man is 12 and exp10 is 0. Processing the leading
//    minus sign is the responsibility of the caller, not this function.
//
// On success, it returns a non-negative int64_t such that the low 63 bits hold
// the 11-bit exponent and 52-bit mantissa.
//
// On failure, it returns a negative value.
//
// The algorithm is based on an original idea by Michael Eisel that was refined
// by Daniel Lemire. See
// https://lemire.me/blog/2020/03/10/fast-float-parsing-in-practice/
// and
// https://nigeltao.github.io/blog/2020/eisel-lemire.html
//
// Preconditions:
//  - man is non-zero.
//  - exp10 is in the range [-307 ..= 288], the same range of the
//    wuffs_base__private_implementation__powers_of_10 array.
//
// The exp10 range (and the fact that man is in the range [1 ..= UINT64_MAX],
// approximately [1 ..= 1.85e+19]) means that (man * (10 ** exp10)) is in the
// range [1e-307 ..= 1.85e+307]. This is entirely within the range of normal
// (neither subnormal nor non-finite) f64 values: DBL_MIN and DBL_MAX are
// approximately 2.23e–308 and 1.80e+308.
static int64_t  //
wuffs_base__private_implementation__parse_number_f64_eisel_lemire(
    uint64_t man,
    int32_t exp10) {
  // Look up the (possibly truncated) base-2 representation of (10 ** exp10).
  // The look-up table was constructed so that it is already normalized: the
  // table entry's mantissa's MSB (most significant bit) is on.
  const uint64_t* po10 =
      &wuffs_base__private_implementation__powers_of_10[exp10 + 307][0];

  // Normalize the man argument. The (man != 0) precondition means that a
  // non-zero bit exists.
  uint32_t clz = wuffs_base__count_leading_zeroes_u64(man);
  man <<= clz;

  // Calculate the return value's base-2 exponent. We might tweak it by ±1
  // later, but its initial value comes from a linear scaling of exp10,
  // converting from power-of-10 to power-of-2, and adjusting by clz.
  //
  // The magic constants are:
  //  - 1087 = 1023 + 64. The 1023 is the f64 exponent bias. The 64 is because
  //    the look-up table uses 64-bit mantissas.
  //  - 217706 is such that the ratio 217706 / 65536 ≈ 3.321930 is close enough
  //    (over the practical range of exp10) to log(10) / log(2) ≈ 3.321928.
  //  - 65536 = 1<<16 is arbitrary but a power of 2, so division is a shift.
  //
  // Equality of the linearly-scaled value and the actual power-of-2, over the
  // range of exp10 arguments that this function accepts, is confirmed by
  // script/print-mpb-powers-of-10.go
  uint64_t ret_exp2 =
      ((uint64_t)(((217706 * exp10) >> 16) + 1087)) - ((uint64_t)clz);

  // Multiply the two mantissas. Normalization means that both mantissas are at
  // least (1<<63), so the 128-bit product must be at least (1<<126). The high
  // 64 bits of the product, x_hi, must therefore be at least (1<<62).
  //
  // As a consequence, x_hi has either 0 or 1 leading zeroes. Shifting x_hi
  // right by either 9 or 10 bits (depending on x_hi's MSB) will therefore
  // leave the top 10 MSBs (bits 54 ..= 63) off and the 11th MSB (bit 53) on.
  wuffs_base__multiply_u64__output x = wuffs_base__multiply_u64(man, po10[1]);
  uint64_t x_hi = x.hi;
  uint64_t x_lo = x.lo;

  // Before we shift right by at least 9 bits, recall that the look-up table
  // entry was possibly truncated. We have so far only calculated a lower bound
  // for the product (man * e), where e is (10 ** exp10). The upper bound would
  // add a further (man * 1) to the 128-bit product, which overflows the lower
  // 64-bit limb if ((x_lo + man) < man).
  //
  // If overflow occurs, that adds 1 to x_hi. Since we're about to shift right
  // by at least 9 bits, that carried 1 can be ignored unless the higher 64-bit
  // limb's low 9 bits are all on.
  //
  // For example, parsing "9999999999999999999" will take the if-true branch
  // here, since:
  //  - x_hi = 0x4563918244F3FFFF
  //  - x_lo = 0x8000000000000000
  //  - man  = 0x8AC7230489E7FFFF
  if (((x_hi & 0x1FF) == 0x1FF) && ((x_lo + man) < man)) {
    // Refine our calculation of (man * e). Before, our approximation of e used
    // a "low resolution" 64-bit mantissa. Now use a "high resolution" 128-bit
    // mantissa. We've already calculated x = (man * bits_0_to_63_incl_of_e).
    // Now calculate y = (man * bits_64_to_127_incl_of_e).
    wuffs_base__multiply_u64__output y = wuffs_base__multiply_u64(man, po10[0]);
    uint64_t y_hi = y.hi;
    uint64_t y_lo = y.lo;

    // Merge the 128-bit x and 128-bit y, which overlap by 64 bits, to
    // calculate the 192-bit product of the 64-bit man by the 128-bit e.
    // As we exit this if-block, we only care about the high 128 bits
    // (merged_hi and merged_lo) of that 192-bit product.
    //
    // For example, parsing "1.234e-45" will take the if-true branch here,
    // since:
    //  - x_hi = 0x70B7E3696DB29FFF
    //  - x_lo = 0xE040000000000000
    //  - y_hi = 0x33718BBEAB0E0D7A
    //  - y_lo = 0xA880000000000000
    uint64_t merged_hi = x_hi;
    uint64_t merged_lo = x_lo + y_hi;
    if (merged_lo < x_lo) {
      merged_hi++;  // Carry the overflow bit.
    }

    // The "high resolution" approximation of e is still a lower bound. Once
    // again, see if the upper bound is large enough to produce a different
    // result. This time, if it does, give up instead of reaching for an even
    // more precise approximation to e.
    //
    // This three-part check is similar to the two-part check that guarded the
    // if block that we're now in, but it has an extra term for the middle 64
    // bits (checking that adding 1 to merged_lo would overflow).
    //
    // For example, parsing "5.9604644775390625e-8" will take the if-true
    // branch here, since:
    //  - merged_hi = 0x7FFFFFFFFFFFFFFF
    //  - merged_lo = 0xFFFFFFFFFFFFFFFF
    //  - y_lo      = 0x4DB3FFC120988200
    //  - man       = 0xD3C21BCECCEDA100
    if (((merged_hi & 0x1FF) == 0x1FF) && ((merged_lo + 1) == 0) &&
        (y_lo + man < man)) {
      return -1;
    }

    // Replace the 128-bit x with merged.
    x_hi = merged_hi;
    x_lo = merged_lo;
  }

  // As mentioned above, shifting x_hi right by either 9 or 10 bits will leave
  // the top 10 MSBs (bits 54 ..= 63) off and the 11th MSB (bit 53) on. If the
  // MSB (before shifting) was on, adjust ret_exp2 for the larger shift.
  //
  // Having bit 53 on (and higher bits off) means that ret_mantissa is a 54-bit
  // number.
  uint64_t msb = x_hi >> 63;
  uint64_t ret_mantissa = x_hi >> (msb + 9);
  ret_exp2 -= 1 ^ msb;

  // IEEE 754 rounds to-nearest with ties rounded to-even. Rounding to-even can
  // be tricky. If we're half-way between two exactly representable numbers
  // (x's low 73 bits are zero and the next 2 bits that matter are "01"), give
  // up instead of trying to pick the winner.
  //
  // Technically, we could tighten the condition by changing "73" to "73 or 74,
  // depending on msb", but a flat "73" is simpler.
  //
  // For example, parsing "1e+23" will take the if-true branch here, since:
  //  - x_hi          = 0x54B40B1F852BDA00
  //  - ret_mantissa  = 0x002A5A058FC295ED
  if ((x_lo == 0) && ((x_hi & 0x1FF) == 0) && ((ret_mantissa & 3) == 1)) {
    return -1;
  }

  // If we're not halfway then it's rounding to-nearest. Starting with a 54-bit
  // number, carry the lowest bit (bit 0) up if it's on. Regardless of whether
  // it was on or off, shifting right by one then produces a 53-bit number. If
  // carrying up overflowed, shift again.
  ret_mantissa += ret_mantissa & 1;
  ret_mantissa >>= 1;
  // This if block is equivalent to (but benchmarks slightly faster than) the
  // following branchless form:
  //    uint64_t overflow_adjustment = ret_mantissa >> 53;
  //    ret_mantissa >>= overflow_adjustment;
  //    ret_exp2 += overflow_adjustment;
  //
  // For example, parsing "7.2057594037927933e+16" will take the if-true
  // branch here, since:
  //  - x_hi          = 0x7FFFFFFFFFFFFE80
  //  - ret_mantissa  = 0x0020000000000000
  if ((ret_mantissa >> 53) > 0) {
    ret_mantissa >>= 1;
    ret_exp2++;
  }

  // Starting with a 53-bit number, IEEE 754 double-precision normal numbers
  // have an implicit mantissa bit. Mask that away and keep the low 52 bits.
  ret_mantissa &= 0x000FFFFFFFFFFFFF;

  // Pack the bits and return.
  return ((int64_t)(ret_mantissa | (ret_exp2 << 52)));
}

// --------

static wuffs_base__result_f64  //
wuffs_base__private_implementation__parse_number_f64_special(
    wuffs_base__slice_u8 s,
    uint32_t options) {
  do {
    if (options & WUFFS_BASE__PARSE_NUMBER_FXX__REJECT_INF_AND_NAN) {
      goto fail;
    }

    uint8_t* p = s.ptr;
    uint8_t* q = s.ptr + s.len;

    for (; (p < q) && (*p == '_'); p++) {
    }
    if (p >= q) {
      goto fail;
    }

    // Parse sign.
    bool negative = false;
    do {
      if (*p == '+') {
        p++;
      } else if (*p == '-') {
        negative = true;
        p++;
      } else {
        break;
      }
      for (; (p < q) && (*p == '_'); p++) {
      }
    } while (0);
    if (p >= q) {
      goto fail;
    }

    bool nan = false;
    switch (p[0]) {
      case 'I':
      case 'i':
        if (((q - p) < 3) ||                     //
            ((p[1] != 'N') && (p[1] != 'n')) ||  //
            ((p[2] != 'F') && (p[2] != 'f'))) {
          goto fail;
        }
        p += 3;

        if ((p >= q) || (*p == '_')) {
          break;
        } else if (((q - p) < 5) ||                     //
                   ((p[0] != 'I') && (p[0] != 'i')) ||  //
                   ((p[1] != 'N') && (p[1] != 'n')) ||  //
                   ((p[2] != 'I') && (p[2] != 'i')) ||  //
                   ((p[3] != 'T') && (p[3] != 't')) ||  //
                   ((p[4] != 'Y') && (p[4] != 'y'))) {
          goto fail;
        }
        p += 5;

        if ((p >= q) || (*p == '_')) {
          break;
        }
        goto fail;

      case 'N':
      case 'n':
        if (((q - p) < 3) ||                     //
            ((p[1] != 'A') && (p[1] != 'a')) ||  //
            ((p[2] != 'N') && (p[2] != 'n'))) {
          goto fail;
        }
        p += 3;

        if ((p >= q) || (*p == '_')) {
          nan = true;
          break;
        }
        goto fail;

      default:
        goto fail;
    }

    // Finish.
    for (; (p < q) && (*p == '_'); p++) {
    }
    if (p != q) {
      goto fail;
    }
    wuffs_base__result_f64 ret;
    ret.status.repr = NULL;
    ret.value = wuffs_base__ieee_754_bit_representation__from_u64_to_f64(
        (nan ? 0x7FFFFFFFFFFFFFFF : 0x7FF0000000000000) |
        (negative ? 0x8000000000000000 : 0));
    return ret;
  } while (0);

fail:
  do {
    wuffs_base__result_f64 ret;
    ret.status.repr = wuffs_base__error__bad_argument;
    ret.value = 0;
    return ret;
  } while (0);
}

WUFFS_BASE__MAYBE_STATIC wuffs_base__result_f64  //
wuffs_base__private_implementation__high_prec_dec__to_f64(
    wuffs_base__private_implementation__high_prec_dec* h,
    uint32_t options) {
  do {
    // powers converts decimal powers of 10 to binary powers of 2. For example,
    // (10000 >> 13) is 1. It stops before the elements exceed 60, also known
    // as WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__SHIFT__MAX_INCL.
    //
    // This rounds down (1<<13 is a lower bound for 1e4). Adding 1 to the array
    // element value rounds up (1<<14 is an upper bound for 1e4) while staying
    // at or below WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__SHIFT__MAX_INCL.
    //
    // When starting in the range [1e+1 .. 1e+2] (i.e. h->decimal_point == +2),
    // powers[2] == 6 and so:
    //  - Right shifting by 6+0 produces the range [10/64 .. 100/64] =
    //    [0.156250 .. 1.56250]. The resultant h->decimal_point is +0 or +1.
    //  - Right shifting by 6+1 produces the range [10/128 .. 100/128] =
    //    [0.078125 .. 0.78125]. The resultant h->decimal_point is -1 or -0.
    //
    // When starting in the range [1e-3 .. 1e-2] (i.e. h->decimal_point == -2),
    // powers[2] == 6 and so:
    //  - Left shifting by 6+0 produces the range [0.001*64 .. 0.01*64] =
    //    [0.064 .. 0.64]. The resultant h->decimal_point is -1 or -0.
    //  - Left shifting by 6+1 produces the range [0.001*128 .. 0.01*128] =
    //    [0.128 .. 1.28]. The resultant h->decimal_point is +0 or +1.
    //
    // Thus, when targeting h->decimal_point being +0 or +1, use (powers[n]+0)
    // when right shifting but (powers[n]+1) when left shifting.
    static const uint32_t num_powers = 19;
    static const uint8_t powers[19] = {
        0,  3,  6,  9,  13, 16, 19, 23, 26, 29,  //
        33, 36, 39, 43, 46, 49, 53, 56, 59,      //
    };

    // Handle zero and obvious extremes. The largest and smallest positive
    // finite f64 values are approximately 1.8e+308 and 4.9e-324.
    if ((h->num_digits == 0) || (h->decimal_point < -326)) {
      goto zero;
    } else if (h->decimal_point > 310) {
      goto infinity;
    }

    // Try the fast Eisel-Lemire algorithm again. Calculating the (man, exp10)
    // pair from the high_prec_dec h is more correct but slower than the
    // approach taken in wuffs_base__parse_number_f64. The latter is optimized
    // for the common cases (e.g. assuming no underscores or a leading '+'
    // sign) rather than the full set of cases allowed by the Wuffs API.
    //
    // When we have 19 or fewer mantissa digits, run Eisel-Lemire once (trying
    // for an exact result). When we have more than 19 mantissa digits, run it
    // twice to get a lower and upper bound. We still have an exact result
    // (within f64's rounding margin) if both bounds are equal (and valid).
    uint32_t i_max = h->num_digits;
    if (i_max > 19) {
      i_max = 19;
    }
    int32_t exp10 = h->decimal_point - ((int32_t)i_max);
    if ((-307 <= exp10) && (exp10 <= 288)) {
      uint64_t man = 0;
      uint32_t i;
      for (i = 0; i < i_max; i++) {
        man = (10 * man) + h->digits[i];
      }
      while (man != 0) {  // The 'while' is just an 'if' that we can 'break'.
        int64_t r0 =
            wuffs_base__private_implementation__parse_number_f64_eisel_lemire(
                man + 0, exp10);
        if (r0 < 0) {
          break;
        } else if (h->num_digits > 19) {
          int64_t r1 =
              wuffs_base__private_implementation__parse_number_f64_eisel_lemire(
                  man + 1, exp10);
          if (r1 != r0) {
            break;
          }
        }
        wuffs_base__result_f64 ret;
        ret.status.repr = NULL;
        ret.value = wuffs_base__ieee_754_bit_representation__from_u64_to_f64(
            ((uint64_t)r0) | (((uint64_t)(h->negative)) << 63));
        return ret;
      }
    }

    // When Eisel-Lemire fails, fall back to Simple Decimal Conversion. See
    // https://nigeltao.github.io/blog/2020/parse-number-f64-simple.html
    //
    // Scale by powers of 2 until we're in the range [0.1 .. 10]. Equivalently,
    // that h->decimal_point is +0 or +1.
    //
    // First we shift right while at or above 10...
    const int32_t f64_bias = -1023;
    int32_t exp2 = 0;
    while (h->decimal_point > 1) {
      uint32_t n = (uint32_t)(+h->decimal_point);
      uint32_t shift =
          (n < num_powers)
              ? powers[n]
              : WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__SHIFT__MAX_INCL;

      wuffs_base__private_implementation__high_prec_dec__small_rshift(h, shift);
      if (h->decimal_point <
          -WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__DECIMAL_POINT__RANGE) {
        goto zero;
      }
      exp2 += (int32_t)shift;
    }
    // ...then we shift left while below 0.1.
    while (h->decimal_point < 0) {
      uint32_t shift;
      uint32_t n = (uint32_t)(-h->decimal_point);
      shift = (n < num_powers)
                  // The +1 is per "when targeting h->decimal_point being +0 or
                  // +1... when left shifting" in the powers comment above.
                  ? (powers[n] + 1)
                  : WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__SHIFT__MAX_INCL;

      wuffs_base__private_implementation__high_prec_dec__small_lshift(h, shift);
      if (h->decimal_point >
          +WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__DECIMAL_POINT__RANGE) {
        goto infinity;
      }
      exp2 -= (int32_t)shift;
    }

    // To get from "in the range [0.1 .. 10]" to "in the range [1 .. 2]" (which
    // will give us our exponent in base-2), the mantissa's first 3 digits will
    // determine the final left shift, equal to 52 (the number of explicit f64
    // bits) plus an additional adjustment.
    int man3 = (100 * h->digits[0]) +
               ((h->num_digits > 1) ? (10 * h->digits[1]) : 0) +
               ((h->num_digits > 2) ? h->digits[2] : 0);
    int32_t additional_lshift = 0;
    if (h->decimal_point == 0) {  // The value is in [0.1 .. 1].
      if (man3 < 125) {
        additional_lshift = +4;
      } else if (man3 < 250) {
        additional_lshift = +3;
      } else if (man3 < 500) {
        additional_lshift = +2;
      } else {
        additional_lshift = +1;
      }
    } else {  // The value is in [1 .. 10].
      if (man3 < 200) {
        additional_lshift = -0;
      } else if (man3 < 400) {
        additional_lshift = -1;
      } else if (man3 < 800) {
        additional_lshift = -2;
      } else {
        additional_lshift = -3;
      }
    }
    exp2 -= additional_lshift;
    uint32_t final_lshift = (uint32_t)(52 + additional_lshift);

    // The minimum normal exponent is (f64_bias + 1).
    while ((f64_bias + 1) > exp2) {
      uint32_t n = (uint32_t)((f64_bias + 1) - exp2);
      if (n > WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__SHIFT__MAX_INCL) {
        n = WUFFS_BASE__PRIVATE_IMPLEMENTATION__HPD__SHIFT__MAX_INCL;
      }
      wuffs_base__private_implementation__high_prec_dec__small_rshift(h, n);
      exp2 += (int32_t)n;
    }

    // Check for overflow.
    if ((exp2 - f64_bias) >= 0x07FF) {  // (1 << 11) - 1.
      goto infinity;
    }

    // Extract 53 bits for the mantissa (in base-2).
    wuffs_base__private_implementation__high_prec_dec__small_lshift(
        h, final_lshift);
    uint64_t man2 =
        wuffs_base__private_implementation__high_prec_dec__rounded_integer(h);

    // Rounding might have added one bit. If so, shift and re-check overflow.
    if ((man2 >> 53) != 0) {
      man2 >>= 1;
      exp2++;
      if ((exp2 - f64_bias) >= 0x07FF) {  // (1 << 11) - 1.
        goto infinity;
      }
    }

    // Handle subnormal numbers.
    if ((man2 >> 52) == 0) {
      exp2 = f64_bias;
    }

    // Pack the bits and return.
    uint64_t exp2_bits =
        (uint64_t)((exp2 - f64_bias) & 0x07FF);              // (1 << 11) - 1.
    uint64_t bits = (man2 & 0x000FFFFFFFFFFFFF) |            // (1 << 52) - 1.
                    (exp2_bits << 52) |                      //
                    (h->negative ? 0x8000000000000000 : 0);  // (1 << 63).

    wuffs_base__result_f64 ret;
    ret.status.repr = NULL;
    ret.value = wuffs_base__ieee_754_bit_representation__from_u64_to_f64(bits);
    return ret;
  } while (0);

zero:
  do {
    uint64_t bits = h->negative ? 0x8000000000000000 : 0;

    wuffs_base__result_f64 ret;
    ret.status.repr = NULL;
    ret.value = wuffs_base__ieee_754_bit_representation__from_u64_to_f64(bits);
    return ret;
  } while (0);

infinity:
  do {
    if (options & WUFFS_BASE__PARSE_NUMBER_FXX__REJECT_INF_AND_NAN) {
      wuffs_base__result_f64 ret;
      ret.status.repr = wuffs_base__error__bad_argument;
      ret.value = 0;
      return ret;
    }

    uint64_t bits = h->negative ? 0xFFF0000000000000 : 0x7FF0000000000000;

    wuffs_base__result_f64 ret;
    ret.status.repr = NULL;
    ret.value = wuffs_base__ieee_754_bit_representation__from_u64_to_f64(bits);
    return ret;
  } while (0);
}

static inline bool  //
wuffs_base__private_implementation__is_decimal_digit(uint8_t c) {
  return ('0' <= c) && (c <= '9');
}

WUFFS_BASE__MAYBE_STATIC wuffs_base__result_f64  //
wuffs_base__parse_number_f64(wuffs_base__slice_u8 s, uint32_t options) {
  // In practice, almost all "dd.ddddE±xxx" numbers can be represented
  // losslessly by a uint64_t mantissa "dddddd" and an int32_t base-10
  // exponent, adjusting "xxx" for the position (if present) of the decimal
  // separator '.' or ','.
  //
  // This (u64 man, i32 exp10) data structure is superficially similar to the
  // "Do It Yourself Floating Point" type from Loitsch (†), but the exponent
  // here is base-10, not base-2.
  //
  // If s's number fits in a (man, exp10), parse that pair with the
  // Eisel-Lemire algorithm. If not, or if Eisel-Lemire fails, parsing s with
  // the fallback algorithm is slower but comprehensive.
  //
  // † "Printing Floating-Point Numbers Quickly and Accurately with Integers"
  // (https://www.cs.tufts.edu/~nr/cs257/archive/florian-loitsch/printf.pdf).
  // Florian Loitsch is also the primary contributor to
  // https://github.com/google/double-conversion
  do {
    // Calculating that (man, exp10) pair needs to stay within s's bounds.
    // Provided that s isn't extremely long, work on a NUL-terminated copy of
    // s's contents. The NUL byte isn't a valid part of "±dd.ddddE±xxx".
    //
    // As the pointer p walks the contents, it's faster to repeatedly check "is
    // *p a valid digit" than "is p within bounds and *p a valid digit".
    if (s.len >= 256) {
      goto fallback;
    }
    uint8_t z[256];
    memcpy(&z[0], s.ptr, s.len);
    z[s.len] = 0;
    const uint8_t* p = &z[0];

    // Look for a leading minus sign. Technically, we could also look for an
    // optional plus sign, but the "script/process-json-numbers.c with -p"
    // benchmark is noticably slower if we do. It's optional and, in practice,
    // usually absent. Let the fallback catch it.
    bool negative = (*p == '-');
    if (negative) {
      p++;
    }

    // After walking "dd.dddd", comparing p later with p now will produce the
    // number of "d"s and "."s.
    const uint8_t* const start_of_digits_ptr = p;

    // Walk the "d"s before a '.', 'E', NUL byte, etc. If it starts with '0',
    // it must be a single '0'. If it starts with a non-zero decimal digit, it
    // can be a sequence of decimal digits.
    //
    // Update the man variable during the walk. It's OK if man overflows now.
    // We'll detect that later.
    uint64_t man;
    if (*p == '0') {
      man = 0;
      p++;
      if (wuffs_base__private_implementation__is_decimal_digit(*p)) {
        goto fallback;
      }
    } else if (wuffs_base__private_implementation__is_decimal_digit(*p)) {
      man = ((uint8_t)(*p - '0'));
      p++;
      for (; wuffs_base__private_implementation__is_decimal_digit(*p); p++) {
        man = (10 * man) + ((uint8_t)(*p - '0'));
      }
    } else {
      goto fallback;
    }

    // Walk the "d"s after the optional decimal separator ('.' or ','),
    // updating the man and exp10 variables.
    int32_t exp10 = 0;
    if (*p ==
        ((options & WUFFS_BASE__PARSE_NUMBER_FXX__DECIMAL_SEPARATOR_IS_A_COMMA)
             ? ','
             : '.')) {
      p++;
      const uint8_t* first_after_separator_ptr = p;
      if (!wuffs_base__private_implementation__is_decimal_digit(*p)) {
        goto fallback;
      }
      man = (10 * man) + ((uint8_t)(*p - '0'));
      p++;
      for (; wuffs_base__private_implementation__is_decimal_digit(*p); p++) {
        man = (10 * man) + ((uint8_t)(*p - '0'));
      }
      exp10 = ((int32_t)(first_after_separator_ptr - p));
    }

    // Count the number of digits:
    //  - for an input of "314159",  digit_count is 6.
    //  - for an input of "3.14159", digit_count is 7.
    //
    // This is off-by-one if there is a decimal separator. That's OK for now.
    // We'll correct for that later. The "script/process-json-numbers.c with
    // -p" benchmark is noticably slower if we try to correct for that now.
    uint32_t digit_count = (uint32_t)(p - start_of_digits_ptr);

    // Update exp10 for the optional exponent, starting with 'E' or 'e'.
    if ((*p | 0x20) == 'e') {
      p++;
      int32_t exp_sign = +1;
      if (*p == '-') {
        p++;
        exp_sign = -1;
      } else if (*p == '+') {
        p++;
      }
      if (!wuffs_base__private_implementation__is_decimal_digit(*p)) {
        goto fallback;
      }
      int32_t exp_num = ((uint8_t)(*p - '0'));
      p++;
      // The rest of the exp_num walking has a peculiar control flow but, once
      // again, the "script/process-json-numbers.c with -p" benchmark is
      // sensitive to alternative formulations.
      if (wuffs_base__private_implementation__is_decimal_digit(*p)) {
        exp_num = (10 * exp_num) + ((uint8_t)(*p - '0'));
        p++;
      }
      if (wuffs_base__private_implementation__is_decimal_digit(*p)) {
        exp_num = (10 * exp_num) + ((uint8_t)(*p - '0'));
        p++;
      }
      while (wuffs_base__private_implementation__is_decimal_digit(*p)) {
        if (exp_num > 0x1000000) {
          goto fallback;
        }
        exp_num = (10 * exp_num) + ((uint8_t)(*p - '0'));
        p++;
      }
      exp10 += exp_sign * exp_num;
    }

    // The Wuffs API is that the original slice has no trailing data. It also
    // allows underscores, which we don't catch here but the fallback should.
    if (p != &z[s.len]) {
      goto fallback;
    }

    // Check that the uint64_t typed man variable has not overflowed, based on
    // digit_count.
    //
    // For reference:
    //   - (1 << 63) is  9223372036854775808, which has 19 decimal digits.
    //   - (1 << 64) is 18446744073709551616, which has 20 decimal digits.
    //   - 19 nines,  9999999999999999999, is  0x8AC7230489E7FFFF, which has 64
    //     bits and 16 hexadecimal digits.
    //   - 20 nines, 99999999999999999999, is 0x56BC75E2D630FFFFF, which has 67
    //     bits and 17 hexadecimal digits.
    if (digit_count > 19) {
      // Even if we have more than 19 pseudo-digits, it's not yet definitely an
      // overflow. Recall that digit_count might be off-by-one (too large) if
      // there's a decimal separator. It will also over-report the number of
      // meaningful digits if the input looks something like "0.000dddExxx".
      //
      // We adjust by the number of leading '0's and '.'s and re-compare to 19.
      // Once again, technically, we could skip ','s too, but that perturbs the
      // "script/process-json-numbers.c with -p" benchmark.
      const uint8_t* q = start_of_digits_ptr;
      for (; (*q == '0') || (*q == '.'); q++) {
      }
      digit_count -= (uint32_t)(q - start_of_digits_ptr);
      if (digit_count > 19) {
        goto fallback;
      }
    }

    // The wuffs_base__private_implementation__parse_number_f64_eisel_lemire
    // preconditions include that exp10 is in the range [-307 ..= 288].
    if ((exp10 < -307) || (288 < exp10)) {
      goto fallback;
    }

    // If both man and (10 ** exp10) are exactly representable by a double, we
    // don't need to run the Eisel-Lemire algorithm.
    if ((-22 <= exp10) && (exp10 <= 22) && ((man >> 53) == 0)) {
      double d = (double)man;
      if (exp10 >= 0) {
        d *= wuffs_base__private_implementation__f64_powers_of_10[+exp10];
      } else {
        d /= wuffs_base__private_implementation__f64_powers_of_10[-exp10];
      }
      wuffs_base__result_f64 ret;
      ret.status.repr = NULL;
      ret.value = negative ? -d : +d;
      return ret;
    }

    // The wuffs_base__private_implementation__parse_number_f64_eisel_lemire
    // preconditions include that man is non-zero. Parsing "0" should be caught
    // by the "If both man and (10 ** exp10)" above, but "0e99" might not.
    if (man == 0) {
      goto fallback;
    }

    // Our man and exp10 are in range. Run the Eisel-Lemire algorithm.
    int64_t r =
        wuffs_base__private_implementation__parse_number_f64_eisel_lemire(
            man, exp10);
    if (r < 0) {
      goto fallback;
    }
    wuffs_base__result_f64 ret;
    ret.status.repr = NULL;
    ret.value = wuffs_base__ieee_754_bit_representation__from_u64_to_f64(
        ((uint64_t)r) | (((uint64_t)negative) << 63));
    return ret;
  } while (0);

fallback:
  do {
    wuffs_base__private_implementation__high_prec_dec h;
    wuffs_base__status status =
        wuffs_base__private_implementation__high_prec_dec__parse(&h, s,
                                                                 options);
    if (status.repr) {
      return wuffs_base__private_implementation__parse_number_f64_special(
          s, options);
    }
    return wuffs_base__private_implementation__high_prec_dec__to_f64(&h,
                                                                     options);
  } while (0);
}

// --------

static inline size_t  //
wuffs_base__private_implementation__render_inf(wuffs_base__slice_u8 dst,
                                               bool neg,
                                               uint32_t options) {
  if (neg) {
    if (dst.len < 4) {
      return 0;
    }
    wuffs_base__poke_u32le__no_bounds_check(dst.ptr, 0x666E492D);  // '-Inf'le.
    return 4;
  }

  if (options & WUFFS_BASE__RENDER_NUMBER_XXX__LEADING_PLUS_SIGN) {
    if (dst.len < 4) {
      return 0;
    }
    wuffs_base__poke_u32le__no_bounds_check(dst.ptr, 0x666E492B);  // '+Inf'le.
    return 4;
  }

  if (dst.len < 3) {
    return 0;
  }
  wuffs_base__poke_u24le__no_bounds_check(dst.ptr, 0x666E49);  // 'Inf'le.
  return 3;
}

static inline size_t  //
wuffs_base__private_implementation__render_nan(wuffs_base__slice_u8 dst) {
  if (dst.len < 3) {
    return 0;
  }
  wuffs_base__poke_u24le__no_bounds_check(dst.ptr, 0x4E614E);  // 'NaN'le.
  return 3;
}

static size_t  //
wuffs_base__private_implementation__high_prec_dec__render_exponent_absent(
    wuffs_base__slice_u8 dst,
    wuffs_base__private_implementation__high_prec_dec* h,
    uint32_t precision,
    uint32_t options) {
  size_t n = (h->negative ||
              (options & WUFFS_BASE__RENDER_NUMBER_XXX__LEADING_PLUS_SIGN))
                 ? 1
                 : 0;
  if (h->decimal_point <= 0) {
    n += 1;
  } else {
    n += (size_t)(h->decimal_point);
  }
  if (precision > 0) {
    n += precision + 1;  // +1 for the '.'.
  }

  // Don't modify dst if the formatted number won't fit.
  if (n > dst.len) {
    return 0;
  }

  // Align-left or align-right.
  uint8_t* ptr = (options & WUFFS_BASE__RENDER_NUMBER_XXX__ALIGN_RIGHT)
                     ? &dst.ptr[dst.len - n]
                     : &dst.ptr[0];

  // Leading "±".
  if (h->negative) {
    *ptr++ = '-';
  } else if (options & WUFFS_BASE__RENDER_NUMBER_XXX__LEADING_PLUS_SIGN) {
    *ptr++ = '+';
  }

  // Integral digits.
  if (h->decimal_point <= 0) {
    *ptr++ = '0';
  } else {
    uint32_t m =
        wuffs_base__u32__min(h->num_digits, (uint32_t)(h->decimal_point));
    uint32_t i = 0;
    for (; i < m; i++) {
      *ptr++ = (uint8_t)('0' | h->digits[i]);
    }
    for (; i < (uint32_t)(h->decimal_point); i++) {
      *ptr++ = '0';
    }
  }

  // Separator and then fractional digits.
  if (precision > 0) {
    *ptr++ =
        (options & WUFFS_BASE__RENDER_NUMBER_FXX__DECIMAL_SEPARATOR_IS_A_COMMA)
            ? ','
            : '.';
    uint32_t i = 0;
    for (; i < precision; i++) {
      uint32_t j = ((uint32_t)(h->decimal_point)) + i;
      *ptr++ = (uint8_t)('0' | ((j < h->num_digits) ? h->digits[j] : 0));
    }
  }

  return n;
}

static size_t  //
wuffs_base__private_implementation__high_prec_dec__render_exponent_present(
    wuffs_base__slice_u8 dst,
    wuffs_base__private_implementation__high_prec_dec* h,
    uint32_t precision,
    uint32_t options) {
  int32_t exp = 0;
  if (h->num_digits > 0) {
    exp = h->decimal_point - 1;
  }
  bool negative_exp = exp < 0;
  if (negative_exp) {
    exp = -exp;
  }

  size_t n = (h->negative ||
              (options & WUFFS_BASE__RENDER_NUMBER_XXX__LEADING_PLUS_SIGN))
                 ? 4
                 : 3;  // Mininum 3 bytes: first digit and then "e±".
  if (precision > 0) {
    n += precision + 1;  // +1 for the '.'.
  }
  n += (exp < 100) ? 2 : 3;

  // Don't modify dst if the formatted number won't fit.
  if (n > dst.len) {
    return 0;
  }

  // Align-left or align-right.
  uint8_t* ptr = (options & WUFFS_BASE__RENDER_NUMBER_XXX__ALIGN_RIGHT)
                     ? &dst.ptr[dst.len - n]
                     : &dst.ptr[0];

  // Leading "±".
  if (h->negative) {
    *ptr++ = '-';
  } else if (options & WUFFS_BASE__RENDER_NUMBER_XXX__LEADING_PLUS_SIGN) {
    *ptr++ = '+';
  }

  // Integral digit.
  if (h->num_digits > 0) {
    *ptr++ = (uint8_t)('0' | h->digits[0]);
  } else {
    *ptr++ = '0';
  }

  // Separator and then fractional digits.
  if (precision > 0) {
    *ptr++ =
        (options & WUFFS_BASE__RENDER_NUMBER_FXX__DECIMAL_SEPARATOR_IS_A_COMMA)
            ? ','
            : '.';
    uint32_t i = 1;
    uint32_t j = wuffs_base__u32__min(h->num_digits, precision + 1);
    for (; i < j; i++) {
      *ptr++ = (uint8_t)('0' | h->digits[i]);
    }
    for (; i <= precision; i++) {
      *ptr++ = '0';
    }
  }

  // Exponent: "e±" and then 2 or 3 digits.
  *ptr++ = 'e';
  *ptr++ = negative_exp ? '-' : '+';
  if (exp < 10) {
    *ptr++ = '0';
    *ptr++ = (uint8_t)('0' | exp);
  } else if (exp < 100) {
    *ptr++ = (uint8_t)('0' | (exp / 10));
    *ptr++ = (uint8_t)('0' | (exp % 10));
  } else {
    int32_t e = exp / 100;
    exp -= e * 100;
    *ptr++ = (uint8_t)('0' | e);
    *ptr++ = (uint8_t)('0' | (exp / 10));
    *ptr++ = (uint8_t)('0' | (exp % 10));
  }

  return n;
}

WUFFS_BASE__MAYBE_STATIC size_t  //
wuffs_base__render_number_f64(wuffs_base__slice_u8 dst,
                              double x,
                              uint32_t precision,
                              uint32_t options) {
  // Decompose x (64 bits) into negativity (1 bit), base-2 exponent (11 bits
  // with a -1023 bias) and mantissa (52 bits).
  uint64_t bits = wuffs_base__ieee_754_bit_representation__from_f64_to_u64(x);
  bool neg = (bits >> 63) != 0;
  int32_t exp2 = ((int32_t)(bits >> 52)) & 0x7FF;
  uint64_t man = bits & 0x000FFFFFFFFFFFFFul;

  // Apply the exponent bias and set the implicit top bit of the mantissa,
  // unless x is subnormal. Also take care of Inf and NaN.
  if (exp2 == 0x7FF) {
    if (man != 0) {
      return wuffs_base__private_implementation__render_nan(dst);
    }
    return wuffs_base__private_implementation__render_inf(dst, neg, options);
  } else if (exp2 == 0) {
    exp2 = -1022;
  } else {
    exp2 -= 1023;
    man |= 0x0010000000000000ul;
  }

  // Ensure that precision isn't too large.
  if (precision > 4095) {
    precision = 4095;
  }

  // Convert from the (neg, exp2, man) tuple to an HPD.
  wuffs_base__private_implementation__high_prec_dec h;
  wuffs_base__private_implementation__high_prec_dec__assign(&h, man, neg);
  if (h.num_digits > 0) {
    wuffs_base__private_implementation__high_prec_dec__lshift(
        &h, exp2 - 52);  // 52 mantissa bits.
  }

  // Handle the "%e" and "%f" formats.
  switch (options & (WUFFS_BASE__RENDER_NUMBER_FXX__EXPONENT_ABSENT |
                     WUFFS_BASE__RENDER_NUMBER_FXX__EXPONENT_PRESENT)) {
    case WUFFS_BASE__RENDER_NUMBER_FXX__EXPONENT_ABSENT:  // The "%"f" format.
      if (options & WUFFS_BASE__RENDER_NUMBER_FXX__JUST_ENOUGH_PRECISION) {
        wuffs_base__private_implementation__high_prec_dec__round_just_enough(
            &h, exp2, man);
        int32_t p = ((int32_t)(h.num_digits)) - h.decimal_point;
        precision = ((uint32_t)(wuffs_base__i32__max(0, p)));
      } else {
        wuffs_base__private_implementation__high_prec_dec__round_nearest(
            &h, ((int32_t)precision) + h.decimal_point);
      }
      return wuffs_base__private_implementation__high_prec_dec__render_exponent_absent(
          dst, &h, precision, options);

    case WUFFS_BASE__RENDER_NUMBER_FXX__EXPONENT_PRESENT:  // The "%e" format.
      if (options & WUFFS_BASE__RENDER_NUMBER_FXX__JUST_ENOUGH_PRECISION) {
        wuffs_base__private_implementation__high_prec_dec__round_just_enough(
            &h, exp2, man);
        precision = (h.num_digits > 0) ? (h.num_digits - 1) : 0;
      } else {
        wuffs_base__private_implementation__high_prec_dec__round_nearest(
            &h, ((int32_t)precision) + 1);
      }
      return wuffs_base__private_implementation__high_prec_dec__render_exponent_present(
          dst, &h, precision, options);
  }

  // We have the "%g" format and so precision means the number of significant
  // digits, not the number of digits after the decimal separator. Perform
  // rounding and determine whether to use "%e" or "%f".
  int32_t e_threshold = 0;
  if (options & WUFFS_BASE__RENDER_NUMBER_FXX__JUST_ENOUGH_PRECISION) {
    wuffs_base__private_implementation__high_prec_dec__round_just_enough(
        &h, exp2, man);
    precision = h.num_digits;
    e_threshold = 6;
  } else {
    if (precision == 0) {
      precision = 1;
    }
    wuffs_base__private_implementation__high_prec_dec__round_nearest(
        &h, ((int32_t)precision));
    e_threshold = ((int32_t)precision);
    int32_t nd = ((int32_t)(h.num_digits));
    if ((e_threshold > nd) && (nd >= h.decimal_point)) {
      e_threshold = nd;
    }
  }

  // Use the "%e" format if the exponent is large.
  int32_t e = h.decimal_point - 1;
  if ((e < -4) || (e_threshold <= e)) {
    uint32_t p = wuffs_base__u32__min(precision, h.num_digits);
    return wuffs_base__private_implementation__high_prec_dec__render_exponent_present(
        dst, &h, (p > 0) ? (p - 1) : 0, options);
  }

  // Use the "%f" format otherwise.
  int32_t p = ((int32_t)precision);
  if (p > h.decimal_point) {
    p = ((int32_t)(h.num_digits));
  }
  precision = ((uint32_t)(wuffs_base__i32__max(0, p - h.decimal_point)));
  return wuffs_base__private_implementation__high_prec_dec__render_exponent_absent(
      dst, &h, precision, options);
}

#endif  // !defined(WUFFS_CONFIG__MODULES) ||
        // defined(WUFFS_CONFIG__MODULE__BASE) ||
        // defined(WUFFS_CONFIG__MODULE__BASE__FLOATCONV)

#if !defined(WUFFS_CONFIG__MODULES) || defined(WUFFS_CONFIG__MODULE__BASE) || \
    defined(WUFFS_CONFIG__MODULE__BASE__INTCONV)

// ---------------- Integer

// wuffs_base__parse_number__foo_digits entries are 0x00 for invalid digits,
// and (0x80 | v) for valid digits, where v is the 4 bit value.

static const uint8_t wuffs_base__parse_number__decimal_digits[256] = {
    // 0     1     2     3     4     5     6     7
    // 8     9     A     B     C     D     E     F
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x00 ..= 0x07.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x08 ..= 0x0F.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x10 ..= 0x17.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x18 ..= 0x1F.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x20 ..= 0x27.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x28 ..= 0x2F.
    0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,  // 0x30 ..= 0x37. '0'-'7'.
    0x88, 0x89, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x38 ..= 0x3F. '8'-'9'.

    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x40 ..= 0x47.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x48 ..= 0x4F.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x50 ..= 0x57.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x58 ..= 0x5F.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x60 ..= 0x67.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x68 ..= 0x6F.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x70 ..= 0x77.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x78 ..= 0x7F.

    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x80 ..= 0x87.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x88 ..= 0x8F.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x90 ..= 0x97.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x98 ..= 0x9F.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0xA0 ..= 0xA7.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0xA8 ..= 0xAF.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0xB0 ..= 0xB7.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0xB8 ..= 0xBF.

    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0xC0 ..= 0xC7.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0xC8 ..= 0xCF.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0xD0 ..= 0xD7.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0xD8 ..= 0xDF.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0xE0 ..= 0xE7.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0xE8 ..= 0xEF.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0xF0 ..= 0xF7.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0xF8 ..= 0xFF.
    // 0     1     2     3     4     5     6     7
    // 8     9     A     B     C     D     E     F
};

static const uint8_t wuffs_base__parse_number__hexadecimal_digits[256] = {
    // 0     1     2     3     4     5     6     7
    // 8     9     A     B     C     D     E     F
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x00 ..= 0x07.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x08 ..= 0x0F.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x10 ..= 0x17.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x18 ..= 0x1F.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x20 ..= 0x27.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x28 ..= 0x2F.
    0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,  // 0x30 ..= 0x37. '0'-'7'.
    0x88, 0x89, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x38 ..= 0x3F. '8'-'9'.

    0x00, 0x8A, 0x8B, 0x8C, 0x8D, 0x8E, 0x8F, 0x00,  // 0x40 ..= 0x47. 'A'-'F'.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x48 ..= 0x4F.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x50 ..= 0x57.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x58 ..= 0x5F.
    0x00, 0x8A, 0x8B, 0x8C, 0x8D, 0x8E, 0x8F, 0x00,  // 0x60 ..= 0x67. 'a'-'f'.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x68 ..= 0x6F.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x70 ..= 0x77.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x78 ..= 0x7F.

    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x80 ..= 0x87.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x88 ..= 0x8F.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x90 ..= 0x97.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x98 ..= 0x9F.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0xA0 ..= 0xA7.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0xA8 ..= 0xAF.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0xB0 ..= 0xB7.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0xB8 ..= 0xBF.

    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0xC0 ..= 0xC7.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0xC8 ..= 0xCF.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0xD0 ..= 0xD7.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0xD8 ..= 0xDF.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0xE0 ..= 0xE7.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0xE8 ..= 0xEF.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0xF0 ..= 0xF7.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0xF8 ..= 0xFF.
    // 0     1     2     3     4     5     6     7
    // 8     9     A     B     C     D     E     F
};

static const uint8_t wuffs_base__private_implementation__encode_base16[16] = {
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,  // 0x00 ..= 0x07.
    0x38, 0x39, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46,  // 0x08 ..= 0x0F.
};

// --------

WUFFS_BASE__MAYBE_STATIC wuffs_base__result_i64  //
wuffs_base__parse_number_i64(wuffs_base__slice_u8 s, uint32_t options) {
  uint8_t* p = s.ptr;
  uint8_t* q = s.ptr + s.len;

  if (options & WUFFS_BASE__PARSE_NUMBER_XXX__ALLOW_UNDERSCORES) {
    for (; (p < q) && (*p == '_'); p++) {
    }
  }

  bool negative = false;
  if (p >= q) {
    goto fail_bad_argument;
  } else if (*p == '-') {
    p++;
    negative = true;
  } else if (*p == '+') {
    p++;
  }

  do {
    wuffs_base__result_u64 r = wuffs_base__parse_number_u64(
        wuffs_base__make_slice_u8(p, (size_t)(q - p)), options);
    if (r.status.repr != NULL) {
      wuffs_base__result_i64 ret;
      ret.status.repr = r.status.repr;
      ret.value = 0;
      return ret;
    } else if (negative) {
      if (r.value < 0x8000000000000000) {
        wuffs_base__result_i64 ret;
        ret.status.repr = NULL;
        ret.value = -(int64_t)(r.value);
        return ret;
      } else if (r.value == 0x8000000000000000) {
        wuffs_base__result_i64 ret;
        ret.status.repr = NULL;
        ret.value = INT64_MIN;
        return ret;
      }
      goto fail_out_of_bounds;
    } else if (r.value > 0x7FFFFFFFFFFFFFFF) {
      goto fail_out_of_bounds;
    } else {
      wuffs_base__result_i64 ret;
      ret.status.repr = NULL;
      ret.value = +(int64_t)(r.value);
      return ret;
    }
  } while (0);

fail_bad_argument:
  do {
    wuffs_base__result_i64 ret;
    ret.status.repr = wuffs_base__error__bad_argument;
    ret.value = 0;
    return ret;
  } while (0);

fail_out_of_bounds:
  do {
    wuffs_base__result_i64 ret;
    ret.status.repr = wuffs_base__error__out_of_bounds;
    ret.value = 0;
    return ret;
  } while (0);
}

WUFFS_BASE__MAYBE_STATIC wuffs_base__result_u64  //
wuffs_base__parse_number_u64(wuffs_base__slice_u8 s, uint32_t options) {
  uint8_t* p = s.ptr;
  uint8_t* q = s.ptr + s.len;

  if (options & WUFFS_BASE__PARSE_NUMBER_XXX__ALLOW_UNDERSCORES) {
    for (; (p < q) && (*p == '_'); p++) {
    }
  }

  if (p >= q) {
    goto fail_bad_argument;

  } else if (*p == '0') {
    p++;
    if (p >= q) {
      goto ok_zero;
    }
    if (options & WUFFS_BASE__PARSE_NUMBER_XXX__ALLOW_UNDERSCORES) {
      if (*p == '_') {
        p++;
        for (; p < q; p++) {
          if (*p != '_') {
            if (options &
                WUFFS_BASE__PARSE_NUMBER_XXX__ALLOW_MULTIPLE_LEADING_ZEROES) {
              goto decimal;
            }
            goto fail_bad_argument;
          }
        }
        goto ok_zero;
      }
    }

    if ((*p == 'x') || (*p == 'X')) {
      p++;
      if (options & WUFFS_BASE__PARSE_NUMBER_XXX__ALLOW_UNDERSCORES) {
        for (; (p < q) && (*p == '_'); p++) {
        }
      }
      if (p < q) {
        goto hexadecimal;
      }

    } else if ((*p == 'd') || (*p == 'D')) {
      p++;
      if (options & WUFFS_BASE__PARSE_NUMBER_XXX__ALLOW_UNDERSCORES) {
        for (; (p < q) && (*p == '_'); p++) {
        }
      }
      if (p < q) {
        goto decimal;
      }
    }

    if (options & WUFFS_BASE__PARSE_NUMBER_XXX__ALLOW_MULTIPLE_LEADING_ZEROES) {
      goto decimal;
    }
    goto fail_bad_argument;
  }

decimal:
  do {
    uint64_t v = wuffs_base__parse_number__decimal_digits[*p++];
    if (v == 0) {
      goto fail_bad_argument;
    }
    v &= 0x0F;

    // UINT64_MAX is 18446744073709551615, which is ((10 * max10) + max1).
    const uint64_t max10 = 1844674407370955161u;
    const uint8_t max1 = 5;

    for (; p < q; p++) {
      if ((*p == '_') &&
          (options & WUFFS_BASE__PARSE_NUMBER_XXX__ALLOW_UNDERSCORES)) {
        continue;
      }
      uint8_t digit = wuffs_base__parse_number__decimal_digits[*p];
      if (digit == 0) {
        goto fail_bad_argument;
      }
      digit &= 0x0F;
      if ((v > max10) || ((v == max10) && (digit > max1))) {
        goto fail_out_of_bounds;
      }
      v = (10 * v) + ((uint64_t)(digit));
    }

    wuffs_base__result_u64 ret;
    ret.status.repr = NULL;
    ret.value = v;
    return ret;
  } while (0);

hexadecimal:
  do {
    uint64_t v = wuffs_base__parse_number__hexadecimal_digits[*p++];
    if (v == 0) {
      goto fail_bad_argument;
    }
    v &= 0x0F;

    for (; p < q; p++) {
      if ((*p == '_') &&
          (options & WUFFS_BASE__PARSE_NUMBER_XXX__ALLOW_UNDERSCORES)) {
        continue;
      }
      uint8_t digit = wuffs_base__parse_number__hexadecimal_digits[*p];
      if (digit == 0) {
        goto fail_bad_argument;
      }
      digit &= 0x0F;
      if ((v >> 60) != 0) {
        goto fail_out_of_bounds;
      }
      v = (v << 4) | ((uint64_t)(digit));
    }

    wuffs_base__result_u64 ret;
    ret.status.repr = NULL;
    ret.value = v;
    return ret;
  } while (0);

ok_zero:
  do {
    wuffs_base__result_u64 ret;
    ret.status.repr = NULL;
    ret.value = 0;
    return ret;
  } while (0);

fail_bad_argument:
  do {
    wuffs_base__result_u64 ret;
    ret.status.repr = wuffs_base__error__bad_argument;
    ret.value = 0;
    return ret;
  } while (0);

fail_out_of_bounds:
  do {
    wuffs_base__result_u64 ret;
    ret.status.repr = wuffs_base__error__out_of_bounds;
    ret.value = 0;
    return ret;
  } while (0);
}

// --------

// wuffs_base__render_number__first_hundred contains the decimal encodings of
// the first one hundred numbers [0 ..= 99].
static const uint8_t wuffs_base__render_number__first_hundred[200] = {
    '0', '0', '0', '1', '0', '2', '0', '3', '0', '4',  //
    '0', '5', '0', '6', '0', '7', '0', '8', '0', '9',  //
    '1', '0', '1', '1', '1', '2', '1', '3', '1', '4',  //
    '1', '5', '1', '6', '1', '7', '1', '8', '1', '9',  //
    '2', '0', '2', '1', '2', '2', '2', '3', '2', '4',  //
    '2', '5', '2', '6', '2', '7', '2', '8', '2', '9',  //
    '3', '0', '3', '1', '3', '2', '3', '3', '3', '4',  //
    '3', '5', '3', '6', '3', '7', '3', '8', '3', '9',  //
    '4', '0', '4', '1', '4', '2', '4', '3', '4', '4',  //
    '4', '5', '4', '6', '4', '7', '4', '8', '4', '9',  //
    '5', '0', '5', '1', '5', '2', '5', '3', '5', '4',  //
    '5', '5', '5', '6', '5', '7', '5', '8', '5', '9',  //
    '6', '0', '6', '1', '6', '2', '6', '3', '6', '4',  //
    '6', '5', '6', '6', '6', '7', '6', '8', '6', '9',  //
    '7', '0', '7', '1', '7', '2', '7', '3', '7', '4',  //
    '7', '5', '7', '6', '7', '7', '7', '8', '7', '9',  //
    '8', '0', '8', '1', '8', '2', '8', '3', '8', '4',  //
    '8', '5', '8', '6', '8', '7', '8', '8', '8', '9',  //
    '9', '0', '9', '1', '9', '2', '9', '3', '9', '4',  //
    '9', '5', '9', '6', '9', '7', '9', '8', '9', '9',  //
};

static size_t  //
wuffs_base__private_implementation__render_number_u64(wuffs_base__slice_u8 dst,
                                                      uint64_t x,
                                                      uint32_t options,
                                                      bool neg) {
  uint8_t buf[WUFFS_BASE__U64__BYTE_LENGTH__MAX_INCL];
  uint8_t* ptr = &buf[0] + sizeof(buf);

  while (x >= 100) {
    size_t index = ((size_t)((x % 100) * 2));
    x /= 100;
    uint8_t s0 = wuffs_base__render_number__first_hundred[index + 0];
    uint8_t s1 = wuffs_base__render_number__first_hundred[index + 1];
    ptr -= 2;
    ptr[0] = s0;
    ptr[1] = s1;
  }

  if (x < 10) {
    ptr -= 1;
    ptr[0] = (uint8_t)('0' + x);
  } else {
    size_t index = ((size_t)(x * 2));
    uint8_t s0 = wuffs_base__render_number__first_hundred[index + 0];
    uint8_t s1 = wuffs_base__render_number__first_hundred[index + 1];
    ptr -= 2;
    ptr[0] = s0;
    ptr[1] = s1;
  }

  if (neg) {
    ptr -= 1;
    ptr[0] = '-';
  } else if (options & WUFFS_BASE__RENDER_NUMBER_XXX__LEADING_PLUS_SIGN) {
    ptr -= 1;
    ptr[0] = '+';
  }

  size_t n = sizeof(buf) - ((size_t)(ptr - &buf[0]));
  if (n > dst.len) {
    return 0;
  }
  memcpy(dst.ptr + ((options & WUFFS_BASE__RENDER_NUMBER_XXX__ALIGN_RIGHT)
                        ? (dst.len - n)
                        : 0),
         ptr, n);
  return n;
}

WUFFS_BASE__MAYBE_STATIC size_t  //
wuffs_base__render_number_i64(wuffs_base__slice_u8 dst,
                              int64_t x,
                              uint32_t options) {
  uint64_t u = (uint64_t)x;
  bool neg = x < 0;
  if (neg) {
    u = 1 + ~u;
  }
  return wuffs_base__private_implementation__render_number_u64(dst, u, options,
                                                               neg);
}

WUFFS_BASE__MAYBE_STATIC size_t  //
wuffs_base__render_number_u64(wuffs_base__slice_u8 dst,
                              uint64_t x,
                              uint32_t options) {
  return wuffs_base__private_implementation__render_number_u64(dst, x, options,
                                                               false);
}

// ---------------- Base-16

WUFFS_BASE__MAYBE_STATIC wuffs_base__transform__output  //
wuffs_base__base_16__decode2(wuffs_base__slice_u8 dst,
                             wuffs_base__slice_u8 src,
                             bool src_closed,
                             uint32_t options) {
  wuffs_base__transform__output o;
  size_t src_len2 = src.len / 2;
  size_t len;
  if (dst.len < src_len2) {
    len = dst.len;
    o.status.repr = wuffs_base__suspension__short_write;
  } else {
    len = src_len2;
    if (!src_closed) {
      o.status.repr = wuffs_base__suspension__short_read;
    } else if (src.len & 1) {
      o.status.repr = wuffs_base__error__bad_data;
    } else {
      o.status.repr = NULL;
    }
  }

  uint8_t* d = dst.ptr;
  uint8_t* s = src.ptr;
  size_t n = len;

  while (n--) {
    *d = (uint8_t)((wuffs_base__parse_number__hexadecimal_digits[s[0]] << 4) |
                   (wuffs_base__parse_number__hexadecimal_digits[s[1]] & 0x0F));
    d += 1;
    s += 2;
  }

  o.num_dst = len;
  o.num_src = len * 2;
  return o;
}

WUFFS_BASE__MAYBE_STATIC wuffs_base__transform__output  //
wuffs_base__base_16__decode4(wuffs_base__slice_u8 dst,
                             wuffs_base__slice_u8 src,
                             bool src_closed,
                             uint32_t options) {
  wuffs_base__transform__output o;
  size_t src_len4 = src.len / 4;
  size_t len = dst.len < src_len4 ? dst.len : src_len4;
  if (dst.len < src_len4) {
    len = dst.len;
    o.status.repr = wuffs_base__suspension__short_write;
  } else {
    len = src_len4;
    if (!src_closed) {
      o.status.repr = wuffs_base__suspension__short_read;
    } else if (src.len & 1) {
      o.status.repr = wuffs_base__error__bad_data;
    } else {
      o.status.repr = NULL;
    }
  }

  uint8_t* d = dst.ptr;
  uint8_t* s = src.ptr;
  size_t n = len;

  while (n--) {
    *d = (uint8_t)((wuffs_base__parse_number__hexadecimal_digits[s[2]] << 4) |
                   (wuffs_base__parse_number__hexadecimal_digits[s[3]] & 0x0F));
    d += 1;
    s += 4;
  }

  o.num_dst = len;
  o.num_src = len * 4;
  return o;
}

WUFFS_BASE__MAYBE_STATIC wuffs_base__transform__output  //
wuffs_base__base_16__encode2(wuffs_base__slice_u8 dst,
                             wuffs_base__slice_u8 src,
                             bool src_closed,
                             uint32_t options) {
  wuffs_base__transform__output o;
  size_t dst_len2 = dst.len / 2;
  size_t len;
  if (dst_len2 < src.len) {
    len = dst_len2;
    o.status.repr = wuffs_base__suspension__short_write;
  } else {
    len = src.len;
    if (!src_closed) {
      o.status.repr = wuffs_base__suspension__short_read;
    } else {
      o.status.repr = NULL;
    }
  }

  uint8_t* d = dst.ptr;
  uint8_t* s = src.ptr;
  size_t n = len;

  while (n--) {
    uint8_t c = *s;
    d[0] = wuffs_base__private_implementation__encode_base16[c >> 4];
    d[1] = wuffs_base__private_implementation__encode_base16[c & 0x0F];
    d += 2;
    s += 1;
  }

  o.num_dst = len * 2;
  o.num_src = len;
  return o;
}

WUFFS_BASE__MAYBE_STATIC wuffs_base__transform__output  //
wuffs_base__base_16__encode4(wuffs_base__slice_u8 dst,
                             wuffs_base__slice_u8 src,
                             bool src_closed,
                             uint32_t options) {
  wuffs_base__transform__output o;
  size_t dst_len4 = dst.len / 4;
  size_t len;
  if (dst_len4 < src.len) {
    len = dst_len4;
    o.status.repr = wuffs_base__suspension__short_write;
  } else {
    len = src.len;
    if (!src_closed) {
      o.status.repr = wuffs_base__suspension__short_read;
    } else {
      o.status.repr = NULL;
    }
  }

  uint8_t* d = dst.ptr;
  uint8_t* s = src.ptr;
  size_t n = len;

  while (n--) {
    uint8_t c = *s;
    d[0] = '\\';
    d[1] = 'x';
    d[2] = wuffs_base__private_implementation__encode_base16[c >> 4];
    d[3] = wuffs_base__private_implementation__encode_base16[c & 0x0F];
    d += 4;
    s += 1;
  }

  o.num_dst = len * 4;
  o.num_src = len;
  return o;
}

// ---------------- Base-64

// The two base-64 alphabets, std and url, differ only in the last two codes.
//  - std: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
//  - url: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"

static const uint8_t wuffs_base__base_64__decode_std[256] = {
    // 0     1     2     3     4     5     6     7
    // 8     9     A     B     C     D     E     F
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0x00 ..= 0x07.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0x08 ..= 0x0F.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0x10 ..= 0x17.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0x18 ..= 0x1F.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0x20 ..= 0x27.
    0x80, 0x80, 0x80, 0x3E, 0x80, 0x80, 0x80, 0x3F,  // 0x28 ..= 0x2F.
    0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B,  // 0x30 ..= 0x37.
    0x3C, 0x3D, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0x38 ..= 0x3F.

    0x80, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,  // 0x40 ..= 0x47.
    0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,  // 0x48 ..= 0x4F.
    0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,  // 0x50 ..= 0x57.
    0x17, 0x18, 0x19, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0x58 ..= 0x5F.
    0x80, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20,  // 0x60 ..= 0x67.
    0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,  // 0x68 ..= 0x6F.
    0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30,  // 0x70 ..= 0x77.
    0x31, 0x32, 0x33, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0x78 ..= 0x7F.

    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0x80 ..= 0x87.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0x88 ..= 0x8F.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0x90 ..= 0x97.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0x98 ..= 0x9F.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0xA0 ..= 0xA7.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0xA8 ..= 0xAF.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0xB0 ..= 0xB7.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0xB8 ..= 0xBF.

    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0xC0 ..= 0xC7.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0xC8 ..= 0xCF.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0xD0 ..= 0xD7.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0xD8 ..= 0xDF.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0xE0 ..= 0xE7.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0xE8 ..= 0xEF.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0xF0 ..= 0xF7.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0xF8 ..= 0xFF.
    // 0     1     2     3     4     5     6     7
    // 8     9     A     B     C     D     E     F
};

static const uint8_t wuffs_base__base_64__decode_url[256] = {
    // 0     1     2     3     4     5     6     7
    // 8     9     A     B     C     D     E     F
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0x00 ..= 0x07.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0x08 ..= 0x0F.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0x10 ..= 0x17.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0x18 ..= 0x1F.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0x20 ..= 0x27.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x3E, 0x80, 0x80,  // 0x28 ..= 0x2F.
    0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B,  // 0x30 ..= 0x37.
    0x3C, 0x3D, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0x38 ..= 0x3F.

    0x80, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,  // 0x40 ..= 0x47.
    0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,  // 0x48 ..= 0x4F.
    0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,  // 0x50 ..= 0x57.
    0x17, 0x18, 0x19, 0x80, 0x80, 0x80, 0x80, 0x3F,  // 0x58 ..= 0x5F.
    0x80, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20,  // 0x60 ..= 0x67.
    0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,  // 0x68 ..= 0x6F.
    0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30,  // 0x70 ..= 0x77.
    0x31, 0x32, 0x33, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0x78 ..= 0x7F.

    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0x80 ..= 0x87.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0x88 ..= 0x8F.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0x90 ..= 0x97.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0x98 ..= 0x9F.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0xA0 ..= 0xA7.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0xA8 ..= 0xAF.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0xB0 ..= 0xB7.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0xB8 ..= 0xBF.

    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0xC0 ..= 0xC7.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0xC8 ..= 0xCF.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0xD0 ..= 0xD7.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0xD8 ..= 0xDF.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0xE0 ..= 0xE7.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0xE8 ..= 0xEF.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0xF0 ..= 0xF7.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0xF8 ..= 0xFF.
    // 0     1     2     3     4     5     6     7
    // 8     9     A     B     C     D     E     F
};

static const uint8_t wuffs_base__base_64__encode_std[64] = {
    // 0     1     2     3     4     5     6     7
    // 8     9     A     B     C     D     E     F
    0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,  // 0x00 ..= 0x07.
    0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F, 0x50,  // 0x08 ..= 0x0F.
    0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,  // 0x10 ..= 0x17.
    0x59, 0x5A, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66,  // 0x18 ..= 0x1F.
    0x67, 0x68, 0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E,  // 0x20 ..= 0x27.
    0x6F, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76,  // 0x28 ..= 0x2F.
    0x77, 0x78, 0x79, 0x7A, 0x30, 0x31, 0x32, 0x33,  // 0x30 ..= 0x37.
    0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x2B, 0x2F,  // 0x38 ..= 0x3F.
};

static const uint8_t wuffs_base__base_64__encode_url[64] = {
    // 0     1     2     3     4     5     6     7
    // 8     9     A     B     C     D     E     F
    0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,  // 0x00 ..= 0x07.
    0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F, 0x50,  // 0x08 ..= 0x0F.
    0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,  // 0x10 ..= 0x17.
    0x59, 0x5A, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66,  // 0x18 ..= 0x1F.
    0x67, 0x68, 0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E,  // 0x20 ..= 0x27.
    0x6F, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76,  // 0x28 ..= 0x2F.
    0x77, 0x78, 0x79, 0x7A, 0x30, 0x31, 0x32, 0x33,  // 0x30 ..= 0x37.
    0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x2D, 0x5F,  // 0x38 ..= 0x3F.
};

// --------

WUFFS_BASE__MAYBE_STATIC wuffs_base__transform__output  //
wuffs_base__base_64__decode(wuffs_base__slice_u8 dst,
                            wuffs_base__slice_u8 src,
                            bool src_closed,
                            uint32_t options) {
  const uint8_t* alphabet = (options & WUFFS_BASE__BASE_64__URL_ALPHABET)
                                ? wuffs_base__base_64__decode_url
                                : wuffs_base__base_64__decode_std;
  wuffs_base__transform__output o;
  uint8_t* d_ptr = dst.ptr;
  size_t d_len = dst.len;
  const uint8_t* s_ptr = src.ptr;
  size_t s_len = src.len;
  bool pad = false;

  while (s_len >= 4) {
    uint32_t s = wuffs_base__peek_u32le__no_bounds_check(s_ptr);
    uint32_t s0 = alphabet[0xFF & (s >> 0)];
    uint32_t s1 = alphabet[0xFF & (s >> 8)];
    uint32_t s2 = alphabet[0xFF & (s >> 16)];
    uint32_t s3 = alphabet[0xFF & (s >> 24)];

    if (((s0 | s1 | s2 | s3) & 0xC0) != 0) {
      if (s_len > 4) {
        o.status.repr = wuffs_base__error__bad_data;
        goto done;
      } else if (!src_closed) {
        o.status.repr = wuffs_base__suspension__short_read;
        goto done;
      } else if ((options & WUFFS_BASE__BASE_64__DECODE_ALLOW_PADDING) &&
                 (s_ptr[3] == '=')) {
        pad = true;
        if (s_ptr[2] == '=') {
          goto src2;
        }
        goto src3;
      }
      o.status.repr = wuffs_base__error__bad_data;
      goto done;
    }

    if (d_len < 3) {
      o.status.repr = wuffs_base__suspension__short_write;
      goto done;
    }

    s_ptr += 4;
    s_len -= 4;
    s = (s0 << 18) | (s1 << 12) | (s2 << 6) | (s3 << 0);
    *d_ptr++ = (uint8_t)(s >> 16);
    *d_ptr++ = (uint8_t)(s >> 8);
    *d_ptr++ = (uint8_t)(s >> 0);
    d_len -= 3;
  }

  if (!src_closed) {
    o.status.repr = wuffs_base__suspension__short_read;
    goto done;
  }

  if (s_len == 0) {
    o.status.repr = NULL;
    goto done;
  } else if (s_len == 1) {
    o.status.repr = wuffs_base__error__bad_data;
    goto done;
  } else if (s_len == 2) {
    goto src2;
  }

src3:
  do {
    uint32_t s = wuffs_base__peek_u24le__no_bounds_check(s_ptr);
    uint32_t s0 = alphabet[0xFF & (s >> 0)];
    uint32_t s1 = alphabet[0xFF & (s >> 8)];
    uint32_t s2 = alphabet[0xFF & (s >> 16)];
    if ((s0 & 0xC0) || (s1 & 0xC0) || (s2 & 0xC3)) {
      o.status.repr = wuffs_base__error__bad_data;
      goto done;
    }
    if (d_len < 2) {
      o.status.repr = wuffs_base__suspension__short_write;
      goto done;
    }
    s_ptr += pad ? 4 : 3;
    s = (s0 << 18) | (s1 << 12) | (s2 << 6);
    *d_ptr++ = (uint8_t)(s >> 16);
    *d_ptr++ = (uint8_t)(s >> 8);
    o.status.repr = NULL;
    goto done;
  } while (0);

src2:
  do {
    uint32_t s = wuffs_base__peek_u16le__no_bounds_check(s_ptr);
    uint32_t s0 = alphabet[0xFF & (s >> 0)];
    uint32_t s1 = alphabet[0xFF & (s >> 8)];
    if ((s0 & 0xC0) || (s1 & 0xCF)) {
      o.status.repr = wuffs_base__error__bad_data;
      goto done;
    }
    if (d_len < 1) {
      o.status.repr = wuffs_base__suspension__short_write;
      goto done;
    }
    s_ptr += pad ? 4 : 2;
    s = (s0 << 18) | (s1 << 12);
    *d_ptr++ = (uint8_t)(s >> 16);
    o.status.repr = NULL;
    goto done;
  } while (0);

done:
  o.num_dst = (size_t)(d_ptr - dst.ptr);
  o.num_src = (size_t)(s_ptr - src.ptr);
  return o;
}

WUFFS_BASE__MAYBE_STATIC wuffs_base__transform__output  //
wuffs_base__base_64__encode(wuffs_base__slice_u8 dst,
                            wuffs_base__slice_u8 src,
                            bool src_closed,
                            uint32_t options) {
  const uint8_t* alphabet = (options & WUFFS_BASE__BASE_64__URL_ALPHABET)
                                ? wuffs_base__base_64__encode_url
                                : wuffs_base__base_64__encode_std;
  wuffs_base__transform__output o;
  uint8_t* d_ptr = dst.ptr;
  size_t d_len = dst.len;
  const uint8_t* s_ptr = src.ptr;
  size_t s_len = src.len;

  do {
    while (s_len >= 3) {
      if (d_len < 4) {
        o.status.repr = wuffs_base__suspension__short_write;
        goto done;
      }
      uint32_t s = wuffs_base__peek_u24be__no_bounds_check(s_ptr);
      s_ptr += 3;
      s_len -= 3;
      *d_ptr++ = alphabet[0x3F & (s >> 18)];
      *d_ptr++ = alphabet[0x3F & (s >> 12)];
      *d_ptr++ = alphabet[0x3F & (s >> 6)];
      *d_ptr++ = alphabet[0x3F & (s >> 0)];
      d_len -= 4;
    }

    if (!src_closed) {
      o.status.repr = wuffs_base__suspension__short_read;
      goto done;
    }

    if (s_len == 2) {
      if (d_len <
          ((options & WUFFS_BASE__BASE_64__ENCODE_EMIT_PADDING) ? 4 : 3)) {
        o.status.repr = wuffs_base__suspension__short_write;
        goto done;
      }
      uint32_t s = ((uint32_t)(wuffs_base__peek_u16be__no_bounds_check(s_ptr)))
                   << 8;
      s_ptr += 2;
      *d_ptr++ = alphabet[0x3F & (s >> 18)];
      *d_ptr++ = alphabet[0x3F & (s >> 12)];
      *d_ptr++ = alphabet[0x3F & (s >> 6)];
      if (options & WUFFS_BASE__BASE_64__ENCODE_EMIT_PADDING) {
        *d_ptr++ = '=';
      }
      o.status.repr = NULL;
      goto done;

    } else if (s_len == 1) {
      if (d_len <
          ((options & WUFFS_BASE__BASE_64__ENCODE_EMIT_PADDING) ? 4 : 2)) {
        o.status.repr = wuffs_base__suspension__short_write;
        goto done;
      }
      uint32_t s = ((uint32_t)(wuffs_base__peek_u8__no_bounds_check(s_ptr)))
                   << 16;
      s_ptr += 1;
      *d_ptr++ = alphabet[0x3F & (s >> 18)];
      *d_ptr++ = alphabet[0x3F & (s >> 12)];
      if (options & WUFFS_BASE__BASE_64__ENCODE_EMIT_PADDING) {
        *d_ptr++ = '=';
        *d_ptr++ = '=';
      }
      o.status.repr = NULL;
      goto done;

    } else {
      o.status.repr = NULL;
      goto done;
    }
  } while (0);

done:
  o.num_dst = (size_t)(d_ptr - dst.ptr);
  o.num_src = (size_t)(s_ptr - src.ptr);
  return o;
}

#endif  // !defined(WUFFS_CONFIG__MODULES) ||
        // defined(WUFFS_CONFIG__MODULE__BASE) ||
        // defined(WUFFS_CONFIG__MODULE__BASE__INTCONV)

#if !defined(WUFFS_CONFIG__MODULES) || defined(WUFFS_CONFIG__MODULE__BASE) || \
    defined(WUFFS_CONFIG__MODULE__BASE__MAGIC)

// ---------------- Magic Numbers

// ICO doesn't start with a magic identifier. Instead, see if the opening bytes
// are plausibly ICO.
//
// Callers should have already verified that (prefix_data.len >= 2) and the
// first two bytes are 0x00.
//
// See:
//  - https://docs.fileformat.com/image/ico/
static int32_t  //
wuffs_base__magic_number_guess_fourcc__maybe_ico(
    wuffs_base__slice_u8 prefix_data,
    bool prefix_closed) {
  // Allow-list for the Image Type field.
  if (prefix_data.len < 4) {
    return prefix_closed ? 0 : -1;
  } else if (prefix_data.ptr[3] != 0) {
    return 0;
  }
  switch (prefix_data.ptr[2]) {
    case 0x01:  // ICO
    case 0x02:  // CUR
      break;
    default:
      return 0;
  }

  // The Number Of Images should be positive.
  if (prefix_data.len < 6) {
    return prefix_closed ? 0 : -1;
  } else if ((prefix_data.ptr[4] == 0) && (prefix_data.ptr[5] == 0)) {
    return 0;
  }

  // The first ICONDIRENTRY's fourth byte should be zero.
  if (prefix_data.len < 10) {
    return prefix_closed ? 0 : -1;
  } else if (prefix_data.ptr[9] != 0) {
    return 0;
  }

  // TODO: have a separate FourCC for CUR?
  return 0x49434F20;  // 'ICO 'be
}

// TGA doesn't start with a magic identifier. Instead, see if the opening bytes
// are plausibly TGA.
//
// Callers should have already verified that (prefix_data.len >= 2) and the
// second byte (prefix_data.ptr[1], the Color Map Type byte), is either 0x00 or
// 0x01.
//
// See:
//  - https://docs.fileformat.com/image/tga/
//  - https://www.dca.fee.unicamp.br/~martino/disciplinas/ea978/tgaffs.pdf
static int32_t  //
wuffs_base__magic_number_guess_fourcc__maybe_tga(
    wuffs_base__slice_u8 prefix_data,
    bool prefix_closed) {
  // Allow-list for the Image Type field.
  if (prefix_data.len < 3) {
    return prefix_closed ? 0 : -1;
  }
  switch (prefix_data.ptr[2]) {
    case 0x01:
    case 0x02:
    case 0x03:
    case 0x09:
    case 0x0A:
    case 0x0B:
      break;
    default:
      // TODO: 0x20 and 0x21 are invalid, according to the spec, but are
      // apparently unofficial extensions.
      return 0;
  }

  // Allow-list for the Color Map Entry Size field (if the Color Map Type field
  // is non-zero) or else all the Color Map fields should be zero.
  if (prefix_data.len < 8) {
    return prefix_closed ? 0 : -1;
  } else if (prefix_data.ptr[1] != 0x00) {
    switch (prefix_data.ptr[7]) {
      case 0x0F:
      case 0x10:
      case 0x18:
      case 0x20:
        break;
      default:
        return 0;
    }
  } else if ((prefix_data.ptr[3] | prefix_data.ptr[4] | prefix_data.ptr[5] |
              prefix_data.ptr[6] | prefix_data.ptr[7]) != 0x00) {
    return 0;
  }

  // Allow-list for the Pixel Depth field.
  if (prefix_data.len < 17) {
    return prefix_closed ? 0 : -1;
  }
  switch (prefix_data.ptr[16]) {
    case 0x01:
    case 0x08:
    case 0x0F:
    case 0x10:
    case 0x18:
    case 0x20:
      break;
    default:
      return 0;
  }

  return 0x54474120;  // 'TGA 'be
}

WUFFS_BASE__MAYBE_STATIC int32_t  //
wuffs_base__magic_number_guess_fourcc(wuffs_base__slice_u8 prefix_data,
                                      bool prefix_closed) {
  // This is similar to (but different from):
  //  - the magic/Magdir tables under https://github.com/file/file
  //  - the MIME Sniffing algorithm at https://mimesniff.spec.whatwg.org/

  // table holds the 'magic numbers' (which are actually variable length
  // strings). The strings may contain NUL bytes, so the "const char* magic"
  // value starts with the length-minus-1 of the 'magic number'.
  //
  // Keep it sorted by magic[1], then magic[0] descending (prioritizing longer
  // matches) and finally by magic[2:]. When multiple entries match, the
  // longest one wins.
  //
  // The fourcc field might be negated, in which case there's further
  // specialization (see § below).
  static struct {
    int32_t fourcc;
    const char* magic;
  } table[] = {
      {-0x30302020, "\x01\x00\x00"},          // '00  'be
      {+0x475A2020, "\x02\x1F\x8B\x08"},      // GZ
      {+0x5A535444, "\x03\x28\xB5\x2F\xFD"},  // ZSTD
      {+0x425A3220, "\x02\x42\x5A\x68"},      // BZ2
      {+0x424D5020, "\x01\x42\x4D"},          // BMP
      {+0x47494620, "\x03\x47\x49\x46\x38"},  // GIF
      {+0x54494646, "\x03\x49\x49\x2A\x00"},  // TIFF (little-endian)
      {+0x54494646, "\x03\x4D\x4D\x00\x2A"},  // TIFF (big-endian)
      {-0x52494646, "\x03\x52\x49\x46\x46"},  // RIFF
      {+0x4E494520, "\x02\x6E\xC3\xAF"},      // NIE
      {+0x514F4920, "\x03\x71\x6F\x69\x66"},  // QOI
      {+0x5A4C4942, "\x01\x78\x9C"},          // ZLIB
      {+0x504E4720, "\x03\x89\x50\x4E\x47"},  // PNG
      {+0x4A504547, "\x01\xFF\xD8"},          // JPEG
  };
  static const size_t table_len = sizeof(table) / sizeof(table[0]);

  if (prefix_data.len == 0) {
    return prefix_closed ? 0 : -1;
  }
  uint8_t pre_first_byte = prefix_data.ptr[0];

  int32_t fourcc = 0;
  size_t i;
  for (i = 0; i < table_len; i++) {
    uint8_t mag_first_byte = ((uint8_t)(table[i].magic[1]));
    if (pre_first_byte < mag_first_byte) {
      break;
    } else if (pre_first_byte > mag_first_byte) {
      continue;
    }
    fourcc = table[i].fourcc;

    uint8_t mag_remaining_len = ((uint8_t)(table[i].magic[0]));
    if (mag_remaining_len == 0) {
      goto match;
    }

    const char* mag_remaining_ptr = table[i].magic + 2;
    uint8_t* pre_remaining_ptr = prefix_data.ptr + 1;
    size_t pre_remaining_len = prefix_data.len - 1;
    if (pre_remaining_len < mag_remaining_len) {
      if (!memcmp(pre_remaining_ptr, mag_remaining_ptr, pre_remaining_len)) {
        return prefix_closed ? 0 : -1;
      }
    } else {
      if (!memcmp(pre_remaining_ptr, mag_remaining_ptr, mag_remaining_len)) {
        goto match;
      }
    }
  }

  if (prefix_data.len < 2) {
    return prefix_closed ? 0 : -1;
  } else if ((prefix_data.ptr[1] == 0x00) || (prefix_data.ptr[1] == 0x01)) {
    return wuffs_base__magic_number_guess_fourcc__maybe_tga(prefix_data,
                                                            prefix_closed);
  }

  return 0;

match:
  // Negative FourCC values (see § above) are further specialized.
  if (fourcc < 0) {
    fourcc = -fourcc;

    if (fourcc == 0x52494646) {  // 'RIFF'be
      if (prefix_data.len < 12) {
        return prefix_closed ? 0 : -1;
      }
      uint32_t x = wuffs_base__peek_u32be__no_bounds_check(prefix_data.ptr + 8);
      if (x == 0x57454250) {  // 'WEBP'be
        return 0x57454250;    // 'WEBP'be
      }

    } else if (fourcc == 0x30302020) {  // '00  'be
      // Binary data starting with multiple 0x00 NUL bytes is quite common.
      // Unfortunately, some file formats also don't start with a magic
      // identifier, so we have to use heuristics (where the order matters, the
      // same as /usr/bin/file's magic/Magdir tables) as best we can. Maybe
      // it's TGA, ICO/CUR, etc. Maybe it's something else.
      int32_t tga = wuffs_base__magic_number_guess_fourcc__maybe_tga(
          prefix_data, prefix_closed);
      if (tga != 0) {
        return tga;
      }
      int32_t ico = wuffs_base__magic_number_guess_fourcc__maybe_ico(
          prefix_data, prefix_closed);
      if (ico != 0) {
        return ico;
      }
      if (prefix_data.len < 4) {
        return prefix_closed ? 0 : -1;
      } else if ((prefix_data.ptr[2] != 0x00) &&
                 ((prefix_data.ptr[2] >= 0x80) ||
                  (prefix_data.ptr[3] != 0x00))) {
        // Roughly speaking, this could be a non-degenerate (non-0-width and
        // non-0-height) WBMP image.
        return 0x57424D50;  // 'WBMP'be
      }
      return 0;
    }
  }
  return fourcc;
}

#endif  // !defined(WUFFS_CONFIG__MODULES) ||
        // defined(WUFFS_CONFIG__MODULE__BASE) ||
        // defined(WUFFS_CONFIG__MODULE__BASE__MAGIC)

#if !defined(WUFFS_CONFIG__MODULES) || defined(WUFFS_CONFIG__MODULE__BASE) || \
    defined(WUFFS_CONFIG__MODULE__BASE__PIXCONV)

// ---------------- Pixel Swizzler

static inline uint32_t  //
wuffs_base__swap_u32_argb_abgr(uint32_t u) {
  uint32_t o = u & 0xFF00FF00ul;
  uint32_t r = u & 0x00FF0000ul;
  uint32_t b = u & 0x000000FFul;
  return o | (r >> 16) | (b << 16);
}

static inline uint64_t  //
wuffs_base__swap_u64_argb_abgr(uint64_t u) {
  uint64_t o = u & 0xFFFF0000FFFF0000ull;
  uint64_t r = u & 0x0000FFFF00000000ull;
  uint64_t b = u & 0x000000000000FFFFull;
  return o | (r >> 32) | (b << 32);
}

static inline uint32_t  //
wuffs_base__color_u64__as__color_u32__swap_u32_argb_abgr(uint64_t c) {
  uint32_t a = ((uint32_t)(0xFF & (c >> 56)));
  uint32_t r = ((uint32_t)(0xFF & (c >> 40)));
  uint32_t g = ((uint32_t)(0xFF & (c >> 24)));
  uint32_t b = ((uint32_t)(0xFF & (c >> 8)));
  return (a << 24) | (b << 16) | (g << 8) | (r << 0);
}

// --------

WUFFS_BASE__MAYBE_STATIC wuffs_base__color_u32_argb_premul  //
wuffs_base__pixel_buffer__color_u32_at(const wuffs_base__pixel_buffer* pb,
                                       uint32_t x,
                                       uint32_t y) {
  if (!pb || (x >= pb->pixcfg.private_impl.width) ||
      (y >= pb->pixcfg.private_impl.height)) {
    return 0;
  }

  if (wuffs_base__pixel_format__is_planar(&pb->pixcfg.private_impl.pixfmt)) {
    // TODO: support planar formats.
    return 0;
  }

  size_t stride = pb->private_impl.planes[0].stride;
  const uint8_t* row = pb->private_impl.planes[0].ptr + (stride * ((size_t)y));

  switch (pb->pixcfg.private_impl.pixfmt.repr) {
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_BINARY:
      return wuffs_base__peek_u32le__no_bounds_check(row + (4 * ((size_t)x)));

    case WUFFS_BASE__PIXEL_FORMAT__INDEXED__BGRA_PREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__INDEXED__BGRA_BINARY: {
      uint8_t* palette = pb->private_impl.planes[3].ptr;
      return wuffs_base__peek_u32le__no_bounds_check(palette +
                                                     (4 * ((size_t)row[x])));
    }

      // Common formats above. Rarer formats below.

    case WUFFS_BASE__PIXEL_FORMAT__Y:
      return 0xFF000000 | (0x00010101 * ((uint32_t)(row[x])));
    case WUFFS_BASE__PIXEL_FORMAT__Y_16LE:
      return 0xFF000000 | (0x00010101 * ((uint32_t)(row[(2 * x) + 1])));
    case WUFFS_BASE__PIXEL_FORMAT__Y_16BE:
      return 0xFF000000 | (0x00010101 * ((uint32_t)(row[(2 * x) + 0])));

    case WUFFS_BASE__PIXEL_FORMAT__INDEXED__BGRA_NONPREMUL: {
      uint8_t* palette = pb->private_impl.planes[3].ptr;
      return wuffs_base__color_u32_argb_nonpremul__as__color_u32_argb_premul(
          wuffs_base__peek_u32le__no_bounds_check(palette +
                                                  (4 * ((size_t)row[x]))));
    }

    case WUFFS_BASE__PIXEL_FORMAT__BGR_565:
      return wuffs_base__color_u16_rgb_565__as__color_u32_argb_premul(
          wuffs_base__peek_u16le__no_bounds_check(row + (2 * ((size_t)x))));
    case WUFFS_BASE__PIXEL_FORMAT__BGR:
      return 0xFF000000 |
             wuffs_base__peek_u24le__no_bounds_check(row + (3 * ((size_t)x)));
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL:
      return wuffs_base__color_u32_argb_nonpremul__as__color_u32_argb_premul(
          wuffs_base__peek_u32le__no_bounds_check(row + (4 * ((size_t)x))));
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL_4X16LE:
      return wuffs_base__color_u64_argb_nonpremul__as__color_u32_argb_premul(
          wuffs_base__peek_u64le__no_bounds_check(row + (8 * ((size_t)x))));
    case WUFFS_BASE__PIXEL_FORMAT__BGRX:
      return 0xFF000000 |
             wuffs_base__peek_u32le__no_bounds_check(row + (4 * ((size_t)x)));

    case WUFFS_BASE__PIXEL_FORMAT__RGB:
      return wuffs_base__swap_u32_argb_abgr(
          0xFF000000 |
          wuffs_base__peek_u24le__no_bounds_check(row + (3 * ((size_t)x))));
    case WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL:
      return wuffs_base__swap_u32_argb_abgr(
          wuffs_base__color_u32_argb_nonpremul__as__color_u32_argb_premul(
              wuffs_base__peek_u32le__no_bounds_check(row +
                                                      (4 * ((size_t)x)))));
    case WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__RGBA_BINARY:
      return wuffs_base__swap_u32_argb_abgr(
          wuffs_base__peek_u32le__no_bounds_check(row + (4 * ((size_t)x))));
    case WUFFS_BASE__PIXEL_FORMAT__RGBX:
      return wuffs_base__swap_u32_argb_abgr(
          0xFF000000 |
          wuffs_base__peek_u32le__no_bounds_check(row + (4 * ((size_t)x))));

    default:
      // TODO: support more formats.
      break;
  }

  return 0;
}

// --------

WUFFS_BASE__MAYBE_STATIC wuffs_base__status  //
wuffs_base__pixel_buffer__set_color_u32_at(
    wuffs_base__pixel_buffer* pb,
    uint32_t x,
    uint32_t y,
    wuffs_base__color_u32_argb_premul color) {
  if (!pb) {
    return wuffs_base__make_status(wuffs_base__error__bad_receiver);
  }
  if ((x >= pb->pixcfg.private_impl.width) ||
      (y >= pb->pixcfg.private_impl.height)) {
    return wuffs_base__make_status(wuffs_base__error__bad_argument);
  }

  if (wuffs_base__pixel_format__is_planar(&pb->pixcfg.private_impl.pixfmt)) {
    // TODO: support planar formats.
    return wuffs_base__make_status(wuffs_base__error__unsupported_option);
  }

  size_t stride = pb->private_impl.planes[0].stride;
  uint8_t* row = pb->private_impl.planes[0].ptr + (stride * ((size_t)y));

  switch (pb->pixcfg.private_impl.pixfmt.repr) {
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__BGRX:
      wuffs_base__poke_u32le__no_bounds_check(row + (4 * ((size_t)x)), color);
      break;

      // Common formats above. Rarer formats below.

    case WUFFS_BASE__PIXEL_FORMAT__Y:
      wuffs_base__poke_u8__no_bounds_check(
          row + ((size_t)x),
          wuffs_base__color_u32_argb_premul__as__color_u8_gray(color));
      break;
    case WUFFS_BASE__PIXEL_FORMAT__Y_16LE:
      wuffs_base__poke_u16le__no_bounds_check(
          row + (2 * ((size_t)x)),
          wuffs_base__color_u32_argb_premul__as__color_u16_gray(color));
      break;
    case WUFFS_BASE__PIXEL_FORMAT__Y_16BE:
      wuffs_base__poke_u16be__no_bounds_check(
          row + (2 * ((size_t)x)),
          wuffs_base__color_u32_argb_premul__as__color_u16_gray(color));
      break;

    case WUFFS_BASE__PIXEL_FORMAT__INDEXED__BGRA_NONPREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__INDEXED__BGRA_PREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__INDEXED__BGRA_BINARY:
      wuffs_base__poke_u8__no_bounds_check(
          row + ((size_t)x), wuffs_base__pixel_palette__closest_element(
                                 wuffs_base__pixel_buffer__palette(pb),
                                 pb->pixcfg.private_impl.pixfmt, color));
      break;

    case WUFFS_BASE__PIXEL_FORMAT__BGR_565:
      wuffs_base__poke_u16le__no_bounds_check(
          row + (2 * ((size_t)x)),
          wuffs_base__color_u32_argb_premul__as__color_u16_rgb_565(color));
      break;
    case WUFFS_BASE__PIXEL_FORMAT__BGR:
      wuffs_base__poke_u24le__no_bounds_check(row + (3 * ((size_t)x)), color);
      break;
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL:
      wuffs_base__poke_u32le__no_bounds_check(
          row + (4 * ((size_t)x)),
          wuffs_base__color_u32_argb_premul__as__color_u32_argb_nonpremul(
              color));
      break;
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL_4X16LE:
      wuffs_base__poke_u64le__no_bounds_check(
          row + (8 * ((size_t)x)),
          wuffs_base__color_u32_argb_premul__as__color_u64_argb_nonpremul(
              color));
      break;

    case WUFFS_BASE__PIXEL_FORMAT__RGB:
      wuffs_base__poke_u24le__no_bounds_check(
          row + (3 * ((size_t)x)), wuffs_base__swap_u32_argb_abgr(color));
      break;
    case WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL:
      wuffs_base__poke_u32le__no_bounds_check(
          row + (4 * ((size_t)x)),
          wuffs_base__color_u32_argb_premul__as__color_u32_argb_nonpremul(
              wuffs_base__swap_u32_argb_abgr(color)));
      break;
    case WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__RGBX:
      wuffs_base__poke_u32le__no_bounds_check(
          row + (4 * ((size_t)x)), wuffs_base__swap_u32_argb_abgr(color));
      break;

    default:
      // TODO: support more formats.
      return wuffs_base__make_status(wuffs_base__error__unsupported_option);
  }

  return wuffs_base__make_status(NULL);
}

// --------

static inline void  //
wuffs_base__pixel_buffer__set_color_u32_fill_rect__xx(
    wuffs_base__pixel_buffer* pb,
    wuffs_base__rect_ie_u32 rect,
    uint16_t color) {
  size_t stride = pb->private_impl.planes[0].stride;
  uint32_t width = wuffs_base__rect_ie_u32__width(&rect);
  if ((stride == (2 * ((uint64_t)width))) && (rect.min_incl_x == 0)) {
    uint8_t* ptr =
        pb->private_impl.planes[0].ptr + (stride * ((size_t)rect.min_incl_y));
    uint32_t height = wuffs_base__rect_ie_u32__height(&rect);
    size_t n;
    for (n = ((size_t)width) * ((size_t)height); n > 0; n--) {
      wuffs_base__poke_u16le__no_bounds_check(ptr, color);
      ptr += 2;
    }
    return;
  }

  uint32_t y;
  for (y = rect.min_incl_y; y < rect.max_excl_y; y++) {
    uint8_t* ptr = pb->private_impl.planes[0].ptr + (stride * ((size_t)y)) +
                   (2 * ((size_t)rect.min_incl_x));
    uint32_t n;
    for (n = width; n > 0; n--) {
      wuffs_base__poke_u16le__no_bounds_check(ptr, color);
      ptr += 2;
    }
  }
}

static inline void  //
wuffs_base__pixel_buffer__set_color_u32_fill_rect__xxx(
    wuffs_base__pixel_buffer* pb,
    wuffs_base__rect_ie_u32 rect,
    uint32_t color) {
  size_t stride = pb->private_impl.planes[0].stride;
  uint32_t width = wuffs_base__rect_ie_u32__width(&rect);
  if ((stride == (3 * ((uint64_t)width))) && (rect.min_incl_x == 0)) {
    uint8_t* ptr =
        pb->private_impl.planes[0].ptr + (stride * ((size_t)rect.min_incl_y));
    uint32_t height = wuffs_base__rect_ie_u32__height(&rect);
    size_t n;
    for (n = ((size_t)width) * ((size_t)height); n > 0; n--) {
      wuffs_base__poke_u24le__no_bounds_check(ptr, color);
      ptr += 3;
    }
    return;
  }

  uint32_t y;
  for (y = rect.min_incl_y; y < rect.max_excl_y; y++) {
    uint8_t* ptr = pb->private_impl.planes[0].ptr + (stride * ((size_t)y)) +
                   (3 * ((size_t)rect.min_incl_x));
    uint32_t n;
    for (n = width; n > 0; n--) {
      wuffs_base__poke_u24le__no_bounds_check(ptr, color);
      ptr += 3;
    }
  }
}

static inline void  //
wuffs_base__pixel_buffer__set_color_u32_fill_rect__xxxx(
    wuffs_base__pixel_buffer* pb,
    wuffs_base__rect_ie_u32 rect,
    uint32_t color) {
  size_t stride = pb->private_impl.planes[0].stride;
  uint32_t width = wuffs_base__rect_ie_u32__width(&rect);
  if ((stride == (4 * ((uint64_t)width))) && (rect.min_incl_x == 0)) {
    uint8_t* ptr =
        pb->private_impl.planes[0].ptr + (stride * ((size_t)rect.min_incl_y));
    uint32_t height = wuffs_base__rect_ie_u32__height(&rect);
    size_t n;
    for (n = ((size_t)width) * ((size_t)height); n > 0; n--) {
      wuffs_base__poke_u32le__no_bounds_check(ptr, color);
      ptr += 4;
    }
    return;
  }

  uint32_t y;
  for (y = rect.min_incl_y; y < rect.max_excl_y; y++) {
    uint8_t* ptr = pb->private_impl.planes[0].ptr + (stride * ((size_t)y)) +
                   (4 * ((size_t)rect.min_incl_x));
    uint32_t n;
    for (n = width; n > 0; n--) {
      wuffs_base__poke_u32le__no_bounds_check(ptr, color);
      ptr += 4;
    }
  }
}

static inline void  //
wuffs_base__pixel_buffer__set_color_u32_fill_rect__xxxxxxxx(
    wuffs_base__pixel_buffer* pb,
    wuffs_base__rect_ie_u32 rect,
    uint64_t color) {
  size_t stride = pb->private_impl.planes[0].stride;
  uint32_t width = wuffs_base__rect_ie_u32__width(&rect);
  if ((stride == (8 * ((uint64_t)width))) && (rect.min_incl_x == 0)) {
    uint8_t* ptr =
        pb->private_impl.planes[0].ptr + (stride * ((size_t)rect.min_incl_y));
    uint32_t height = wuffs_base__rect_ie_u32__height(&rect);
    size_t n;
    for (n = ((size_t)width) * ((size_t)height); n > 0; n--) {
      wuffs_base__poke_u64le__no_bounds_check(ptr, color);
      ptr += 8;
    }
    return;
  }

  uint32_t y;
  for (y = rect.min_incl_y; y < rect.max_excl_y; y++) {
    uint8_t* ptr = pb->private_impl.planes[0].ptr + (stride * ((size_t)y)) +
                   (8 * ((size_t)rect.min_incl_x));
    uint32_t n;
    for (n = width; n > 0; n--) {
      wuffs_base__poke_u64le__no_bounds_check(ptr, color);
      ptr += 8;
    }
  }
}

WUFFS_BASE__MAYBE_STATIC wuffs_base__status  //
wuffs_base__pixel_buffer__set_color_u32_fill_rect(
    wuffs_base__pixel_buffer* pb,
    wuffs_base__rect_ie_u32 rect,
    wuffs_base__color_u32_argb_premul color) {
  if (!pb) {
    return wuffs_base__make_status(wuffs_base__error__bad_receiver);
  } else if (wuffs_base__rect_ie_u32__is_empty(&rect)) {
    return wuffs_base__make_status(NULL);
  }
  wuffs_base__rect_ie_u32 bounds =
      wuffs_base__pixel_config__bounds(&pb->pixcfg);
  if (!wuffs_base__rect_ie_u32__contains_rect(&bounds, rect)) {
    return wuffs_base__make_status(wuffs_base__error__bad_argument);
  }

  if (wuffs_base__pixel_format__is_planar(&pb->pixcfg.private_impl.pixfmt)) {
    // TODO: support planar formats.
    return wuffs_base__make_status(wuffs_base__error__unsupported_option);
  }

  switch (pb->pixcfg.private_impl.pixfmt.repr) {
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__BGRX:
      wuffs_base__pixel_buffer__set_color_u32_fill_rect__xxxx(pb, rect, color);
      return wuffs_base__make_status(NULL);

      // Common formats above. Rarer formats below.

    case WUFFS_BASE__PIXEL_FORMAT__BGR_565:
      wuffs_base__pixel_buffer__set_color_u32_fill_rect__xx(
          pb, rect,
          wuffs_base__color_u32_argb_premul__as__color_u16_rgb_565(color));
      return wuffs_base__make_status(NULL);

    case WUFFS_BASE__PIXEL_FORMAT__BGR:
      wuffs_base__pixel_buffer__set_color_u32_fill_rect__xxx(pb, rect, color);
      return wuffs_base__make_status(NULL);

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL:
      wuffs_base__pixel_buffer__set_color_u32_fill_rect__xxxx(
          pb, rect,
          wuffs_base__color_u32_argb_premul__as__color_u32_argb_nonpremul(
              color));
      return wuffs_base__make_status(NULL);

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL_4X16LE:
      wuffs_base__pixel_buffer__set_color_u32_fill_rect__xxxxxxxx(
          pb, rect,
          wuffs_base__color_u32_argb_premul__as__color_u64_argb_nonpremul(
              color));
      return wuffs_base__make_status(NULL);

    case WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL:
      wuffs_base__pixel_buffer__set_color_u32_fill_rect__xxxx(
          pb, rect,
          wuffs_base__color_u32_argb_premul__as__color_u32_argb_nonpremul(
              wuffs_base__swap_u32_argb_abgr(color)));
      return wuffs_base__make_status(NULL);

    case WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__RGBX:
      wuffs_base__pixel_buffer__set_color_u32_fill_rect__xxxx(
          pb, rect, wuffs_base__swap_u32_argb_abgr(color));
      return wuffs_base__make_status(NULL);
  }

  uint32_t y;
  for (y = rect.min_incl_y; y < rect.max_excl_y; y++) {
    uint32_t x;
    for (x = rect.min_incl_x; x < rect.max_excl_x; x++) {
      wuffs_base__pixel_buffer__set_color_u32_at(pb, x, y, color);
    }
  }
  return wuffs_base__make_status(NULL);
}

// --------

WUFFS_BASE__MAYBE_STATIC uint8_t  //
wuffs_base__pixel_palette__closest_element(
    wuffs_base__slice_u8 palette_slice,
    wuffs_base__pixel_format palette_format,
    wuffs_base__color_u32_argb_premul c) {
  size_t n = palette_slice.len / 4;
  if (n > (WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH / 4)) {
    n = (WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH / 4);
  }
  size_t best_index = 0;
  uint64_t best_score = 0xFFFFFFFFFFFFFFFF;

  // Work in 16-bit color.
  uint32_t ca = 0x101 * (0xFF & (c >> 24));
  uint32_t cr = 0x101 * (0xFF & (c >> 16));
  uint32_t cg = 0x101 * (0xFF & (c >> 8));
  uint32_t cb = 0x101 * (0xFF & (c >> 0));

  switch (palette_format.repr) {
    case WUFFS_BASE__PIXEL_FORMAT__INDEXED__BGRA_NONPREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__INDEXED__BGRA_PREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__INDEXED__BGRA_BINARY: {
      bool nonpremul = palette_format.repr ==
                       WUFFS_BASE__PIXEL_FORMAT__INDEXED__BGRA_NONPREMUL;

      size_t i;
      for (i = 0; i < n; i++) {
        // Work in 16-bit color.
        uint32_t pb = 0x101 * ((uint32_t)(palette_slice.ptr[(4 * i) + 0]));
        uint32_t pg = 0x101 * ((uint32_t)(palette_slice.ptr[(4 * i) + 1]));
        uint32_t pr = 0x101 * ((uint32_t)(palette_slice.ptr[(4 * i) + 2]));
        uint32_t pa = 0x101 * ((uint32_t)(palette_slice.ptr[(4 * i) + 3]));

        // Convert to premultiplied alpha.
        if (nonpremul && (pa != 0xFFFF)) {
          pb = (pb * pa) / 0xFFFF;
          pg = (pg * pa) / 0xFFFF;
          pr = (pr * pa) / 0xFFFF;
        }

        // These deltas are conceptually int32_t (signed) but after squaring,
        // it's equivalent to work in uint32_t (unsigned).
        pb -= cb;
        pg -= cg;
        pr -= cr;
        pa -= ca;
        uint64_t score = ((uint64_t)(pb * pb)) + ((uint64_t)(pg * pg)) +
                         ((uint64_t)(pr * pr)) + ((uint64_t)(pa * pa));
        if (best_score > score) {
          best_score = score;
          best_index = i;
        }
      }
      break;
    }
  }

  return (uint8_t)best_index;
}

// --------

static inline uint32_t  //
wuffs_base__composite_nonpremul_nonpremul_u32_axxx(uint32_t dst_nonpremul,
                                                   uint32_t src_nonpremul) {
  // Extract 16-bit color components.
  //
  // If the destination is transparent then SRC_OVER is equivalent to SRC: just
  // return src_nonpremul. This isn't just an optimization (skipping the rest
  // of the function's computation). It also preserves the nonpremul
  // distinction between e.g. transparent red and transparent blue that would
  // otherwise be lost by converting from nonpremul to premul and back.
  uint32_t da = 0x101 * (0xFF & (dst_nonpremul >> 24));
  if (da == 0) {
    return src_nonpremul;
  }
  uint32_t dr = 0x101 * (0xFF & (dst_nonpremul >> 16));
  uint32_t dg = 0x101 * (0xFF & (dst_nonpremul >> 8));
  uint32_t db = 0x101 * (0xFF & (dst_nonpremul >> 0));
  uint32_t sa = 0x101 * (0xFF & (src_nonpremul >> 24));
  uint32_t sr = 0x101 * (0xFF & (src_nonpremul >> 16));
  uint32_t sg = 0x101 * (0xFF & (src_nonpremul >> 8));
  uint32_t sb = 0x101 * (0xFF & (src_nonpremul >> 0));

  // Convert dst from nonpremul to premul.
  dr = (dr * da) / 0xFFFF;
  dg = (dg * da) / 0xFFFF;
  db = (db * da) / 0xFFFF;

  // Calculate the inverse of the src-alpha: how much of the dst to keep.
  uint32_t ia = 0xFFFF - sa;

  // Composite src (nonpremul) over dst (premul).
  da = sa + ((da * ia) / 0xFFFF);
  dr = ((sr * sa) + (dr * ia)) / 0xFFFF;
  dg = ((sg * sa) + (dg * ia)) / 0xFFFF;
  db = ((sb * sa) + (db * ia)) / 0xFFFF;

  // Convert dst from premul to nonpremul.
  if (da != 0) {
    dr = (dr * 0xFFFF) / da;
    dg = (dg * 0xFFFF) / da;
    db = (db * 0xFFFF) / da;
  }

  // Convert from 16-bit color to 8-bit color.
  da >>= 8;
  dr >>= 8;
  dg >>= 8;
  db >>= 8;

  // Combine components.
  return (db << 0) | (dg << 8) | (dr << 16) | (da << 24);
}

static inline uint64_t  //
wuffs_base__composite_nonpremul_nonpremul_u64_axxx(uint64_t dst_nonpremul,
                                                   uint64_t src_nonpremul) {
  // Extract components.
  //
  // If the destination is transparent then SRC_OVER is equivalent to SRC: just
  // return src_nonpremul. This isn't just an optimization (skipping the rest
  // of the function's computation). It also preserves the nonpremul
  // distinction between e.g. transparent red and transparent blue that would
  // otherwise be lost by converting from nonpremul to premul and back.
  uint64_t da = 0xFFFF & (dst_nonpremul >> 48);
  if (da == 0) {
    return src_nonpremul;
  }
  uint64_t dr = 0xFFFF & (dst_nonpremul >> 32);
  uint64_t dg = 0xFFFF & (dst_nonpremul >> 16);
  uint64_t db = 0xFFFF & (dst_nonpremul >> 0);
  uint64_t sa = 0xFFFF & (src_nonpremul >> 48);
  uint64_t sr = 0xFFFF & (src_nonpremul >> 32);
  uint64_t sg = 0xFFFF & (src_nonpremul >> 16);
  uint64_t sb = 0xFFFF & (src_nonpremul >> 0);

  // Convert dst from nonpremul to premul.
  dr = (dr * da) / 0xFFFF;
  dg = (dg * da) / 0xFFFF;
  db = (db * da) / 0xFFFF;

  // Calculate the inverse of the src-alpha: how much of the dst to keep.
  uint64_t ia = 0xFFFF - sa;

  // Composite src (nonpremul) over dst (premul).
  da = sa + ((da * ia) / 0xFFFF);
  dr = ((sr * sa) + (dr * ia)) / 0xFFFF;
  dg = ((sg * sa) + (dg * ia)) / 0xFFFF;
  db = ((sb * sa) + (db * ia)) / 0xFFFF;

  // Convert dst from premul to nonpremul.
  if (da != 0) {
    dr = (dr * 0xFFFF) / da;
    dg = (dg * 0xFFFF) / da;
    db = (db * 0xFFFF) / da;
  }

  // Combine components.
  return (db << 0) | (dg << 16) | (dr << 32) | (da << 48);
}

static inline uint32_t  //
wuffs_base__composite_nonpremul_premul_u32_axxx(uint32_t dst_nonpremul,
                                                uint32_t src_premul) {
  // Extract 16-bit color components.
  uint32_t da = 0x101 * (0xFF & (dst_nonpremul >> 24));
  uint32_t dr = 0x101 * (0xFF & (dst_nonpremul >> 16));
  uint32_t dg = 0x101 * (0xFF & (dst_nonpremul >> 8));
  uint32_t db = 0x101 * (0xFF & (dst_nonpremul >> 0));
  uint32_t sa = 0x101 * (0xFF & (src_premul >> 24));
  uint32_t sr = 0x101 * (0xFF & (src_premul >> 16));
  uint32_t sg = 0x101 * (0xFF & (src_premul >> 8));
  uint32_t sb = 0x101 * (0xFF & (src_premul >> 0));

  // Convert dst from nonpremul to premul.
  dr = (dr * da) / 0xFFFF;
  dg = (dg * da) / 0xFFFF;
  db = (db * da) / 0xFFFF;

  // Calculate the inverse of the src-alpha: how much of the dst to keep.
  uint32_t ia = 0xFFFF - sa;

  // Composite src (premul) over dst (premul).
  da = sa + ((da * ia) / 0xFFFF);
  dr = sr + ((dr * ia) / 0xFFFF);
  dg = sg + ((dg * ia) / 0xFFFF);
  db = sb + ((db * ia) / 0xFFFF);

  // Convert dst from premul to nonpremul.
  if (da != 0) {
    dr = (dr * 0xFFFF) / da;
    dg = (dg * 0xFFFF) / da;
    db = (db * 0xFFFF) / da;
  }

  // Convert from 16-bit color to 8-bit color.
  da >>= 8;
  dr >>= 8;
  dg >>= 8;
  db >>= 8;

  // Combine components.
  return (db << 0) | (dg << 8) | (dr << 16) | (da << 24);
}

static inline uint64_t  //
wuffs_base__composite_nonpremul_premul_u64_axxx(uint64_t dst_nonpremul,
                                                uint64_t src_premul) {
  // Extract components.
  uint64_t da = 0xFFFF & (dst_nonpremul >> 48);
  uint64_t dr = 0xFFFF & (dst_nonpremul >> 32);
  uint64_t dg = 0xFFFF & (dst_nonpremul >> 16);
  uint64_t db = 0xFFFF & (dst_nonpremul >> 0);
  uint64_t sa = 0xFFFF & (src_premul >> 48);
  uint64_t sr = 0xFFFF & (src_premul >> 32);
  uint64_t sg = 0xFFFF & (src_premul >> 16);
  uint64_t sb = 0xFFFF & (src_premul >> 0);

  // Convert dst from nonpremul to premul.
  dr = (dr * da) / 0xFFFF;
  dg = (dg * da) / 0xFFFF;
  db = (db * da) / 0xFFFF;

  // Calculate the inverse of the src-alpha: how much of the dst to keep.
  uint64_t ia = 0xFFFF - sa;

  // Composite src (premul) over dst (premul).
  da = sa + ((da * ia) / 0xFFFF);
  dr = sr + ((dr * ia) / 0xFFFF);
  dg = sg + ((dg * ia) / 0xFFFF);
  db = sb + ((db * ia) / 0xFFFF);

  // Convert dst from premul to nonpremul.
  if (da != 0) {
    dr = (dr * 0xFFFF) / da;
    dg = (dg * 0xFFFF) / da;
    db = (db * 0xFFFF) / da;
  }

  // Combine components.
  return (db << 0) | (dg << 16) | (dr << 32) | (da << 48);
}

static inline uint32_t  //
wuffs_base__composite_premul_nonpremul_u32_axxx(uint32_t dst_premul,
                                                uint32_t src_nonpremul) {
  // Extract 16-bit color components.
  uint32_t da = 0x101 * (0xFF & (dst_premul >> 24));
  uint32_t dr = 0x101 * (0xFF & (dst_premul >> 16));
  uint32_t dg = 0x101 * (0xFF & (dst_premul >> 8));
  uint32_t db = 0x101 * (0xFF & (dst_premul >> 0));
  uint32_t sa = 0x101 * (0xFF & (src_nonpremul >> 24));
  uint32_t sr = 0x101 * (0xFF & (src_nonpremul >> 16));
  uint32_t sg = 0x101 * (0xFF & (src_nonpremul >> 8));
  uint32_t sb = 0x101 * (0xFF & (src_nonpremul >> 0));

  // Calculate the inverse of the src-alpha: how much of the dst to keep.
  uint32_t ia = 0xFFFF - sa;

  // Composite src (nonpremul) over dst (premul).
  da = sa + ((da * ia) / 0xFFFF);
  dr = ((sr * sa) + (dr * ia)) / 0xFFFF;
  dg = ((sg * sa) + (dg * ia)) / 0xFFFF;
  db = ((sb * sa) + (db * ia)) / 0xFFFF;

  // Convert from 16-bit color to 8-bit color.
  da >>= 8;
  dr >>= 8;
  dg >>= 8;
  db >>= 8;

  // Combine components.
  return (db << 0) | (dg << 8) | (dr << 16) | (da << 24);
}

static inline uint64_t  //
wuffs_base__composite_premul_nonpremul_u64_axxx(uint64_t dst_premul,
                                                uint64_t src_nonpremul) {
  // Extract components.
  uint64_t da = 0xFFFF & (dst_premul >> 48);
  uint64_t dr = 0xFFFF & (dst_premul >> 32);
  uint64_t dg = 0xFFFF & (dst_premul >> 16);
  uint64_t db = 0xFFFF & (dst_premul >> 0);
  uint64_t sa = 0xFFFF & (src_nonpremul >> 48);
  uint64_t sr = 0xFFFF & (src_nonpremul >> 32);
  uint64_t sg = 0xFFFF & (src_nonpremul >> 16);
  uint64_t sb = 0xFFFF & (src_nonpremul >> 0);

  // Calculate the inverse of the src-alpha: how much of the dst to keep.
  uint64_t ia = 0xFFFF - sa;

  // Composite src (nonpremul) over dst (premul).
  da = sa + ((da * ia) / 0xFFFF);
  dr = ((sr * sa) + (dr * ia)) / 0xFFFF;
  dg = ((sg * sa) + (dg * ia)) / 0xFFFF;
  db = ((sb * sa) + (db * ia)) / 0xFFFF;

  // Combine components.
  return (db << 0) | (dg << 16) | (dr << 32) | (da << 48);
}

static inline uint32_t  //
wuffs_base__composite_premul_premul_u32_axxx(uint32_t dst_premul,
                                             uint32_t src_premul) {
  // Extract 16-bit color components.
  uint32_t da = 0x101 * (0xFF & (dst_premul >> 24));
  uint32_t dr = 0x101 * (0xFF & (dst_premul >> 16));
  uint32_t dg = 0x101 * (0xFF & (dst_premul >> 8));
  uint32_t db = 0x101 * (0xFF & (dst_premul >> 0));
  uint32_t sa = 0x101 * (0xFF & (src_premul >> 24));
  uint32_t sr = 0x101 * (0xFF & (src_premul >> 16));
  uint32_t sg = 0x101 * (0xFF & (src_premul >> 8));
  uint32_t sb = 0x101 * (0xFF & (src_premul >> 0));

  // Calculate the inverse of the src-alpha: how much of the dst to keep.
  uint32_t ia = 0xFFFF - sa;

  // Composite src (premul) over dst (premul).
  da = sa + ((da * ia) / 0xFFFF);
  dr = sr + ((dr * ia) / 0xFFFF);
  dg = sg + ((dg * ia) / 0xFFFF);
  db = sb + ((db * ia) / 0xFFFF);

  // Convert from 16-bit color to 8-bit color.
  da >>= 8;
  dr >>= 8;
  dg >>= 8;
  db >>= 8;

  // Combine components.
  return (db << 0) | (dg << 8) | (dr << 16) | (da << 24);
}

// --------

static uint64_t  //
wuffs_base__pixel_swizzler__squash_align4_bgr_565_8888(uint8_t* dst_ptr,
                                                       size_t dst_len,
                                                       const uint8_t* src_ptr,
                                                       size_t src_len,
                                                       bool nonpremul) {
  size_t len = (dst_len < src_len ? dst_len : src_len) / 4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;

  size_t n = len;
  while (n--) {
    uint32_t argb = wuffs_base__peek_u32le__no_bounds_check(s);
    if (nonpremul) {
      argb =
          wuffs_base__color_u32_argb_nonpremul__as__color_u32_argb_premul(argb);
    }
    uint32_t b5 = 0x1F & (argb >> (8 - 5));
    uint32_t g6 = 0x3F & (argb >> (16 - 6));
    uint32_t r5 = 0x1F & (argb >> (24 - 5));
    uint32_t alpha = argb & 0xFF000000;
    wuffs_base__poke_u32le__no_bounds_check(
        d, alpha | (r5 << 11) | (g6 << 5) | (b5 << 0));
    s += 4;
    d += 4;
  }
  return len;
}

// --------

static uint64_t  //
wuffs_base__pixel_swizzler__swap_rgb_bgr(uint8_t* dst_ptr,
                                         size_t dst_len,
                                         uint8_t* dst_palette_ptr,
                                         size_t dst_palette_len,
                                         const uint8_t* src_ptr,
                                         size_t src_len) {
  size_t len = (dst_len < src_len ? dst_len : src_len) / 3;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;

  size_t n = len;
  while (n--) {
    uint8_t s0 = s[0];
    uint8_t s1 = s[1];
    uint8_t s2 = s[2];
    d[0] = s2;
    d[1] = s1;
    d[2] = s0;
    s += 3;
    d += 3;
  }
  return len;
}

// ‼ WUFFS MULTI-FILE SECTION +x86_sse42
#if defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
WUFFS_BASE__MAYBE_ATTRIBUTE_TARGET("pclmul,popcnt,sse4.2")
static uint64_t  //
wuffs_base__pixel_swizzler__swap_rgbx_bgrx__sse42(uint8_t* dst_ptr,
                                                  size_t dst_len,
                                                  uint8_t* dst_palette_ptr,
                                                  size_t dst_palette_len,
                                                  const uint8_t* src_ptr,
                                                  size_t src_len) {
  size_t len = (dst_len < src_len ? dst_len : src_len) / 4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  __m128i shuffle = _mm_set_epi8(+0x0F, +0x0C, +0x0D, +0x0E,  //
                                 +0x0B, +0x08, +0x09, +0x0A,  //
                                 +0x07, +0x04, +0x05, +0x06,  //
                                 +0x03, +0x00, +0x01, +0x02);

  while (n >= 4) {
    __m128i x;
    x = _mm_lddqu_si128((const __m128i*)(const void*)s);
    x = _mm_shuffle_epi8(x, shuffle);
    _mm_storeu_si128((__m128i*)(void*)d, x);

    s += 4 * 4;
    d += 4 * 4;
    n -= 4;
  }

  while (n--) {
    uint8_t s0 = s[0];
    uint8_t s1 = s[1];
    uint8_t s2 = s[2];
    uint8_t s3 = s[3];
    d[0] = s2;
    d[1] = s1;
    d[2] = s0;
    d[3] = s3;
    s += 4;
    d += 4;
  }
  return len;
}
#endif  // defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
// ‼ WUFFS MULTI-FILE SECTION -x86_sse42

static uint64_t  //
wuffs_base__pixel_swizzler__swap_rgbx_bgrx(uint8_t* dst_ptr,
                                           size_t dst_len,
                                           uint8_t* dst_palette_ptr,
                                           size_t dst_palette_len,
                                           const uint8_t* src_ptr,
                                           size_t src_len) {
  size_t len = (dst_len < src_len ? dst_len : src_len) / 4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;

  size_t n = len;
  while (n--) {
    uint8_t s0 = s[0];
    uint8_t s1 = s[1];
    uint8_t s2 = s[2];
    uint8_t s3 = s[3];
    d[0] = s2;
    d[1] = s1;
    d[2] = s0;
    d[3] = s3;
    s += 4;
    d += 4;
  }
  return len;
}

// --------

static uint64_t  //
wuffs_base__pixel_swizzler__copy_1_1(uint8_t* dst_ptr,
                                     size_t dst_len,
                                     uint8_t* dst_palette_ptr,
                                     size_t dst_palette_len,
                                     const uint8_t* src_ptr,
                                     size_t src_len) {
  size_t len = (dst_len < src_len) ? dst_len : src_len;
  if (len > 0) {
    memmove(dst_ptr, src_ptr, len);
  }
  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__copy_2_2(uint8_t* dst_ptr,
                                     size_t dst_len,
                                     uint8_t* dst_palette_ptr,
                                     size_t dst_palette_len,
                                     const uint8_t* src_ptr,
                                     size_t src_len) {
  size_t dst_len2 = dst_len / 2;
  size_t src_len2 = src_len / 2;
  size_t len = (dst_len2 < src_len2) ? dst_len2 : src_len2;
  if (len > 0) {
    memmove(dst_ptr, src_ptr, len * 2);
  }
  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__copy_3_3(uint8_t* dst_ptr,
                                     size_t dst_len,
                                     uint8_t* dst_palette_ptr,
                                     size_t dst_palette_len,
                                     const uint8_t* src_ptr,
                                     size_t src_len) {
  size_t dst_len3 = dst_len / 3;
  size_t src_len3 = src_len / 3;
  size_t len = (dst_len3 < src_len3) ? dst_len3 : src_len3;
  if (len > 0) {
    memmove(dst_ptr, src_ptr, len * 3);
  }
  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__copy_4_4(uint8_t* dst_ptr,
                                     size_t dst_len,
                                     uint8_t* dst_palette_ptr,
                                     size_t dst_palette_len,
                                     const uint8_t* src_ptr,
                                     size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len4 < src_len4) ? dst_len4 : src_len4;
  if (len > 0) {
    memmove(dst_ptr, src_ptr, len * 4);
  }
  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__copy_8_8(uint8_t* dst_ptr,
                                     size_t dst_len,
                                     uint8_t* dst_palette_ptr,
                                     size_t dst_palette_len,
                                     const uint8_t* src_ptr,
                                     size_t src_len) {
  size_t dst_len8 = dst_len / 8;
  size_t src_len8 = src_len / 8;
  size_t len = (dst_len8 < src_len8) ? dst_len8 : src_len8;
  if (len > 0) {
    memmove(dst_ptr, src_ptr, len * 8);
  }
  return len;
}

// --------

static uint64_t  //
wuffs_base__pixel_swizzler__bgr_565__bgr(uint8_t* dst_ptr,
                                         size_t dst_len,
                                         uint8_t* dst_palette_ptr,
                                         size_t dst_palette_len,
                                         const uint8_t* src_ptr,
                                         size_t src_len) {
  size_t dst_len2 = dst_len / 2;
  size_t src_len3 = src_len / 3;
  size_t len = (dst_len2 < src_len3) ? dst_len2 : src_len3;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    uint32_t b5 = s[0] >> 3;
    uint32_t g6 = s[1] >> 2;
    uint32_t r5 = s[2] >> 3;
    uint32_t rgb_565 = (r5 << 11) | (g6 << 5) | (b5 << 0);
    wuffs_base__poke_u16le__no_bounds_check(d + (0 * 2), (uint16_t)rgb_565);

    s += 1 * 3;
    d += 1 * 2;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr_565__bgrx(uint8_t* dst_ptr,
                                          size_t dst_len,
                                          uint8_t* dst_palette_ptr,
                                          size_t dst_palette_len,
                                          const uint8_t* src_ptr,
                                          size_t src_len) {
  size_t dst_len2 = dst_len / 2;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len2 < src_len4) ? dst_len2 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    uint32_t b5 = s[0] >> 3;
    uint32_t g6 = s[1] >> 2;
    uint32_t r5 = s[2] >> 3;
    uint32_t rgb_565 = (r5 << 11) | (g6 << 5) | (b5 << 0);
    wuffs_base__poke_u16le__no_bounds_check(d + (0 * 2), (uint16_t)rgb_565);

    s += 1 * 4;
    d += 1 * 2;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr_565__bgra_nonpremul__src(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len2 = dst_len / 2;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len2 < src_len4) ? dst_len2 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    wuffs_base__poke_u16le__no_bounds_check(
        d + (0 * 2),
        wuffs_base__color_u32_argb_premul__as__color_u16_rgb_565(
            wuffs_base__color_u32_argb_nonpremul__as__color_u32_argb_premul(
                wuffs_base__peek_u32le__no_bounds_check(s + (0 * 4)))));

    s += 1 * 4;
    d += 1 * 2;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr_565__bgra_nonpremul_4x16le__src(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len2 = dst_len / 2;
  size_t src_len8 = src_len / 8;
  size_t len = (dst_len2 < src_len8) ? dst_len2 : src_len8;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    wuffs_base__poke_u16le__no_bounds_check(
        d + (0 * 2),
        wuffs_base__color_u32_argb_premul__as__color_u16_rgb_565(
            wuffs_base__color_u64_argb_nonpremul__as__color_u32_argb_premul(
                wuffs_base__peek_u64le__no_bounds_check(s + (0 * 8)))));

    s += 1 * 8;
    d += 1 * 2;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr_565__bgra_nonpremul__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len2 = dst_len / 2;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len2 < src_len4) ? dst_len2 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    // Extract 16-bit color components.
    uint32_t sa = 0x101 * ((uint32_t)s[3]);
    uint32_t sr = 0x101 * ((uint32_t)s[2]);
    uint32_t sg = 0x101 * ((uint32_t)s[1]);
    uint32_t sb = 0x101 * ((uint32_t)s[0]);

    // Convert from 565 color to 16-bit color.
    uint32_t old_rgb_565 = wuffs_base__peek_u16le__no_bounds_check(d + (0 * 2));
    uint32_t old_r5 = 0x1F & (old_rgb_565 >> 11);
    uint32_t dr = (0x8421 * old_r5) >> 4;
    uint32_t old_g6 = 0x3F & (old_rgb_565 >> 5);
    uint32_t dg = (0x1041 * old_g6) >> 2;
    uint32_t old_b5 = 0x1F & (old_rgb_565 >> 0);
    uint32_t db = (0x8421 * old_b5) >> 4;

    // Calculate the inverse of the src-alpha: how much of the dst to keep.
    uint32_t ia = 0xFFFF - sa;

    // Composite src (nonpremul) over dst (premul).
    dr = ((sr * sa) + (dr * ia)) / 0xFFFF;
    dg = ((sg * sa) + (dg * ia)) / 0xFFFF;
    db = ((sb * sa) + (db * ia)) / 0xFFFF;

    // Convert from 16-bit color to 565 color and combine the components.
    uint32_t new_r5 = 0x1F & (dr >> 11);
    uint32_t new_g6 = 0x3F & (dg >> 10);
    uint32_t new_b5 = 0x1F & (db >> 11);
    uint32_t new_rgb_565 = (new_r5 << 11) | (new_g6 << 5) | (new_b5 << 0);
    wuffs_base__poke_u16le__no_bounds_check(d + (0 * 2), (uint16_t)new_rgb_565);

    s += 1 * 4;
    d += 1 * 2;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr_565__bgra_nonpremul_4x16le__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len2 = dst_len / 2;
  size_t src_len8 = src_len / 8;
  size_t len = (dst_len2 < src_len8) ? dst_len2 : src_len8;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    // Extract 16-bit color components.
    uint32_t sa = ((uint32_t)wuffs_base__peek_u16le__no_bounds_check(s + 6));
    uint32_t sr = ((uint32_t)wuffs_base__peek_u16le__no_bounds_check(s + 4));
    uint32_t sg = ((uint32_t)wuffs_base__peek_u16le__no_bounds_check(s + 2));
    uint32_t sb = ((uint32_t)wuffs_base__peek_u16le__no_bounds_check(s + 0));

    // Convert from 565 color to 16-bit color.
    uint32_t old_rgb_565 = wuffs_base__peek_u16le__no_bounds_check(d + (0 * 2));
    uint32_t old_r5 = 0x1F & (old_rgb_565 >> 11);
    uint32_t dr = (0x8421 * old_r5) >> 4;
    uint32_t old_g6 = 0x3F & (old_rgb_565 >> 5);
    uint32_t dg = (0x1041 * old_g6) >> 2;
    uint32_t old_b5 = 0x1F & (old_rgb_565 >> 0);
    uint32_t db = (0x8421 * old_b5) >> 4;

    // Calculate the inverse of the src-alpha: how much of the dst to keep.
    uint32_t ia = 0xFFFF - sa;

    // Composite src (nonpremul) over dst (premul).
    dr = ((sr * sa) + (dr * ia)) / 0xFFFF;
    dg = ((sg * sa) + (dg * ia)) / 0xFFFF;
    db = ((sb * sa) + (db * ia)) / 0xFFFF;

    // Convert from 16-bit color to 565 color and combine the components.
    uint32_t new_r5 = 0x1F & (dr >> 11);
    uint32_t new_g6 = 0x3F & (dg >> 10);
    uint32_t new_b5 = 0x1F & (db >> 11);
    uint32_t new_rgb_565 = (new_r5 << 11) | (new_g6 << 5) | (new_b5 << 0);
    wuffs_base__poke_u16le__no_bounds_check(d + (0 * 2), (uint16_t)new_rgb_565);

    s += 1 * 8;
    d += 1 * 2;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr_565__bgra_premul__src(uint8_t* dst_ptr,
                                                      size_t dst_len,
                                                      uint8_t* dst_palette_ptr,
                                                      size_t dst_palette_len,
                                                      const uint8_t* src_ptr,
                                                      size_t src_len) {
  size_t dst_len2 = dst_len / 2;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len2 < src_len4) ? dst_len2 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    wuffs_base__poke_u16le__no_bounds_check(
        d + (0 * 2), wuffs_base__color_u32_argb_premul__as__color_u16_rgb_565(
                         wuffs_base__peek_u32le__no_bounds_check(s + (0 * 4))));

    s += 1 * 4;
    d += 1 * 2;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr_565__bgra_premul__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len2 = dst_len / 2;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len2 < src_len4) ? dst_len2 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    // Extract 16-bit color components.
    uint32_t sa = 0x101 * ((uint32_t)s[3]);
    uint32_t sr = 0x101 * ((uint32_t)s[2]);
    uint32_t sg = 0x101 * ((uint32_t)s[1]);
    uint32_t sb = 0x101 * ((uint32_t)s[0]);

    // Convert from 565 color to 16-bit color.
    uint32_t old_rgb_565 = wuffs_base__peek_u16le__no_bounds_check(d + (0 * 2));
    uint32_t old_r5 = 0x1F & (old_rgb_565 >> 11);
    uint32_t dr = (0x8421 * old_r5) >> 4;
    uint32_t old_g6 = 0x3F & (old_rgb_565 >> 5);
    uint32_t dg = (0x1041 * old_g6) >> 2;
    uint32_t old_b5 = 0x1F & (old_rgb_565 >> 0);
    uint32_t db = (0x8421 * old_b5) >> 4;

    // Calculate the inverse of the src-alpha: how much of the dst to keep.
    uint32_t ia = 0xFFFF - sa;

    // Composite src (premul) over dst (premul).
    dr = sr + ((dr * ia) / 0xFFFF);
    dg = sg + ((dg * ia) / 0xFFFF);
    db = sb + ((db * ia) / 0xFFFF);

    // Convert from 16-bit color to 565 color and combine the components.
    uint32_t new_r5 = 0x1F & (dr >> 11);
    uint32_t new_g6 = 0x3F & (dg >> 10);
    uint32_t new_b5 = 0x1F & (db >> 11);
    uint32_t new_rgb_565 = (new_r5 << 11) | (new_g6 << 5) | (new_b5 << 0);
    wuffs_base__poke_u16le__no_bounds_check(d + (0 * 2), (uint16_t)new_rgb_565);

    s += 1 * 4;
    d += 1 * 2;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr_565__rgb(uint8_t* dst_ptr,
                                         size_t dst_len,
                                         uint8_t* dst_palette_ptr,
                                         size_t dst_palette_len,
                                         const uint8_t* src_ptr,
                                         size_t src_len) {
  size_t dst_len2 = dst_len / 2;
  size_t src_len3 = src_len / 3;
  size_t len = (dst_len2 < src_len3) ? dst_len2 : src_len3;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    uint32_t r5 = s[0] >> 3;
    uint32_t g6 = s[1] >> 2;
    uint32_t b5 = s[2] >> 3;
    uint32_t rgb_565 = (r5 << 11) | (g6 << 5) | (b5 << 0);
    wuffs_base__poke_u16le__no_bounds_check(d + (0 * 2), (uint16_t)rgb_565);

    s += 1 * 3;
    d += 1 * 2;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr_565__rgba_nonpremul__src(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len2 = dst_len / 2;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len2 < src_len4) ? dst_len2 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    wuffs_base__poke_u16le__no_bounds_check(
        d + (0 * 2),
        wuffs_base__color_u32_argb_premul__as__color_u16_rgb_565(
            wuffs_base__swap_u32_argb_abgr(
                wuffs_base__color_u32_argb_nonpremul__as__color_u32_argb_premul(
                    wuffs_base__peek_u32le__no_bounds_check(s + (0 * 4))))));

    s += 1 * 4;
    d += 1 * 2;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr_565__rgba_nonpremul__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len2 = dst_len / 2;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len2 < src_len4) ? dst_len2 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    // Extract 16-bit color components.
    uint32_t sa = 0x101 * ((uint32_t)s[3]);
    uint32_t sb = 0x101 * ((uint32_t)s[2]);
    uint32_t sg = 0x101 * ((uint32_t)s[1]);
    uint32_t sr = 0x101 * ((uint32_t)s[0]);

    // Convert from 565 color to 16-bit color.
    uint32_t old_rgb_565 = wuffs_base__peek_u16le__no_bounds_check(d + (0 * 2));
    uint32_t old_r5 = 0x1F & (old_rgb_565 >> 11);
    uint32_t dr = (0x8421 * old_r5) >> 4;
    uint32_t old_g6 = 0x3F & (old_rgb_565 >> 5);
    uint32_t dg = (0x1041 * old_g6) >> 2;
    uint32_t old_b5 = 0x1F & (old_rgb_565 >> 0);
    uint32_t db = (0x8421 * old_b5) >> 4;

    // Calculate the inverse of the src-alpha: how much of the dst to keep.
    uint32_t ia = 0xFFFF - sa;

    // Composite src (nonpremul) over dst (premul).
    dr = ((sr * sa) + (dr * ia)) / 0xFFFF;
    dg = ((sg * sa) + (dg * ia)) / 0xFFFF;
    db = ((sb * sa) + (db * ia)) / 0xFFFF;

    // Convert from 16-bit color to 565 color and combine the components.
    uint32_t new_r5 = 0x1F & (dr >> 11);
    uint32_t new_g6 = 0x3F & (dg >> 10);
    uint32_t new_b5 = 0x1F & (db >> 11);
    uint32_t new_rgb_565 = (new_r5 << 11) | (new_g6 << 5) | (new_b5 << 0);
    wuffs_base__poke_u16le__no_bounds_check(d + (0 * 2), (uint16_t)new_rgb_565);

    s += 1 * 4;
    d += 1 * 2;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr_565__rgba_premul__src(uint8_t* dst_ptr,
                                                      size_t dst_len,
                                                      uint8_t* dst_palette_ptr,
                                                      size_t dst_palette_len,
                                                      const uint8_t* src_ptr,
                                                      size_t src_len) {
  size_t dst_len2 = dst_len / 2;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len2 < src_len4) ? dst_len2 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    wuffs_base__poke_u16le__no_bounds_check(
        d + (0 * 2),
        wuffs_base__color_u32_argb_premul__as__color_u16_rgb_565(
            wuffs_base__swap_u32_argb_abgr(
                wuffs_base__peek_u32le__no_bounds_check(s + (0 * 4)))));

    s += 1 * 4;
    d += 1 * 2;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr_565__rgba_premul__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len2 = dst_len / 2;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len2 < src_len4) ? dst_len2 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    // Extract 16-bit color components.
    uint32_t sa = 0x101 * ((uint32_t)s[3]);
    uint32_t sb = 0x101 * ((uint32_t)s[2]);
    uint32_t sg = 0x101 * ((uint32_t)s[1]);
    uint32_t sr = 0x101 * ((uint32_t)s[0]);

    // Convert from 565 color to 16-bit color.
    uint32_t old_rgb_565 = wuffs_base__peek_u16le__no_bounds_check(d + (0 * 2));
    uint32_t old_r5 = 0x1F & (old_rgb_565 >> 11);
    uint32_t dr = (0x8421 * old_r5) >> 4;
    uint32_t old_g6 = 0x3F & (old_rgb_565 >> 5);
    uint32_t dg = (0x1041 * old_g6) >> 2;
    uint32_t old_b5 = 0x1F & (old_rgb_565 >> 0);
    uint32_t db = (0x8421 * old_b5) >> 4;

    // Calculate the inverse of the src-alpha: how much of the dst to keep.
    uint32_t ia = 0xFFFF - sa;

    // Composite src (premul) over dst (premul).
    dr = sr + ((dr * ia) / 0xFFFF);
    dg = sg + ((dg * ia) / 0xFFFF);
    db = sb + ((db * ia) / 0xFFFF);

    // Convert from 16-bit color to 565 color and combine the components.
    uint32_t new_r5 = 0x1F & (dr >> 11);
    uint32_t new_g6 = 0x3F & (dg >> 10);
    uint32_t new_b5 = 0x1F & (db >> 11);
    uint32_t new_rgb_565 = (new_r5 << 11) | (new_g6 << 5) | (new_b5 << 0);
    wuffs_base__poke_u16le__no_bounds_check(d + (0 * 2), (uint16_t)new_rgb_565);

    s += 1 * 4;
    d += 1 * 2;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr_565__y(uint8_t* dst_ptr,
                                       size_t dst_len,
                                       uint8_t* dst_palette_ptr,
                                       size_t dst_palette_len,
                                       const uint8_t* src_ptr,
                                       size_t src_len) {
  size_t dst_len2 = dst_len / 2;
  size_t len = (dst_len2 < src_len) ? dst_len2 : src_len;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    uint32_t y5 = s[0] >> 3;
    uint32_t y6 = s[0] >> 2;
    uint32_t rgb_565 = (y5 << 11) | (y6 << 5) | (y5 << 0);
    wuffs_base__poke_u16le__no_bounds_check(d + (0 * 2), (uint16_t)rgb_565);

    s += 1 * 1;
    d += 1 * 2;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr_565__y_16be(uint8_t* dst_ptr,
                                            size_t dst_len,
                                            uint8_t* dst_palette_ptr,
                                            size_t dst_palette_len,
                                            const uint8_t* src_ptr,
                                            size_t src_len) {
  size_t dst_len2 = dst_len / 2;
  size_t src_len2 = src_len / 2;
  size_t len = (dst_len2 < src_len2) ? dst_len2 : src_len2;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    uint32_t y5 = s[0] >> 3;
    uint32_t y6 = s[0] >> 2;
    uint32_t rgb_565 = (y5 << 11) | (y6 << 5) | (y5 << 0);
    wuffs_base__poke_u16le__no_bounds_check(d + (0 * 2), (uint16_t)rgb_565);

    s += 1 * 2;
    d += 1 * 2;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr_565__index__src(uint8_t* dst_ptr,
                                                size_t dst_len,
                                                uint8_t* dst_palette_ptr,
                                                size_t dst_palette_len,
                                                const uint8_t* src_ptr,
                                                size_t src_len) {
  if (dst_palette_len !=
      WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH) {
    return 0;
  }
  size_t dst_len2 = dst_len / 2;
  size_t len = (dst_len2 < src_len) ? dst_len2 : src_len;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  const size_t loop_unroll_count = 4;

  while (n >= loop_unroll_count) {
    wuffs_base__poke_u16le__no_bounds_check(
        d + (0 * 2), wuffs_base__peek_u16le__no_bounds_check(
                         dst_palette_ptr + ((size_t)s[0] * 4)));
    wuffs_base__poke_u16le__no_bounds_check(
        d + (1 * 2), wuffs_base__peek_u16le__no_bounds_check(
                         dst_palette_ptr + ((size_t)s[1] * 4)));
    wuffs_base__poke_u16le__no_bounds_check(
        d + (2 * 2), wuffs_base__peek_u16le__no_bounds_check(
                         dst_palette_ptr + ((size_t)s[2] * 4)));
    wuffs_base__poke_u16le__no_bounds_check(
        d + (3 * 2), wuffs_base__peek_u16le__no_bounds_check(
                         dst_palette_ptr + ((size_t)s[3] * 4)));

    s += loop_unroll_count * 1;
    d += loop_unroll_count * 2;
    n -= loop_unroll_count;
  }

  while (n >= 1) {
    wuffs_base__poke_u16le__no_bounds_check(
        d + (0 * 2), wuffs_base__peek_u16le__no_bounds_check(
                         dst_palette_ptr + ((size_t)s[0] * 4)));

    s += 1 * 1;
    d += 1 * 2;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr_565__index_bgra_nonpremul__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  if (dst_palette_len !=
      WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH) {
    return 0;
  }
  size_t dst_len2 = dst_len / 2;
  size_t len = (dst_len2 < src_len) ? dst_len2 : src_len;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    uint32_t d0 = wuffs_base__color_u16_rgb_565__as__color_u32_argb_premul(
        wuffs_base__peek_u16le__no_bounds_check(d + (0 * 2)));
    uint32_t s0 = wuffs_base__peek_u32le__no_bounds_check(dst_palette_ptr +
                                                          ((size_t)s[0] * 4));
    wuffs_base__poke_u16le__no_bounds_check(
        d + (0 * 2),
        wuffs_base__color_u32_argb_premul__as__color_u16_rgb_565(
            wuffs_base__composite_premul_nonpremul_u32_axxx(d0, s0)));

    s += 1 * 1;
    d += 1 * 2;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr_565__index_binary_alpha__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  if (dst_palette_len !=
      WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH) {
    return 0;
  }
  size_t dst_len2 = dst_len / 2;
  size_t len = (dst_len2 < src_len) ? dst_len2 : src_len;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    uint32_t s0 = wuffs_base__peek_u32le__no_bounds_check(dst_palette_ptr +
                                                          ((size_t)s[0] * 4));
    if (s0) {
      wuffs_base__poke_u16le__no_bounds_check(d + (0 * 2), (uint16_t)s0);
    }

    s += 1 * 1;
    d += 1 * 2;
    n -= 1;
  }

  return len;
}

// --------

static uint64_t  //
wuffs_base__pixel_swizzler__bgr__bgr_565(uint8_t* dst_ptr,
                                         size_t dst_len,
                                         uint8_t* dst_palette_ptr,
                                         size_t dst_palette_len,
                                         const uint8_t* src_ptr,
                                         size_t src_len) {
  size_t dst_len3 = dst_len / 3;
  size_t src_len2 = src_len / 2;
  size_t len = (dst_len3 < src_len2) ? dst_len3 : src_len2;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    uint32_t s0 = wuffs_base__color_u16_rgb_565__as__color_u32_argb_premul(
        wuffs_base__peek_u16le__no_bounds_check(s + (0 * 2)));
    wuffs_base__poke_u24le__no_bounds_check(d + (0 * 3), s0);

    s += 1 * 2;
    d += 1 * 3;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr__bgra_nonpremul__src(uint8_t* dst_ptr,
                                                     size_t dst_len,
                                                     uint8_t* dst_palette_ptr,
                                                     size_t dst_palette_len,
                                                     const uint8_t* src_ptr,
                                                     size_t src_len) {
  size_t dst_len3 = dst_len / 3;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len3 < src_len4) ? dst_len3 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    uint32_t s0 =
        wuffs_base__color_u32_argb_nonpremul__as__color_u32_argb_premul(
            wuffs_base__peek_u32le__no_bounds_check(s + (0 * 4)));
    wuffs_base__poke_u24le__no_bounds_check(d + (0 * 3), s0);

    s += 1 * 4;
    d += 1 * 3;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr__bgra_nonpremul_4x16le__src(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len3 = dst_len / 3;
  size_t src_len8 = src_len / 8;
  size_t len = (dst_len3 < src_len8) ? dst_len3 : src_len8;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    uint32_t s0 =
        wuffs_base__color_u64_argb_nonpremul__as__color_u32_argb_premul(
            wuffs_base__peek_u64le__no_bounds_check(s + (0 * 8)));
    wuffs_base__poke_u24le__no_bounds_check(d + (0 * 3), s0);

    s += 1 * 8;
    d += 1 * 3;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr__bgra_nonpremul__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len3 = dst_len / 3;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len3 < src_len4) ? dst_len3 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    // Extract 16-bit color components.
    uint32_t dr = 0x101 * ((uint32_t)d[2]);
    uint32_t dg = 0x101 * ((uint32_t)d[1]);
    uint32_t db = 0x101 * ((uint32_t)d[0]);
    uint32_t sa = 0x101 * ((uint32_t)s[3]);
    uint32_t sr = 0x101 * ((uint32_t)s[2]);
    uint32_t sg = 0x101 * ((uint32_t)s[1]);
    uint32_t sb = 0x101 * ((uint32_t)s[0]);

    // Calculate the inverse of the src-alpha: how much of the dst to keep.
    uint32_t ia = 0xFFFF - sa;

    // Composite src (nonpremul) over dst (premul).
    dr = ((sr * sa) + (dr * ia)) / 0xFFFF;
    dg = ((sg * sa) + (dg * ia)) / 0xFFFF;
    db = ((sb * sa) + (db * ia)) / 0xFFFF;

    // Convert from 16-bit color to 8-bit color.
    d[0] = (uint8_t)(db >> 8);
    d[1] = (uint8_t)(dg >> 8);
    d[2] = (uint8_t)(dr >> 8);

    s += 1 * 4;
    d += 1 * 3;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr__bgra_nonpremul_4x16le__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len3 = dst_len / 3;
  size_t src_len8 = src_len / 8;
  size_t len = (dst_len3 < src_len8) ? dst_len3 : src_len8;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    // Extract 16-bit color components.
    uint32_t dr = 0x101 * ((uint32_t)d[2]);
    uint32_t dg = 0x101 * ((uint32_t)d[1]);
    uint32_t db = 0x101 * ((uint32_t)d[0]);
    uint32_t sa = ((uint32_t)wuffs_base__peek_u16le__no_bounds_check(s + 6));
    uint32_t sr = ((uint32_t)wuffs_base__peek_u16le__no_bounds_check(s + 4));
    uint32_t sg = ((uint32_t)wuffs_base__peek_u16le__no_bounds_check(s + 2));
    uint32_t sb = ((uint32_t)wuffs_base__peek_u16le__no_bounds_check(s + 0));

    // Calculate the inverse of the src-alpha: how much of the dst to keep.
    uint32_t ia = 0xFFFF - sa;

    // Composite src (nonpremul) over dst (premul).
    dr = ((sr * sa) + (dr * ia)) / 0xFFFF;
    dg = ((sg * sa) + (dg * ia)) / 0xFFFF;
    db = ((sb * sa) + (db * ia)) / 0xFFFF;

    // Convert from 16-bit color to 8-bit color.
    d[0] = (uint8_t)(db >> 8);
    d[1] = (uint8_t)(dg >> 8);
    d[2] = (uint8_t)(dr >> 8);

    s += 1 * 8;
    d += 1 * 3;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr__bgra_premul__src(uint8_t* dst_ptr,
                                                  size_t dst_len,
                                                  uint8_t* dst_palette_ptr,
                                                  size_t dst_palette_len,
                                                  const uint8_t* src_ptr,
                                                  size_t src_len) {
  size_t dst_len3 = dst_len / 3;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len3 < src_len4) ? dst_len3 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    uint8_t s0 = s[0];
    uint8_t s1 = s[1];
    uint8_t s2 = s[2];
    d[0] = s0;
    d[1] = s1;
    d[2] = s2;

    s += 1 * 4;
    d += 1 * 3;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr__bgra_premul__src_over(uint8_t* dst_ptr,
                                                       size_t dst_len,
                                                       uint8_t* dst_palette_ptr,
                                                       size_t dst_palette_len,
                                                       const uint8_t* src_ptr,
                                                       size_t src_len) {
  size_t dst_len3 = dst_len / 3;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len3 < src_len4) ? dst_len3 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    // Extract 16-bit color components.
    uint32_t dr = 0x101 * ((uint32_t)d[2]);
    uint32_t dg = 0x101 * ((uint32_t)d[1]);
    uint32_t db = 0x101 * ((uint32_t)d[0]);
    uint32_t sa = 0x101 * ((uint32_t)s[3]);
    uint32_t sr = 0x101 * ((uint32_t)s[2]);
    uint32_t sg = 0x101 * ((uint32_t)s[1]);
    uint32_t sb = 0x101 * ((uint32_t)s[0]);

    // Calculate the inverse of the src-alpha: how much of the dst to keep.
    uint32_t ia = 0xFFFF - sa;

    // Composite src (premul) over dst (premul).
    dr = sr + ((dr * ia) / 0xFFFF);
    dg = sg + ((dg * ia) / 0xFFFF);
    db = sb + ((db * ia) / 0xFFFF);

    // Convert from 16-bit color to 8-bit color.
    d[0] = (uint8_t)(db >> 8);
    d[1] = (uint8_t)(dg >> 8);
    d[2] = (uint8_t)(dr >> 8);

    s += 1 * 4;
    d += 1 * 3;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr__rgba_nonpremul__src(uint8_t* dst_ptr,
                                                     size_t dst_len,
                                                     uint8_t* dst_palette_ptr,
                                                     size_t dst_palette_len,
                                                     const uint8_t* src_ptr,
                                                     size_t src_len) {
  size_t dst_len3 = dst_len / 3;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len3 < src_len4) ? dst_len3 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    uint32_t s0 = wuffs_base__swap_u32_argb_abgr(
        wuffs_base__color_u32_argb_nonpremul__as__color_u32_argb_premul(
            wuffs_base__peek_u32le__no_bounds_check(s + (0 * 4))));
    wuffs_base__poke_u24le__no_bounds_check(d + (0 * 3), s0);

    s += 1 * 4;
    d += 1 * 3;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr__rgba_nonpremul__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len3 = dst_len / 3;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len3 < src_len4) ? dst_len3 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    // Extract 16-bit color components.
    uint32_t dr = 0x101 * ((uint32_t)d[2]);
    uint32_t dg = 0x101 * ((uint32_t)d[1]);
    uint32_t db = 0x101 * ((uint32_t)d[0]);
    uint32_t sa = 0x101 * ((uint32_t)s[3]);
    uint32_t sb = 0x101 * ((uint32_t)s[2]);
    uint32_t sg = 0x101 * ((uint32_t)s[1]);
    uint32_t sr = 0x101 * ((uint32_t)s[0]);

    // Calculate the inverse of the src-alpha: how much of the dst to keep.
    uint32_t ia = 0xFFFF - sa;

    // Composite src (nonpremul) over dst (premul).
    dr = ((sr * sa) + (dr * ia)) / 0xFFFF;
    dg = ((sg * sa) + (dg * ia)) / 0xFFFF;
    db = ((sb * sa) + (db * ia)) / 0xFFFF;

    // Convert from 16-bit color to 8-bit color.
    d[0] = (uint8_t)(db >> 8);
    d[1] = (uint8_t)(dg >> 8);
    d[2] = (uint8_t)(dr >> 8);

    s += 1 * 4;
    d += 1 * 3;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr__rgba_premul__src(uint8_t* dst_ptr,
                                                  size_t dst_len,
                                                  uint8_t* dst_palette_ptr,
                                                  size_t dst_palette_len,
                                                  const uint8_t* src_ptr,
                                                  size_t src_len) {
  size_t dst_len3 = dst_len / 3;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len3 < src_len4) ? dst_len3 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    uint8_t s0 = s[0];
    uint8_t s1 = s[1];
    uint8_t s2 = s[2];
    d[0] = s2;
    d[1] = s1;
    d[2] = s0;

    s += 1 * 4;
    d += 1 * 3;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgr__rgba_premul__src_over(uint8_t* dst_ptr,
                                                       size_t dst_len,
                                                       uint8_t* dst_palette_ptr,
                                                       size_t dst_palette_len,
                                                       const uint8_t* src_ptr,
                                                       size_t src_len) {
  size_t dst_len3 = dst_len / 3;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len3 < src_len4) ? dst_len3 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    // Extract 16-bit color components.
    uint32_t dr = 0x101 * ((uint32_t)d[2]);
    uint32_t dg = 0x101 * ((uint32_t)d[1]);
    uint32_t db = 0x101 * ((uint32_t)d[0]);
    uint32_t sa = 0x101 * ((uint32_t)s[3]);
    uint32_t sb = 0x101 * ((uint32_t)s[2]);
    uint32_t sg = 0x101 * ((uint32_t)s[1]);
    uint32_t sr = 0x101 * ((uint32_t)s[0]);

    // Calculate the inverse of the src-alpha: how much of the dst to keep.
    uint32_t ia = 0xFFFF - sa;

    // Composite src (premul) over dst (premul).
    dr = sr + ((dr * ia) / 0xFFFF);
    dg = sg + ((dg * ia) / 0xFFFF);
    db = sb + ((db * ia) / 0xFFFF);

    // Convert from 16-bit color to 8-bit color.
    d[0] = (uint8_t)(db >> 8);
    d[1] = (uint8_t)(dg >> 8);
    d[2] = (uint8_t)(dr >> 8);

    s += 1 * 4;
    d += 1 * 3;
    n -= 1;
  }

  return len;
}

// --------

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_nonpremul__bgra_nonpremul__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len4 < src_len4) ? dst_len4 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    uint32_t d0 = wuffs_base__peek_u32le__no_bounds_check(d + (0 * 4));
    uint32_t s0 = wuffs_base__peek_u32le__no_bounds_check(s + (0 * 4));
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4),
        wuffs_base__composite_nonpremul_nonpremul_u32_axxx(d0, s0));

    s += 1 * 4;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_nonpremul__bgra_nonpremul_4x16le__src(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len8 = src_len / 8;
  size_t len = (dst_len4 < src_len8) ? dst_len4 : src_len8;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;

  size_t n = len;
  while (n >= 1) {
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4), wuffs_base__color_u64__as__color_u32(
                         wuffs_base__peek_u64le__no_bounds_check(s + (0 * 8))));

    s += 1 * 8;
    d += 1 * 4;
    n -= 1;
  }
  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_nonpremul__bgra_nonpremul_4x16le__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len8 = src_len / 8;
  size_t len = (dst_len4 < src_len8) ? dst_len4 : src_len8;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    uint64_t d0 = wuffs_base__color_u32__as__color_u64(
        wuffs_base__peek_u32le__no_bounds_check(d + (0 * 4)));
    uint64_t s0 = wuffs_base__peek_u64le__no_bounds_check(s + (0 * 8));
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4),
        wuffs_base__color_u64__as__color_u32(
            wuffs_base__composite_nonpremul_nonpremul_u64_axxx(d0, s0)));

    s += 1 * 8;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_nonpremul__bgra_premul__src(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len4 < src_len4) ? dst_len4 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    uint32_t s0 = wuffs_base__peek_u32le__no_bounds_check(s + (0 * 4));
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4),
        wuffs_base__color_u32_argb_premul__as__color_u32_argb_nonpremul(s0));

    s += 1 * 4;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_nonpremul__bgra_premul__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len4 < src_len4) ? dst_len4 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    uint32_t d0 = wuffs_base__peek_u32le__no_bounds_check(d + (0 * 4));
    uint32_t s0 = wuffs_base__peek_u32le__no_bounds_check(s + (0 * 4));
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4), wuffs_base__composite_nonpremul_premul_u32_axxx(d0, s0));

    s += 1 * 4;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_nonpremul__index_bgra_nonpremul__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  if (dst_palette_len !=
      WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH) {
    return 0;
  }
  size_t dst_len4 = dst_len / 4;
  size_t len = (dst_len4 < src_len) ? dst_len4 : src_len;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    uint32_t d0 = wuffs_base__peek_u32le__no_bounds_check(d + (0 * 4));
    uint32_t s0 = wuffs_base__peek_u32le__no_bounds_check(dst_palette_ptr +
                                                          ((size_t)s[0] * 4));
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4),
        wuffs_base__composite_nonpremul_nonpremul_u32_axxx(d0, s0));

    s += 1 * 1;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_nonpremul__rgba_nonpremul__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len4 < src_len4) ? dst_len4 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    uint32_t d0 = wuffs_base__peek_u32le__no_bounds_check(d + (0 * 4));
    uint32_t s0 = wuffs_base__swap_u32_argb_abgr(
        wuffs_base__peek_u32le__no_bounds_check(s + (0 * 4)));
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4),
        wuffs_base__composite_nonpremul_nonpremul_u32_axxx(d0, s0));

    s += 1 * 4;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_nonpremul__rgba_premul__src(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len4 < src_len4) ? dst_len4 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    uint32_t s0 = wuffs_base__swap_u32_argb_abgr(
        wuffs_base__peek_u32le__no_bounds_check(s + (0 * 4)));
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4),
        wuffs_base__color_u32_argb_premul__as__color_u32_argb_nonpremul(s0));

    s += 1 * 4;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_nonpremul__rgba_premul__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len4 < src_len4) ? dst_len4 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    uint32_t d0 = wuffs_base__peek_u32le__no_bounds_check(d + (0 * 4));
    uint32_t s0 = wuffs_base__swap_u32_argb_abgr(
        wuffs_base__peek_u32le__no_bounds_check(s + (0 * 4)));
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4), wuffs_base__composite_nonpremul_premul_u32_axxx(d0, s0));

    s += 1 * 4;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

// --------

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_nonpremul_4x16le__bgra_nonpremul__src(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len8 = dst_len / 8;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len8 < src_len4) ? dst_len8 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;

  size_t n = len;
  while (n >= 1) {
    uint8_t s0 = s[0];
    uint8_t s1 = s[1];
    uint8_t s2 = s[2];
    uint8_t s3 = s[3];
    d[0] = s0;
    d[1] = s0;
    d[2] = s1;
    d[3] = s1;
    d[4] = s2;
    d[5] = s2;
    d[6] = s3;
    d[7] = s3;

    s += 1 * 4;
    d += 1 * 8;
    n -= 1;
  }
  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_nonpremul_4x16le__bgra_nonpremul__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len8 = dst_len / 8;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len8 < src_len4) ? dst_len8 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;

  size_t n = len;
  while (n >= 1) {
    uint64_t d0 = wuffs_base__peek_u64le__no_bounds_check(d + (0 * 8));
    uint64_t s0 = wuffs_base__color_u32__as__color_u64(
        wuffs_base__peek_u32le__no_bounds_check(s + (0 * 4)));
    wuffs_base__poke_u64le__no_bounds_check(
        d + (0 * 8),
        wuffs_base__composite_nonpremul_nonpremul_u64_axxx(d0, s0));

    s += 1 * 4;
    d += 1 * 8;
    n -= 1;
  }
  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_nonpremul_4x16le__bgra_nonpremul_4x16le__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len8 = dst_len / 8;
  size_t src_len8 = src_len / 8;
  size_t len = (dst_len8 < src_len8) ? dst_len8 : src_len8;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;

  size_t n = len;
  while (n >= 1) {
    uint64_t d0 = wuffs_base__peek_u64le__no_bounds_check(d + (0 * 8));
    uint64_t s0 = wuffs_base__peek_u64le__no_bounds_check(s + (0 * 8));
    wuffs_base__poke_u64le__no_bounds_check(
        d + (0 * 8),
        wuffs_base__composite_nonpremul_nonpremul_u64_axxx(d0, s0));

    s += 1 * 8;
    d += 1 * 8;
    n -= 1;
  }
  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_nonpremul_4x16le__bgra_premul__src(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len8 = dst_len / 8;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len8 < src_len4) ? dst_len8 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;

  size_t n = len;
  while (n >= 1) {
    uint64_t s0 = wuffs_base__color_u32__as__color_u64(
        wuffs_base__color_u32_argb_premul__as__color_u32_argb_nonpremul(
            wuffs_base__peek_u32le__no_bounds_check(s + (0 * 4))));
    wuffs_base__poke_u64le__no_bounds_check(d + (0 * 8), s0);

    s += 1 * 4;
    d += 1 * 8;
    n -= 1;
  }
  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_nonpremul_4x16le__bgra_premul__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len8 = dst_len / 8;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len8 < src_len4) ? dst_len8 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;

  size_t n = len;
  while (n >= 1) {
    uint64_t d0 = wuffs_base__peek_u64le__no_bounds_check(d + (0 * 8));
    uint64_t s0 = wuffs_base__color_u32__as__color_u64(
        wuffs_base__peek_u32le__no_bounds_check(s + (0 * 4)));
    wuffs_base__poke_u64le__no_bounds_check(
        d + (0 * 8), wuffs_base__composite_nonpremul_premul_u64_axxx(d0, s0));

    s += 1 * 4;
    d += 1 * 8;
    n -= 1;
  }
  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_nonpremul_4x16le__index_bgra_nonpremul__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  if (dst_palette_len !=
      WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH) {
    return 0;
  }
  size_t dst_len8 = dst_len / 8;
  size_t len = (dst_len8 < src_len) ? dst_len8 : src_len;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    uint64_t d0 = wuffs_base__peek_u64le__no_bounds_check(d + (0 * 8));
    uint64_t s0 = wuffs_base__color_u32__as__color_u64(
        wuffs_base__peek_u32le__no_bounds_check(dst_palette_ptr +
                                                ((size_t)s[0] * 4)));
    wuffs_base__poke_u64le__no_bounds_check(
        d + (0 * 8),
        wuffs_base__composite_nonpremul_nonpremul_u64_axxx(d0, s0));

    s += 1 * 1;
    d += 1 * 8;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_nonpremul_4x16le__rgba_nonpremul__src(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len8 = dst_len / 8;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len8 < src_len4) ? dst_len8 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;

  size_t n = len;
  while (n >= 1) {
    uint8_t s0 = s[0];
    uint8_t s1 = s[1];
    uint8_t s2 = s[2];
    uint8_t s3 = s[3];
    d[0] = s2;
    d[1] = s2;
    d[2] = s1;
    d[3] = s1;
    d[4] = s0;
    d[5] = s0;
    d[6] = s3;
    d[7] = s3;

    s += 1 * 4;
    d += 1 * 8;
    n -= 1;
  }
  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_nonpremul_4x16le__rgba_nonpremul__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len8 = dst_len / 8;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len8 < src_len4) ? dst_len8 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;

  size_t n = len;
  while (n >= 1) {
    uint64_t d0 = wuffs_base__peek_u64le__no_bounds_check(d + (0 * 8));
    uint64_t s0 =
        wuffs_base__color_u32__as__color_u64(wuffs_base__swap_u32_argb_abgr(
            wuffs_base__peek_u32le__no_bounds_check(s + (0 * 4))));
    wuffs_base__poke_u64le__no_bounds_check(
        d + (0 * 8),
        wuffs_base__composite_nonpremul_nonpremul_u64_axxx(d0, s0));

    s += 1 * 4;
    d += 1 * 8;
    n -= 1;
  }
  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_nonpremul_4x16le__rgba_premul__src(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len8 = dst_len / 8;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len8 < src_len4) ? dst_len8 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;

  size_t n = len;
  while (n >= 1) {
    uint64_t s0 = wuffs_base__color_u32__as__color_u64(
        wuffs_base__color_u32_argb_premul__as__color_u32_argb_nonpremul(
            wuffs_base__swap_u32_argb_abgr(
                wuffs_base__peek_u32le__no_bounds_check(s + (0 * 4)))));
    wuffs_base__poke_u64le__no_bounds_check(d + (0 * 8), s0);

    s += 1 * 4;
    d += 1 * 8;
    n -= 1;
  }
  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_nonpremul_4x16le__rgba_premul__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len8 = dst_len / 8;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len8 < src_len4) ? dst_len8 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;

  size_t n = len;
  while (n >= 1) {
    uint64_t d0 = wuffs_base__peek_u64le__no_bounds_check(d + (0 * 8));
    uint64_t s0 =
        wuffs_base__color_u32__as__color_u64(wuffs_base__swap_u32_argb_abgr(
            wuffs_base__peek_u32le__no_bounds_check(s + (0 * 4))));
    wuffs_base__poke_u64le__no_bounds_check(
        d + (0 * 8), wuffs_base__composite_nonpremul_premul_u64_axxx(d0, s0));

    s += 1 * 4;
    d += 1 * 8;
    n -= 1;
  }
  return len;
}

// --------

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_premul__bgra_nonpremul__src(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len4 < src_len4) ? dst_len4 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    uint32_t s0 = wuffs_base__peek_u32le__no_bounds_check(s + (0 * 4));
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4),
        wuffs_base__color_u32_argb_nonpremul__as__color_u32_argb_premul(s0));

    s += 1 * 4;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_premul__bgra_nonpremul_4x16le__src(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len8 = src_len / 8;
  size_t len = (dst_len4 < src_len8) ? dst_len4 : src_len8;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    uint64_t s0 = wuffs_base__peek_u64le__no_bounds_check(s + (0 * 8));
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4),
        wuffs_base__color_u64_argb_nonpremul__as__color_u32_argb_premul(s0));

    s += 1 * 8;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_premul__bgra_nonpremul__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len4 < src_len4) ? dst_len4 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    uint32_t d0 = wuffs_base__peek_u32le__no_bounds_check(d + (0 * 4));
    uint32_t s0 = wuffs_base__peek_u32le__no_bounds_check(s + (0 * 4));
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4), wuffs_base__composite_premul_nonpremul_u32_axxx(d0, s0));

    s += 1 * 4;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_premul__bgra_nonpremul_4x16le__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len8 = src_len / 8;
  size_t len = (dst_len4 < src_len8) ? dst_len4 : src_len8;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    uint64_t d0 = wuffs_base__color_u32__as__color_u64(
        wuffs_base__peek_u32le__no_bounds_check(d + (0 * 4)));
    uint64_t s0 = wuffs_base__peek_u64le__no_bounds_check(s + (0 * 8));
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4),
        wuffs_base__color_u64__as__color_u32(
            wuffs_base__composite_premul_nonpremul_u64_axxx(d0, s0)));

    s += 1 * 8;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_premul__bgra_premul__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len4 < src_len4) ? dst_len4 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    uint32_t d0 = wuffs_base__peek_u32le__no_bounds_check(d + (0 * 4));
    uint32_t s0 = wuffs_base__peek_u32le__no_bounds_check(s + (0 * 4));
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4), wuffs_base__composite_premul_premul_u32_axxx(d0, s0));

    s += 1 * 4;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_premul__index_bgra_nonpremul__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  if (dst_palette_len !=
      WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH) {
    return 0;
  }
  size_t dst_len4 = dst_len / 4;
  size_t len = (dst_len4 < src_len) ? dst_len4 : src_len;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    uint32_t d0 = wuffs_base__peek_u32le__no_bounds_check(d + (0 * 4));
    uint32_t s0 = wuffs_base__peek_u32le__no_bounds_check(dst_palette_ptr +
                                                          ((size_t)s[0] * 4));
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4), wuffs_base__composite_premul_nonpremul_u32_axxx(d0, s0));

    s += 1 * 1;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_premul__rgba_nonpremul__src(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len4 < src_len4) ? dst_len4 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    uint32_t s0 = wuffs_base__swap_u32_argb_abgr(
        wuffs_base__peek_u32le__no_bounds_check(s + (0 * 4)));
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4),
        wuffs_base__color_u32_argb_nonpremul__as__color_u32_argb_premul(s0));

    s += 1 * 4;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_premul__rgba_nonpremul__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len4 < src_len4) ? dst_len4 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    uint32_t d0 = wuffs_base__peek_u32le__no_bounds_check(d + (0 * 4));
    uint32_t s0 = wuffs_base__swap_u32_argb_abgr(
        wuffs_base__peek_u32le__no_bounds_check(s + (0 * 4)));
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4), wuffs_base__composite_premul_nonpremul_u32_axxx(d0, s0));

    s += 1 * 4;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_premul__rgba_nonpremul_4x16le__src(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len8 = src_len / 8;
  size_t len = (dst_len4 < src_len8) ? dst_len4 : src_len8;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    uint64_t s0 = wuffs_base__peek_u64le__no_bounds_check(s + (0 * 8));
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4),
        wuffs_base__swap_u32_argb_abgr(
            wuffs_base__color_u64_argb_nonpremul__as__color_u32_argb_premul(
                s0)));

    s += 1 * 8;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_premul__rgba_nonpremul_4x16le__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len8 = src_len / 8;
  size_t len = (dst_len4 < src_len8) ? dst_len4 : src_len8;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    uint64_t d0 = wuffs_base__color_u32__as__color_u64(
        wuffs_base__peek_u32le__no_bounds_check(d + (0 * 4)));
    uint64_t s0 = wuffs_base__swap_u64_argb_abgr(
        wuffs_base__peek_u64le__no_bounds_check(s + (0 * 8)));
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4),
        wuffs_base__color_u64__as__color_u32(
            wuffs_base__composite_premul_nonpremul_u64_axxx(d0, s0)));

    s += 1 * 8;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgra_premul__rgba_premul__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len4 < src_len4) ? dst_len4 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    uint32_t d0 = wuffs_base__peek_u32le__no_bounds_check(d + (0 * 4));
    uint32_t s0 = wuffs_base__swap_u32_argb_abgr(
        wuffs_base__peek_u32le__no_bounds_check(s + (0 * 4)));
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4), wuffs_base__composite_premul_premul_u32_axxx(d0, s0));

    s += 1 * 4;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

// --------

static uint64_t  //
wuffs_base__pixel_swizzler__bgrw__bgr(uint8_t* dst_ptr,
                                      size_t dst_len,
                                      uint8_t* dst_palette_ptr,
                                      size_t dst_palette_len,
                                      const uint8_t* src_ptr,
                                      size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len3 = src_len / 3;
  size_t len = (dst_len4 < src_len3) ? dst_len4 : src_len3;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4),
        0xFF000000 | wuffs_base__peek_u24le__no_bounds_check(s + (0 * 3)));

    s += 1 * 3;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgrw__bgr_565(uint8_t* dst_ptr,
                                          size_t dst_len,
                                          uint8_t* dst_palette_ptr,
                                          size_t dst_palette_len,
                                          const uint8_t* src_ptr,
                                          size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len2 = src_len / 2;
  size_t len = (dst_len4 < src_len2) ? dst_len4 : src_len2;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4), wuffs_base__color_u16_rgb_565__as__color_u32_argb_premul(
                         wuffs_base__peek_u16le__no_bounds_check(s + (0 * 2))));

    s += 1 * 2;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgrw__bgrx(uint8_t* dst_ptr,
                                       size_t dst_len,
                                       uint8_t* dst_palette_ptr,
                                       size_t dst_palette_len,
                                       const uint8_t* src_ptr,
                                       size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len4 < src_len4) ? dst_len4 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4),
        0xFF000000 | wuffs_base__peek_u32le__no_bounds_check(s + (0 * 4)));

    s += 1 * 4;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

// ‼ WUFFS MULTI-FILE SECTION +x86_sse42
#if defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
WUFFS_BASE__MAYBE_ATTRIBUTE_TARGET("pclmul,popcnt,sse4.2")
static uint64_t  //
wuffs_base__pixel_swizzler__bgrw__rgb__sse42(uint8_t* dst_ptr,
                                             size_t dst_len,
                                             uint8_t* dst_palette_ptr,
                                             size_t dst_palette_len,
                                             const uint8_t* src_ptr,
                                             size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len3 = src_len / 3;
  size_t len = (dst_len4 < src_len3) ? dst_len4 : src_len3;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  __m128i shuffle = _mm_set_epi8(+0x00, +0x09, +0x0A, +0x0B,  //
                                 +0x00, +0x06, +0x07, +0x08,  //
                                 +0x00, +0x03, +0x04, +0x05,  //
                                 +0x00, +0x00, +0x01, +0x02);
  __m128i or_ff = _mm_set_epi8(-0x01, +0x00, +0x00, +0x00,  //
                               -0x01, +0x00, +0x00, +0x00,  //
                               -0x01, +0x00, +0x00, +0x00,  //
                               -0x01, +0x00, +0x00, +0x00);

  while (n >= 6) {
    __m128i x;
    x = _mm_lddqu_si128((const __m128i*)(const void*)s);
    x = _mm_shuffle_epi8(x, shuffle);
    x = _mm_or_si128(x, or_ff);
    _mm_storeu_si128((__m128i*)(void*)d, x);

    s += 4 * 3;
    d += 4 * 4;
    n -= 4;
  }

  while (n >= 1) {
    uint8_t b0 = s[0];
    uint8_t b1 = s[1];
    uint8_t b2 = s[2];
    d[0] = b2;
    d[1] = b1;
    d[2] = b0;
    d[3] = 0xFF;

    s += 1 * 3;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}
#endif  // defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
// ‼ WUFFS MULTI-FILE SECTION -x86_sse42

static uint64_t  //
wuffs_base__pixel_swizzler__bgrw__rgb(uint8_t* dst_ptr,
                                      size_t dst_len,
                                      uint8_t* dst_palette_ptr,
                                      size_t dst_palette_len,
                                      const uint8_t* src_ptr,
                                      size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len3 = src_len / 3;
  size_t len = (dst_len4 < src_len3) ? dst_len4 : src_len3;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    uint8_t b0 = s[0];
    uint8_t b1 = s[1];
    uint8_t b2 = s[2];
    d[0] = b2;
    d[1] = b1;
    d[2] = b0;
    d[3] = 0xFF;

    s += 1 * 3;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgrw__rgbx(uint8_t* dst_ptr,
                                       size_t dst_len,
                                       uint8_t* dst_palette_ptr,
                                       size_t dst_palette_len,
                                       const uint8_t* src_ptr,
                                       size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len4 < src_len4) ? dst_len4 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    uint8_t b0 = s[0];
    uint8_t b1 = s[1];
    uint8_t b2 = s[2];
    d[0] = b2;
    d[1] = b1;
    d[2] = b0;
    d[3] = 0xFF;

    s += 1 * 4;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

// --------

static uint64_t  //
wuffs_base__pixel_swizzler__bgrw_4x16le__bgr(uint8_t* dst_ptr,
                                             size_t dst_len,
                                             uint8_t* dst_palette_ptr,
                                             size_t dst_palette_len,
                                             const uint8_t* src_ptr,
                                             size_t src_len) {
  size_t dst_len8 = dst_len / 8;
  size_t src_len3 = src_len / 3;
  size_t len = (dst_len8 < src_len3) ? dst_len8 : src_len3;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    uint8_t s0 = s[0];
    uint8_t s1 = s[1];
    uint8_t s2 = s[2];
    d[0] = s0;
    d[1] = s0;
    d[2] = s1;
    d[3] = s1;
    d[4] = s2;
    d[5] = s2;
    d[6] = 0xFF;
    d[7] = 0xFF;

    s += 1 * 3;
    d += 1 * 8;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgrw_4x16le__bgr_565(uint8_t* dst_ptr,
                                                 size_t dst_len,
                                                 uint8_t* dst_palette_ptr,
                                                 size_t dst_palette_len,
                                                 const uint8_t* src_ptr,
                                                 size_t src_len) {
  size_t dst_len8 = dst_len / 8;
  size_t src_len2 = src_len / 2;
  size_t len = (dst_len8 < src_len2) ? dst_len8 : src_len2;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    wuffs_base__poke_u64le__no_bounds_check(
        d + (0 * 8),
        wuffs_base__color_u32__as__color_u64(
            wuffs_base__color_u16_rgb_565__as__color_u32_argb_premul(
                wuffs_base__peek_u16le__no_bounds_check(s + (0 * 2)))));

    s += 1 * 2;
    d += 1 * 8;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgrw_4x16le__bgrx(uint8_t* dst_ptr,
                                              size_t dst_len,
                                              uint8_t* dst_palette_ptr,
                                              size_t dst_palette_len,
                                              const uint8_t* src_ptr,
                                              size_t src_len) {
  size_t dst_len8 = dst_len / 8;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len8 < src_len4) ? dst_len8 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    uint8_t s0 = s[0];
    uint8_t s1 = s[1];
    uint8_t s2 = s[2];
    d[0] = s0;
    d[1] = s0;
    d[2] = s1;
    d[3] = s1;
    d[4] = s2;
    d[5] = s2;
    d[6] = 0xFF;
    d[7] = 0xFF;

    s += 1 * 4;
    d += 1 * 8;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__bgrw_4x16le__rgb(uint8_t* dst_ptr,
                                             size_t dst_len,
                                             uint8_t* dst_palette_ptr,
                                             size_t dst_palette_len,
                                             const uint8_t* src_ptr,
                                             size_t src_len) {
  size_t dst_len8 = dst_len / 8;
  size_t src_len3 = src_len / 3;
  size_t len = (dst_len8 < src_len3) ? dst_len8 : src_len3;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    uint8_t s0 = s[0];
    uint8_t s1 = s[1];
    uint8_t s2 = s[2];
    d[0] = s2;
    d[1] = s2;
    d[2] = s1;
    d[3] = s1;
    d[4] = s0;
    d[5] = s0;
    d[6] = 0xFF;
    d[7] = 0xFF;

    s += 1 * 3;
    d += 1 * 8;
    n -= 1;
  }

  return len;
}

// --------

static uint64_t  //
wuffs_base__pixel_swizzler__rgba_nonpremul__bgra_nonpremul_4x16le__src(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len8 = src_len / 8;
  size_t len = (dst_len4 < src_len8) ? dst_len4 : src_len8;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;

  size_t n = len;
  while (n >= 1) {
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4), wuffs_base__color_u64__as__color_u32__swap_u32_argb_abgr(
                         wuffs_base__peek_u64le__no_bounds_check(s + (0 * 8))));

    s += 1 * 8;
    d += 1 * 4;
    n -= 1;
  }
  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__rgba_nonpremul__bgra_nonpremul_4x16le__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len8 = src_len / 8;
  size_t len = (dst_len4 < src_len8) ? dst_len4 : src_len8;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    uint64_t d0 = wuffs_base__color_u32__as__color_u64(
        wuffs_base__peek_u32le__no_bounds_check(d + (0 * 4)));
    uint64_t s0 = wuffs_base__swap_u64_argb_abgr(
        wuffs_base__peek_u64le__no_bounds_check(s + (0 * 8)));
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4),
        wuffs_base__color_u64__as__color_u32(
            wuffs_base__composite_nonpremul_nonpremul_u64_axxx(d0, s0)));

    s += 1 * 8;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

// --------

static uint64_t  //
wuffs_base__pixel_swizzler__rgbw__bgr_565(uint8_t* dst_ptr,
                                          size_t dst_len,
                                          uint8_t* dst_palette_ptr,
                                          size_t dst_palette_len,
                                          const uint8_t* src_ptr,
                                          size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len2 = src_len / 2;
  size_t len = (dst_len4 < src_len2) ? dst_len4 : src_len2;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4),
        wuffs_base__swap_u32_argb_abgr(
            wuffs_base__color_u16_rgb_565__as__color_u32_argb_premul(
                wuffs_base__peek_u16le__no_bounds_check(s + (0 * 2)))));

    s += 1 * 2;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

// --------

static uint64_t  //
wuffs_base__pixel_swizzler__xxx__index__src(uint8_t* dst_ptr,
                                            size_t dst_len,
                                            uint8_t* dst_palette_ptr,
                                            size_t dst_palette_len,
                                            const uint8_t* src_ptr,
                                            size_t src_len) {
  if (dst_palette_len !=
      WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH) {
    return 0;
  }
  size_t dst_len3 = dst_len / 3;
  size_t len = (dst_len3 < src_len) ? dst_len3 : src_len;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  const size_t loop_unroll_count = 4;

  // The comparison in the while condition is ">", not ">=", because with
  // ">=", the last 4-byte store could write past the end of the dst slice.
  //
  // Each 4-byte store writes one too many bytes, but a subsequent store
  // will overwrite that with the correct byte. There is always another
  // store, whether a 4-byte store in this loop or a 1-byte store in the
  // next loop.
  while (n > loop_unroll_count) {
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 3), wuffs_base__peek_u32le__no_bounds_check(
                         dst_palette_ptr + ((size_t)s[0] * 4)));
    wuffs_base__poke_u32le__no_bounds_check(
        d + (1 * 3), wuffs_base__peek_u32le__no_bounds_check(
                         dst_palette_ptr + ((size_t)s[1] * 4)));
    wuffs_base__poke_u32le__no_bounds_check(
        d + (2 * 3), wuffs_base__peek_u32le__no_bounds_check(
                         dst_palette_ptr + ((size_t)s[2] * 4)));
    wuffs_base__poke_u32le__no_bounds_check(
        d + (3 * 3), wuffs_base__peek_u32le__no_bounds_check(
                         dst_palette_ptr + ((size_t)s[3] * 4)));

    s += loop_unroll_count * 1;
    d += loop_unroll_count * 3;
    n -= loop_unroll_count;
  }

  while (n >= 1) {
    uint32_t s0 = wuffs_base__peek_u32le__no_bounds_check(dst_palette_ptr +
                                                          ((size_t)s[0] * 4));
    wuffs_base__poke_u24le__no_bounds_check(d + (0 * 3), s0);

    s += 1 * 1;
    d += 1 * 3;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__xxx__index_bgra_nonpremul__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  if (dst_palette_len !=
      WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH) {
    return 0;
  }
  size_t dst_len3 = dst_len / 3;
  size_t len = (dst_len3 < src_len) ? dst_len3 : src_len;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    uint32_t d0 =
        wuffs_base__peek_u24le__no_bounds_check(d + (0 * 3)) | 0xFF000000;
    uint32_t s0 = wuffs_base__peek_u32le__no_bounds_check(dst_palette_ptr +
                                                          ((size_t)s[0] * 4));
    wuffs_base__poke_u24le__no_bounds_check(
        d + (0 * 3), wuffs_base__composite_premul_nonpremul_u32_axxx(d0, s0));

    s += 1 * 1;
    d += 1 * 3;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__xxx__index_binary_alpha__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  if (dst_palette_len !=
      WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH) {
    return 0;
  }
  size_t dst_len3 = dst_len / 3;
  size_t len = (dst_len3 < src_len) ? dst_len3 : src_len;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  const size_t loop_unroll_count = 4;

  while (n >= loop_unroll_count) {
    uint32_t s0 = wuffs_base__peek_u32le__no_bounds_check(dst_palette_ptr +
                                                          ((size_t)s[0] * 4));
    if (s0) {
      wuffs_base__poke_u24le__no_bounds_check(d + (0 * 3), s0);
    }
    uint32_t s1 = wuffs_base__peek_u32le__no_bounds_check(dst_palette_ptr +
                                                          ((size_t)s[1] * 4));
    if (s1) {
      wuffs_base__poke_u24le__no_bounds_check(d + (1 * 3), s1);
    }
    uint32_t s2 = wuffs_base__peek_u32le__no_bounds_check(dst_palette_ptr +
                                                          ((size_t)s[2] * 4));
    if (s2) {
      wuffs_base__poke_u24le__no_bounds_check(d + (2 * 3), s2);
    }
    uint32_t s3 = wuffs_base__peek_u32le__no_bounds_check(dst_palette_ptr +
                                                          ((size_t)s[3] * 4));
    if (s3) {
      wuffs_base__poke_u24le__no_bounds_check(d + (3 * 3), s3);
    }

    s += loop_unroll_count * 1;
    d += loop_unroll_count * 3;
    n -= loop_unroll_count;
  }

  while (n >= 1) {
    uint32_t s0 = wuffs_base__peek_u32le__no_bounds_check(dst_palette_ptr +
                                                          ((size_t)s[0] * 4));
    if (s0) {
      wuffs_base__poke_u24le__no_bounds_check(d + (0 * 3), s0);
    }

    s += 1 * 1;
    d += 1 * 3;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__xxx__xxxx(uint8_t* dst_ptr,
                                      size_t dst_len,
                                      uint8_t* dst_palette_ptr,
                                      size_t dst_palette_len,
                                      const uint8_t* src_ptr,
                                      size_t src_len) {
  size_t dst_len3 = dst_len / 3;
  size_t src_len4 = src_len / 4;
  size_t len = (dst_len3 < src_len4) ? dst_len3 : src_len4;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    wuffs_base__poke_u24le__no_bounds_check(
        d + (0 * 3), wuffs_base__peek_u32le__no_bounds_check(s + (0 * 4)));

    s += 1 * 4;
    d += 1 * 3;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__xxx__y(uint8_t* dst_ptr,
                                   size_t dst_len,
                                   uint8_t* dst_palette_ptr,
                                   size_t dst_palette_len,
                                   const uint8_t* src_ptr,
                                   size_t src_len) {
  size_t dst_len3 = dst_len / 3;
  size_t len = (dst_len3 < src_len) ? dst_len3 : src_len;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    uint8_t s0 = s[0];
    d[0] = s0;
    d[1] = s0;
    d[2] = s0;

    s += 1 * 1;
    d += 1 * 3;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__xxx__y_16be(uint8_t* dst_ptr,
                                        size_t dst_len,
                                        uint8_t* dst_palette_ptr,
                                        size_t dst_palette_len,
                                        const uint8_t* src_ptr,
                                        size_t src_len) {
  size_t dst_len3 = dst_len / 3;
  size_t src_len2 = src_len / 2;
  size_t len = (dst_len3 < src_len2) ? dst_len3 : src_len2;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  // TODO: unroll.

  while (n >= 1) {
    uint8_t s0 = s[0];
    d[0] = s0;
    d[1] = s0;
    d[2] = s0;

    s += 1 * 2;
    d += 1 * 3;
    n -= 1;
  }

  return len;
}

// --------

static uint64_t  //
wuffs_base__pixel_swizzler__xxxx__index__src(uint8_t* dst_ptr,
                                             size_t dst_len,
                                             uint8_t* dst_palette_ptr,
                                             size_t dst_palette_len,
                                             const uint8_t* src_ptr,
                                             size_t src_len) {
  if (dst_palette_len !=
      WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH) {
    return 0;
  }
  size_t dst_len4 = dst_len / 4;
  size_t len = (dst_len4 < src_len) ? dst_len4 : src_len;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  const size_t loop_unroll_count = 4;

  while (n >= loop_unroll_count) {
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4), wuffs_base__peek_u32le__no_bounds_check(
                         dst_palette_ptr + ((size_t)s[0] * 4)));
    wuffs_base__poke_u32le__no_bounds_check(
        d + (1 * 4), wuffs_base__peek_u32le__no_bounds_check(
                         dst_palette_ptr + ((size_t)s[1] * 4)));
    wuffs_base__poke_u32le__no_bounds_check(
        d + (2 * 4), wuffs_base__peek_u32le__no_bounds_check(
                         dst_palette_ptr + ((size_t)s[2] * 4)));
    wuffs_base__poke_u32le__no_bounds_check(
        d + (3 * 4), wuffs_base__peek_u32le__no_bounds_check(
                         dst_palette_ptr + ((size_t)s[3] * 4)));

    s += loop_unroll_count * 1;
    d += loop_unroll_count * 4;
    n -= loop_unroll_count;
  }

  while (n >= 1) {
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4), wuffs_base__peek_u32le__no_bounds_check(
                         dst_palette_ptr + ((size_t)s[0] * 4)));

    s += 1 * 1;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__xxxx__index_binary_alpha__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  if (dst_palette_len !=
      WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH) {
    return 0;
  }
  size_t dst_len4 = dst_len / 4;
  size_t len = (dst_len4 < src_len) ? dst_len4 : src_len;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  const size_t loop_unroll_count = 4;

  while (n >= loop_unroll_count) {
    uint32_t s0 = wuffs_base__peek_u32le__no_bounds_check(dst_palette_ptr +
                                                          ((size_t)s[0] * 4));
    if (s0) {
      wuffs_base__poke_u32le__no_bounds_check(d + (0 * 4), s0);
    }
    uint32_t s1 = wuffs_base__peek_u32le__no_bounds_check(dst_palette_ptr +
                                                          ((size_t)s[1] * 4));
    if (s1) {
      wuffs_base__poke_u32le__no_bounds_check(d + (1 * 4), s1);
    }
    uint32_t s2 = wuffs_base__peek_u32le__no_bounds_check(dst_palette_ptr +
                                                          ((size_t)s[2] * 4));
    if (s2) {
      wuffs_base__poke_u32le__no_bounds_check(d + (2 * 4), s2);
    }
    uint32_t s3 = wuffs_base__peek_u32le__no_bounds_check(dst_palette_ptr +
                                                          ((size_t)s[3] * 4));
    if (s3) {
      wuffs_base__poke_u32le__no_bounds_check(d + (3 * 4), s3);
    }

    s += loop_unroll_count * 1;
    d += loop_unroll_count * 4;
    n -= loop_unroll_count;
  }

  while (n >= 1) {
    uint32_t s0 = wuffs_base__peek_u32le__no_bounds_check(dst_palette_ptr +
                                                          ((size_t)s[0] * 4));
    if (s0) {
      wuffs_base__poke_u32le__no_bounds_check(d + (0 * 4), s0);
    }

    s += 1 * 1;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

// ‼ WUFFS MULTI-FILE SECTION +x86_sse42
#if defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
WUFFS_BASE__MAYBE_ATTRIBUTE_TARGET("pclmul,popcnt,sse4.2")
static uint64_t  //
wuffs_base__pixel_swizzler__xxxx__y__sse42(uint8_t* dst_ptr,
                                           size_t dst_len,
                                           uint8_t* dst_palette_ptr,
                                           size_t dst_palette_len,
                                           const uint8_t* src_ptr,
                                           size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t len = (dst_len4 < src_len) ? dst_len4 : src_len;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  __m128i shuffle = _mm_set_epi8(+0x03, +0x03, +0x03, +0x03,  //
                                 +0x02, +0x02, +0x02, +0x02,  //
                                 +0x01, +0x01, +0x01, +0x01,  //
                                 +0x00, +0x00, +0x00, +0x00);
  __m128i or_ff = _mm_set_epi8(-0x01, +0x00, +0x00, +0x00,  //
                               -0x01, +0x00, +0x00, +0x00,  //
                               -0x01, +0x00, +0x00, +0x00,  //
                               -0x01, +0x00, +0x00, +0x00);

  while (n >= 4) {
    __m128i x;
    x = _mm_cvtsi32_si128((int)(wuffs_base__peek_u32le__no_bounds_check(s)));
    x = _mm_shuffle_epi8(x, shuffle);
    x = _mm_or_si128(x, or_ff);
    _mm_storeu_si128((__m128i*)(void*)d, x);

    s += 4 * 1;
    d += 4 * 4;
    n -= 4;
  }

  while (n >= 1) {
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4), 0xFF000000 | (0x010101 * (uint32_t)s[0]));

    s += 1 * 1;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}
#endif  // defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
// ‼ WUFFS MULTI-FILE SECTION -x86_sse42

static uint64_t  //
wuffs_base__pixel_swizzler__xxxx__y(uint8_t* dst_ptr,
                                    size_t dst_len,
                                    uint8_t* dst_palette_ptr,
                                    size_t dst_palette_len,
                                    const uint8_t* src_ptr,
                                    size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t len = (dst_len4 < src_len) ? dst_len4 : src_len;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4), 0xFF000000 | (0x010101 * (uint32_t)s[0]));

    s += 1 * 1;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__xxxx__y_16be(uint8_t* dst_ptr,
                                         size_t dst_len,
                                         uint8_t* dst_palette_ptr,
                                         size_t dst_palette_len,
                                         const uint8_t* src_ptr,
                                         size_t src_len) {
  size_t dst_len4 = dst_len / 4;
  size_t src_len2 = src_len / 2;
  size_t len = (dst_len4 < src_len2) ? dst_len4 : src_len2;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    wuffs_base__poke_u32le__no_bounds_check(
        d + (0 * 4), 0xFF000000 | (0x010101 * (uint32_t)s[0]));

    s += 1 * 2;
    d += 1 * 4;
    n -= 1;
  }

  return len;
}

// --------

static uint64_t  //
wuffs_base__pixel_swizzler__xxxxxxxx__index__src(uint8_t* dst_ptr,
                                                 size_t dst_len,
                                                 uint8_t* dst_palette_ptr,
                                                 size_t dst_palette_len,
                                                 const uint8_t* src_ptr,
                                                 size_t src_len) {
  if (dst_palette_len !=
      WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH) {
    return 0;
  }
  size_t dst_len8 = dst_len / 8;
  size_t len = (dst_len8 < src_len) ? dst_len8 : src_len;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    wuffs_base__poke_u64le__no_bounds_check(
        d + (0 * 8), wuffs_base__color_u32__as__color_u64(
                         wuffs_base__peek_u32le__no_bounds_check(
                             dst_palette_ptr + ((size_t)s[0] * 4))));

    s += 1 * 1;
    d += 1 * 8;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__xxxxxxxx__index_binary_alpha__src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    const uint8_t* src_ptr,
    size_t src_len) {
  if (dst_palette_len !=
      WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH) {
    return 0;
  }
  size_t dst_len8 = dst_len / 8;
  size_t len = (dst_len8 < src_len) ? dst_len8 : src_len;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    uint32_t s0 = wuffs_base__peek_u32le__no_bounds_check(dst_palette_ptr +
                                                          ((size_t)s[0] * 4));
    if (s0) {
      wuffs_base__poke_u64le__no_bounds_check(
          d + (0 * 8), wuffs_base__color_u32__as__color_u64(s0));
    }

    s += 1 * 1;
    d += 1 * 8;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__xxxxxxxx__y(uint8_t* dst_ptr,
                                        size_t dst_len,
                                        uint8_t* dst_palette_ptr,
                                        size_t dst_palette_len,
                                        const uint8_t* src_ptr,
                                        size_t src_len) {
  size_t dst_len8 = dst_len / 8;
  size_t len = (dst_len8 < src_len) ? dst_len8 : src_len;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    wuffs_base__poke_u64le__no_bounds_check(
        d + (0 * 8), 0xFFFF000000000000 | (0x010101010101 * (uint64_t)s[0]));

    s += 1 * 1;
    d += 1 * 8;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__xxxxxxxx__y_16be(uint8_t* dst_ptr,
                                             size_t dst_len,
                                             uint8_t* dst_palette_ptr,
                                             size_t dst_palette_len,
                                             const uint8_t* src_ptr,
                                             size_t src_len) {
  size_t dst_len8 = dst_len / 8;
  size_t src_len2 = src_len / 2;
  size_t len = (dst_len8 < src_len2) ? dst_len8 : src_len2;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    uint64_t s0 =
        ((uint64_t)(wuffs_base__peek_u16be__no_bounds_check(s + (0 * 2))));
    wuffs_base__poke_u64le__no_bounds_check(
        d + (0 * 8), 0xFFFF000000000000 | (0x000100010001 * s0));

    s += 1 * 2;
    d += 1 * 8;
    n -= 1;
  }

  return len;
}

// --------

static uint64_t  //
wuffs_base__pixel_swizzler__y__y_16be(uint8_t* dst_ptr,
                                      size_t dst_len,
                                      uint8_t* dst_palette_ptr,
                                      size_t dst_palette_len,
                                      const uint8_t* src_ptr,
                                      size_t src_len) {
  size_t src_len2 = src_len / 2;
  size_t len = (dst_len < src_len2) ? dst_len : src_len2;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    d[0] = s[0];

    s += 1 * 2;
    d += 1 * 1;
    n -= 1;
  }

  return len;
}

static uint64_t  //
wuffs_base__pixel_swizzler__y_16le__y_16be(uint8_t* dst_ptr,
                                           size_t dst_len,
                                           uint8_t* dst_palette_ptr,
                                           size_t dst_palette_len,
                                           const uint8_t* src_ptr,
                                           size_t src_len) {
  size_t dst_len2 = dst_len / 2;
  size_t src_len2 = src_len / 2;
  size_t len = (dst_len2 < src_len2) ? dst_len2 : src_len2;
  uint8_t* d = dst_ptr;
  const uint8_t* s = src_ptr;
  size_t n = len;

  while (n >= 1) {
    uint8_t s0 = s[0];
    uint8_t s1 = s[1];
    d[0] = s1;
    d[1] = s0;

    s += 1 * 2;
    d += 1 * 2;
    n -= 1;
  }

  return len;
}

// --------

static uint64_t  //
wuffs_base__pixel_swizzler__transparent_black_src(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    uint64_t num_pixels,
    uint32_t dst_pixfmt_bytes_per_pixel) {
  uint64_t n = ((uint64_t)dst_len) / dst_pixfmt_bytes_per_pixel;
  if (n > num_pixels) {
    n = num_pixels;
  }
  memset(dst_ptr, 0, ((size_t)(n * dst_pixfmt_bytes_per_pixel)));
  return n;
}

static uint64_t  //
wuffs_base__pixel_swizzler__transparent_black_src_over(
    uint8_t* dst_ptr,
    size_t dst_len,
    uint8_t* dst_palette_ptr,
    size_t dst_palette_len,
    uint64_t num_pixels,
    uint32_t dst_pixfmt_bytes_per_pixel) {
  uint64_t n = ((uint64_t)dst_len) / dst_pixfmt_bytes_per_pixel;
  if (n > num_pixels) {
    n = num_pixels;
  }
  return n;
}

// --------

static wuffs_base__pixel_swizzler__func  //
wuffs_base__pixel_swizzler__prepare__y(wuffs_base__pixel_swizzler* p,
                                       wuffs_base__pixel_format dst_pixfmt,
                                       wuffs_base__slice_u8 dst_palette,
                                       wuffs_base__slice_u8 src_palette,
                                       wuffs_base__pixel_blend blend) {
  switch (dst_pixfmt.repr) {
    case WUFFS_BASE__PIXEL_FORMAT__Y:
      return wuffs_base__pixel_swizzler__copy_1_1;

    case WUFFS_BASE__PIXEL_FORMAT__BGR_565:
      return wuffs_base__pixel_swizzler__bgr_565__y;

    case WUFFS_BASE__PIXEL_FORMAT__BGR:
    case WUFFS_BASE__PIXEL_FORMAT__RGB:
      return wuffs_base__pixel_swizzler__xxx__y;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_BINARY:
    case WUFFS_BASE__PIXEL_FORMAT__BGRX:
    case WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__RGBA_BINARY:
    case WUFFS_BASE__PIXEL_FORMAT__RGBX:
#if defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
      if (wuffs_base__cpu_arch__have_x86_sse42()) {
        return wuffs_base__pixel_swizzler__xxxx__y__sse42;
      }
#endif
      return wuffs_base__pixel_swizzler__xxxx__y;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL_4X16LE:
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL_4X16LE:
    case WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL_4X16LE:
    case WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL_4X16LE:
      return wuffs_base__pixel_swizzler__xxxxxxxx__y;
  }
  return NULL;
}

static wuffs_base__pixel_swizzler__func  //
wuffs_base__pixel_swizzler__prepare__y_16be(wuffs_base__pixel_swizzler* p,
                                            wuffs_base__pixel_format dst_pixfmt,
                                            wuffs_base__slice_u8 dst_palette,
                                            wuffs_base__slice_u8 src_palette,
                                            wuffs_base__pixel_blend blend) {
  switch (dst_pixfmt.repr) {
    case WUFFS_BASE__PIXEL_FORMAT__Y:
      return wuffs_base__pixel_swizzler__y__y_16be;

    case WUFFS_BASE__PIXEL_FORMAT__Y_16LE:
      return wuffs_base__pixel_swizzler__y_16le__y_16be;

    case WUFFS_BASE__PIXEL_FORMAT__Y_16BE:
      return wuffs_base__pixel_swizzler__copy_2_2;

    case WUFFS_BASE__PIXEL_FORMAT__BGR_565:
      return wuffs_base__pixel_swizzler__bgr_565__y_16be;

    case WUFFS_BASE__PIXEL_FORMAT__BGR:
    case WUFFS_BASE__PIXEL_FORMAT__RGB:
      return wuffs_base__pixel_swizzler__xxx__y_16be;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_BINARY:
    case WUFFS_BASE__PIXEL_FORMAT__BGRX:
    case WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__RGBA_BINARY:
    case WUFFS_BASE__PIXEL_FORMAT__RGBX:
      return wuffs_base__pixel_swizzler__xxxx__y_16be;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL_4X16LE:
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL_4X16LE:
    case WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL_4X16LE:
    case WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL_4X16LE:
      return wuffs_base__pixel_swizzler__xxxxxxxx__y_16be;
  }
  return NULL;
}

static wuffs_base__pixel_swizzler__func  //
wuffs_base__pixel_swizzler__prepare__indexed__bgra_nonpremul(
    wuffs_base__pixel_swizzler* p,
    wuffs_base__pixel_format dst_pixfmt,
    wuffs_base__slice_u8 dst_palette,
    wuffs_base__slice_u8 src_palette,
    wuffs_base__pixel_blend blend) {
  switch (dst_pixfmt.repr) {
    case WUFFS_BASE__PIXEL_FORMAT__INDEXED__BGRA_NONPREMUL:
      if (wuffs_base__slice_u8__copy_from_slice(dst_palette, src_palette) !=
          WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH) {
        return NULL;
      }
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__copy_1_1;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGR_565:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          if (wuffs_base__pixel_swizzler__squash_align4_bgr_565_8888(
                  dst_palette.ptr, dst_palette.len, src_palette.ptr,
                  src_palette.len, true) !=
              (WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH / 4)) {
            return NULL;
          }
          return wuffs_base__pixel_swizzler__bgr_565__index__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          if (wuffs_base__slice_u8__copy_from_slice(dst_palette, src_palette) !=
              WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH) {
            return NULL;
          }
          return wuffs_base__pixel_swizzler__bgr_565__index_bgra_nonpremul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGR:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          if (wuffs_base__pixel_swizzler__bgra_premul__bgra_nonpremul__src(
                  dst_palette.ptr, dst_palette.len, NULL, 0, src_palette.ptr,
                  src_palette.len) !=
              (WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH / 4)) {
            return NULL;
          }
          return wuffs_base__pixel_swizzler__xxx__index__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          if (wuffs_base__slice_u8__copy_from_slice(dst_palette, src_palette) !=
              WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH) {
            return NULL;
          }
          return wuffs_base__pixel_swizzler__xxx__index_bgra_nonpremul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL:
      if (wuffs_base__slice_u8__copy_from_slice(dst_palette, src_palette) !=
          WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH) {
        return NULL;
      }
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__xxxx__index__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_nonpremul__index_bgra_nonpremul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL_4X16LE:
      if (wuffs_base__slice_u8__copy_from_slice(dst_palette, src_palette) !=
          WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH) {
        return NULL;
      }
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__xxxxxxxx__index__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_nonpremul_4x16le__index_bgra_nonpremul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          if (wuffs_base__pixel_swizzler__bgra_premul__bgra_nonpremul__src(
                  dst_palette.ptr, dst_palette.len, NULL, 0, src_palette.ptr,
                  src_palette.len) !=
              (WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH / 4)) {
            return NULL;
          }
          return wuffs_base__pixel_swizzler__xxxx__index__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          if (wuffs_base__slice_u8__copy_from_slice(dst_palette, src_palette) !=
              WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH) {
            return NULL;
          }
          return wuffs_base__pixel_swizzler__bgra_premul__index_bgra_nonpremul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__RGB:
      // TODO.
      break;

    case WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL:
      if (wuffs_base__pixel_swizzler__swap_rgbx_bgrx(
              dst_palette.ptr, dst_palette.len, NULL, 0, src_palette.ptr,
              src_palette.len) !=
          (WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH / 4)) {
        return NULL;
      }
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__xxxx__index__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_nonpremul__index_bgra_nonpremul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          if (wuffs_base__pixel_swizzler__bgra_premul__rgba_nonpremul__src(
                  dst_palette.ptr, dst_palette.len, NULL, 0, src_palette.ptr,
                  src_palette.len) !=
              (WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH / 4)) {
            return NULL;
          }
          return wuffs_base__pixel_swizzler__xxxx__index__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          if (wuffs_base__pixel_swizzler__swap_rgbx_bgrx(
                  dst_palette.ptr, dst_palette.len, NULL, 0, src_palette.ptr,
                  src_palette.len) !=
              (WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH / 4)) {
            return NULL;
          }
          return wuffs_base__pixel_swizzler__bgra_premul__index_bgra_nonpremul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__RGBX:
      // TODO.
      break;
  }
  return NULL;
}

static wuffs_base__pixel_swizzler__func  //
wuffs_base__pixel_swizzler__prepare__indexed__bgra_binary(
    wuffs_base__pixel_swizzler* p,
    wuffs_base__pixel_format dst_pixfmt,
    wuffs_base__slice_u8 dst_palette,
    wuffs_base__slice_u8 src_palette,
    wuffs_base__pixel_blend blend) {
  switch (dst_pixfmt.repr) {
    case WUFFS_BASE__PIXEL_FORMAT__INDEXED__BGRA_NONPREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__INDEXED__BGRA_PREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__INDEXED__BGRA_BINARY:
      if (wuffs_base__slice_u8__copy_from_slice(dst_palette, src_palette) !=
          WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH) {
        return NULL;
      }
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__copy_1_1;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGR_565:
      if (wuffs_base__pixel_swizzler__squash_align4_bgr_565_8888(
              dst_palette.ptr, dst_palette.len, src_palette.ptr,
              src_palette.len, false) !=
          (WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH / 4)) {
        return NULL;
      }
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__bgr_565__index__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgr_565__index_binary_alpha__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGR:
      if (wuffs_base__slice_u8__copy_from_slice(dst_palette, src_palette) !=
          WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH) {
        return NULL;
      }
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__xxx__index__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__xxx__index_binary_alpha__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_BINARY:
      if (wuffs_base__slice_u8__copy_from_slice(dst_palette, src_palette) !=
          WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH) {
        return NULL;
      }
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__xxxx__index__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__xxxx__index_binary_alpha__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL_4X16LE:
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL_4X16LE:
      if (wuffs_base__slice_u8__copy_from_slice(dst_palette, src_palette) !=
          WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH) {
        return NULL;
      }
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__xxxxxxxx__index__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__xxxxxxxx__index_binary_alpha__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__RGB:
      if (wuffs_base__pixel_swizzler__swap_rgbx_bgrx(
              dst_palette.ptr, dst_palette.len, NULL, 0, src_palette.ptr,
              src_palette.len) !=
          (WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH / 4)) {
        return NULL;
      }
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__xxx__index__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__xxx__index_binary_alpha__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__RGBA_BINARY:
      if (wuffs_base__pixel_swizzler__swap_rgbx_bgrx(
              dst_palette.ptr, dst_palette.len, NULL, 0, src_palette.ptr,
              src_palette.len) !=
          (WUFFS_BASE__PIXEL_FORMAT__INDEXED__PALETTE_BYTE_LENGTH / 4)) {
        return NULL;
      }
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__xxxx__index__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__xxxx__index_binary_alpha__src_over;
      }
      return NULL;
  }
  return NULL;
}

static wuffs_base__pixel_swizzler__func  //
wuffs_base__pixel_swizzler__prepare__bgr_565(
    wuffs_base__pixel_swizzler* p,
    wuffs_base__pixel_format dst_pixfmt,
    wuffs_base__slice_u8 dst_palette,
    wuffs_base__slice_u8 src_palette,
    wuffs_base__pixel_blend blend) {
  switch (dst_pixfmt.repr) {
    case WUFFS_BASE__PIXEL_FORMAT__BGR_565:
      return wuffs_base__pixel_swizzler__copy_2_2;

    case WUFFS_BASE__PIXEL_FORMAT__BGR:
      return wuffs_base__pixel_swizzler__bgr__bgr_565;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_BINARY:
    case WUFFS_BASE__PIXEL_FORMAT__BGRX:
      return wuffs_base__pixel_swizzler__bgrw__bgr_565;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL_4X16LE:
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL_4X16LE:
      return wuffs_base__pixel_swizzler__bgrw_4x16le__bgr_565;

    case WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__RGBA_BINARY:
    case WUFFS_BASE__PIXEL_FORMAT__RGBX:
      return wuffs_base__pixel_swizzler__rgbw__bgr_565;
  }
  return NULL;
}

static wuffs_base__pixel_swizzler__func  //
wuffs_base__pixel_swizzler__prepare__bgr(wuffs_base__pixel_swizzler* p,
                                         wuffs_base__pixel_format dst_pixfmt,
                                         wuffs_base__slice_u8 dst_palette,
                                         wuffs_base__slice_u8 src_palette,
                                         wuffs_base__pixel_blend blend) {
  switch (dst_pixfmt.repr) {
    case WUFFS_BASE__PIXEL_FORMAT__BGR_565:
      return wuffs_base__pixel_swizzler__bgr_565__bgr;

    case WUFFS_BASE__PIXEL_FORMAT__BGR:
      return wuffs_base__pixel_swizzler__copy_3_3;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_BINARY:
    case WUFFS_BASE__PIXEL_FORMAT__BGRX:
      return wuffs_base__pixel_swizzler__bgrw__bgr;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL_4X16LE:
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL_4X16LE:
      return wuffs_base__pixel_swizzler__bgrw_4x16le__bgr;

    case WUFFS_BASE__PIXEL_FORMAT__RGB:
      return wuffs_base__pixel_swizzler__swap_rgb_bgr;

    case WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__RGBA_BINARY:
    case WUFFS_BASE__PIXEL_FORMAT__RGBX:
#if defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
      if (wuffs_base__cpu_arch__have_x86_sse42()) {
        return wuffs_base__pixel_swizzler__bgrw__rgb__sse42;
      }
#endif
      return wuffs_base__pixel_swizzler__bgrw__rgb;
  }
  return NULL;
}

static wuffs_base__pixel_swizzler__func  //
wuffs_base__pixel_swizzler__prepare__bgra_nonpremul(
    wuffs_base__pixel_swizzler* p,
    wuffs_base__pixel_format dst_pixfmt,
    wuffs_base__slice_u8 dst_palette,
    wuffs_base__slice_u8 src_palette,
    wuffs_base__pixel_blend blend) {
  switch (dst_pixfmt.repr) {
    case WUFFS_BASE__PIXEL_FORMAT__BGR_565:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__bgr_565__bgra_nonpremul__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgr_565__bgra_nonpremul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGR:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__bgr__bgra_nonpremul__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgr__bgra_nonpremul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__copy_4_4;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_nonpremul__bgra_nonpremul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL_4X16LE:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__bgra_nonpremul_4x16le__bgra_nonpremul__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_nonpremul_4x16le__bgra_nonpremul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__bgra_premul__bgra_nonpremul__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_premul__bgra_nonpremul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_BINARY:
    case WUFFS_BASE__PIXEL_FORMAT__BGRX:
      // TODO.
      break;

    case WUFFS_BASE__PIXEL_FORMAT__RGB:
      // TODO.
      break;

    case WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
#if defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
          if (wuffs_base__cpu_arch__have_x86_sse42()) {
            return wuffs_base__pixel_swizzler__swap_rgbx_bgrx__sse42;
          }
#endif
          return wuffs_base__pixel_swizzler__swap_rgbx_bgrx;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_nonpremul__rgba_nonpremul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__bgra_premul__rgba_nonpremul__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_premul__rgba_nonpremul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__RGBA_BINARY:
    case WUFFS_BASE__PIXEL_FORMAT__RGBX:
      // TODO.
      break;
  }
  return NULL;
}

static wuffs_base__pixel_swizzler__func  //
wuffs_base__pixel_swizzler__prepare__bgra_nonpremul_4x16le(
    wuffs_base__pixel_swizzler* p,
    wuffs_base__pixel_format dst_pixfmt,
    wuffs_base__slice_u8 dst_palette,
    wuffs_base__slice_u8 src_palette,
    wuffs_base__pixel_blend blend) {
  switch (dst_pixfmt.repr) {
    case WUFFS_BASE__PIXEL_FORMAT__BGR_565:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__bgr_565__bgra_nonpremul_4x16le__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgr_565__bgra_nonpremul_4x16le__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGR:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__bgr__bgra_nonpremul_4x16le__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgr__bgra_nonpremul_4x16le__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__bgra_nonpremul__bgra_nonpremul_4x16le__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_nonpremul__bgra_nonpremul_4x16le__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL_4X16LE:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__copy_8_8;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_nonpremul_4x16le__bgra_nonpremul_4x16le__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__bgra_premul__bgra_nonpremul_4x16le__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_premul__bgra_nonpremul_4x16le__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_BINARY:
    case WUFFS_BASE__PIXEL_FORMAT__BGRX:
      // TODO.
      break;

    case WUFFS_BASE__PIXEL_FORMAT__RGB:
      // TODO.
      break;

    case WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__rgba_nonpremul__bgra_nonpremul_4x16le__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__rgba_nonpremul__bgra_nonpremul_4x16le__src_over;
      }
      break;

    case WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__bgra_premul__rgba_nonpremul_4x16le__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_premul__rgba_nonpremul_4x16le__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__RGBA_BINARY:
    case WUFFS_BASE__PIXEL_FORMAT__RGBX:
      // TODO.
      break;
  }
  return NULL;
}

static wuffs_base__pixel_swizzler__func  //
wuffs_base__pixel_swizzler__prepare__bgra_premul(
    wuffs_base__pixel_swizzler* p,
    wuffs_base__pixel_format dst_pixfmt,
    wuffs_base__slice_u8 dst_palette,
    wuffs_base__slice_u8 src_palette,
    wuffs_base__pixel_blend blend) {
  switch (dst_pixfmt.repr) {
    case WUFFS_BASE__PIXEL_FORMAT__BGR_565:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__bgr_565__bgra_premul__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgr_565__bgra_premul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGR:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__bgr__bgra_premul__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgr__bgra_premul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__bgra_nonpremul__bgra_premul__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_nonpremul__bgra_premul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL_4X16LE:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__bgra_nonpremul_4x16le__bgra_premul__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_nonpremul_4x16le__bgra_premul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__copy_4_4;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_premul__bgra_premul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__bgra_nonpremul__rgba_premul__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_nonpremul__rgba_premul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
#if defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
          if (wuffs_base__cpu_arch__have_x86_sse42()) {
            return wuffs_base__pixel_swizzler__swap_rgbx_bgrx__sse42;
          }
#endif
          return wuffs_base__pixel_swizzler__swap_rgbx_bgrx;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_premul__rgba_premul__src_over;
      }
      return NULL;
  }
  return NULL;
}

static wuffs_base__pixel_swizzler__func  //
wuffs_base__pixel_swizzler__prepare__bgrx(wuffs_base__pixel_swizzler* p,
                                          wuffs_base__pixel_format dst_pixfmt,
                                          wuffs_base__slice_u8 dst_palette,
                                          wuffs_base__slice_u8 src_palette,
                                          wuffs_base__pixel_blend blend) {
  switch (dst_pixfmt.repr) {
    case WUFFS_BASE__PIXEL_FORMAT__BGR_565:
      return wuffs_base__pixel_swizzler__bgr_565__bgrx;

    case WUFFS_BASE__PIXEL_FORMAT__BGR:
      return wuffs_base__pixel_swizzler__xxx__xxxx;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_BINARY:
      return wuffs_base__pixel_swizzler__bgrw__bgrx;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL_4X16LE:
      return wuffs_base__pixel_swizzler__bgrw_4x16le__bgrx;

    case WUFFS_BASE__PIXEL_FORMAT__BGRX:
      return wuffs_base__pixel_swizzler__copy_4_4;

    case WUFFS_BASE__PIXEL_FORMAT__RGB:
      // TODO.
      break;

    case WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__RGBA_BINARY:
    case WUFFS_BASE__PIXEL_FORMAT__RGBX:
      return wuffs_base__pixel_swizzler__bgrw__rgbx;
  }
  return NULL;
}

static wuffs_base__pixel_swizzler__func  //
wuffs_base__pixel_swizzler__prepare__rgb(wuffs_base__pixel_swizzler* p,
                                         wuffs_base__pixel_format dst_pixfmt,
                                         wuffs_base__slice_u8 dst_palette,
                                         wuffs_base__slice_u8 src_palette,
                                         wuffs_base__pixel_blend blend) {
  switch (dst_pixfmt.repr) {
    case WUFFS_BASE__PIXEL_FORMAT__BGR_565:
      return wuffs_base__pixel_swizzler__bgr_565__rgb;

    case WUFFS_BASE__PIXEL_FORMAT__BGR:
      return wuffs_base__pixel_swizzler__swap_rgb_bgr;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__BGRA_BINARY:
    case WUFFS_BASE__PIXEL_FORMAT__BGRX:
#if defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
      if (wuffs_base__cpu_arch__have_x86_sse42()) {
        return wuffs_base__pixel_swizzler__bgrw__rgb__sse42;
      }
#endif
      return wuffs_base__pixel_swizzler__bgrw__rgb;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL_4X16LE:
      return wuffs_base__pixel_swizzler__bgrw_4x16le__rgb;

    case WUFFS_BASE__PIXEL_FORMAT__RGB:
      return wuffs_base__pixel_swizzler__copy_3_3;

    case WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL:
    case WUFFS_BASE__PIXEL_FORMAT__RGBA_BINARY:
    case WUFFS_BASE__PIXEL_FORMAT__RGBX:
      return wuffs_base__pixel_swizzler__bgrw__bgr;
  }
  return NULL;
}

static wuffs_base__pixel_swizzler__func  //
wuffs_base__pixel_swizzler__prepare__rgba_nonpremul(
    wuffs_base__pixel_swizzler* p,
    wuffs_base__pixel_format dst_pixfmt,
    wuffs_base__slice_u8 dst_palette,
    wuffs_base__slice_u8 src_palette,
    wuffs_base__pixel_blend blend) {
  switch (dst_pixfmt.repr) {
    case WUFFS_BASE__PIXEL_FORMAT__BGR_565:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__bgr_565__rgba_nonpremul__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgr_565__rgba_nonpremul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGR:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__bgr__rgba_nonpremul__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgr__rgba_nonpremul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
#if defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
          if (wuffs_base__cpu_arch__have_x86_sse42()) {
            return wuffs_base__pixel_swizzler__swap_rgbx_bgrx__sse42;
          }
#endif
          return wuffs_base__pixel_swizzler__swap_rgbx_bgrx;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_nonpremul__rgba_nonpremul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL_4X16LE:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__bgra_nonpremul_4x16le__rgba_nonpremul__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_nonpremul_4x16le__rgba_nonpremul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__bgra_premul__rgba_nonpremul__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_premul__rgba_nonpremul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_BINARY:
    case WUFFS_BASE__PIXEL_FORMAT__BGRX:
      // TODO.
      break;

    case WUFFS_BASE__PIXEL_FORMAT__RGB:
      // TODO.
      break;

    case WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__copy_4_4;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_nonpremul__bgra_nonpremul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__bgra_premul__bgra_nonpremul__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_premul__bgra_nonpremul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__RGBA_BINARY:
    case WUFFS_BASE__PIXEL_FORMAT__RGBX:
      // TODO.
      break;
  }
  return NULL;
}

static wuffs_base__pixel_swizzler__func  //
wuffs_base__pixel_swizzler__prepare__rgba_premul(
    wuffs_base__pixel_swizzler* p,
    wuffs_base__pixel_format dst_pixfmt,
    wuffs_base__slice_u8 dst_palette,
    wuffs_base__slice_u8 src_palette,
    wuffs_base__pixel_blend blend) {
  switch (dst_pixfmt.repr) {
    case WUFFS_BASE__PIXEL_FORMAT__BGR_565:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__bgr_565__rgba_premul__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgr_565__rgba_premul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGR:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__bgr__rgba_premul__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgr__rgba_premul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__bgra_nonpremul__rgba_premul__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_nonpremul__rgba_premul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL_4X16LE:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__bgra_nonpremul_4x16le__rgba_premul__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_nonpremul_4x16le__rgba_premul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
#if defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
          if (wuffs_base__cpu_arch__have_x86_sse42()) {
            return wuffs_base__pixel_swizzler__swap_rgbx_bgrx__sse42;
          }
#endif
          return wuffs_base__pixel_swizzler__swap_rgbx_bgrx;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_premul__rgba_premul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__bgra_nonpremul__bgra_premul__src;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_nonpremul__bgra_premul__src_over;
      }
      return NULL;

    case WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL:
      switch (blend) {
        case WUFFS_BASE__PIXEL_BLEND__SRC:
          return wuffs_base__pixel_swizzler__copy_4_4;
        case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
          return wuffs_base__pixel_swizzler__bgra_premul__bgra_premul__src_over;
      }
      return NULL;
  }
  return NULL;
}

// --------

WUFFS_BASE__MAYBE_STATIC wuffs_base__status  //
wuffs_base__pixel_swizzler__prepare(wuffs_base__pixel_swizzler* p,
                                    wuffs_base__pixel_format dst_pixfmt,
                                    wuffs_base__slice_u8 dst_palette,
                                    wuffs_base__pixel_format src_pixfmt,
                                    wuffs_base__slice_u8 src_palette,
                                    wuffs_base__pixel_blend blend) {
  if (!p) {
    return wuffs_base__make_status(wuffs_base__error__bad_receiver);
  }
  p->private_impl.func = NULL;
  p->private_impl.transparent_black_func = NULL;
  p->private_impl.dst_pixfmt_bytes_per_pixel = 0;
  p->private_impl.src_pixfmt_bytes_per_pixel = 0;

  wuffs_base__pixel_swizzler__func func = NULL;
  wuffs_base__pixel_swizzler__transparent_black_func transparent_black_func =
      NULL;

  uint32_t dst_pixfmt_bits_per_pixel =
      wuffs_base__pixel_format__bits_per_pixel(&dst_pixfmt);
  if ((dst_pixfmt_bits_per_pixel == 0) ||
      ((dst_pixfmt_bits_per_pixel & 7) != 0)) {
    return wuffs_base__make_status(
        wuffs_base__error__unsupported_pixel_swizzler_option);
  }

  uint32_t src_pixfmt_bits_per_pixel =
      wuffs_base__pixel_format__bits_per_pixel(&src_pixfmt);
  if ((src_pixfmt_bits_per_pixel == 0) ||
      ((src_pixfmt_bits_per_pixel & 7) != 0)) {
    return wuffs_base__make_status(
        wuffs_base__error__unsupported_pixel_swizzler_option);
  }

  // TODO: support many more formats.

  switch (blend) {
    case WUFFS_BASE__PIXEL_BLEND__SRC:
      transparent_black_func =
          wuffs_base__pixel_swizzler__transparent_black_src;
      break;

    case WUFFS_BASE__PIXEL_BLEND__SRC_OVER:
      transparent_black_func =
          wuffs_base__pixel_swizzler__transparent_black_src_over;
      break;
  }

  switch (src_pixfmt.repr) {
    case WUFFS_BASE__PIXEL_FORMAT__Y:
      func = wuffs_base__pixel_swizzler__prepare__y(p, dst_pixfmt, dst_palette,
                                                    src_palette, blend);
      break;

    case WUFFS_BASE__PIXEL_FORMAT__Y_16BE:
      func = wuffs_base__pixel_swizzler__prepare__y_16be(
          p, dst_pixfmt, dst_palette, src_palette, blend);
      break;

    case WUFFS_BASE__PIXEL_FORMAT__INDEXED__BGRA_NONPREMUL:
      func = wuffs_base__pixel_swizzler__prepare__indexed__bgra_nonpremul(
          p, dst_pixfmt, dst_palette, src_palette, blend);
      break;

    case WUFFS_BASE__PIXEL_FORMAT__INDEXED__BGRA_BINARY:
      func = wuffs_base__pixel_swizzler__prepare__indexed__bgra_binary(
          p, dst_pixfmt, dst_palette, src_palette, blend);
      break;

    case WUFFS_BASE__PIXEL_FORMAT__BGR_565:
      func = wuffs_base__pixel_swizzler__prepare__bgr_565(
          p, dst_pixfmt, dst_palette, src_palette, blend);
      break;

    case WUFFS_BASE__PIXEL_FORMAT__BGR:
      func = wuffs_base__pixel_swizzler__prepare__bgr(
          p, dst_pixfmt, dst_palette, src_palette, blend);
      break;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL:
      func = wuffs_base__pixel_swizzler__prepare__bgra_nonpremul(
          p, dst_pixfmt, dst_palette, src_palette, blend);
      break;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL_4X16LE:
      func = wuffs_base__pixel_swizzler__prepare__bgra_nonpremul_4x16le(
          p, dst_pixfmt, dst_palette, src_palette, blend);
      break;

    case WUFFS_BASE__PIXEL_FORMAT__BGRA_PREMUL:
      func = wuffs_base__pixel_swizzler__prepare__bgra_premul(
          p, dst_pixfmt, dst_palette, src_palette, blend);
      break;

    case WUFFS_BASE__PIXEL_FORMAT__BGRX:
      func = wuffs_base__pixel_swizzler__prepare__bgrx(
          p, dst_pixfmt, dst_palette, src_palette, blend);
      break;

    case WUFFS_BASE__PIXEL_FORMAT__RGB:
      func = wuffs_base__pixel_swizzler__prepare__rgb(
          p, dst_pixfmt, dst_palette, src_palette, blend);
      break;

    case WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL:
      func = wuffs_base__pixel_swizzler__prepare__rgba_nonpremul(
          p, dst_pixfmt, dst_palette, src_palette, blend);
      break;

    case WUFFS_BASE__PIXEL_FORMAT__RGBA_PREMUL:
      func = wuffs_base__pixel_swizzler__prepare__rgba_premul(
          p, dst_pixfmt, dst_palette, src_palette, blend);
      break;
  }

  p->private_impl.func = func;
  p->private_impl.transparent_black_func = transparent_black_func;
  p->private_impl.dst_pixfmt_bytes_per_pixel = dst_pixfmt_bits_per_pixel / 8;
  p->private_impl.src_pixfmt_bytes_per_pixel = src_pixfmt_bits_per_pixel / 8;
  return wuffs_base__make_status(
      func ? NULL : wuffs_base__error__unsupported_pixel_swizzler_option);
}

WUFFS_BASE__MAYBE_STATIC uint64_t  //
wuffs_base__pixel_swizzler__limited_swizzle_u32_interleaved_from_reader(
    const wuffs_base__pixel_swizzler* p,
    uint32_t up_to_num_pixels,
    wuffs_base__slice_u8 dst,
    wuffs_base__slice_u8 dst_palette,
    const uint8_t** ptr_iop_r,
    const uint8_t* io2_r) {
  if (p && p->private_impl.func) {
    const uint8_t* iop_r = *ptr_iop_r;
    uint64_t src_len = wuffs_base__u64__min(
        ((uint64_t)up_to_num_pixels) *
            ((uint64_t)p->private_impl.src_pixfmt_bytes_per_pixel),
        ((uint64_t)(io2_r - iop_r)));
    uint64_t n =
        (*p->private_impl.func)(dst.ptr, dst.len, dst_palette.ptr,
                                dst_palette.len, iop_r, (size_t)src_len);
    *ptr_iop_r += n * p->private_impl.src_pixfmt_bytes_per_pixel;
    return n;
  }
  return 0;
}

WUFFS_BASE__MAYBE_STATIC uint64_t  //
wuffs_base__pixel_swizzler__swizzle_interleaved_from_reader(
    const wuffs_base__pixel_swizzler* p,
    wuffs_base__slice_u8 dst,
    wuffs_base__slice_u8 dst_palette,
    const uint8_t** ptr_iop_r,
    const uint8_t* io2_r) {
  if (p && p->private_impl.func) {
    const uint8_t* iop_r = *ptr_iop_r;
    uint64_t src_len = ((uint64_t)(io2_r - iop_r));
    uint64_t n =
        (*p->private_impl.func)(dst.ptr, dst.len, dst_palette.ptr,
                                dst_palette.len, iop_r, (size_t)src_len);
    *ptr_iop_r += n * p->private_impl.src_pixfmt_bytes_per_pixel;
    return n;
  }
  return 0;
}

WUFFS_BASE__MAYBE_STATIC uint64_t  //
wuffs_base__pixel_swizzler__swizzle_interleaved_from_slice(
    const wuffs_base__pixel_swizzler* p,
    wuffs_base__slice_u8 dst,
    wuffs_base__slice_u8 dst_palette,
    wuffs_base__slice_u8 src) {
  if (p && p->private_impl.func) {
    return (*p->private_impl.func)(dst.ptr, dst.len, dst_palette.ptr,
                                   dst_palette.len, src.ptr, src.len);
  }
  return 0;
}

WUFFS_BASE__MAYBE_STATIC uint64_t  //
wuffs_base__pixel_swizzler__swizzle_interleaved_transparent_black(
    const wuffs_base__pixel_swizzler* p,
    wuffs_base__slice_u8 dst,
    wuffs_base__slice_u8 dst_palette,
    uint64_t num_pixels) {
  if (p && p->private_impl.transparent_black_func) {
    return (*p->private_impl.transparent_black_func)(
        dst.ptr, dst.len, dst_palette.ptr, dst_palette.len, num_pixels,
        p->private_impl.dst_pixfmt_bytes_per_pixel);
  }
  return 0;
}

#endif  // !defined(WUFFS_CONFIG__MODULES) ||
        // defined(WUFFS_CONFIG__MODULE__BASE) ||
        // defined(WUFFS_CONFIG__MODULE__BASE__PIXCONV)

#if !defined(WUFFS_CONFIG__MODULES) || defined(WUFFS_CONFIG__MODULE__BASE) || \
    defined(WUFFS_CONFIG__MODULE__BASE__UTF8)

// ---------------- Unicode and UTF-8

WUFFS_BASE__MAYBE_STATIC size_t  //
wuffs_base__utf_8__encode(wuffs_base__slice_u8 dst, uint32_t code_point) {
  if (code_point <= 0x7F) {
    if (dst.len >= 1) {
      dst.ptr[0] = (uint8_t)(code_point);
      return 1;
    }

  } else if (code_point <= 0x07FF) {
    if (dst.len >= 2) {
      dst.ptr[0] = (uint8_t)(0xC0 | ((code_point >> 6)));
      dst.ptr[1] = (uint8_t)(0x80 | ((code_point >> 0) & 0x3F));
      return 2;
    }

  } else if (code_point <= 0xFFFF) {
    if ((dst.len >= 3) && ((code_point < 0xD800) || (0xDFFF < code_point))) {
      dst.ptr[0] = (uint8_t)(0xE0 | ((code_point >> 12)));
      dst.ptr[1] = (uint8_t)(0x80 | ((code_point >> 6) & 0x3F));
      dst.ptr[2] = (uint8_t)(0x80 | ((code_point >> 0) & 0x3F));
      return 3;
    }

  } else if (code_point <= 0x10FFFF) {
    if (dst.len >= 4) {
      dst.ptr[0] = (uint8_t)(0xF0 | ((code_point >> 18)));
      dst.ptr[1] = (uint8_t)(0x80 | ((code_point >> 12) & 0x3F));
      dst.ptr[2] = (uint8_t)(0x80 | ((code_point >> 6) & 0x3F));
      dst.ptr[3] = (uint8_t)(0x80 | ((code_point >> 0) & 0x3F));
      return 4;
    }
  }

  return 0;
}

// wuffs_base__utf_8__byte_length_minus_1 is the byte length (minus 1) of a
// UTF-8 encoded code point, based on the encoding's initial byte.
//  - 0x00 is 1-byte UTF-8 (ASCII).
//  - 0x01 is the start of 2-byte UTF-8.
//  - 0x02 is the start of 3-byte UTF-8.
//  - 0x03 is the start of 4-byte UTF-8.
//  - 0x40 is a UTF-8 tail byte.
//  - 0x80 is invalid UTF-8.
//
// RFC 3629 (UTF-8) gives this grammar for valid UTF-8:
//    UTF8-1      = %x00-7F
//    UTF8-2      = %xC2-DF UTF8-tail
//    UTF8-3      = %xE0 %xA0-BF UTF8-tail / %xE1-EC 2( UTF8-tail ) /
//                  %xED %x80-9F UTF8-tail / %xEE-EF 2( UTF8-tail )
//    UTF8-4      = %xF0 %x90-BF 2( UTF8-tail ) / %xF1-F3 3( UTF8-tail ) /
//                  %xF4 %x80-8F 2( UTF8-tail )
//    UTF8-tail   = %x80-BF
static const uint8_t wuffs_base__utf_8__byte_length_minus_1[256] = {
    // 0     1     2     3     4     5     6     7
    // 8     9     A     B     C     D     E     F
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x00 ..= 0x07.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x08 ..= 0x0F.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x10 ..= 0x17.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x18 ..= 0x1F.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x20 ..= 0x27.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x28 ..= 0x2F.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x30 ..= 0x37.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x38 ..= 0x3F.

    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x40 ..= 0x47.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x48 ..= 0x4F.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x50 ..= 0x57.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x58 ..= 0x5F.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x60 ..= 0x67.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x68 ..= 0x6F.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x70 ..= 0x77.
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // 0x78 ..= 0x7F.

    0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  // 0x80 ..= 0x87.
    0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  // 0x88 ..= 0x8F.
    0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  // 0x90 ..= 0x97.
    0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  // 0x98 ..= 0x9F.
    0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  // 0xA0 ..= 0xA7.
    0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  // 0xA8 ..= 0xAF.
    0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  // 0xB0 ..= 0xB7.
    0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  // 0xB8 ..= 0xBF.

    0x80, 0x80, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,  // 0xC0 ..= 0xC7.
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,  // 0xC8 ..= 0xCF.
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,  // 0xD0 ..= 0xD7.
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,  // 0xD8 ..= 0xDF.
    0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,  // 0xE0 ..= 0xE7.
    0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,  // 0xE8 ..= 0xEF.
    0x03, 0x03, 0x03, 0x03, 0x03, 0x80, 0x80, 0x80,  // 0xF0 ..= 0xF7.
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,  // 0xF8 ..= 0xFF.
    // 0     1     2     3     4     5     6     7
    // 8     9     A     B     C     D     E     F
};

WUFFS_BASE__MAYBE_STATIC wuffs_base__utf_8__next__output  //
wuffs_base__utf_8__next(const uint8_t* s_ptr, size_t s_len) {
  if (s_len == 0) {
    return wuffs_base__make_utf_8__next__output(0, 0);
  }
  uint32_t c = s_ptr[0];
  switch (wuffs_base__utf_8__byte_length_minus_1[c & 0xFF]) {
    case 0:
      return wuffs_base__make_utf_8__next__output(c, 1);

    case 1:
      if (s_len < 2) {
        break;
      }
      c = wuffs_base__peek_u16le__no_bounds_check(s_ptr);
      if ((c & 0xC000) != 0x8000) {
        break;
      }
      c = (0x0007C0 & (c << 6)) | (0x00003F & (c >> 8));
      return wuffs_base__make_utf_8__next__output(c, 2);

    case 2:
      if (s_len < 3) {
        break;
      }
      c = wuffs_base__peek_u24le__no_bounds_check(s_ptr);
      if ((c & 0xC0C000) != 0x808000) {
        break;
      }
      c = (0x00F000 & (c << 12)) | (0x000FC0 & (c >> 2)) |
          (0x00003F & (c >> 16));
      if ((c <= 0x07FF) || ((0xD800 <= c) && (c <= 0xDFFF))) {
        break;
      }
      return wuffs_base__make_utf_8__next__output(c, 3);

    case 3:
      if (s_len < 4) {
        break;
      }
      c = wuffs_base__peek_u32le__no_bounds_check(s_ptr);
      if ((c & 0xC0C0C000) != 0x80808000) {
        break;
      }
      c = (0x1C0000 & (c << 18)) | (0x03F000 & (c << 4)) |
          (0x000FC0 & (c >> 10)) | (0x00003F & (c >> 24));
      if ((c <= 0xFFFF) || (0x110000 <= c)) {
        break;
      }
      return wuffs_base__make_utf_8__next__output(c, 4);
  }

  return wuffs_base__make_utf_8__next__output(
      WUFFS_BASE__UNICODE_REPLACEMENT_CHARACTER, 1);
}

WUFFS_BASE__MAYBE_STATIC wuffs_base__utf_8__next__output  //
wuffs_base__utf_8__next_from_end(const uint8_t* s_ptr, size_t s_len) {
  if (s_len == 0) {
    return wuffs_base__make_utf_8__next__output(0, 0);
  }
  const uint8_t* ptr = &s_ptr[s_len - 1];
  if (*ptr < 0x80) {
    return wuffs_base__make_utf_8__next__output(*ptr, 1);

  } else if (*ptr < 0xC0) {
    const uint8_t* too_far = &s_ptr[(s_len > 4) ? (s_len - 4) : 0];
    uint32_t n = 1;
    while (ptr != too_far) {
      ptr--;
      n++;
      if (*ptr < 0x80) {
        break;
      } else if (*ptr < 0xC0) {
        continue;
      }
      wuffs_base__utf_8__next__output o = wuffs_base__utf_8__next(ptr, n);
      if (o.byte_length != n) {
        break;
      }
      return o;
    }
  }

  return wuffs_base__make_utf_8__next__output(
      WUFFS_BASE__UNICODE_REPLACEMENT_CHARACTER, 1);
}

WUFFS_BASE__MAYBE_STATIC size_t  //
wuffs_base__utf_8__longest_valid_prefix(const uint8_t* s_ptr, size_t s_len) {
  // TODO: possibly optimize the all-ASCII case (4 or 8 bytes at a time).
  //
  // TODO: possibly optimize this by manually inlining the
  // wuffs_base__utf_8__next calls.
  size_t original_len = s_len;
  while (s_len > 0) {
    wuffs_base__utf_8__next__output o = wuffs_base__utf_8__next(s_ptr, s_len);
    if ((o.code_point > 0x7F) && (o.byte_length == 1)) {
      break;
    }
    s_ptr += o.byte_length;
    s_len -= o.byte_length;
  }
  return original_len - s_len;
}

WUFFS_BASE__MAYBE_STATIC size_t  //
wuffs_base__ascii__longest_valid_prefix(const uint8_t* s_ptr, size_t s_len) {
  // TODO: possibly optimize this by checking 4 or 8 bytes at a time.
  const uint8_t* original_ptr = s_ptr;
  const uint8_t* p = s_ptr;
  const uint8_t* q = s_ptr + s_len;
  for (; (p != q) && ((*p & 0x80) == 0); p++) {
  }
  return (size_t)(p - original_ptr);
}

#endif  // !defined(WUFFS_CONFIG__MODULES) ||
        // defined(WUFFS_CONFIG__MODULE__BASE) ||
        // defined(WUFFS_CONFIG__MODULE__BASE__UTF8)

#ifdef __cplusplus
}  // extern "C"
#endif

#if !defined(WUFFS_CONFIG__MODULES) || defined(WUFFS_CONFIG__MODULE__ADLER32)

// ---------------- Status Codes Implementations

// ---------------- Private Consts

// ---------------- Private Initializer Prototypes

// ---------------- Private Function Prototypes

static wuffs_base__empty_struct
wuffs_adler32__hasher__up(
    wuffs_adler32__hasher* self,
    wuffs_base__slice_u8 a_x);

static wuffs_base__empty_struct
wuffs_adler32__hasher__up__choosy_default(
    wuffs_adler32__hasher* self,
    wuffs_base__slice_u8 a_x);

#if defined(WUFFS_BASE__CPU_ARCH__ARM_NEON)
static wuffs_base__empty_struct
wuffs_adler32__hasher__up_arm_neon(
    wuffs_adler32__hasher* self,
    wuffs_base__slice_u8 a_x);
#endif  // defined(WUFFS_BASE__CPU_ARCH__ARM_NEON)

#if defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
static wuffs_base__empty_struct
wuffs_adler32__hasher__up_x86_sse42(
    wuffs_adler32__hasher* self,
    wuffs_base__slice_u8 a_x);
#endif  // defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)

// ---------------- VTables

const wuffs_base__hasher_u32__func_ptrs
wuffs_adler32__hasher__func_ptrs_for__wuffs_base__hasher_u32 = {
  (wuffs_base__empty_struct(*)(void*,
      uint32_t,
      bool))(&wuffs_adler32__hasher__set_quirk_enabled),
  (uint32_t(*)(void*,
      wuffs_base__slice_u8))(&wuffs_adler32__hasher__update_u32),
};

// ---------------- Initializer Implementations

wuffs_base__status WUFFS_BASE__WARN_UNUSED_RESULT
wuffs_adler32__hasher__initialize(
    wuffs_adler32__hasher* self,
    size_t sizeof_star_self,
    uint64_t wuffs_version,
    uint32_t options){
  if (!self) {
    return wuffs_base__make_status(wuffs_base__error__bad_receiver);
  }
  if (sizeof(*self) != sizeof_star_self) {
    return wuffs_base__make_status(wuffs_base__error__bad_sizeof_receiver);
  }
  if (((wuffs_version >> 32) != WUFFS_VERSION_MAJOR) ||
      (((wuffs_version >> 16) & 0xFFFF) > WUFFS_VERSION_MINOR)) {
    return wuffs_base__make_status(wuffs_base__error__bad_wuffs_version);
  }

  if ((options & WUFFS_INITIALIZE__ALREADY_ZEROED) != 0) {
    // The whole point of this if-check is to detect an uninitialized *self.
    // We disable the warning on GCC. Clang-5.0 does not have this warning.
#if !defined(__clang__) && defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
    if (self->private_impl.magic != 0) {
      return wuffs_base__make_status(wuffs_base__error__initialize_falsely_claimed_already_zeroed);
    }
#if !defined(__clang__) && defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
  } else {
    if ((options & WUFFS_INITIALIZE__LEAVE_INTERNAL_BUFFERS_UNINITIALIZED) == 0) {
      memset(self, 0, sizeof(*self));
      options |= WUFFS_INITIALIZE__ALREADY_ZEROED;
    } else {
      memset(&(self->private_impl), 0, sizeof(self->private_impl));
    }
  }

  self->private_impl.choosy_up = &wuffs_adler32__hasher__up__choosy_default;

  self->private_impl.magic = WUFFS_BASE__MAGIC;
  self->private_impl.vtable_for__wuffs_base__hasher_u32.vtable_name =
      wuffs_base__hasher_u32__vtable_name;
  self->private_impl.vtable_for__wuffs_base__hasher_u32.function_pointers =
      (const void*)(&wuffs_adler32__hasher__func_ptrs_for__wuffs_base__hasher_u32);
  return wuffs_base__make_status(NULL);
}

wuffs_adler32__hasher*
wuffs_adler32__hasher__alloc() {
  wuffs_adler32__hasher* x =
      (wuffs_adler32__hasher*)(calloc(sizeof(wuffs_adler32__hasher), 1));
  if (!x) {
    return NULL;
  }
  if (wuffs_adler32__hasher__initialize(
      x, sizeof(wuffs_adler32__hasher), WUFFS_VERSION, WUFFS_INITIALIZE__ALREADY_ZEROED).repr) {
    free(x);
    return NULL;
  }
  return x;
}

size_t
sizeof__wuffs_adler32__hasher() {
  return sizeof(wuffs_adler32__hasher);
}

// ---------------- Function Implementations

// -------- func adler32.hasher.set_quirk_enabled

WUFFS_BASE__MAYBE_STATIC wuffs_base__empty_struct
wuffs_adler32__hasher__set_quirk_enabled(
    wuffs_adler32__hasher* self,
    uint32_t a_quirk,
    bool a_enabled) {
  return wuffs_base__make_empty_struct();
}

// -------- func adler32.hasher.update_u32

WUFFS_BASE__MAYBE_STATIC uint32_t
wuffs_adler32__hasher__update_u32(
    wuffs_adler32__hasher* self,
    wuffs_base__slice_u8 a_x) {
  if (!self) {
    return 0;
  }
  if (self->private_impl.magic != WUFFS_BASE__MAGIC) {
    return 0;
  }

  if ( ! self->private_impl.f_started) {
    self->private_impl.f_started = true;
    self->private_impl.f_state = 1;
    self->private_impl.choosy_up = (
#if defined(WUFFS_BASE__CPU_ARCH__ARM_NEON)
        wuffs_base__cpu_arch__have_arm_neon() ? &wuffs_adler32__hasher__up_arm_neon :
#endif
#if defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
        wuffs_base__cpu_arch__have_x86_sse42() ? &wuffs_adler32__hasher__up_x86_sse42 :
#endif
        self->private_impl.choosy_up);
  }
  wuffs_adler32__hasher__up(self, a_x);
  return self->private_impl.f_state;
}

// -------- func adler32.hasher.up

static wuffs_base__empty_struct
wuffs_adler32__hasher__up(
    wuffs_adler32__hasher* self,
    wuffs_base__slice_u8 a_x) {
  return (*self->private_impl.choosy_up)(self, a_x);
}

static wuffs_base__empty_struct
wuffs_adler32__hasher__up__choosy_default(
    wuffs_adler32__hasher* self,
    wuffs_base__slice_u8 a_x) {
  uint32_t v_s1 = 0;
  uint32_t v_s2 = 0;
  wuffs_base__slice_u8 v_remaining = {0};
  wuffs_base__slice_u8 v_p = {0};

  v_s1 = ((self->private_impl.f_state) & 0xFFFF);
  v_s2 = ((self->private_impl.f_state) >> (32 - (16)));
  while (((uint64_t)(a_x.len)) > 0) {
    v_remaining = wuffs_base__slice_u8__subslice_j(a_x, 0);
    if (((uint64_t)(a_x.len)) > 5552) {
      v_remaining = wuffs_base__slice_u8__subslice_i(a_x, 5552);
      a_x = wuffs_base__slice_u8__subslice_j(a_x, 5552);
    }
    {
      wuffs_base__slice_u8 i_slice_p = a_x;
      v_p.ptr = i_slice_p.ptr;
      v_p.len = 1;
      uint8_t* i_end0_p = v_p.ptr + (((i_slice_p.len - (size_t)(v_p.ptr - i_slice_p.ptr)) / 8) * 8);
      while (v_p.ptr < i_end0_p) {
        v_s1 += ((uint32_t)(v_p.ptr[0]));
        v_s2 += v_s1;
        v_p.ptr += 1;
        v_s1 += ((uint32_t)(v_p.ptr[0]));
        v_s2 += v_s1;
        v_p.ptr += 1;
        v_s1 += ((uint32_t)(v_p.ptr[0]));
        v_s2 += v_s1;
        v_p.ptr += 1;
        v_s1 += ((uint32_t)(v_p.ptr[0]));
        v_s2 += v_s1;
        v_p.ptr += 1;
        v_s1 += ((uint32_t)(v_p.ptr[0]));
        v_s2 += v_s1;
        v_p.ptr += 1;
        v_s1 += ((uint32_t)(v_p.ptr[0]));
        v_s2 += v_s1;
        v_p.ptr += 1;
        v_s1 += ((uint32_t)(v_p.ptr[0]));
        v_s2 += v_s1;
        v_p.ptr += 1;
        v_s1 += ((uint32_t)(v_p.ptr[0]));
        v_s2 += v_s1;
        v_p.ptr += 1;
      }
      v_p.len = 1;
      uint8_t* i_end1_p = i_slice_p.ptr + i_slice_p.len;
      while (v_p.ptr < i_end1_p) {
        v_s1 += ((uint32_t)(v_p.ptr[0]));
        v_s2 += v_s1;
        v_p.ptr += 1;
      }
      v_p.len = 0;
    }
    v_s1 %= 65521;
    v_s2 %= 65521;
    a_x = v_remaining;
  }
  self->private_impl.f_state = (((v_s2 & 65535) << 16) | (v_s1 & 65535));
  return wuffs_base__make_empty_struct();
}

// ‼ WUFFS MULTI-FILE SECTION +arm_neon
// -------- func adler32.hasher.up_arm_neon

#if defined(WUFFS_BASE__CPU_ARCH__ARM_NEON)
static wuffs_base__empty_struct
wuffs_adler32__hasher__up_arm_neon(
    wuffs_adler32__hasher* self,
    wuffs_base__slice_u8 a_x) {
  uint32_t v_s1 = 0;
  uint32_t v_s2 = 0;
  wuffs_base__slice_u8 v_remaining = {0};
  wuffs_base__slice_u8 v_p = {0};
  uint8x16_t v_p__left = {0};
  uint8x16_t v_p_right = {0};
  uint32x4_t v_v1 = {0};
  uint32x4_t v_v2 = {0};
  uint16x8_t v_col0 = {0};
  uint16x8_t v_col1 = {0};
  uint16x8_t v_col2 = {0};
  uint16x8_t v_col3 = {0};
  uint32x2_t v_sum1 = {0};
  uint32x2_t v_sum2 = {0};
  uint32x2_t v_sum12 = {0};
  uint32_t v_num_iterate_bytes = 0;
  uint64_t v_tail_index = 0;

  v_s1 = ((self->private_impl.f_state) & 0xFFFF);
  v_s2 = ((self->private_impl.f_state) >> (32 - (16)));
  while ((((uint64_t)(a_x.len)) > 0) && ((15 & ((uint32_t)(0xFFF & (uintptr_t)(a_x.ptr)))) != 0)) {
    v_s1 += ((uint32_t)(a_x.ptr[0]));
    v_s2 += v_s1;
    a_x = wuffs_base__slice_u8__subslice_i(a_x, 1);
  }
  v_s1 %= 65521;
  v_s2 %= 65521;
  while (((uint64_t)(a_x.len)) > 0) {
    v_remaining = wuffs_base__slice_u8__subslice_j(a_x, 0);
    if (((uint64_t)(a_x.len)) > 5536) {
      v_remaining = wuffs_base__slice_u8__subslice_i(a_x, 5536);
      a_x = wuffs_base__slice_u8__subslice_j(a_x, 5536);
    }
    v_num_iterate_bytes = ((uint32_t)((((uint64_t)(a_x.len)) & 4294967264)));
    v_s2 += ((uint32_t)(v_s1 * v_num_iterate_bytes));
    v_v1 = vdupq_n_u32(0);
    v_v2 = vdupq_n_u32(0);
    v_col0 = vdupq_n_u16(0);
    v_col1 = vdupq_n_u16(0);
    v_col2 = vdupq_n_u16(0);
    v_col3 = vdupq_n_u16(0);
    {
      wuffs_base__slice_u8 i_slice_p = a_x;
      v_p.ptr = i_slice_p.ptr;
      v_p.len = 32;
      uint8_t* i_end0_p = v_p.ptr + (((i_slice_p.len - (size_t)(v_p.ptr - i_slice_p.ptr)) / 32) * 32);
      while (v_p.ptr < i_end0_p) {
        v_p__left = vld1q_u8(v_p.ptr);
        v_p_right = vld1q_u8(v_p.ptr + 16);
        v_v2 = vaddq_u32(v_v2, v_v1);
        v_v1 = vpadalq_u16(v_v1, vpadalq_u8(vpaddlq_u8(v_p__left), v_p_right));
        v_col0 = vaddw_u8(v_col0, vget_low_u8(v_p__left));
        v_col1 = vaddw_u8(v_col1, vget_high_u8(v_p__left));
        v_col2 = vaddw_u8(v_col2, vget_low_u8(v_p_right));
        v_col3 = vaddw_u8(v_col3, vget_high_u8(v_p_right));
        v_p.ptr += 32;
      }
      v_p.len = 0;
    }
    v_v2 = vshlq_n_u32(v_v2, 5);
    v_v2 = vmlal_u16(v_v2, vget_low_u16(v_col0), ((uint16x4_t){32, 31, 30, 29}));
    v_v2 = vmlal_u16(v_v2, vget_high_u16(v_col0), ((uint16x4_t){28, 27, 26, 25}));
    v_v2 = vmlal_u16(v_v2, vget_low_u16(v_col1), ((uint16x4_t){24, 23, 22, 21}));
    v_v2 = vmlal_u16(v_v2, vget_high_u16(v_col1), ((uint16x4_t){20, 19, 18, 17}));
    v_v2 = vmlal_u16(v_v2, vget_low_u16(v_col2), ((uint16x4_t){16, 15, 14, 13}));
    v_v2 = vmlal_u16(v_v2, vget_high_u16(v_col2), ((uint16x4_t){12, 11, 10, 9}));
    v_v2 = vmlal_u16(v_v2, vget_low_u16(v_col3), ((uint16x4_t){8, 7, 6, 5}));
    v_v2 = vmlal_u16(v_v2, vget_high_u16(v_col3), ((uint16x4_t){4, 3, 2, 1}));
    v_sum1 = vpadd_u32(vget_low_u32(v_v1), vget_high_u32(v_v1));
    v_sum2 = vpadd_u32(vget_low_u32(v_v2), vget_high_u32(v_v2));
    v_sum12 = vpadd_u32(v_sum1, v_sum2);
    v_s1 += vget_lane_u32(v_sum12, 0);
    v_s2 += vget_lane_u32(v_sum12, 1);
    v_tail_index = (((uint64_t)(a_x.len)) & 18446744073709551584u);
    if (v_tail_index < ((uint64_t)(a_x.len))) {
      {
        wuffs_base__slice_u8 i_slice_p = wuffs_base__slice_u8__subslice_i(a_x, v_tail_index);
        v_p.ptr = i_slice_p.ptr;
        v_p.len = 1;
        uint8_t* i_end0_p = i_slice_p.ptr + i_slice_p.len;
        while (v_p.ptr < i_end0_p) {
          v_s1 += ((uint32_t)(v_p.ptr[0]));
          v_s2 += v_s1;
          v_p.ptr += 1;
        }
        v_p.len = 0;
      }
    }
    v_s1 %= 65521;
    v_s2 %= 65521;
    a_x = v_remaining;
  }
  self->private_impl.f_state = (((v_s2 & 65535) << 16) | (v_s1 & 65535));
  return wuffs_base__make_empty_struct();
}
#endif  // defined(WUFFS_BASE__CPU_ARCH__ARM_NEON)
// ‼ WUFFS MULTI-FILE SECTION -arm_neon

// ‼ WUFFS MULTI-FILE SECTION +x86_sse42
// -------- func adler32.hasher.up_x86_sse42

#if defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
WUFFS_BASE__MAYBE_ATTRIBUTE_TARGET("pclmul,popcnt,sse4.2")
static wuffs_base__empty_struct
wuffs_adler32__hasher__up_x86_sse42(
    wuffs_adler32__hasher* self,
    wuffs_base__slice_u8 a_x) {
  uint32_t v_s1 = 0;
  uint32_t v_s2 = 0;
  wuffs_base__slice_u8 v_remaining = {0};
  wuffs_base__slice_u8 v_p = {0};
  __m128i v_zeroes = {0};
  __m128i v_ones = {0};
  __m128i v_weights__left = {0};
  __m128i v_weights_right = {0};
  __m128i v_q__left = {0};
  __m128i v_q_right = {0};
  __m128i v_v1 = {0};
  __m128i v_v2 = {0};
  __m128i v_v2j = {0};
  __m128i v_v2k = {0};
  uint32_t v_num_iterate_bytes = 0;
  uint64_t v_tail_index = 0;

  v_zeroes = _mm_set1_epi16((int16_t)(0));
  v_ones = _mm_set1_epi16((int16_t)(1));
  v_weights__left = _mm_set_epi8((int8_t)(17), (int8_t)(18), (int8_t)(19), (int8_t)(20), (int8_t)(21), (int8_t)(22), (int8_t)(23), (int8_t)(24), (int8_t)(25), (int8_t)(26), (int8_t)(27), (int8_t)(28), (int8_t)(29), (int8_t)(30), (int8_t)(31), (int8_t)(32));
  v_weights_right = _mm_set_epi8((int8_t)(1), (int8_t)(2), (int8_t)(3), (int8_t)(4), (int8_t)(5), (int8_t)(6), (int8_t)(7), (int8_t)(8), (int8_t)(9), (int8_t)(10), (int8_t)(11), (int8_t)(12), (int8_t)(13), (int8_t)(14), (int8_t)(15), (int8_t)(16));
  v_s1 = ((self->private_impl.f_state) & 0xFFFF);
  v_s2 = ((self->private_impl.f_state) >> (32 - (16)));
  while (((uint64_t)(a_x.len)) > 0) {
    v_remaining = wuffs_base__slice_u8__subslice_j(a_x, 0);
    if (((uint64_t)(a_x.len)) > 5536) {
      v_remaining = wuffs_base__slice_u8__subslice_i(a_x, 5536);
      a_x = wuffs_base__slice_u8__subslice_j(a_x, 5536);
    }
    v_num_iterate_bytes = ((uint32_t)((((uint64_t)(a_x.len)) & 4294967264)));
    v_s2 += ((uint32_t)(v_s1 * v_num_iterate_bytes));
    v_v1 = _mm_setzero_si128();
    v_v2j = _mm_setzero_si128();
    v_v2k = _mm_setzero_si128();
    {
      wuffs_base__slice_u8 i_slice_p = a_x;
      v_p.ptr = i_slice_p.ptr;
      v_p.len = 32;
      uint8_t* i_end0_p = v_p.ptr + (((i_slice_p.len - (size_t)(v_p.ptr - i_slice_p.ptr)) / 32) * 32);
      while (v_p.ptr < i_end0_p) {
        v_q__left = _mm_lddqu_si128((const __m128i*)(const void*)(v_p.ptr));
        v_q_right = _mm_lddqu_si128((const __m128i*)(const void*)(v_p.ptr + 16));
        v_v2j = _mm_add_epi32(v_v2j, v_v1);
        v_v1 = _mm_add_epi32(v_v1, _mm_sad_epu8(v_q__left, v_zeroes));
        v_v1 = _mm_add_epi32(v_v1, _mm_sad_epu8(v_q_right, v_zeroes));
        v_v2k = _mm_add_epi32(v_v2k, _mm_madd_epi16(v_ones, _mm_maddubs_epi16(v_q__left, v_weights__left)));
        v_v2k = _mm_add_epi32(v_v2k, _mm_madd_epi16(v_ones, _mm_maddubs_epi16(v_q_right, v_weights_right)));
        v_p.ptr += 32;
      }
      v_p.len = 0;
    }
    v_v1 = _mm_add_epi32(v_v1, _mm_shuffle_epi32(v_v1, (int32_t)(177)));
    v_v1 = _mm_add_epi32(v_v1, _mm_shuffle_epi32(v_v1, (int32_t)(78)));
    v_s1 += ((uint32_t)(_mm_cvtsi128_si32(v_v1)));
    v_v2 = _mm_add_epi32(v_v2k, _mm_slli_epi32(v_v2j, (int32_t)(5)));
    v_v2 = _mm_add_epi32(v_v2, _mm_shuffle_epi32(v_v2, (int32_t)(177)));
    v_v2 = _mm_add_epi32(v_v2, _mm_shuffle_epi32(v_v2, (int32_t)(78)));
    v_s2 += ((uint32_t)(_mm_cvtsi128_si32(v_v2)));
    v_tail_index = (((uint64_t)(a_x.len)) & 18446744073709551584u);
    if (v_tail_index < ((uint64_t)(a_x.len))) {
      {
        wuffs_base__slice_u8 i_slice_p = wuffs_base__slice_u8__subslice_i(a_x, v_tail_index);
        v_p.ptr = i_slice_p.ptr;
        v_p.len = 1;
        uint8_t* i_end0_p = i_slice_p.ptr + i_slice_p.len;
        while (v_p.ptr < i_end0_p) {
          v_s1 += ((uint32_t)(v_p.ptr[0]));
          v_s2 += v_s1;
          v_p.ptr += 1;
        }
        v_p.len = 0;
      }
    }
    v_s1 %= 65521;
    v_s2 %= 65521;
    a_x = v_remaining;
  }
  self->private_impl.f_state = (((v_s2 & 65535) << 16) | (v_s1 & 65535));
  return wuffs_base__make_empty_struct();
}
#endif  // defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
// ‼ WUFFS MULTI-FILE SECTION -x86_sse42

#endif  // !defined(WUFFS_CONFIG__MODULES) || defined(WUFFS_CONFIG__MODULE__ADLER32)

#if !defined(WUFFS_CONFIG__MODULES) || defined(WUFFS_CONFIG__MODULE__CRC32)

// ---------------- Status Codes Implementations

// ---------------- Private Consts

static const uint32_t
WUFFS_CRC32__IEEE_TABLE[16][256] WUFFS_BASE__POTENTIALLY_UNUSED = {
  {
    0, 1996959894, 3993919788, 2567524794, 124634137, 1886057615, 3915621685, 2657392035,
    249268274, 2044508324, 3772115230, 2547177864, 162941995, 2125561021, 3887607047, 2428444049,
    498536548, 1789927666, 4089016648, 2227061214, 450548861, 1843258603, 4107580753, 2211677639,
    325883990, 1684777152, 4251122042, 2321926636, 335633487, 1661365465, 4195302755, 2366115317,
    997073096, 1281953886, 3579855332, 2724688242, 1006888145, 1258607687, 3524101629, 2768942443,
    901097722, 1119000684, 3686517206, 2898065728, 853044451, 1172266101, 3705015759, 2882616665,
    651767980, 1373503546, 3369554304, 3218104598, 565507253, 1454621731, 3485111705, 3099436303,
    671266974, 1594198024, 3322730930, 2970347812, 795835527, 1483230225, 3244367275, 3060149565,
    1994146192, 31158534, 2563907772, 4023717930, 1907459465, 112637215, 2680153253, 3904427059,
    2013776290, 251722036, 2517215374, 3775830040, 2137656763, 141376813, 2439277719, 3865271297,
    1802195444, 476864866, 2238001368, 4066508878, 1812370925, 453092731, 2181625025, 4111451223,
    1706088902, 314042704, 2344532202, 4240017532, 1658658271, 366619977, 2362670323, 4224994405,
    1303535960, 984961486, 2747007092, 3569037538, 1256170817, 1037604311, 2765210733, 3554079995,
    1131014506, 879679996, 2909243462, 3663771856, 1141124467, 855842277, 2852801631, 3708648649,
    1342533948, 654459306, 3188396048, 3373015174, 1466479909, 544179635, 3110523913, 3462522015,
    1591671054, 702138776, 2966460450, 3352799412, 1504918807, 783551873, 3082640443, 3233442989,
    3988292384, 2596254646, 62317068, 1957810842, 3939845945, 2647816111, 81470997, 1943803523,
    3814918930, 2489596804, 225274430, 2053790376, 3826175755, 2466906013, 167816743, 2097651377,
    4027552580, 2265490386, 503444072, 1762050814, 4150417245, 2154129355, 426522225, 1852507879,
    4275313526, 2312317920, 282753626, 1742555852, 4189708143, 2394877945, 397917763, 1622183637,
    3604390888, 2714866558, 953729732, 1340076626, 3518719985, 2797360999, 1068828381, 1219638859,
    3624741850, 2936675148, 906185462, 1090812512, 3747672003, 2825379669, 829329135, 1181335161,
    3412177804, 3160834842, 628085408, 1382605366, 3423369109, 3138078467, 570562233, 1426400815,
    3317316542, 2998733608, 733239954, 1555261956, 3268935591, 3050360625, 752459403, 1541320221,
    2607071920, 3965973030, 1969922972, 40735498, 2617837225, 3943577151, 1913087877, 83908371,
    2512341634, 3803740692, 2075208622, 213261112, 2463272603, 3855990285, 2094854071, 198958881,
    2262029012, 4057260610, 1759359992, 534414190, 2176718541, 4139329115, 1873836001, 414664567,
    2282248934, 4279200368, 1711684554, 285281116, 2405801727, 4167216745, 1634467795, 376229701,
    2685067896, 3608007406, 1308918612, 956543938, 2808555105, 3495958263, 1231636301, 1047427035,
    2932959818, 3654703836, 1088359270, 936918000, 2847714899, 3736837829, 1202900863, 817233897,
    3183342108, 3401237130, 1404277552, 615818150, 3134207493, 3453421203, 1423857449, 601450431,
    3009837614, 3294710456, 1567103746, 711928724, 3020668471, 3272380065, 1510334235, 755167117,
  }, {
    0, 421212481, 842424962, 724390851, 1684849924, 2105013317, 1448781702, 1329698503,
    3369699848, 3519200073, 4210026634, 3824474571, 2897563404, 3048111693, 2659397006, 2274893007,
    1254232657, 1406739216, 2029285587, 1643069842, 783210325, 934667796, 479770071, 92505238,
    2182846553, 2600511768, 2955803355, 2838940570, 3866582365, 4285295644, 3561045983, 3445231262,
    2508465314, 2359236067, 2813478432, 3198777185, 4058571174, 3908292839, 3286139684, 3670389349,
    1566420650, 1145479147, 1869335592, 1987116393, 959540142, 539646703, 185010476, 303839341,
    3745920755, 3327985586, 3983561841, 4100678960, 3140154359, 2721170102, 2300350837, 2416418868,
    396344571, 243568058, 631889529, 1018359608, 1945336319, 1793607870, 1103436669, 1490954812,
    4034481925, 3915546180, 3259968903, 3679722694, 2484439553, 2366552896, 2787371139, 3208174018,
    950060301, 565965900, 177645455, 328046286, 1556873225, 1171730760, 1861902987, 2011255754,
    3132841300, 2745199637, 2290958294, 2442530455, 3738671184, 3352078609, 3974232786, 4126854035,
    1919080284, 1803150877, 1079293406, 1498383519, 370020952, 253043481, 607678682, 1025720731,
    1711106983, 2095471334, 1472923941, 1322268772, 26324643, 411738082, 866634785, 717028704,
    2904875439, 3024081134, 2668790573, 2248782444, 3376948395, 3495106026, 4219356713, 3798300520,
    792689142, 908347575, 487136116, 68299317, 1263779058, 1380486579, 2036719216, 1618931505,
    3890672638, 4278043327, 3587215740, 3435896893, 2206873338, 2593195963, 2981909624, 2829542713,
    998479947, 580430090, 162921161, 279890824, 1609522511, 1190423566, 1842954189, 1958874764,
    4082766403, 3930137346, 3245109441, 3631694208, 2536953671, 2385372678, 2768287173, 3155920004,
    1900120602, 1750776667, 1131931800, 1517083097, 355290910, 204897887, 656092572, 1040194781,
    3113746450, 2692952403, 2343461520, 2461357009, 3723805974, 3304059991, 4022511508, 4141455061,
    2919742697, 3072101800, 2620513899, 2234183466, 3396041197, 3547351212, 4166851439, 3779471918,
    1725839073, 2143618976, 1424512099, 1307796770, 45282277, 464110244, 813994343, 698327078,
    3838160568, 4259225593, 3606301754, 3488152955, 2158586812, 2578602749, 2996767038, 2877569151,
    740041904, 889656817, 506086962, 120682355, 1215357364, 1366020341, 2051441462, 1667084919,
    3422213966, 3538019855, 4190942668, 3772220557, 2945847882, 3062702859, 2644537544, 2226864521,
    52649286, 439905287, 823476164, 672009861, 1733269570, 2119477507, 1434057408, 1281543041,
    2167981343, 2552493150, 3004082077, 2853541596, 3847487515, 4233048410, 3613549209, 3464057816,
    1239502615, 1358593622, 2077699477, 1657543892, 764250643, 882293586, 532408465, 111204816,
    1585378284, 1197851309, 1816695150, 1968414767, 974272232, 587794345, 136598634, 289367339,
    2527558116, 2411481253, 2760973158, 3179948583, 4073438432, 3956313505, 3237863010, 3655790371,
    347922877, 229101820, 646611775, 1066513022, 1892689081, 1774917112, 1122387515, 1543337850,
    3697634229, 3313392372, 3998419255, 4148705398, 3087642289, 2702352368, 2319436851, 2468674930,
  }, {
    0, 29518391, 59036782, 38190681, 118073564, 114017003, 76381362, 89069189,
    236147128, 265370511, 228034006, 206958561, 152762724, 148411219, 178138378, 190596925,
    472294256, 501532999, 530741022, 509615401, 456068012, 451764635, 413917122, 426358261,
    305525448, 334993663, 296822438, 275991697, 356276756, 352202787, 381193850, 393929805,
    944588512, 965684439, 1003065998, 973863097, 1061482044, 1049003019, 1019230802, 1023561829,
    912136024, 933002607, 903529270, 874031361, 827834244, 815125939, 852716522, 856752605,
    611050896, 631869351, 669987326, 640506825, 593644876, 580921211, 551983394, 556069653,
    712553512, 733666847, 704405574, 675154545, 762387700, 749958851, 787859610, 792175277,
    1889177024, 1901651959, 1931368878, 1927033753, 2006131996, 1985040171, 1947726194, 1976933189,
    2122964088, 2135668303, 2098006038, 2093965857, 2038461604, 2017599123, 2047123658, 2076625661,
    1824272048, 1836991623, 1866005214, 1861914857, 1807058540, 1786244187, 1748062722, 1777547317,
    1655668488, 1668093247, 1630251878, 1625932113, 1705433044, 1684323811, 1713505210, 1742760333,
    1222101792, 1226154263, 1263738702, 1251046777, 1339974652, 1310460363, 1281013650, 1301863845,
    1187289752, 1191637167, 1161842422, 1149379777, 1103966788, 1074747507, 1112139306, 1133218845,
    1425107024, 1429406311, 1467333694, 1454888457, 1408811148, 1379576507, 1350309090, 1371438805,
    1524775400, 1528845279, 1499917702, 1487177649, 1575719220, 1546255107, 1584350554, 1605185389,
    3778354048, 3774312887, 3803303918, 3816007129, 3862737756, 3892238699, 3854067506, 3833203973,
    4012263992, 4007927823, 3970080342, 3982554209, 3895452388, 3924658387, 3953866378, 3932773565,
    4245928176, 4241609415, 4271336606, 4283762345, 4196012076, 4225268251, 4187931714, 4166823541,
    4076923208, 4072833919, 4035198246, 4047918865, 4094247316, 4123732899, 4153251322, 4132437965,
    3648544096, 3636082519, 3673983246, 3678331705, 3732010428, 3753090955, 3723829714, 3694611429,
    3614117080, 3601426159, 3572488374, 3576541825, 3496125444, 3516976691, 3555094634, 3525581405,
    3311336976, 3298595879, 3336186494, 3340255305, 3260503756, 3281337595, 3251864226, 3222399125,
    3410866088, 3398419871, 3368647622, 3372945905, 3427010420, 3448139075, 3485520666, 3456284973,
    2444203584, 2423127159, 2452308526, 2481530905, 2527477404, 2539934891, 2502093554, 2497740997,
    2679949304, 2659102159, 2620920726, 2650438049, 2562027300, 2574714131, 2603727690, 2599670141,
    2374579504, 2353749767, 2383274334, 2412743529, 2323684844, 2336421851, 2298759554, 2294686645,
    2207933576, 2186809023, 2149495014, 2178734801, 2224278612, 2236720739, 2266437690, 2262135309,
    2850214048, 2820717207, 2858812622, 2879680249, 2934667388, 2938704459, 2909776914, 2897069605,
    2817622296, 2788420399, 2759153014, 2780249921, 2700618180, 2704950259, 2742877610, 2730399645,
    3049550800, 3020298727, 3057690558, 3078802825, 2999835404, 3004150075, 2974355298, 2961925461,
    3151438440, 3121956959, 3092510214, 3113327665, 3168701108, 3172786307, 3210370778, 3197646061,
  }, {
    0, 3099354981, 2852767883, 313896942, 2405603159, 937357362, 627793884, 2648127673,
    3316918511, 2097696650, 1874714724, 3607201537, 1255587768, 4067088605, 3772741427, 1482887254,
    1343838111, 3903140090, 4195393300, 1118632049, 3749429448, 1741137837, 1970407491, 3452858150,
    2511175536, 756094997, 1067759611, 2266550430, 449832999, 2725482306, 2965774508, 142231497,
    2687676222, 412010587, 171665333, 2995192016, 793786473, 2548850444, 2237264098, 1038456711,
    1703315409, 3711623348, 3482275674, 1999841343, 3940814982, 1381529571, 1089329165, 4166106984,
    4029413537, 1217896388, 1512189994, 3802027855, 2135519222, 3354724499, 3577784189, 1845280792,
    899665998, 2367928107, 2677414085, 657096608, 3137160985, 37822588, 284462994, 2823350519,
    2601801789, 598228824, 824021174, 2309093331, 343330666, 2898962447, 3195996129, 113467524,
    1587572946, 3860600759, 4104763481, 1276501820, 3519211397, 1769898208, 2076913422, 3279374443,
    3406630818, 1941006535, 1627703081, 3652755532, 1148164341, 4241751952, 3999682686, 1457141531,
    247015245, 3053797416, 2763059142, 470583459, 2178658330, 963106687, 735213713, 2473467892,
    992409347, 2207944806, 2435792776, 697522413, 3024379988, 217581361, 508405983, 2800865210,
    4271038444, 1177467017, 1419450215, 3962007554, 1911572667, 3377213406, 3690561584, 1665525589,
    1799331996, 3548628985, 3241568279, 2039091058, 3831314379, 1558270126, 1314193216, 4142438437,
    2928380019, 372764438, 75645176, 3158189981, 568925988, 2572515393, 2346768303, 861712586,
    3982079547, 1441124702, 1196457648, 4293663189, 1648042348, 3666298377, 3358779879, 1888390786,
    686661332, 2421291441, 2196002399, 978858298, 2811169155, 523464422, 226935048, 3040519789,
    3175145892, 100435649, 390670639, 2952089162, 841119475, 2325614998, 2553003640, 546822429,
    2029308235, 3225988654, 3539796416, 1782671013, 4153826844, 1328167289, 1570739863, 3844338162,
    1298864389, 4124540512, 3882013070, 1608431339, 3255406162, 2058742071, 1744848601, 3501990332,
    2296328682, 811816591, 584513889, 2590678532, 129869501, 3204563416, 2914283062, 352848211,
    494030490, 2781751807, 3078325777, 264757620, 2450577869, 715964072, 941166918, 2158327331,
    3636881013, 1618608400, 1926213374, 3396585883, 1470427426, 4011365959, 4255988137, 1158766284,
    1984818694, 3471935843, 3695453837, 1693991400, 4180638033, 1100160564, 1395044826, 3952793279,
    3019491049, 189112716, 435162722, 2706139399, 1016811966, 2217162459, 2526189877, 774831696,
    643086745, 2666061564, 2354934034, 887166583, 2838900430, 294275499, 54519365, 3145957664,
    3823145334, 1532818963, 1240029693, 4048895640, 1820460577, 3560857924, 3331051178, 2117577167,
    3598663992, 1858283101, 2088143283, 3301633750, 1495127663, 3785470218, 4078182116, 1269332353,
    332098007, 2876706482, 3116540252, 25085497, 2628386432, 605395429, 916469259, 2384220526,
    2254837415, 1054503362, 745528876, 2496903497, 151290352, 2981684885, 2735556987, 464596510,
    1137851976, 4218313005, 3923506883, 1365741990, 3434129695, 1946996346, 1723425172, 3724871409,
  }, {
    0, 1029712304, 2059424608, 1201699536, 4118849216, 3370159984, 2403399072, 2988497936,
    812665793, 219177585, 1253054625, 2010132753, 3320900865, 4170237105, 3207642721, 2186319825,
    1625331586, 1568718386, 438355170, 658566482, 2506109250, 2818578674, 4020265506, 3535817618,
    1351670851, 1844508147, 709922595, 389064339, 2769320579, 2557498163, 3754961379, 3803185235,
    3250663172, 4238411444, 3137436772, 2254525908, 876710340, 153198708, 1317132964, 1944187668,
    4054934725, 3436268917, 2339452837, 3054575125, 70369797, 961670069, 2129760613, 1133623509,
    2703341702, 2621542710, 3689016294, 3867263574, 1419845190, 1774270454, 778128678, 318858390,
    2438067015, 2888948471, 3952189479, 3606153623, 1691440519, 1504803895, 504432359, 594620247,
    1492342857, 1704161785, 573770537, 525542041, 2910060169, 2417219385, 3618876905, 3939730521,
    1753420680, 1440954936, 306397416, 790849880, 2634265928, 2690882808, 3888375336, 3668168600,
    940822475, 91481723, 1121164459, 2142483739, 3448989963, 4042473659, 3075684971, 2318603227,
    140739594, 889433530, 1923340138, 1338244826, 4259521226, 3229813626, 2267247018, 3124975642,
    2570221389, 2756861693, 3824297005, 3734113693, 1823658381, 1372780605, 376603373, 722643805,
    2839690380, 2485261628, 3548540908, 4007806556, 1556257356, 1638052860, 637716780, 459464860,
    4191346895, 3300051327, 2199040943, 3195181599, 206718479, 825388991, 1989285231, 1274166495,
    3382881038, 4106388158, 3009607790, 2382549470, 1008864718, 21111934, 1189240494, 2072147742,
    2984685714, 2357631266, 3408323570, 4131834434, 1147541074, 2030452706, 1051084082, 63335554,
    2174155603, 3170292451, 4216760371, 3325460867, 1947622803, 1232499747, 248909555, 867575619,
    3506841360, 3966111392, 2881909872, 2527485376, 612794832, 434546784, 1581699760, 1663499008,
    3782634705, 3692447073, 2612412337, 2799048193, 351717905, 697754529, 1849071985, 1398190273,
    1881644950, 1296545318, 182963446, 931652934, 2242328918, 3100053734, 4284967478, 3255255942,
    1079497815, 2100821479, 983009079, 133672583, 3050795671, 2293717799, 3474399735, 4067887175,
    281479188, 765927844, 1778867060, 1466397380, 3846680276, 3626469220, 2676489652, 2733102084,
    548881365, 500656741, 1517752501, 1729575173, 3577210133, 3898068133, 2952246901, 2459410373,
    3910527195, 3564487019, 2480257979, 2931134987, 479546907, 569730987, 1716854139, 1530213579,
    3647316762, 3825568426, 2745561210, 2663766474, 753206746, 293940330, 1445287610, 1799716618,
    2314567513, 3029685993, 4080348217, 3461678473, 2088098201, 1091956777, 112560889, 1003856713,
    3112514712, 2229607720, 3276105720, 4263857736, 1275433560, 1902492648, 918929720, 195422344,
    685033439, 364179055, 1377080511, 1869921551, 3713294623, 3761522863, 2811507327, 2599689167,
    413436958, 633644462, 1650777982, 1594160846, 3978570462, 3494118254, 2548332990, 2860797966,
    1211387997, 1968470509, 854852413, 261368461, 3182753437, 2161434413, 3346310653, 4195650637,
    2017729436, 1160000044, 42223868, 1071931724, 2378480988, 2963576044, 4144295484, 3395602316,
  }, {
    0, 3411858341, 1304994059, 2257875630, 2609988118, 1355649459, 3596215069, 486879416,
    3964895853, 655315400, 2711298918, 1791488195, 2009251963, 3164476382, 973758832, 4048990933,
    64357019, 3364540734, 1310630800, 2235723829, 2554806413, 1394316072, 3582976390, 517157411,
    4018503926, 618222419, 2722963965, 1762783832, 1947517664, 3209171269, 970744811, 4068520014,
    128714038, 3438335635, 1248109629, 2167961496, 2621261600, 1466012805, 3522553387, 447296910,
    3959392091, 547575038, 2788632144, 1835791861, 1886307661, 3140622056, 1034314822, 4143626211,
    75106221, 3475428360, 1236444838, 2196665603, 2682996155, 1421317662, 3525567664, 427767573,
    3895035328, 594892389, 2782995659, 1857943406, 1941489622, 3101955187, 1047553757, 4113347960,
    257428076, 3288652233, 1116777319, 2311878850, 2496219258, 1603640287, 3640781169, 308099796,
    3809183745, 676813732, 2932025610, 1704983215, 2023410199, 3016104370, 894593820, 4262377657,
    210634999, 3352484690, 1095150076, 2316991065, 2535410401, 1547934020, 3671583722, 294336591,
    3772615322, 729897279, 2903845777, 1716123700, 2068629644, 2953845545, 914647431, 4258839074,
    150212442, 3282623743, 1161604689, 2388688372, 2472889676, 1480171241, 3735940167, 368132066,
    3836185911, 805002898, 2842635324, 1647574937, 2134298401, 3026852996, 855535146, 4188192143,
    186781121, 3229539940, 1189784778, 2377547631, 2427670487, 1542429810, 3715886812, 371670393,
    3882979244, 741170185, 2864262823, 1642462466, 2095107514, 3082559007, 824732849, 4201955092,
    514856152, 3589064573, 1400419795, 2552522358, 2233554638, 1316849003, 3370776517, 62202976,
    4075001525, 968836368, 3207280574, 1954014235, 1769133219, 2720925446, 616199592, 4024870413,
    493229635, 3594175974, 1353627464, 2616354029, 2264355925, 1303087088, 3409966430, 6498043,
    4046820398, 979978123, 3170710821, 2007099008, 1789187640, 2717386141, 661419827, 3962610838,
    421269998, 3527459403, 1423225061, 2676515648, 2190300152, 1238466653, 3477467891, 68755798,
    4115633027, 1041448998, 3095868040, 1943789869, 1860096405, 2776760880, 588673182, 3897205563,
    449450869, 3516317904, 1459794558, 2623431131, 2170245475, 1242006214, 3432247400, 131015629,
    4137259288, 1036337853, 3142660115, 1879958454, 1829294862, 2790523051, 549483013, 3952910752,
    300424884, 3669282065, 1545650111, 2541513754, 2323209378, 1092980487, 3350330793, 216870412,
    4256931033, 921128828, 2960342482, 2066738807, 1714085583, 2910195050, 736264132, 3770592353,
    306060335, 3647131530, 1610005796, 2494197377, 2309971513, 1123257756, 3295149874, 255536279,
    4268596802, 892423655, 3013951305, 2029645036, 1711070292, 2929725425, 674528607, 3815288570,
    373562242, 3709388839, 1535949449, 2429577516, 2379569556, 1183418929, 3223189663, 188820282,
    4195850735, 827017802, 3084859620, 2089020225, 1636228089, 2866415708, 743340786, 3876759895,
    361896217, 3738094268, 1482340370, 2466671543, 2382584591, 1163888810, 3284924932, 144124321,
    4190215028, 849168593, 3020503679, 2136336858, 1649465698, 2836138695, 798521449, 3838094284,
  }, {
    0, 2792819636, 2543784233, 837294749, 4098827283, 1379413927, 1674589498, 3316072078,
    871321191, 2509784531, 2758827854, 34034938, 3349178996, 1641505216, 1346337629, 4131942633,
    1742642382, 3249117050, 4030828007, 1446413907, 2475800797, 904311657, 68069876, 2725880384,
    1412551337, 4064729373, 3283010432, 1708771380, 2692675258, 101317902, 937551763, 2442587175,
    3485284764, 1774858792, 1478633653, 4266992385, 1005723023, 2642744891, 2892827814, 169477906,
    4233263099, 1512406095, 1808623314, 3451546982, 136139752, 2926205020, 2676114113, 972376437,
    2825102674, 236236518, 1073525883, 2576072655, 1546420545, 4200303349, 3417542760, 1841601500,
    2609703733, 1039917185, 202635804, 2858742184, 1875103526, 3384067218, 4166835727, 1579931067,
    1141601657, 3799809741, 3549717584, 1977839588, 2957267306, 372464350, 668680259, 2175552503,
    2011446046, 3516084394, 3766168119, 1175200131, 2209029901, 635180217, 338955812, 2990736784,
    601221559, 2242044419, 3024812190, 306049834, 3617246628, 1911408144, 1074125965, 3866285881,
    272279504, 3058543716, 2275784441, 567459149, 3832906691, 1107462263, 1944752874, 3583875422,
    2343980261, 767641425, 472473036, 3126744696, 2147051766, 3649987394, 3899029983, 1309766251,
    3092841090, 506333494, 801510315, 2310084639, 1276520081, 3932237093, 3683203000, 2113813516,
    3966292011, 1243601823, 2079834370, 3716205238, 405271608, 3192979340, 2411259153, 701492901,
    3750207052, 2045810168, 1209569125, 4000285905, 734575199, 2378150379, 3159862134, 438345922,
    2283203314, 778166598, 529136603, 3120492655, 2086260449, 3660498261, 3955679176, 1303499900,
    3153699989, 495890209, 744928700, 2316418568, 1337360518, 3921775410, 3626602927, 2120129051,
    4022892092, 1237286280, 2018993941, 3726666913, 461853231, 3186645403, 2350400262, 711936178,
    3693557851, 2052076527, 1270360434, 3989775046, 677911624, 2384402428, 3220639073, 427820757,
    1202443118, 3789347034, 3493118535, 1984154099, 3018127229, 362020041, 612099668, 2181885408,
    1950653705, 3526596285, 3822816288, 1168934804, 2148251930, 645706414, 395618355, 2984485767,
    544559008, 2248295444, 3085590153, 295523645, 3560598451, 1917673479, 1134918298, 3855773998,
    328860103, 3052210803, 2214924526, 577903450, 3889505748, 1101147744, 1883911421, 3594338121,
    3424493451, 1785369663, 1535282850, 4260726038, 944946072, 2653270060, 2949491377, 163225861,
    4294103532, 1501944408, 1752023237, 3457862513, 196998655, 2915761739, 2619532502, 978710370,
    2881684293, 229902577, 1012666988, 2586515928, 1603020630, 4193987810, 3356702335, 1852063179,
    2553040162, 1046169238, 263412747, 2848217023, 1818454321, 3390333573, 4227627032, 1569420204,
    60859927, 2782375331, 2487203646, 843627658, 4159668740, 1368951216, 1617990445, 3322386585,
    810543216, 2520310724, 2815490393, 27783917, 3288386659, 1652017111, 1402985802, 4125677310,
    1685994201, 3255382381, 4091620336, 1435902020, 2419138250, 910562686, 128847843, 2715354199,
    1469150398, 4058414858, 3222168983, 1719234083, 2749255853, 94984985, 876691844, 2453031472,
  }, {
    0, 3433693342, 1109723005, 2391738339, 2219446010, 1222643300, 3329165703, 180685081,
    3555007413, 525277995, 2445286600, 1567235158, 1471092047, 2600801745, 361370162, 3642757804,
    2092642603, 2953916853, 1050555990, 4063508168, 4176560081, 878395215, 3134470316, 1987983410,
    2942184094, 1676945920, 3984272867, 567356797, 722740324, 3887998202, 1764827929, 2778407815,
    4185285206, 903635656, 3142804779, 2012833205, 2101111980, 2979425330, 1058630609, 4088621903,
    714308067, 3862526333, 1756790430, 2753330688, 2933487385, 1651734407, 3975966820, 542535930,
    2244825981, 1231508451, 3353891840, 188896414, 25648519, 3442302233, 1134713594, 2399689316,
    1445480648, 2592229462, 336416693, 3634843435, 3529655858, 516441772, 2420588879, 1559052753,
    698204909, 3845636723, 1807271312, 2803025166, 2916600855, 1635634313, 4025666410, 593021940,
    4202223960, 919787974, 3093159461, 1962401467, 2117261218, 2996361020, 1008193759, 4038971457,
    1428616134, 2576151384, 386135227, 3685348389, 3513580860, 499580322, 2471098945, 1608776415,
    2260985971, 1248454893, 3303468814, 139259792, 42591881, 3458459159, 1085071860, 2349261162,
    3505103035, 474062885, 2463016902, 1583654744, 1419882049, 2550902495, 377792828, 3660491170,
    51297038, 3483679632, 1093385331, 2374089965, 2269427188, 1273935210, 3311514249, 164344343,
    2890961296, 1627033870, 4000683757, 585078387, 672833386, 3836780532, 1782552599, 2794821769,
    2142603813, 3005188795, 1032883544, 4047146438, 4227826911, 928351297, 3118105506, 1970307900,
    1396409818, 2677114180, 287212199, 3719594553, 3614542624, 467372990, 2505346141, 1509854403,
    2162073199, 1282711281, 3271268626, 240228748, 76845205, 3359543307, 1186043880, 2317064054,
    796964081, 3811226735, 1839575948, 2702160658, 2882189835, 1734392469, 3924802934, 625327592,
    4234522436, 818917338, 3191908409, 1927981223, 2016387518, 3028656416, 973776579, 4137723485,
    2857232268, 1726474002, 3899187441, 616751215, 772270454, 3803048424, 1814228491, 2693328533,
    2041117753, 3036871847, 999160644, 4146592730, 4259508931, 826864221, 3217552830, 1936586016,
    3606501031, 442291769, 2496909786, 1484378436, 1388107869, 2652297411, 278519584, 3694387134,
    85183762, 3384397196, 1194773103, 2342308593, 2170143720, 1307820918, 3279733909, 265733131,
    2057717559, 3054258089, 948125770, 4096344276, 4276898253, 843467091, 3167309488, 1885556270,
    2839764098, 1709792284, 3949353983, 667704161, 755585656, 3785577190, 1865176325, 2743489947,
    102594076, 3401021058, 1144549729, 2291298815, 2186770662, 1325234296, 3228729243, 215514885,
    3589828009, 424832311, 2547870420, 1534552650, 1370645331, 2635621325, 328688686, 3745342640,
    2211456353, 1333405183, 3254067740, 224338562, 127544219, 3408931589, 1170156774, 2299866232,
    1345666772, 2627681866, 303053225, 3736746295, 3565105198, 416624816, 2522494803, 1525692365,
    4285207626, 868291796, 3176010551, 1910772649, 2065767088, 3079346734, 956571085, 4121828691,
    747507711, 3760459617, 1856702594, 2717976604, 2831417605, 1684930971, 3940615800, 642451174,
  },
  {
    0, 393942083, 787884166, 965557445, 1575768332, 1251427663, 1931114890, 1684106697,
    3151536664, 2896410203, 2502855326, 2186649309, 3862229780, 4048545623, 3368213394, 3753496529,
    2898281073, 3149616690, 2184604407, 2504883892, 4046197629, 3864463166, 3755621371, 3366006712,
    387506281, 6550570, 971950319, 781573292, 1257550181, 1569695014, 1677892067, 1937345952,
    2196865699, 2508887776, 2886183461, 3145514598, 3743273903, 3362179052, 4058774313, 3868258154,
    958996667, 777139448, 400492605, 10755198, 1690661303, 1941857780, 1244879153, 1565019506,
    775012562, 961205393, 13101140, 398261271, 1943900638, 1688634781, 1563146584, 1246801179,
    2515100362, 2190636681, 3139390028, 2892258831, 3355784134, 3749586821, 3874691904, 4052225795,
    3734110983, 3387496260, 4033096577, 3877584834, 2206093835, 2483373640, 2911402637, 3136515790,
    1699389727, 1915860316, 1270647193, 1556585946, 950464531, 803071056, 374397077, 19647702,
    1917993334, 1697207605, 1554278896, 1272937907, 800985210, 952435769, 21510396, 372452543,
    3381322606, 3740399405, 3883715560, 4027047851, 2489758306, 2199758369, 3130039012, 2917895847,
    1550025124, 1259902439, 1922410786, 1710144865, 26202280, 385139947, 796522542, 939715693,
    3887801276, 4039129087, 3377269562, 3728088953, 3126293168, 2905368307, 2493602358, 2212122229,
    4037264341, 3889747862, 3730172755, 3375300368, 2907673305, 3124004506, 2209987167, 2495786524,
    1266377165, 1543533966, 1703758155, 1928748296, 379007169, 32253058, 945887303, 790236164,
    1716846671, 1898845196, 1218652361, 1608006794, 1002000707, 750929152, 357530053, 36990342,
    3717046871, 3405166100, 4084959953, 3825245842, 2153902939, 2535122712, 2929187805, 3119304606,
    3398779454, 3723384445, 3831720632, 4078468859, 2541294386, 2147616625, 3113171892, 2935238647,
    1900929062, 1714877541, 1606142112, 1220599011, 748794154, 1004184937, 39295404, 355241455,
    3835986668, 4091516591, 3394415210, 3710500393, 3108557792, 2922629027, 2545875814, 2160455461,
    1601970420, 1208431799, 1904871538, 1727077425, 43020792, 367748539, 744905086, 991776061,
    1214562461, 1595921630, 1720903707, 1911159896, 361271697, 49513938, 998160663, 738569556,
    4089209477, 3838277318, 3712633347, 3392233024, 2924491657, 3106613194, 2158369551, 2547846988,
    3100050248, 2948339467, 2519804878, 2169126797, 3844821572, 4065347079, 3420289730, 3701894785,
    52404560, 342144275, 770279894, 982687125, 1593045084, 1233708063, 1879431386, 1736363161,
    336019769, 58479994, 988899775, 764050940, 1240141877, 1586496630, 1729968307, 1885744368,
    2950685473, 3097818978, 2166999975, 2522013668, 4063474221, 3846743662, 3703937707, 3418263272,
    976650731, 760059304, 348170605, 62635310, 1742393575, 1889649828, 1227683937, 1582820386,
    2179867635, 2526361520, 2937588597, 3093503798, 3691148031, 3413731004, 4076100217, 3851374138,
    2532754330, 2173556697, 3087067932, 2944139103, 3407516310, 3697379029, 3857496592, 4070026835,
    758014338, 978679233, 64506116, 346250567, 1891774606, 1740186829, 1580472328, 1229917259,
  }, {
    0, 4022496062, 83218493, 3946298115, 166436986, 3861498692, 220098631, 3806075769,
    332873972, 4229245898, 388141257, 4175494135, 440197262, 4127099824, 516501683, 4044053389,
    665747944, 3362581206, 593187285, 3432594155, 776282514, 3246869164, 716239279, 3312622225,
    880394524, 3686509090, 814485793, 3746462239, 1033003366, 3528460888, 963096923, 3601193573,
    1331495888, 2694801646, 1269355501, 2758457555, 1186374570, 2843003028, 1111716759, 2910918825,
    1552565028, 3007850522, 1484755737, 3082680359, 1432478558, 3131279456, 1368666979, 3193329757,
    1760789048, 2268195078, 1812353541, 2210675003, 1628971586, 2396670332, 1710092927, 2318375233,
    2066006732, 2498144754, 2144408305, 2417195471, 1926193846, 2634877320, 1983558283, 2583222709,
    2662991776, 1903717534, 2588923805, 1972223139, 2538711002, 2022952164, 2477029351, 2087066841,
    2372749140, 1655647338, 2308478825, 1717238871, 2223433518, 1799654416, 2155034387, 1873894445,
    3105130056, 1456926070, 3185661557, 1378041163, 2969511474, 1597852940, 3020617231, 1539874097,
    2864957116, 1157737858, 2922780289, 1106542015, 2737333958, 1290407416, 2816325371, 1210047941,
    3521578096, 1042640718, 3574781005, 986759027, 3624707082, 936300340, 3707335735, 859512585,
    3257943172, 770846650, 3334837433, 688390023, 3420185854, 605654976, 3475911875, 552361981,
    4132013464, 428600998, 4072428965, 494812827, 4288816610, 274747100, 4216845791, 345349857,
    3852387692, 173846098, 3781891409, 245988975, 3967116566, 62328360, 3900749099, 121822741,
    3859089665, 164061759, 3807435068, 221426178, 4025395579, 2933317, 3944446278, 81334904,
    4124199413, 437265099, 4045904328, 518386422, 4231653775, 335250097, 4174133682, 386814604,
    3249244393, 778691543, 3311294676, 714879978, 3359647891, 662848429, 3434477742, 595039120,
    3531393053, 1035903779, 3599308832, 961245982, 3684132967, 877986649, 3747788890, 815846244,
    2841119441, 1184522735, 2913852140, 1114616274, 2696129195, 1332855189, 2756082326, 1266946472,
    3129952805, 1431118107, 3195705880, 1371074854, 3009735263, 1554415969, 3079748194, 1481855324,
    2398522169, 1630855175, 2315475716, 1707159610, 2266835779, 1759461501, 2213084030, 1814728768,
    2636237773, 1927520499, 2580814832, 1981182158, 2496293815, 2064121993, 2420095882, 2147340468,
    2025787041, 2541577631, 2085281436, 2475210146, 1901375195, 2660681189, 1973518054, 2590184920,
    1801997909, 2225743211, 1872600680, 2153772374, 1652813359, 2369881361, 1719025170, 2310296876,
    1594986313, 2966676599, 1541693300, 3022402634, 1459236659, 3107472397, 1376780046, 3184366640,
    1288097725, 2734990467, 1211309952, 2817619134, 1160605639, 2867791097, 1104723962, 2920993988,
    937561457, 3626001999, 857201996, 3704993394, 1040821515, 3519792693, 989625654, 3577615880,
    607473029, 3421972155, 549494200, 3473077894, 769584639, 3256649409, 690699714, 3337180924,
    273452185, 4287555495, 347692196, 4219156378, 430386403, 4133832669, 491977950, 4069562336,
    60542061, 3965298515, 124656720, 3903616878, 175139863, 3853649705, 243645482, 3779581716,
  }, {
    0, 3247366080, 1483520449, 2581751297, 2967040898, 1901571138, 3904227907, 691737987,
    3133399365, 2068659845, 3803142276, 589399876, 169513671, 3415493895, 1383475974, 2482566342,
    2935407819, 1870142219, 4137319690, 924099274, 506443593, 3751897225, 1178799752, 2278412616,
    339027342, 3585866318, 1280941135, 2379694991, 2766951948, 1700956620, 4236308429, 1024339981,
    2258407383, 1192382487, 3740284438, 528411094, 910556245, 4157285269, 1848198548, 2946996820,
    1012887186, 4258378066, 1681119059, 2780629139, 2357599504, 1292419792, 3572147409, 358906641,
    678054684, 3924071644, 1879503581, 2978491677, 2561882270, 1497229150, 3235873119, 22109855,
    2460592729, 1395094937, 3401913240, 189516888, 577821147, 3825075739, 2048679962, 3146956762,
    3595049455, 398902831, 2384764974, 1336573934, 1720805997, 2803873197, 1056822188, 4285729900,
    1821112490, 2902796138, 887570795, 4117339819, 3696397096, 500978920, 2218668777, 1169222953,
    2025774372, 3106931428, 550659301, 3780950821, 3362238118, 166293862, 2416645991, 1367722151,
    3262987361, 66315169, 2584839584, 1537170016, 1923370979, 3005911075, 717813282, 3947244002,
    1356109368, 2438613496, 146288633, 3375820857, 3759007162, 562248314, 3093388411, 2045739963,
    3927406461, 731490493, 2994458300, 1945440636, 1523451135, 2604718911, 44219710, 3274466046,
    4263662323, 1068272947, 2790189874, 1740649714, 1325080945, 2406874801, 379033776, 3608758128,
    1155642294, 2238671990, 479005303, 3708016055, 4097359924, 901128180, 2891217397, 1843045941,
    2011248031, 3060787807, 797805662, 3993195422, 3342353949, 112630237, 2673147868, 1591353372,
    3441611994, 212601626, 2504944923, 1421914843, 2113644376, 3161815192, 630660761, 3826893145,
    3642224980, 412692116, 2172340373, 1089836885, 1775141590, 2822790422, 832715543, 4029474007,
    1674842129, 2723860433, 1001957840, 4197873168, 3540870035, 310623315, 2338445906, 1257178514,
    4051548744, 821257608, 2836464521, 1755307081, 1101318602, 2150241802, 432566283, 3628511179,
    1270766349, 2318435533, 332587724, 3529260300, 4217841807, 988411727, 2735444302, 1652903566,
    1602977411, 2651169091, 132630338, 3328776322, 4015131905, 786223809, 3074340032, 1991273216,
    3846741958, 616972294, 3173262855, 2091579847, 1435626564, 2485072772, 234706309, 3430124101,
    2712218736, 1613231024, 4190475697, 944458353, 292577266, 3506339890, 1226630707, 2291284467,
    459984181, 3672380149, 1124496628, 2189994804, 2880683703, 1782407543, 4091479926, 844224694,
    257943739, 3469817723, 1462980986, 2529005242, 3213269817, 2114471161, 3890881272, 644152632,
    3046902270, 1947391550, 3991973951, 746483711, 88439420, 3301680572, 1563018173, 2628197501,
    657826727, 3871046759, 2136545894, 3201811878, 2548879397, 1449267173, 3481299428, 235845156,
    2650161890, 1551408418, 3315268387, 68429027, 758067552, 3970035360, 1967360161, 3033356129,
    2311284588, 1213053100, 3517963949, 270598509, 958010606, 4170500910, 1635167535, 2700636911,
    855672361, 4069415401, 1802256360, 2866995240, 2212099499, 1113008747, 3686091882, 440112042,
  }, {
    0, 2611301487, 3963330207, 2006897392, 50740095, 2560849680, 4013794784, 1956178319,
    101480190, 2645113489, 3929532513, 1905435662, 84561281, 2662269422, 3912356638, 1922342769,
    202960380, 2545787283, 3760419683, 2072395532, 253679235, 2495322860, 3810871324, 2021655667,
    169122562, 2444351341, 3861841309, 2106214898, 152215677, 2461527058, 3844685538, 2123133581,
    405920760, 2207553431, 4094313831, 1873742088, 456646791, 2157096168, 4144791064, 1823027831,
    507358470, 2241388905, 4060492697, 1772322806, 490444409, 2258557462, 4043311334, 1789215881,
    338245124, 2408348267, 4161972379, 1672996084, 388959611, 2357870868, 4212429796, 1622269835,
    304431354, 2306870421, 4263435877, 1706791434, 287538053, 2324051946, 4246267162, 1723705717,
    811841520, 2881944479, 3696765295, 1207788800, 862293135, 2831204576, 3747484176, 1157324415,
    913293582, 2915732833, 3662962577, 1106318334, 896137841, 2932651550, 3646055662, 1123494017,
    1014716940, 2816349795, 3493905555, 1273334012, 1065181555, 2765630748, 3544645612, 1222882179,
    980888818, 2714919069, 3595350637, 1307180546, 963712909, 2731826146, 3578431762, 1324336509,
    676490248, 3019317351, 3295277719, 1607253752, 726947703, 2968591128, 3345992168, 1556776327,
    777919222, 3053147801, 3261432937, 1505806342, 760750473, 3070062054, 3244539670, 1522987897,
    608862708, 3220163995, 3362856811, 1406423812, 659339915, 3169449700, 3413582868, 1355966587,
    575076106, 3118709605, 3464325525, 1440228858, 557894773, 3135602714, 3447411434, 1457397381,
    1623683040, 4217512847, 2365387135, 391757072, 1673614495, 4167309552, 2415577600, 341804655,
    1724586270, 4251866481, 2331019137, 290835438, 1707942497, 4268256782, 2314648830, 307490961,
    1826587164, 4152020595, 2162433155, 457265388, 1876539747, 4101829900, 2212636668, 407333779,
    1792275682, 4051089549, 2263378557, 491595282, 1775619997, 4067460082, 2246988034, 508239213,
    2029433880, 3813931127, 2496473735, 258500328, 2079362919, 3763716872, 2546668024, 208559511,
    2130363110, 3848244873, 2462145657, 157552662, 2113730969, 3864638966, 2445764358, 174205801,
    1961777636, 4014675339, 2564147067, 57707284, 2011718299, 3964481268, 2614361092, 7778411,
    1927425818, 3913769845, 2665066885, 92077546, 1910772837, 3930150922, 2648673018, 108709525,
    1352980496, 3405878399, 3164554895, 658115296, 1403183983, 3355946752, 3214507504, 607924639,
    1453895406, 3440239233, 3130208369, 557218846, 1437504913, 3456883198, 3113552654, 573589345,
    1555838444, 3340335491, 2961681267, 723707676, 1606028947, 3290383100, 3011612684, 673504355,
    1521500946, 3239382909, 3062619533, 758026722, 1505130605, 3256038402, 3045975794, 774417053,
    1217725416, 3543158663, 2762906999, 1057739032, 1267939479, 3493229816, 2812847624, 1007544935,
    1318679830, 3577493881, 2728586121, 956803046, 1302285929, 3594125830, 2711933174, 973184153,
    1150152212, 3743982203, 2830528651, 856898788, 1200346475, 3694041348, 2880457716, 806684571,
    1115789546, 3643069573, 2931426933, 891243034, 1099408277, 3659722746, 2914794762, 907637093,
  }, {
    0, 3717650821, 1616688459, 3184159950, 3233376918, 489665299, 2699419613, 2104690264,
    1510200173, 2274691816, 979330598, 3888758691, 2595928571, 1194090622, 4209380528, 661706037,
    3020400346, 1771143007, 3562738577, 164481556, 1958661196, 2837976521, 350386439, 3379863682,
    3993269687, 865250354, 2388181244, 1406015865, 784146209, 4079732388, 1323412074, 2474079215,
    3011398645, 1860735600, 3542286014, 246687547, 1942430051, 2924607718, 328963112, 3456978349,
    3917322392, 887832861, 2300653011, 1421341782, 700772878, 4099025803, 1234716485, 2483986112,
    125431087, 3673109674, 1730500708, 3132326369, 3351283641, 441867836, 2812031730, 2047535991,
    1568292418, 2163009479, 1025936137, 3769651852, 2646824148, 1079348561, 4255113631, 537475098,
    3180171691, 1612400686, 3721471200, 4717925, 2100624189, 2694980280, 493375094, 3237910515,
    3884860102, 974691139, 2278750093, 1514417672, 657926224, 4204917205, 1198234907, 2600289438,
    160053105, 3558665972, 1775665722, 3024116671, 3375586791, 346391650, 2842683564, 1962488105,
    1401545756, 2384412057, 869618007, 3997403346, 2469432970, 1319524111, 4083956673, 788193860,
    250862174, 3546612699, 1856990997, 3006903952, 3461001416, 333211981, 2920678787, 1937824774,
    1425017139, 2305216694, 883735672, 3912918525, 2487837605, 1239398944, 4095071982, 696455019,
    3136584836, 1734518017, 3668494799, 121507914, 2051872274, 2816200599, 437363545, 3347544796,
    3774328809, 1029797484, 2158697122, 1564328743, 542033279, 4258798842, 1074950196, 2642717105,
    2691310871, 2113731730, 3224801372, 497043929, 1624461185, 3175454212, 9435850, 3709412175,
    4201248378, 671035391, 2587181873, 1201904308, 986750188, 3880142185, 1519135143, 2266689570,
    342721485, 3388693064, 1949382278, 2846355203, 3570723163, 155332830, 3028835344, 1763607957,
    1315852448, 2482538789, 775087595, 4087626862, 2396469814, 1396827059, 4002123645, 857560824,
    320106210, 3464673127, 1934154665, 2933785132, 3551331444, 238804465, 3018961215, 1852270778,
    1226292623, 2491507722, 692783300, 4108177729, 2309936921, 1412959900, 3924976210, 879016919,
    2803091512, 2055541181, 3343875443, 450471158, 1739236014, 3124525867, 133568485, 3663777376,
    4245691221, 545702608, 2639048222, 1088059291, 1034514883, 3762268230, 1576387720, 2153979149,
    501724348, 3228659001, 2109407735, 2687359090, 3713981994, 13109167, 3171052385, 1620357860,
    1206151121, 2591211092, 666423962, 4197321503, 2271022407, 1523307714, 3875649548, 982999433,
    2850034278, 1953942499, 3384583981, 338329256, 1767471344, 3033506165, 151375291, 3566408766,
    4091789579, 779425934, 2478797888, 1311354309, 861580189, 4006375960, 1392910038, 2391852883,
    2929327945, 1930372812, 3469036034, 324244359, 1847629279, 3015068762, 243015828, 3555391761,
    4103744548, 688715169, 2496043375, 1229996266, 874727090, 3920994103, 1417671673, 2313759356,
    446585235, 3339223062, 2059594968, 2807313757, 3660002053, 129100416, 3128657486, 1743609803,
    1084066558, 2634765179, 549535669, 4250396208, 2149900392, 1571961325, 3765982499, 1039043750,
  }, {
    0, 2635063670, 3782132909, 2086741467, 430739227, 2225303149, 4173482934, 1707977408,
    861478454, 2924937024, 3526875803, 1329085421, 720736557, 3086643291, 3415954816, 1452586230,
    1722956908, 4223524122, 2279405761, 450042295, 2132718455, 3792785921, 2658170842, 58693292,
    1441473114, 3370435372, 3028674295, 696911745, 1279765825, 3511176247, 2905172460, 807831706,
    3445913816, 1349228974, 738901109, 2969918723, 3569940419, 1237784245, 900084590, 2829701656,
    4265436910, 1664255896, 525574723, 2187084597, 3885099509, 2057177219, 117386584, 2616249390,
    2882946228, 920233410, 1253605401, 3619119471, 2994391983, 796207833, 1393823490, 3457937012,
    2559531650, 92322804, 2044829231, 3840835417, 2166609305, 472659183, 1615663412, 4249022530,
    1102706673, 3702920839, 2698457948, 1037619754, 1477802218, 3306854812, 3111894087, 611605809,
    1927342535, 4025419953, 2475568490, 243387420, 1800169180, 4131620778, 2317525617, 388842247,
    655084445, 3120835307, 3328511792, 1533734470, 1051149446, 2745738736, 3754524715, 1120297309,
    340972971, 2304586973, 4114354438, 1748234352, 234773168, 2431761350, 3968900637, 1906278251,
    2363330345, 299003487, 1840466820, 4038896370, 2507210802, 142532932, 1948239007, 3910149609,
    3213136159, 579563625, 1592415666, 3286611140, 2787646980, 992477042, 1195825833, 3662232543,
    3933188933, 2002801203, 184645608, 2517538462, 4089658462, 1858919720, 313391347, 2409765253,
    3644239219, 1144605701, 945318366, 2773977256, 3231326824, 1570095902, 569697989, 3170568115,
    2205413346, 511446676, 1646078799, 4279421497, 2598330617, 131105167, 2075239508, 3871229218,
    2955604436, 757403810, 1363424633, 3427521551, 2844163791, 881434553, 1223211618, 3588709140,
    3854685070, 2026779384, 78583587, 2577462869, 4235025557, 1633861091, 486774840, 2148301134,
    3600338360, 1268198606, 938871061, 2868504675, 3476308643, 1379640277, 777684494, 3008718712,
    1310168890, 3541595724, 2943964055, 846639841, 1471879201, 3400857943, 3067468940, 735723002,
    2102298892, 3762382970, 2619362721, 19901655, 1692534295, 4193118049, 2240594618, 411247564,
    681945942, 3047836192, 3385552891, 1422167693, 822682701, 2886124859, 3496468704, 1298661782,
    469546336, 2264093718, 4203901389, 1738379451, 38812283, 2673859341, 3812556502, 2117148576,
    3268024339, 1606809957, 598006974, 3198893512, 3680933640, 1181316734, 973624229, 2802299603,
    4052944421, 1822222163, 285065864, 2381456382, 3896478014, 1966106696, 156323219, 2489232613,
    2759337087, 964150537, 1159127250, 3625517476, 3184831332, 551242258, 1555722185, 3249901247,
    2535537225, 170842943, 1984954084, 3946848146, 2391651666, 327308324, 1877176831, 4075589769,
    263086283, 2460058045, 4005602406, 1942963472, 369291216, 2332888742, 4151061373, 1784924683,
    1022852861, 2717425547, 3717839440, 1083595558, 626782694, 3092517008, 3291821387, 1497027645,
    1763466407, 4094934481, 2289211402, 360544636, 1890636732, 3988730570, 2447251217, 215086695,
    1514488465, 3343557607, 3140191804, 639919946, 1139395978, 3739626748, 2726758695, 1065936977,
  }, {
    0, 3120290792, 2827399569, 293431929, 2323408227, 864534155, 586863858, 2600537882,
    3481914503, 1987188591, 1729068310, 3740575486, 1173727716, 4228805132, 3983743093, 1418249117,
    1147313999, 4254680231, 3974377182, 1428157750, 3458136620, 2011505092, 1721256893, 3747844181,
    2347455432, 839944224, 594403929, 2593536433, 26687147, 3094146371, 2836498234, 283794642,
    2294627998, 826205558, 541298447, 2578994407, 45702141, 3141697557, 2856315500, 331624836,
    1196225049, 4273416689, 4023010184, 1446090848, 3442513786, 1959480466, 1706436331, 3696098563,
    3433538001, 1968994873, 1679888448, 3722103720, 1188807858, 4280295258, 3999102243, 1470541515,
    53374294, 3134568126, 2879970503, 307431215, 2303854645, 816436189, 567589284, 2553242188,
    3405478781, 1929420949, 1652411116, 3682996484, 1082596894, 4185703926, 3892424591, 1375368295,
    91404282, 3163122706, 2918450795, 336584067, 2400113305, 922028401, 663249672, 2658384096,
    2392450098, 929185754, 639587747, 2682555979, 82149713, 3172883129, 2892181696, 362343208,
    1091578037, 4176212829, 3918960932, 1349337804, 3412872662, 1922537022, 1676344391, 3658557359,
    1111377379, 4224032267, 3937989746, 1396912026, 3359776896, 1908013928, 1623494929, 3644803833,
    2377615716, 877417100, 623982837, 2630542109, 130804743, 3190831087, 2941083030, 381060734,
    106748588, 3215393092, 2933549885, 388083925, 2350956495, 903570471, 614862430, 2640172470,
    3386185259, 1882115523, 1632872378, 3634920530, 1135178568, 4199721120, 3945775833, 1389631793,
    1317531835, 4152109907, 3858841898, 1610259138, 3304822232, 2097172016, 1820140617, 3582394273,
    2165193788, 955639764, 696815021, 2423477829, 192043359, 2995356343, 2750736590, 437203750,
    182808564, 3005133852, 2724453989, 462947725, 2157513367, 962777471, 673168134, 2447663342,
    3312231283, 2090301595, 1844056802, 3557935370, 1326499344, 4142603768, 3885397889, 1584245865,
    3326266917, 2142836173, 1858371508, 3611272284, 1279175494, 4123357358, 3837270743, 1564721471,
    164299426, 2955991370, 2706223923, 414607579, 2209834945, 978107433, 724686416, 2462715320,
    2183156074, 1004243586, 715579643, 2472360723, 140260361, 2980573153, 2698675608, 421617264,
    1302961645, 4099032581, 3845074044, 1557460884, 3352688782, 2116952934, 1867729183, 3601371895,
    2222754758, 1032278062, 754596439, 2499928511, 234942117, 3086693709, 2793824052, 528319708,
    1274365761, 4061043881, 3816027856, 1518873912, 3246989858, 2020800970, 1762628531, 3505670235,
    3223196809, 2045103969, 1754834200, 3512958704, 1247965674, 4086934018, 3806642299, 1528765331,
    261609486, 3060532198, 2802936223, 518697591, 2246819181, 1007707781, 762121468, 2492913428,
    213497176, 3041029808, 2755593417, 499441441, 2261110843, 1061030867, 776167850, 2545465922,
    3274734047, 2060165687, 1807140942, 3528266662, 1229724860, 4038575956, 3788156205, 1479636677,
    1222322711, 4045468159, 3764231046, 1504067694, 3265744756, 2069664924, 1780612837, 3554288909,
    2270357136, 1051278712, 802445057, 2519698665, 221152243, 3033880603, 2779263586, 475261322,
  }, {
    0, 2926088593, 2275419491, 701019378, 3560000647, 2052709654, 1402038756, 4261017717,
    1930665807, 3715829470, 4105419308, 1524313021, 2804077512, 155861593, 545453739, 2397726522,
    3861331614, 1213181711, 1636244477, 3488582252, 840331801, 2625561480, 3048626042, 467584747,
    2503254481, 995897408, 311723186, 3170637091, 1090907478, 4016929991, 3332753461, 1758288292,
    390036349, 3109546732, 2426363422, 1056427919, 3272488954, 1835443819, 1152258713, 3938878216,
    1680663602, 3393484195, 3817652561, 1306808512, 2954733749, 510998820, 935169494, 2580880455,
    4044899811, 1601229938, 1991794816, 3637571857, 623446372, 2336332021, 2726898695, 216120726,
    2181814956, 744704829, 95158223, 2881711710, 1446680107, 4166125498, 3516576584, 2146575065,
    780072698, 2148951915, 2849952665, 129384968, 4199529085, 1411853292, 2112855838, 3548843663,
    1567451573, 4077254692, 3670887638, 1957027143, 2304517426, 657765539, 251396177, 2694091200,
    3361327204, 1714510325, 1341779207, 3784408214, 476611811, 2986349938, 2613617024, 899690513,
    3142211371, 354600634, 1021997640, 2458051545, 1870338988, 3239283261, 3906682575, 1186180958,
    960597383, 2536053782, 3202459876, 277428597, 3983589632, 1125666961, 1792074851, 3300423154,
    1246892744, 3829039961, 3455203243, 1671079482, 2657312335, 806080478, 432241452, 3081497277,
    3748049689, 1896751752, 1489409658, 4138600427, 190316446, 2772397583, 2365053693, 580864876,
    2893360214, 35503559, 735381813, 2243795108, 2017747153, 3593269568, 4293150130, 1368183843,
    1560145396, 4069882981, 3680356503, 1966430470, 2295112051, 648294626, 258769936, 2701399425,
    804156091, 2173100842, 2823706584, 103204425, 4225711676, 1438101421, 2088704863, 3524758222,
    3134903146, 347226875, 1031468553, 2467456920, 1860935661, 3229814396, 3914054286, 1193487135,
    3385412645, 1738661300, 1315531078, 3758225623, 502792354, 3012596019, 2589468097, 875607120,
    1271043721, 3853125400, 3429020650, 1644831355, 2683558414, 832261023, 408158061, 3057348348,
    953223622, 2528745559, 3211865253, 286899508, 3974120769, 1116263632, 1799381026, 3307794867,
    2917509143, 59586950, 709201268, 2217549029, 2043995280, 3619452161, 4269064691, 1344032866,
    3740677976, 1889445577, 1498812987, 4148069290, 180845535, 2762992206, 2372361916, 588238637,
    1921194766, 3706423967, 4112727661, 1531686908, 2796705673, 148555288, 554857194, 2407195515,
    26248257, 2952271312, 2251333922, 676868275, 3584149702, 2076793175, 1375858085, 4234771508,
    2493785488, 986493953, 319029491, 3178008930, 1083533591, 4009621638, 3342158964, 1767759333,
    3887577823, 1239362382, 1612160956, 3464433197, 864482904, 2649647049, 3022443323, 441336490,
    1706844275, 3419730402, 3793503504, 1282724993, 2978819316, 535149925, 908921239, 2554697734,
    380632892, 3100077741, 2433735263, 1063734222, 3265180603, 1828069930, 1161729752, 3948283721,
    2207997677, 770953084, 71007118, 2857626143, 1470763626, 4190274555, 3490330377, 2120394392,
    4035494306, 1591758899, 1999168705, 3644880208, 616140069, 2328960180, 2736367686, 225524183,
  },
};

static const uint8_t
WUFFS_CRC32__IEEE_X86_SSE42_K1K2[16] WUFFS_BASE__POTENTIALLY_UNUSED = {
  212, 43, 68, 84, 1, 0, 0, 0,
  150, 21, 228, 198, 1, 0, 0, 0,
};

static const uint8_t
WUFFS_CRC32__IEEE_X86_SSE42_K3K4[16] WUFFS_BASE__POTENTIALLY_UNUSED = {
  208, 151, 25, 117, 1, 0, 0, 0,
  158, 0, 170, 204, 0, 0, 0, 0,
};

static const uint8_t
WUFFS_CRC32__IEEE_X86_SSE42_K5ZZ[16] WUFFS_BASE__POTENTIALLY_UNUSED = {
  36, 97, 205, 99, 1, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
};

static const uint8_t
WUFFS_CRC32__IEEE_X86_SSE42_PXMU[16] WUFFS_BASE__POTENTIALLY_UNUSED = {
  65, 6, 113, 219, 1, 0, 0, 0,
  65, 22, 1, 247, 1, 0, 0, 0,
};

// ---------------- Private Initializer Prototypes

// ---------------- Private Function Prototypes

static wuffs_base__empty_struct
wuffs_crc32__ieee_hasher__up(
    wuffs_crc32__ieee_hasher* self,
    wuffs_base__slice_u8 a_x);

static wuffs_base__empty_struct
wuffs_crc32__ieee_hasher__up__choosy_default(
    wuffs_crc32__ieee_hasher* self,
    wuffs_base__slice_u8 a_x);

#if defined(WUFFS_BASE__CPU_ARCH__ARM_CRC32)
static wuffs_base__empty_struct
wuffs_crc32__ieee_hasher__up_arm_crc32(
    wuffs_crc32__ieee_hasher* self,
    wuffs_base__slice_u8 a_x);
#endif  // defined(WUFFS_BASE__CPU_ARCH__ARM_CRC32)

#if defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
static wuffs_base__empty_struct
wuffs_crc32__ieee_hasher__up_x86_avx2(
    wuffs_crc32__ieee_hasher* self,
    wuffs_base__slice_u8 a_x);
#endif  // defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)

#if defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
static wuffs_base__empty_struct
wuffs_crc32__ieee_hasher__up_x86_sse42(
    wuffs_crc32__ieee_hasher* self,
    wuffs_base__slice_u8 a_x);
#endif  // defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)

// ---------------- VTables

const wuffs_base__hasher_u32__func_ptrs
wuffs_crc32__ieee_hasher__func_ptrs_for__wuffs_base__hasher_u32 = {
  (wuffs_base__empty_struct(*)(void*,
      uint32_t,
      bool))(&wuffs_crc32__ieee_hasher__set_quirk_enabled),
  (uint32_t(*)(void*,
      wuffs_base__slice_u8))(&wuffs_crc32__ieee_hasher__update_u32),
};

// ---------------- Initializer Implementations

wuffs_base__status WUFFS_BASE__WARN_UNUSED_RESULT
wuffs_crc32__ieee_hasher__initialize(
    wuffs_crc32__ieee_hasher* self,
    size_t sizeof_star_self,
    uint64_t wuffs_version,
    uint32_t options){
  if (!self) {
    return wuffs_base__make_status(wuffs_base__error__bad_receiver);
  }
  if (sizeof(*self) != sizeof_star_self) {
    return wuffs_base__make_status(wuffs_base__error__bad_sizeof_receiver);
  }
  if (((wuffs_version >> 32) != WUFFS_VERSION_MAJOR) ||
      (((wuffs_version >> 16) & 0xFFFF) > WUFFS_VERSION_MINOR)) {
    return wuffs_base__make_status(wuffs_base__error__bad_wuffs_version);
  }

  if ((options & WUFFS_INITIALIZE__ALREADY_ZEROED) != 0) {
    // The whole point of this if-check is to detect an uninitialized *self.
    // We disable the warning on GCC. Clang-5.0 does not have this warning.
#if !defined(__clang__) && defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
    if (self->private_impl.magic != 0) {
      return wuffs_base__make_status(wuffs_base__error__initialize_falsely_claimed_already_zeroed);
    }
#if !defined(__clang__) && defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
  } else {
    if ((options & WUFFS_INITIALIZE__LEAVE_INTERNAL_BUFFERS_UNINITIALIZED) == 0) {
      memset(self, 0, sizeof(*self));
      options |= WUFFS_INITIALIZE__ALREADY_ZEROED;
    } else {
      memset(&(self->private_impl), 0, sizeof(self->private_impl));
    }
  }

  self->private_impl.choosy_up = &wuffs_crc32__ieee_hasher__up__choosy_default;

  self->private_impl.magic = WUFFS_BASE__MAGIC;
  self->private_impl.vtable_for__wuffs_base__hasher_u32.vtable_name =
      wuffs_base__hasher_u32__vtable_name;
  self->private_impl.vtable_for__wuffs_base__hasher_u32.function_pointers =
      (const void*)(&wuffs_crc32__ieee_hasher__func_ptrs_for__wuffs_base__hasher_u32);
  return wuffs_base__make_status(NULL);
}

wuffs_crc32__ieee_hasher*
wuffs_crc32__ieee_hasher__alloc() {
  wuffs_crc32__ieee_hasher* x =
      (wuffs_crc32__ieee_hasher*)(calloc(sizeof(wuffs_crc32__ieee_hasher), 1));
  if (!x) {
    return NULL;
  }
  if (wuffs_crc32__ieee_hasher__initialize(
      x, sizeof(wuffs_crc32__ieee_hasher), WUFFS_VERSION, WUFFS_INITIALIZE__ALREADY_ZEROED).repr) {
    free(x);
    return NULL;
  }
  return x;
}

size_t
sizeof__wuffs_crc32__ieee_hasher() {
  return sizeof(wuffs_crc32__ieee_hasher);
}

// ---------------- Function Implementations

// -------- func crc32.ieee_hasher.set_quirk_enabled

WUFFS_BASE__MAYBE_STATIC wuffs_base__empty_struct
wuffs_crc32__ieee_hasher__set_quirk_enabled(
    wuffs_crc32__ieee_hasher* self,
    uint32_t a_quirk,
    bool a_enabled) {
  return wuffs_base__make_empty_struct();
}

// -------- func crc32.ieee_hasher.update_u32

WUFFS_BASE__MAYBE_STATIC uint32_t
wuffs_crc32__ieee_hasher__update_u32(
    wuffs_crc32__ieee_hasher* self,
    wuffs_base__slice_u8 a_x) {
  if (!self) {
    return 0;
  }
  if (self->private_impl.magic != WUFFS_BASE__MAGIC) {
    return 0;
  }

  if (self->private_impl.f_state == 0) {
    self->private_impl.choosy_up = (
#if defined(WUFFS_BASE__CPU_ARCH__ARM_CRC32)
        wuffs_base__cpu_arch__have_arm_crc32() ? &wuffs_crc32__ieee_hasher__up_arm_crc32 :
#endif
#if defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
        wuffs_base__cpu_arch__have_x86_avx2() ? &wuffs_crc32__ieee_hasher__up_x86_avx2 :
#endif
#if defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
        wuffs_base__cpu_arch__have_x86_sse42() ? &wuffs_crc32__ieee_hasher__up_x86_sse42 :
#endif
        self->private_impl.choosy_up);
  }
  wuffs_crc32__ieee_hasher__up(self, a_x);
  return self->private_impl.f_state;
}

// -------- func crc32.ieee_hasher.up

static wuffs_base__empty_struct
wuffs_crc32__ieee_hasher__up(
    wuffs_crc32__ieee_hasher* self,
    wuffs_base__slice_u8 a_x) {
  return (*self->private_impl.choosy_up)(self, a_x);
}

static wuffs_base__empty_struct
wuffs_crc32__ieee_hasher__up__choosy_default(
    wuffs_crc32__ieee_hasher* self,
    wuffs_base__slice_u8 a_x) {
  uint32_t v_s = 0;
  wuffs_base__slice_u8 v_p = {0};

  v_s = (4294967295 ^ self->private_impl.f_state);
  {
    wuffs_base__slice_u8 i_slice_p = a_x;
    v_p.ptr = i_slice_p.ptr;
    v_p.len = 16;
    uint8_t* i_end0_p = v_p.ptr + (((i_slice_p.len - (size_t)(v_p.ptr - i_slice_p.ptr)) / 32) * 32);
    while (v_p.ptr < i_end0_p) {
      v_s ^= ((((uint32_t)(v_p.ptr[0])) << 0) |
          (((uint32_t)(v_p.ptr[1])) << 8) |
          (((uint32_t)(v_p.ptr[2])) << 16) |
          (((uint32_t)(v_p.ptr[3])) << 24));
      v_s = (WUFFS_CRC32__IEEE_TABLE[0][v_p.ptr[15]] ^
          WUFFS_CRC32__IEEE_TABLE[1][v_p.ptr[14]] ^
          WUFFS_CRC32__IEEE_TABLE[2][v_p.ptr[13]] ^
          WUFFS_CRC32__IEEE_TABLE[3][v_p.ptr[12]] ^
          WUFFS_CRC32__IEEE_TABLE[4][v_p.ptr[11]] ^
          WUFFS_CRC32__IEEE_TABLE[5][v_p.ptr[10]] ^
          WUFFS_CRC32__IEEE_TABLE[6][v_p.ptr[9]] ^
          WUFFS_CRC32__IEEE_TABLE[7][v_p.ptr[8]] ^
          WUFFS_CRC32__IEEE_TABLE[8][v_p.ptr[7]] ^
          WUFFS_CRC32__IEEE_TABLE[9][v_p.ptr[6]] ^
          WUFFS_CRC32__IEEE_TABLE[10][v_p.ptr[5]] ^
          WUFFS_CRC32__IEEE_TABLE[11][v_p.ptr[4]] ^
          WUFFS_CRC32__IEEE_TABLE[12][(255 & (v_s >> 24))] ^
          WUFFS_CRC32__IEEE_TABLE[13][(255 & (v_s >> 16))] ^
          WUFFS_CRC32__IEEE_TABLE[14][(255 & (v_s >> 8))] ^
          WUFFS_CRC32__IEEE_TABLE[15][(255 & (v_s >> 0))]);
      v_p.ptr += 16;
      v_s ^= ((((uint32_t)(v_p.ptr[0])) << 0) |
          (((uint32_t)(v_p.ptr[1])) << 8) |
          (((uint32_t)(v_p.ptr[2])) << 16) |
          (((uint32_t)(v_p.ptr[3])) << 24));
      v_s = (WUFFS_CRC32__IEEE_TABLE[0][v_p.ptr[15]] ^
          WUFFS_CRC32__IEEE_TABLE[1][v_p.ptr[14]] ^
          WUFFS_CRC32__IEEE_TABLE[2][v_p.ptr[13]] ^
          WUFFS_CRC32__IEEE_TABLE[3][v_p.ptr[12]] ^
          WUFFS_CRC32__IEEE_TABLE[4][v_p.ptr[11]] ^
          WUFFS_CRC32__IEEE_TABLE[5][v_p.ptr[10]] ^
          WUFFS_CRC32__IEEE_TABLE[6][v_p.ptr[9]] ^
          WUFFS_CRC32__IEEE_TABLE[7][v_p.ptr[8]] ^
          WUFFS_CRC32__IEEE_TABLE[8][v_p.ptr[7]] ^
          WUFFS_CRC32__IEEE_TABLE[9][v_p.ptr[6]] ^
          WUFFS_CRC32__IEEE_TABLE[10][v_p.ptr[5]] ^
          WUFFS_CRC32__IEEE_TABLE[11][v_p.ptr[4]] ^
          WUFFS_CRC32__IEEE_TABLE[12][(255 & (v_s >> 24))] ^
          WUFFS_CRC32__IEEE_TABLE[13][(255 & (v_s >> 16))] ^
          WUFFS_CRC32__IEEE_TABLE[14][(255 & (v_s >> 8))] ^
          WUFFS_CRC32__IEEE_TABLE[15][(255 & (v_s >> 0))]);
      v_p.ptr += 16;
    }
    v_p.len = 16;
    uint8_t* i_end1_p = v_p.ptr + (((i_slice_p.len - (size_t)(v_p.ptr - i_slice_p.ptr)) / 16) * 16);
    while (v_p.ptr < i_end1_p) {
      v_s ^= ((((uint32_t)(v_p.ptr[0])) << 0) |
          (((uint32_t)(v_p.ptr[1])) << 8) |
          (((uint32_t)(v_p.ptr[2])) << 16) |
          (((uint32_t)(v_p.ptr[3])) << 24));
      v_s = (WUFFS_CRC32__IEEE_TABLE[0][v_p.ptr[15]] ^
          WUFFS_CRC32__IEEE_TABLE[1][v_p.ptr[14]] ^
          WUFFS_CRC32__IEEE_TABLE[2][v_p.ptr[13]] ^
          WUFFS_CRC32__IEEE_TABLE[3][v_p.ptr[12]] ^
          WUFFS_CRC32__IEEE_TABLE[4][v_p.ptr[11]] ^
          WUFFS_CRC32__IEEE_TABLE[5][v_p.ptr[10]] ^
          WUFFS_CRC32__IEEE_TABLE[6][v_p.ptr[9]] ^
          WUFFS_CRC32__IEEE_TABLE[7][v_p.ptr[8]] ^
          WUFFS_CRC32__IEEE_TABLE[8][v_p.ptr[7]] ^
          WUFFS_CRC32__IEEE_TABLE[9][v_p.ptr[6]] ^
          WUFFS_CRC32__IEEE_TABLE[10][v_p.ptr[5]] ^
          WUFFS_CRC32__IEEE_TABLE[11][v_p.ptr[4]] ^
          WUFFS_CRC32__IEEE_TABLE[12][(255 & (v_s >> 24))] ^
          WUFFS_CRC32__IEEE_TABLE[13][(255 & (v_s >> 16))] ^
          WUFFS_CRC32__IEEE_TABLE[14][(255 & (v_s >> 8))] ^
          WUFFS_CRC32__IEEE_TABLE[15][(255 & (v_s >> 0))]);
      v_p.ptr += 16;
    }
    v_p.len = 1;
    uint8_t* i_end2_p = i_slice_p.ptr + i_slice_p.len;
    while (v_p.ptr < i_end2_p) {
      v_s = (WUFFS_CRC32__IEEE_TABLE[0][(((uint8_t)((v_s & 255))) ^ v_p.ptr[0])] ^ (v_s >> 8));
      v_p.ptr += 1;
    }
    v_p.len = 0;
  }
  self->private_impl.f_state = (4294967295 ^ v_s);
  return wuffs_base__make_empty_struct();
}

// ‼ WUFFS MULTI-FILE SECTION +arm_crc32
// -------- func crc32.ieee_hasher.up_arm_crc32

#if defined(WUFFS_BASE__CPU_ARCH__ARM_CRC32)
static wuffs_base__empty_struct
wuffs_crc32__ieee_hasher__up_arm_crc32(
    wuffs_crc32__ieee_hasher* self,
    wuffs_base__slice_u8 a_x) {
  wuffs_base__slice_u8 v_p = {0};
  uint32_t v_s = 0;

  v_s = (4294967295 ^ self->private_impl.f_state);
  while ((((uint64_t)(a_x.len)) > 0) && ((15 & ((uint32_t)(0xFFF & (uintptr_t)(a_x.ptr)))) != 0)) {
    v_s = __crc32b(v_s, a_x.ptr[0]);
    a_x = wuffs_base__slice_u8__subslice_i(a_x, 1);
  }
  {
    wuffs_base__slice_u8 i_slice_p = a_x;
    v_p.ptr = i_slice_p.ptr;
    v_p.len = 8;
    uint8_t* i_end0_p = v_p.ptr + (((i_slice_p.len - (size_t)(v_p.ptr - i_slice_p.ptr)) / 128) * 128);
    while (v_p.ptr < i_end0_p) {
      v_s = __crc32d(v_s, wuffs_base__peek_u64le__no_bounds_check(v_p.ptr));
      v_p.ptr += 8;
      v_s = __crc32d(v_s, wuffs_base__peek_u64le__no_bounds_check(v_p.ptr));
      v_p.ptr += 8;
      v_s = __crc32d(v_s, wuffs_base__peek_u64le__no_bounds_check(v_p.ptr));
      v_p.ptr += 8;
      v_s = __crc32d(v_s, wuffs_base__peek_u64le__no_bounds_check(v_p.ptr));
      v_p.ptr += 8;
      v_s = __crc32d(v_s, wuffs_base__peek_u64le__no_bounds_check(v_p.ptr));
      v_p.ptr += 8;
      v_s = __crc32d(v_s, wuffs_base__peek_u64le__no_bounds_check(v_p.ptr));
      v_p.ptr += 8;
      v_s = __crc32d(v_s, wuffs_base__peek_u64le__no_bounds_check(v_p.ptr));
      v_p.ptr += 8;
      v_s = __crc32d(v_s, wuffs_base__peek_u64le__no_bounds_check(v_p.ptr));
      v_p.ptr += 8;
      v_s = __crc32d(v_s, wuffs_base__peek_u64le__no_bounds_check(v_p.ptr));
      v_p.ptr += 8;
      v_s = __crc32d(v_s, wuffs_base__peek_u64le__no_bounds_check(v_p.ptr));
      v_p.ptr += 8;
      v_s = __crc32d(v_s, wuffs_base__peek_u64le__no_bounds_check(v_p.ptr));
      v_p.ptr += 8;
      v_s = __crc32d(v_s, wuffs_base__peek_u64le__no_bounds_check(v_p.ptr));
      v_p.ptr += 8;
      v_s = __crc32d(v_s, wuffs_base__peek_u64le__no_bounds_check(v_p.ptr));
      v_p.ptr += 8;
      v_s = __crc32d(v_s, wuffs_base__peek_u64le__no_bounds_check(v_p.ptr));
      v_p.ptr += 8;
      v_s = __crc32d(v_s, wuffs_base__peek_u64le__no_bounds_check(v_p.ptr));
      v_p.ptr += 8;
      v_s = __crc32d(v_s, wuffs_base__peek_u64le__no_bounds_check(v_p.ptr));
      v_p.ptr += 8;
    }
    v_p.len = 8;
    uint8_t* i_end1_p = v_p.ptr + (((i_slice_p.len - (size_t)(v_p.ptr - i_slice_p.ptr)) / 8) * 8);
    while (v_p.ptr < i_end1_p) {
      v_s = __crc32d(v_s, wuffs_base__peek_u64le__no_bounds_check(v_p.ptr));
      v_p.ptr += 8;
    }
    v_p.len = 1;
    uint8_t* i_end2_p = i_slice_p.ptr + i_slice_p.len;
    while (v_p.ptr < i_end2_p) {
      v_s = __crc32b(v_s, v_p.ptr[0]);
      v_p.ptr += 1;
    }
    v_p.len = 0;
  }
  self->private_impl.f_state = (4294967295 ^ v_s);
  return wuffs_base__make_empty_struct();
}
#endif  // defined(WUFFS_BASE__CPU_ARCH__ARM_CRC32)
// ‼ WUFFS MULTI-FILE SECTION -arm_crc32

// ‼ WUFFS MULTI-FILE SECTION +x86_avx2
// -------- func crc32.ieee_hasher.up_x86_avx2

#if defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
WUFFS_BASE__MAYBE_ATTRIBUTE_TARGET("pclmul,popcnt,sse4.2,avx2")
static wuffs_base__empty_struct
wuffs_crc32__ieee_hasher__up_x86_avx2(
    wuffs_crc32__ieee_hasher* self,
    wuffs_base__slice_u8 a_x) {
  uint32_t v_s = 0;
  wuffs_base__slice_u8 v_p = {0};
  __m128i v_k = {0};
  __m128i v_x0 = {0};
  __m128i v_x1 = {0};
  __m128i v_x2 = {0};
  __m128i v_x3 = {0};
  __m128i v_y0 = {0};
  __m128i v_y1 = {0};
  __m128i v_y2 = {0};
  __m128i v_y3 = {0};
  uint64_t v_tail_index = 0;

  v_s = (4294967295 ^ self->private_impl.f_state);
  while ((((uint64_t)(a_x.len)) > 0) && ((15 & ((uint32_t)(0xFFF & (uintptr_t)(a_x.ptr)))) != 0)) {
    v_s = (WUFFS_CRC32__IEEE_TABLE[0][(((uint8_t)((v_s & 255))) ^ a_x.ptr[0])] ^ (v_s >> 8));
    a_x = wuffs_base__slice_u8__subslice_i(a_x, 1);
  }
  if (((uint64_t)(a_x.len)) < 64) {
    {
      wuffs_base__slice_u8 i_slice_p = a_x;
      v_p.ptr = i_slice_p.ptr;
      v_p.len = 1;
      uint8_t* i_end0_p = i_slice_p.ptr + i_slice_p.len;
      while (v_p.ptr < i_end0_p) {
        v_s = (WUFFS_CRC32__IEEE_TABLE[0][(((uint8_t)((v_s & 255))) ^ v_p.ptr[0])] ^ (v_s >> 8));
        v_p.ptr += 1;
      }
      v_p.len = 0;
    }
    self->private_impl.f_state = (4294967295 ^ v_s);
    return wuffs_base__make_empty_struct();
  }
  v_x0 = _mm_lddqu_si128((const __m128i*)(const void*)(a_x.ptr + 0));
  v_x1 = _mm_lddqu_si128((const __m128i*)(const void*)(a_x.ptr + 16));
  v_x2 = _mm_lddqu_si128((const __m128i*)(const void*)(a_x.ptr + 32));
  v_x3 = _mm_lddqu_si128((const __m128i*)(const void*)(a_x.ptr + 48));
  v_x0 = _mm_xor_si128(v_x0, _mm_cvtsi32_si128((int32_t)(v_s)));
  v_k = _mm_lddqu_si128((const __m128i*)(const void*)(WUFFS_CRC32__IEEE_X86_SSE42_K1K2));
  {
    wuffs_base__slice_u8 i_slice_p = wuffs_base__slice_u8__subslice_i(a_x, 64);
    v_p.ptr = i_slice_p.ptr;
    v_p.len = 64;
    uint8_t* i_end0_p = v_p.ptr + (((i_slice_p.len - (size_t)(v_p.ptr - i_slice_p.ptr)) / 64) * 64);
    while (v_p.ptr < i_end0_p) {
      v_y0 = _mm_clmulepi64_si128(v_x0, v_k, (int32_t)(0));
      v_y1 = _mm_clmulepi64_si128(v_x1, v_k, (int32_t)(0));
      v_y2 = _mm_clmulepi64_si128(v_x2, v_k, (int32_t)(0));
      v_y3 = _mm_clmulepi64_si128(v_x3, v_k, (int32_t)(0));
      v_x0 = _mm_clmulepi64_si128(v_x0, v_k, (int32_t)(17));
      v_x1 = _mm_clmulepi64_si128(v_x1, v_k, (int32_t)(17));
      v_x2 = _mm_clmulepi64_si128(v_x2, v_k, (int32_t)(17));
      v_x3 = _mm_clmulepi64_si128(v_x3, v_k, (int32_t)(17));
      v_x0 = _mm_xor_si128(_mm_xor_si128(v_x0, v_y0), _mm_lddqu_si128((const __m128i*)(const void*)(v_p.ptr + 0)));
      v_x1 = _mm_xor_si128(_mm_xor_si128(v_x1, v_y1), _mm_lddqu_si128((const __m128i*)(const void*)(v_p.ptr + 16)));
      v_x2 = _mm_xor_si128(_mm_xor_si128(v_x2, v_y2), _mm_lddqu_si128((const __m128i*)(const void*)(v_p.ptr + 32)));
      v_x3 = _mm_xor_si128(_mm_xor_si128(v_x3, v_y3), _mm_lddqu_si128((const __m128i*)(const void*)(v_p.ptr + 48)));
      v_p.ptr += 64;
    }
    v_p.len = 0;
  }
  v_k = _mm_lddqu_si128((const __m128i*)(const void*)(WUFFS_CRC32__IEEE_X86_SSE42_K3K4));
  v_y0 = _mm_clmulepi64_si128(v_x0, v_k, (int32_t)(0));
  v_x0 = _mm_clmulepi64_si128(v_x0, v_k, (int32_t)(17));
  v_x0 = _mm_xor_si128(v_x0, v_x1);
  v_x0 = _mm_xor_si128(v_x0, v_y0);
  v_y0 = _mm_clmulepi64_si128(v_x0, v_k, (int32_t)(0));
  v_x0 = _mm_clmulepi64_si128(v_x0, v_k, (int32_t)(17));
  v_x0 = _mm_xor_si128(v_x0, v_x2);
  v_x0 = _mm_xor_si128(v_x0, v_y0);
  v_y0 = _mm_clmulepi64_si128(v_x0, v_k, (int32_t)(0));
  v_x0 = _mm_clmulepi64_si128(v_x0, v_k, (int32_t)(17));
  v_x0 = _mm_xor_si128(v_x0, v_x3);
  v_x0 = _mm_xor_si128(v_x0, v_y0);
  v_x1 = _mm_clmulepi64_si128(v_x0, v_k, (int32_t)(16));
  v_x2 = _mm_set_epi32((int32_t)(0), (int32_t)(4294967295), (int32_t)(0), (int32_t)(4294967295));
  v_x0 = _mm_srli_si128(v_x0, (int32_t)(8));
  v_x0 = _mm_xor_si128(v_x0, v_x1);
  v_k = _mm_lddqu_si128((const __m128i*)(const void*)(WUFFS_CRC32__IEEE_X86_SSE42_K5ZZ));
  v_x1 = _mm_srli_si128(v_x0, (int32_t)(4));
  v_x0 = _mm_and_si128(v_x0, v_x2);
  v_x0 = _mm_clmulepi64_si128(v_x0, v_k, (int32_t)(0));
  v_x0 = _mm_xor_si128(v_x0, v_x1);
  v_k = _mm_lddqu_si128((const __m128i*)(const void*)(WUFFS_CRC32__IEEE_X86_SSE42_PXMU));
  v_x1 = _mm_and_si128(v_x0, v_x2);
  v_x1 = _mm_clmulepi64_si128(v_x1, v_k, (int32_t)(16));
  v_x1 = _mm_and_si128(v_x1, v_x2);
  v_x1 = _mm_clmulepi64_si128(v_x1, v_k, (int32_t)(0));
  v_x0 = _mm_xor_si128(v_x0, v_x1);
  v_s = ((uint32_t)(_mm_extract_epi32(v_x0, (int32_t)(1))));
  v_tail_index = (((uint64_t)(a_x.len)) & 18446744073709551552u);
  if (v_tail_index < ((uint64_t)(a_x.len))) {
    {
      wuffs_base__slice_u8 i_slice_p = wuffs_base__slice_u8__subslice_i(a_x, v_tail_index);
      v_p.ptr = i_slice_p.ptr;
      v_p.len = 1;
      uint8_t* i_end0_p = i_slice_p.ptr + i_slice_p.len;
      while (v_p.ptr < i_end0_p) {
        v_s = (WUFFS_CRC32__IEEE_TABLE[0][(((uint8_t)((v_s & 255))) ^ v_p.ptr[0])] ^ (v_s >> 8));
        v_p.ptr += 1;
      }
      v_p.len = 0;
    }
  }
  self->private_impl.f_state = (4294967295 ^ v_s);
  return wuffs_base__make_empty_struct();
}
#endif  // defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
// ‼ WUFFS MULTI-FILE SECTION -x86_avx2

// ‼ WUFFS MULTI-FILE SECTION +x86_sse42
// -------- func crc32.ieee_hasher.up_x86_sse42

#if defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
WUFFS_BASE__MAYBE_ATTRIBUTE_TARGET("pclmul,popcnt,sse4.2")
static wuffs_base__empty_struct
wuffs_crc32__ieee_hasher__up_x86_sse42(
    wuffs_crc32__ieee_hasher* self,
    wuffs_base__slice_u8 a_x) {
  uint32_t v_s = 0;
  wuffs_base__slice_u8 v_p = {0};
  __m128i v_k = {0};
  __m128i v_x0 = {0};
  __m128i v_x1 = {0};
  __m128i v_x2 = {0};
  __m128i v_x3 = {0};
  __m128i v_y0 = {0};
  __m128i v_y1 = {0};
  __m128i v_y2 = {0};
  __m128i v_y3 = {0};
  uint64_t v_tail_index = 0;

  v_s = (4294967295 ^ self->private_impl.f_state);
  while ((((uint64_t)(a_x.len)) > 0) && ((15 & ((uint32_t)(0xFFF & (uintptr_t)(a_x.ptr)))) != 0)) {
    v_s = (WUFFS_CRC32__IEEE_TABLE[0][(((uint8_t)((v_s & 255))) ^ a_x.ptr[0])] ^ (v_s >> 8));
    a_x = wuffs_base__slice_u8__subslice_i(a_x, 1);
  }
  if (((uint64_t)(a_x.len)) < 64) {
    {
      wuffs_base__slice_u8 i_slice_p = a_x;
      v_p.ptr = i_slice_p.ptr;
      v_p.len = 1;
      uint8_t* i_end0_p = i_slice_p.ptr + i_slice_p.len;
      while (v_p.ptr < i_end0_p) {
        v_s = (WUFFS_CRC32__IEEE_TABLE[0][(((uint8_t)((v_s & 255))) ^ v_p.ptr[0])] ^ (v_s >> 8));
        v_p.ptr += 1;
      }
      v_p.len = 0;
    }
    self->private_impl.f_state = (4294967295 ^ v_s);
    return wuffs_base__make_empty_struct();
  }
  v_x0 = _mm_lddqu_si128((const __m128i*)(const void*)(a_x.ptr + 0));
  v_x1 = _mm_lddqu_si128((const __m128i*)(const void*)(a_x.ptr + 16));
  v_x2 = _mm_lddqu_si128((const __m128i*)(const void*)(a_x.ptr + 32));
  v_x3 = _mm_lddqu_si128((const __m128i*)(const void*)(a_x.ptr + 48));
  v_x0 = _mm_xor_si128(v_x0, _mm_cvtsi32_si128((int32_t)(v_s)));
  v_k = _mm_lddqu_si128((const __m128i*)(const void*)(WUFFS_CRC32__IEEE_X86_SSE42_K1K2));
  {
    wuffs_base__slice_u8 i_slice_p = wuffs_base__slice_u8__subslice_i(a_x, 64);
    v_p.ptr = i_slice_p.ptr;
    v_p.len = 64;
    uint8_t* i_end0_p = v_p.ptr + (((i_slice_p.len - (size_t)(v_p.ptr - i_slice_p.ptr)) / 64) * 64);
    while (v_p.ptr < i_end0_p) {
      v_y0 = _mm_clmulepi64_si128(v_x0, v_k, (int32_t)(0));
      v_y1 = _mm_clmulepi64_si128(v_x1, v_k, (int32_t)(0));
      v_y2 = _mm_clmulepi64_si128(v_x2, v_k, (int32_t)(0));
      v_y3 = _mm_clmulepi64_si128(v_x3, v_k, (int32_t)(0));
      v_x0 = _mm_clmulepi64_si128(v_x0, v_k, (int32_t)(17));
      v_x1 = _mm_clmulepi64_si128(v_x1, v_k, (int32_t)(17));
      v_x2 = _mm_clmulepi64_si128(v_x2, v_k, (int32_t)(17));
      v_x3 = _mm_clmulepi64_si128(v_x3, v_k, (int32_t)(17));
      v_x0 = _mm_xor_si128(_mm_xor_si128(v_x0, v_y0), _mm_lddqu_si128((const __m128i*)(const void*)(v_p.ptr + 0)));
      v_x1 = _mm_xor_si128(_mm_xor_si128(v_x1, v_y1), _mm_lddqu_si128((const __m128i*)(const void*)(v_p.ptr + 16)));
      v_x2 = _mm_xor_si128(_mm_xor_si128(v_x2, v_y2), _mm_lddqu_si128((const __m128i*)(const void*)(v_p.ptr + 32)));
      v_x3 = _mm_xor_si128(_mm_xor_si128(v_x3, v_y3), _mm_lddqu_si128((const __m128i*)(const void*)(v_p.ptr + 48)));
      v_p.ptr += 64;
    }
    v_p.len = 0;
  }
  v_k = _mm_lddqu_si128((const __m128i*)(const void*)(WUFFS_CRC32__IEEE_X86_SSE42_K3K4));
  v_y0 = _mm_clmulepi64_si128(v_x0, v_k, (int32_t)(0));
  v_x0 = _mm_clmulepi64_si128(v_x0, v_k, (int32_t)(17));
  v_x0 = _mm_xor_si128(v_x0, v_x1);
  v_x0 = _mm_xor_si128(v_x0, v_y0);
  v_y0 = _mm_clmulepi64_si128(v_x0, v_k, (int32_t)(0));
  v_x0 = _mm_clmulepi64_si128(v_x0, v_k, (int32_t)(17));
  v_x0 = _mm_xor_si128(v_x0, v_x2);
  v_x0 = _mm_xor_si128(v_x0, v_y0);
  v_y0 = _mm_clmulepi64_si128(v_x0, v_k, (int32_t)(0));
  v_x0 = _mm_clmulepi64_si128(v_x0, v_k, (int32_t)(17));
  v_x0 = _mm_xor_si128(v_x0, v_x3);
  v_x0 = _mm_xor_si128(v_x0, v_y0);
  v_x1 = _mm_clmulepi64_si128(v_x0, v_k, (int32_t)(16));
  v_x2 = _mm_set_epi32((int32_t)(0), (int32_t)(4294967295), (int32_t)(0), (int32_t)(4294967295));
  v_x0 = _mm_srli_si128(v_x0, (int32_t)(8));
  v_x0 = _mm_xor_si128(v_x0, v_x1);
  v_k = _mm_lddqu_si128((const __m128i*)(const void*)(WUFFS_CRC32__IEEE_X86_SSE42_K5ZZ));
  v_x1 = _mm_srli_si128(v_x0, (int32_t)(4));
  v_x0 = _mm_and_si128(v_x0, v_x2);
  v_x0 = _mm_clmulepi64_si128(v_x0, v_k, (int32_t)(0));
  v_x0 = _mm_xor_si128(v_x0, v_x1);
  v_k = _mm_lddqu_si128((const __m128i*)(const void*)(WUFFS_CRC32__IEEE_X86_SSE42_PXMU));
  v_x1 = _mm_and_si128(v_x0, v_x2);
  v_x1 = _mm_clmulepi64_si128(v_x1, v_k, (int32_t)(16));
  v_x1 = _mm_and_si128(v_x1, v_x2);
  v_x1 = _mm_clmulepi64_si128(v_x1, v_k, (int32_t)(0));
  v_x0 = _mm_xor_si128(v_x0, v_x1);
  v_s = ((uint32_t)(_mm_extract_epi32(v_x0, (int32_t)(1))));
  v_tail_index = (((uint64_t)(a_x.len)) & 18446744073709551552u);
  if (v_tail_index < ((uint64_t)(a_x.len))) {
    {
      wuffs_base__slice_u8 i_slice_p = wuffs_base__slice_u8__subslice_i(a_x, v_tail_index);
      v_p.ptr = i_slice_p.ptr;
      v_p.len = 1;
      uint8_t* i_end0_p = i_slice_p.ptr + i_slice_p.len;
      while (v_p.ptr < i_end0_p) {
        v_s = (WUFFS_CRC32__IEEE_TABLE[0][(((uint8_t)((v_s & 255))) ^ v_p.ptr[0])] ^ (v_s >> 8));
        v_p.ptr += 1;
      }
      v_p.len = 0;
    }
  }
  self->private_impl.f_state = (4294967295 ^ v_s);
  return wuffs_base__make_empty_struct();
}
#endif  // defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
// ‼ WUFFS MULTI-FILE SECTION -x86_sse42

#endif  // !defined(WUFFS_CONFIG__MODULES) || defined(WUFFS_CONFIG__MODULE__CRC32)

#if !defined(WUFFS_CONFIG__MODULES) || defined(WUFFS_CONFIG__MODULE__DEFLATE)

// ---------------- Status Codes Implementations

const char wuffs_deflate__error__bad_huffman_code_over_subscribed[] = "#deflate: bad Huffman code (over-subscribed)";
const char wuffs_deflate__error__bad_huffman_code_under_subscribed[] = "#deflate: bad Huffman code (under-subscribed)";
const char wuffs_deflate__error__bad_huffman_code_length_count[] = "#deflate: bad Huffman code length count";
const char wuffs_deflate__error__bad_huffman_code_length_repetition[] = "#deflate: bad Huffman code length repetition";
const char wuffs_deflate__error__bad_huffman_code[] = "#deflate: bad Huffman code";
const char wuffs_deflate__error__bad_huffman_minimum_code_length[] = "#deflate: bad Huffman minimum code length";
const char wuffs_deflate__error__bad_block[] = "#deflate: bad block";
const char wuffs_deflate__error__bad_distance[] = "#deflate: bad distance";
const char wuffs_deflate__error__bad_distance_code_count[] = "#deflate: bad distance code count";
const char wuffs_deflate__error__bad_literal_length_code_count[] = "#deflate: bad literal/length code count";
const char wuffs_deflate__error__inconsistent_stored_block_length[] = "#deflate: inconsistent stored block length";
const char wuffs_deflate__error__missing_end_of_block_code[] = "#deflate: missing end-of-block code";
const char wuffs_deflate__error__no_huffman_codes[] = "#deflate: no Huffman codes";
const char wuffs_deflate__error__truncated_input[] = "#deflate: truncated input";
const char wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state[] = "#deflate: internal error: inconsistent Huffman decoder state";
const char wuffs_deflate__error__internal_error_inconsistent_i_o[] = "#deflate: internal error: inconsistent I/O";
const char wuffs_deflate__error__internal_error_inconsistent_distance[] = "#deflate: internal error: inconsistent distance";
const char wuffs_deflate__error__internal_error_inconsistent_n_bits[] = "#deflate: internal error: inconsistent n_bits";

// ---------------- Private Consts

static const uint8_t
WUFFS_DEFLATE__CODE_ORDER[19] WUFFS_BASE__POTENTIALLY_UNUSED = {
  16, 17, 18, 0, 8, 7, 9, 6,
  10, 5, 11, 4, 12, 3, 13, 2,
  14, 1, 15,
};

static const uint8_t
WUFFS_DEFLATE__REVERSE8[256] WUFFS_BASE__POTENTIALLY_UNUSED = {
  0, 128, 64, 192, 32, 160, 96, 224,
  16, 144, 80, 208, 48, 176, 112, 240,
  8, 136, 72, 200, 40, 168, 104, 232,
  24, 152, 88, 216, 56, 184, 120, 248,
  4, 132, 68, 196, 36, 164, 100, 228,
  20, 148, 84, 212, 52, 180, 116, 244,
  12, 140, 76, 204, 44, 172, 108, 236,
  28, 156, 92, 220, 60, 188, 124, 252,
  2, 130, 66, 194, 34, 162, 98, 226,
  18, 146, 82, 210, 50, 178, 114, 242,
  10, 138, 74, 202, 42, 170, 106, 234,
  26, 154, 90, 218, 58, 186, 122, 250,
  6, 134, 70, 198, 38, 166, 102, 230,
  22, 150, 86, 214, 54, 182, 118, 246,
  14, 142, 78, 206, 46, 174, 110, 238,
  30, 158, 94, 222, 62, 190, 126, 254,
  1, 129, 65, 193, 33, 161, 97, 225,
  17, 145, 81, 209, 49, 177, 113, 241,
  9, 137, 73, 201, 41, 169, 105, 233,
  25, 153, 89, 217, 57, 185, 121, 249,
  5, 133, 69, 197, 37, 165, 101, 229,
  21, 149, 85, 213, 53, 181, 117, 245,
  13, 141, 77, 205, 45, 173, 109, 237,
  29, 157, 93, 221, 61, 189, 125, 253,
  3, 131, 67, 195, 35, 163, 99, 227,
  19, 147, 83, 211, 51, 179, 115, 243,
  11, 139, 75, 203, 43, 171, 107, 235,
  27, 155, 91, 219, 59, 187, 123, 251,
  7, 135, 71, 199, 39, 167, 103, 231,
  23, 151, 87, 215, 55, 183, 119, 247,
  15, 143, 79, 207, 47, 175, 111, 239,
  31, 159, 95, 223, 63, 191, 127, 255,
};

static const uint32_t
WUFFS_DEFLATE__LCODE_MAGIC_NUMBERS[32] WUFFS_BASE__POTENTIALLY_UNUSED = {
  1073741824, 1073742080, 1073742336, 1073742592, 1073742848, 1073743104, 1073743360, 1073743616,
  1073743888, 1073744400, 1073744912, 1073745424, 1073745952, 1073746976, 1073748000, 1073749024,
  1073750064, 1073752112, 1073754160, 1073756208, 1073758272, 1073762368, 1073766464, 1073770560,
  1073774672, 1073782864, 1073791056, 1073799248, 1073807104, 134217728, 134217728, 134217728,
};

static const uint32_t
WUFFS_DEFLATE__DCODE_MAGIC_NUMBERS[32] WUFFS_BASE__POTENTIALLY_UNUSED = {
  1073741824, 1073742080, 1073742336, 1073742592, 1073742864, 1073743376, 1073743904, 1073744928,
  1073745968, 1073748016, 1073750080, 1073754176, 1073758288, 1073766480, 1073774688, 1073791072,
  1073807472, 1073840240, 1073873024, 1073938560, 1074004112, 1074135184, 1074266272, 1074528416,
  1074790576, 1075314864, 1075839168, 1076887744, 1077936336, 1080033488, 134217728, 134217728,
};

#define WUFFS_DEFLATE__HUFFS_TABLE_SIZE 1024

#define WUFFS_DEFLATE__HUFFS_TABLE_MASK 1023

// ---------------- Private Initializer Prototypes

// ---------------- Private Function Prototypes

static wuffs_base__status
wuffs_deflate__decoder__do_transform_io(
    wuffs_deflate__decoder* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__io_buffer* a_src,
    wuffs_base__slice_u8 a_workbuf);

static wuffs_base__status
wuffs_deflate__decoder__decode_blocks(
    wuffs_deflate__decoder* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__io_buffer* a_src);

static wuffs_base__status
wuffs_deflate__decoder__decode_uncompressed(
    wuffs_deflate__decoder* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__io_buffer* a_src);

static wuffs_base__status
wuffs_deflate__decoder__init_fixed_huffman(
    wuffs_deflate__decoder* self);

static wuffs_base__status
wuffs_deflate__decoder__init_dynamic_huffman(
    wuffs_deflate__decoder* self,
    wuffs_base__io_buffer* a_src);

static wuffs_base__status
wuffs_deflate__decoder__init_huff(
    wuffs_deflate__decoder* self,
    uint32_t a_which,
    uint32_t a_n_codes0,
    uint32_t a_n_codes1,
    uint32_t a_base_symbol);

#if defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
static wuffs_base__status
wuffs_deflate__decoder__decode_huffman_bmi2(
    wuffs_deflate__decoder* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__io_buffer* a_src);
#endif  // defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)

static wuffs_base__status
wuffs_deflate__decoder__decode_huffman_fast32(
    wuffs_deflate__decoder* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__io_buffer* a_src);

static wuffs_base__status
wuffs_deflate__decoder__decode_huffman_fast64(
    wuffs_deflate__decoder* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__io_buffer* a_src);

static wuffs_base__status
wuffs_deflate__decoder__decode_huffman_fast64__choosy_default(
    wuffs_deflate__decoder* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__io_buffer* a_src);

static wuffs_base__status
wuffs_deflate__decoder__decode_huffman_slow(
    wuffs_deflate__decoder* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__io_buffer* a_src);

// ---------------- VTables

const wuffs_base__io_transformer__func_ptrs
wuffs_deflate__decoder__func_ptrs_for__wuffs_base__io_transformer = {
  (wuffs_base__empty_struct(*)(void*,
      uint32_t,
      bool))(&wuffs_deflate__decoder__set_quirk_enabled),
  (wuffs_base__status(*)(void*,
      wuffs_base__io_buffer*,
      wuffs_base__io_buffer*,
      wuffs_base__slice_u8))(&wuffs_deflate__decoder__transform_io),
  (wuffs_base__range_ii_u64(*)(const void*))(&wuffs_deflate__decoder__workbuf_len),
};

// ---------------- Initializer Implementations

wuffs_base__status WUFFS_BASE__WARN_UNUSED_RESULT
wuffs_deflate__decoder__initialize(
    wuffs_deflate__decoder* self,
    size_t sizeof_star_self,
    uint64_t wuffs_version,
    uint32_t options){
  if (!self) {
    return wuffs_base__make_status(wuffs_base__error__bad_receiver);
  }
  if (sizeof(*self) != sizeof_star_self) {
    return wuffs_base__make_status(wuffs_base__error__bad_sizeof_receiver);
  }
  if (((wuffs_version >> 32) != WUFFS_VERSION_MAJOR) ||
      (((wuffs_version >> 16) & 0xFFFF) > WUFFS_VERSION_MINOR)) {
    return wuffs_base__make_status(wuffs_base__error__bad_wuffs_version);
  }

  if ((options & WUFFS_INITIALIZE__ALREADY_ZEROED) != 0) {
    // The whole point of this if-check is to detect an uninitialized *self.
    // We disable the warning on GCC. Clang-5.0 does not have this warning.
#if !defined(__clang__) && defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
    if (self->private_impl.magic != 0) {
      return wuffs_base__make_status(wuffs_base__error__initialize_falsely_claimed_already_zeroed);
    }
#if !defined(__clang__) && defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
  } else {
    if ((options & WUFFS_INITIALIZE__LEAVE_INTERNAL_BUFFERS_UNINITIALIZED) == 0) {
      memset(self, 0, sizeof(*self));
      options |= WUFFS_INITIALIZE__ALREADY_ZEROED;
    } else {
      memset(&(self->private_impl), 0, sizeof(self->private_impl));
    }
  }

  self->private_impl.choosy_decode_huffman_fast64 = &wuffs_deflate__decoder__decode_huffman_fast64__choosy_default;

  self->private_impl.magic = WUFFS_BASE__MAGIC;
  self->private_impl.vtable_for__wuffs_base__io_transformer.vtable_name =
      wuffs_base__io_transformer__vtable_name;
  self->private_impl.vtable_for__wuffs_base__io_transformer.function_pointers =
      (const void*)(&wuffs_deflate__decoder__func_ptrs_for__wuffs_base__io_transformer);
  return wuffs_base__make_status(NULL);
}

wuffs_deflate__decoder*
wuffs_deflate__decoder__alloc() {
  wuffs_deflate__decoder* x =
      (wuffs_deflate__decoder*)(calloc(sizeof(wuffs_deflate__decoder), 1));
  if (!x) {
    return NULL;
  }
  if (wuffs_deflate__decoder__initialize(
      x, sizeof(wuffs_deflate__decoder), WUFFS_VERSION, WUFFS_INITIALIZE__ALREADY_ZEROED).repr) {
    free(x);
    return NULL;
  }
  return x;
}

size_t
sizeof__wuffs_deflate__decoder() {
  return sizeof(wuffs_deflate__decoder);
}

// ---------------- Function Implementations

// -------- func deflate.decoder.add_history

WUFFS_BASE__MAYBE_STATIC wuffs_base__empty_struct
wuffs_deflate__decoder__add_history(
    wuffs_deflate__decoder* self,
    wuffs_base__slice_u8 a_hist) {
  if (!self) {
    return wuffs_base__make_empty_struct();
  }
  if (self->private_impl.magic != WUFFS_BASE__MAGIC) {
    return wuffs_base__make_empty_struct();
  }

  wuffs_base__slice_u8 v_s = {0};
  uint64_t v_n_copied = 0;
  uint32_t v_already_full = 0;

  v_s = a_hist;
  if (((uint64_t)(v_s.len)) >= 32768) {
    v_s = wuffs_base__slice_u8__suffix(v_s, 32768);
    wuffs_base__slice_u8__copy_from_slice(wuffs_base__make_slice_u8(self->private_data.f_history, 32768), v_s);
    self->private_impl.f_history_index = 32768;
  } else {
    v_n_copied = wuffs_base__slice_u8__copy_from_slice(wuffs_base__make_slice_u8_ij(self->private_data.f_history, (self->private_impl.f_history_index & 32767), 32768), v_s);
    if (v_n_copied < ((uint64_t)(v_s.len))) {
      v_s = wuffs_base__slice_u8__subslice_i(v_s, v_n_copied);
      v_n_copied = wuffs_base__slice_u8__copy_from_slice(wuffs_base__make_slice_u8(self->private_data.f_history, 32768), v_s);
      self->private_impl.f_history_index = (((uint32_t)((v_n_copied & 32767))) + 32768);
    } else {
      v_already_full = 0;
      if (self->private_impl.f_history_index >= 32768) {
        v_already_full = 32768;
      }
      self->private_impl.f_history_index = ((self->private_impl.f_history_index & 32767) + ((uint32_t)((v_n_copied & 32767))) + v_already_full);
    }
  }
  wuffs_base__slice_u8__copy_from_slice(wuffs_base__make_slice_u8_ij(self->private_data.f_history, 32768, 33025), wuffs_base__make_slice_u8(self->private_data.f_history, 33025));
  return wuffs_base__make_empty_struct();
}

// -------- func deflate.decoder.set_quirk_enabled

WUFFS_BASE__MAYBE_STATIC wuffs_base__empty_struct
wuffs_deflate__decoder__set_quirk_enabled(
    wuffs_deflate__decoder* self,
    uint32_t a_quirk,
    bool a_enabled) {
  return wuffs_base__make_empty_struct();
}

// -------- func deflate.decoder.workbuf_len

WUFFS_BASE__MAYBE_STATIC wuffs_base__range_ii_u64
wuffs_deflate__decoder__workbuf_len(
    const wuffs_deflate__decoder* self) {
  if (!self) {
    return wuffs_base__utility__empty_range_ii_u64();
  }
  if ((self->private_impl.magic != WUFFS_BASE__MAGIC) &&
      (self->private_impl.magic != WUFFS_BASE__DISABLED)) {
    return wuffs_base__utility__empty_range_ii_u64();
  }

  return wuffs_base__utility__make_range_ii_u64(1, 1);
}

// -------- func deflate.decoder.transform_io

WUFFS_BASE__MAYBE_STATIC wuffs_base__status
wuffs_deflate__decoder__transform_io(
    wuffs_deflate__decoder* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__io_buffer* a_src,
    wuffs_base__slice_u8 a_workbuf) {
  if (!self) {
    return wuffs_base__make_status(wuffs_base__error__bad_receiver);
  }
  if (self->private_impl.magic != WUFFS_BASE__MAGIC) {
    return wuffs_base__make_status(
        (self->private_impl.magic == WUFFS_BASE__DISABLED)
        ? wuffs_base__error__disabled_by_previous_error
        : wuffs_base__error__initialize_not_called);
  }
  if (!a_dst || !a_src) {
    self->private_impl.magic = WUFFS_BASE__DISABLED;
    return wuffs_base__make_status(wuffs_base__error__bad_argument);
  }
  if ((self->private_impl.active_coroutine != 0) &&
      (self->private_impl.active_coroutine != 1)) {
    self->private_impl.magic = WUFFS_BASE__DISABLED;
    return wuffs_base__make_status(wuffs_base__error__interleaved_coroutine_calls);
  }
  self->private_impl.active_coroutine = 0;
  wuffs_base__status status = wuffs_base__make_status(NULL);

  wuffs_base__status v_status = wuffs_base__make_status(NULL);

  uint32_t coro_susp_point = self->private_impl.p_transform_io[0];
  switch (coro_susp_point) {
    WUFFS_BASE__COROUTINE_SUSPENSION_POINT_0;

    while (true) {
      {
        wuffs_base__status t_0 = wuffs_deflate__decoder__do_transform_io(self, a_dst, a_src, a_workbuf);
        v_status = t_0;
      }
      if ((v_status.repr == wuffs_base__suspension__short_read) && (a_src && a_src->meta.closed)) {
        status = wuffs_base__make_status(wuffs_deflate__error__truncated_input);
        goto exit;
      }
      status = v_status;
      WUFFS_BASE__COROUTINE_SUSPENSION_POINT_MAYBE_SUSPEND(1);
    }

    ok:
    self->private_impl.p_transform_io[0] = 0;
    goto exit;
  }

  goto suspend;
  suspend:
  self->private_impl.p_transform_io[0] = wuffs_base__status__is_suspension(&status) ? coro_susp_point : 0;
  self->private_impl.active_coroutine = wuffs_base__status__is_suspension(&status) ? 1 : 0;

  goto exit;
  exit:
  if (wuffs_base__status__is_error(&status)) {
    self->private_impl.magic = WUFFS_BASE__DISABLED;
  }
  return status;
}

// -------- func deflate.decoder.do_transform_io

static wuffs_base__status
wuffs_deflate__decoder__do_transform_io(
    wuffs_deflate__decoder* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__io_buffer* a_src,
    wuffs_base__slice_u8 a_workbuf) {
  wuffs_base__status status = wuffs_base__make_status(NULL);

  uint64_t v_mark = 0;
  wuffs_base__status v_status = wuffs_base__make_status(NULL);

  uint8_t* iop_a_dst = NULL;
  uint8_t* io0_a_dst WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  uint8_t* io1_a_dst WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  uint8_t* io2_a_dst WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  if (a_dst && a_dst->data.ptr) {
    io0_a_dst = a_dst->data.ptr;
    io1_a_dst = io0_a_dst + a_dst->meta.wi;
    iop_a_dst = io1_a_dst;
    io2_a_dst = io0_a_dst + a_dst->data.len;
    if (a_dst->meta.closed) {
      io2_a_dst = iop_a_dst;
    }
  }

  uint32_t coro_susp_point = self->private_impl.p_do_transform_io[0];
  switch (coro_susp_point) {
    WUFFS_BASE__COROUTINE_SUSPENSION_POINT_0;

    self->private_impl.choosy_decode_huffman_fast64 = (
#if defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
        wuffs_base__cpu_arch__have_x86_bmi2() ? &wuffs_deflate__decoder__decode_huffman_bmi2 :
#endif
        self->private_impl.choosy_decode_huffman_fast64);
    while (true) {
      v_mark = ((uint64_t)(iop_a_dst - io0_a_dst));
      {
        if (a_dst) {
          a_dst->meta.wi = ((size_t)(iop_a_dst - a_dst->data.ptr));
        }
        wuffs_base__status t_0 = wuffs_deflate__decoder__decode_blocks(self, a_dst, a_src);
        v_status = t_0;
        if (a_dst) {
          iop_a_dst = a_dst->data.ptr + a_dst->meta.wi;
        }
      }
      if ( ! wuffs_base__status__is_suspension(&v_status)) {
        status = v_status;
        if (wuffs_base__status__is_error(&status)) {
          goto exit;
        } else if (wuffs_base__status__is_suspension(&status)) {
          status = wuffs_base__make_status(wuffs_base__error__cannot_return_a_suspension);
          goto exit;
        }
        goto ok;
      }
      wuffs_base__u64__sat_add_indirect(&self->private_impl.f_transformed_history_count, wuffs_base__io__count_since(v_mark, ((uint64_t)(iop_a_dst - io0_a_dst))));
      wuffs_deflate__decoder__add_history(self, wuffs_base__io__since(v_mark, ((uint64_t)(iop_a_dst - io0_a_dst)), io0_a_dst));
      status = v_status;
      WUFFS_BASE__COROUTINE_SUSPENSION_POINT_MAYBE_SUSPEND(1);
    }

    ok:
    self->private_impl.p_do_transform_io[0] = 0;
    goto exit;
  }

  goto suspend;
  suspend:
  self->private_impl.p_do_transform_io[0] = wuffs_base__status__is_suspension(&status) ? coro_susp_point : 0;

  goto exit;
  exit:
  if (a_dst && a_dst->data.ptr) {
    a_dst->meta.wi = ((size_t)(iop_a_dst - a_dst->data.ptr));
  }

  return status;
}

// -------- func deflate.decoder.decode_blocks

static wuffs_base__status
wuffs_deflate__decoder__decode_blocks(
    wuffs_deflate__decoder* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__io_buffer* a_src) {
  wuffs_base__status status = wuffs_base__make_status(NULL);

  uint32_t v_final = 0;
  uint32_t v_b0 = 0;
  uint32_t v_type = 0;
  wuffs_base__status v_status = wuffs_base__make_status(NULL);

  const uint8_t* iop_a_src = NULL;
  const uint8_t* io0_a_src WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  const uint8_t* io1_a_src WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  const uint8_t* io2_a_src WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  if (a_src && a_src->data.ptr) {
    io0_a_src = a_src->data.ptr;
    io1_a_src = io0_a_src + a_src->meta.ri;
    iop_a_src = io1_a_src;
    io2_a_src = io0_a_src + a_src->meta.wi;
  }

  uint32_t coro_susp_point = self->private_impl.p_decode_blocks[0];
  if (coro_susp_point) {
    v_final = self->private_data.s_decode_blocks[0].v_final;
  }
  switch (coro_susp_point) {
    WUFFS_BASE__COROUTINE_SUSPENSION_POINT_0;

    label__outer__continue:;
    while (v_final == 0) {
      while (self->private_impl.f_n_bits < 3) {
        {
          WUFFS_BASE__COROUTINE_SUSPENSION_POINT(1);
          if (WUFFS_BASE__UNLIKELY(iop_a_src == io2_a_src)) {
            status = wuffs_base__make_status(wuffs_base__suspension__short_read);
            goto suspend;
          }
          uint32_t t_0 = *iop_a_src++;
          v_b0 = t_0;
        }
        self->private_impl.f_bits |= (v_b0 << (self->private_impl.f_n_bits & 3));
        self->private_impl.f_n_bits = ((self->private_impl.f_n_bits & 3) + 8);
      }
      v_final = (self->private_impl.f_bits & 1);
      v_type = ((self->private_impl.f_bits >> 1) & 3);
      self->private_impl.f_bits >>= 3;
      self->private_impl.f_n_bits -= 3;
      if (v_type == 0) {
        if (a_src) {
          a_src->meta.ri = ((size_t)(iop_a_src - a_src->data.ptr));
        }
        WUFFS_BASE__COROUTINE_SUSPENSION_POINT(2);
        status = wuffs_deflate__decoder__decode_uncompressed(self, a_dst, a_src);
        if (a_src) {
          iop_a_src = a_src->data.ptr + a_src->meta.ri;
        }
        if (status.repr) {
          goto suspend;
        }
        goto label__outer__continue;
      } else if (v_type == 1) {
        v_status = wuffs_deflate__decoder__init_fixed_huffman(self);
        if ( ! wuffs_base__status__is_ok(&v_status)) {
          status = v_status;
          if (wuffs_base__status__is_error(&status)) {
            goto exit;
          } else if (wuffs_base__status__is_suspension(&status)) {
            status = wuffs_base__make_status(wuffs_base__error__cannot_return_a_suspension);
            goto exit;
          }
          goto ok;
        }
      } else if (v_type == 2) {
        if (a_src) {
          a_src->meta.ri = ((size_t)(iop_a_src - a_src->data.ptr));
        }
        WUFFS_BASE__COROUTINE_SUSPENSION_POINT(3);
        status = wuffs_deflate__decoder__init_dynamic_huffman(self, a_src);
        if (a_src) {
          iop_a_src = a_src->data.ptr + a_src->meta.ri;
        }
        if (status.repr) {
          goto suspend;
        }
      } else {
        status = wuffs_base__make_status(wuffs_deflate__error__bad_block);
        goto exit;
      }
      self->private_impl.f_end_of_block = false;
      while (true) {
        if (sizeof(void*) == 4) {
          if (a_src) {
            a_src->meta.ri = ((size_t)(iop_a_src - a_src->data.ptr));
          }
          v_status = wuffs_deflate__decoder__decode_huffman_fast32(self, a_dst, a_src);
          if (a_src) {
            iop_a_src = a_src->data.ptr + a_src->meta.ri;
          }
        } else {
          if (a_src) {
            a_src->meta.ri = ((size_t)(iop_a_src - a_src->data.ptr));
          }
          v_status = wuffs_deflate__decoder__decode_huffman_fast64(self, a_dst, a_src);
          if (a_src) {
            iop_a_src = a_src->data.ptr + a_src->meta.ri;
          }
        }
        if (wuffs_base__status__is_error(&v_status)) {
          status = v_status;
          goto exit;
        }
        if (self->private_impl.f_end_of_block) {
          goto label__outer__continue;
        }
        if (a_src) {
          a_src->meta.ri = ((size_t)(iop_a_src - a_src->data.ptr));
        }
        WUFFS_BASE__COROUTINE_SUSPENSION_POINT(4);
        status = wuffs_deflate__decoder__decode_huffman_slow(self, a_dst, a_src);
        if (a_src) {
          iop_a_src = a_src->data.ptr + a_src->meta.ri;
        }
        if (status.repr) {
          goto suspend;
        }
        if (self->private_impl.f_end_of_block) {
          goto label__outer__continue;
        }
      }
    }

    ok:
    self->private_impl.p_decode_blocks[0] = 0;
    goto exit;
  }

  goto suspend;
  suspend:
  self->private_impl.p_decode_blocks[0] = wuffs_base__status__is_suspension(&status) ? coro_susp_point : 0;
  self->private_data.s_decode_blocks[0].v_final = v_final;

  goto exit;
  exit:
  if (a_src && a_src->data.ptr) {
    a_src->meta.ri = ((size_t)(iop_a_src - a_src->data.ptr));
  }

  return status;
}

// -------- func deflate.decoder.decode_uncompressed

static wuffs_base__status
wuffs_deflate__decoder__decode_uncompressed(
    wuffs_deflate__decoder* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__io_buffer* a_src) {
  wuffs_base__status status = wuffs_base__make_status(NULL);

  uint32_t v_length = 0;
  uint32_t v_n_copied = 0;

  uint8_t* iop_a_dst = NULL;
  uint8_t* io0_a_dst WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  uint8_t* io1_a_dst WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  uint8_t* io2_a_dst WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  if (a_dst && a_dst->data.ptr) {
    io0_a_dst = a_dst->data.ptr;
    io1_a_dst = io0_a_dst + a_dst->meta.wi;
    iop_a_dst = io1_a_dst;
    io2_a_dst = io0_a_dst + a_dst->data.len;
    if (a_dst->meta.closed) {
      io2_a_dst = iop_a_dst;
    }
  }
  const uint8_t* iop_a_src = NULL;
  const uint8_t* io0_a_src WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  const uint8_t* io1_a_src WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  const uint8_t* io2_a_src WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  if (a_src && a_src->data.ptr) {
    io0_a_src = a_src->data.ptr;
    io1_a_src = io0_a_src + a_src->meta.ri;
    iop_a_src = io1_a_src;
    io2_a_src = io0_a_src + a_src->meta.wi;
  }

  uint32_t coro_susp_point = self->private_impl.p_decode_uncompressed[0];
  if (coro_susp_point) {
    v_length = self->private_data.s_decode_uncompressed[0].v_length;
  }
  switch (coro_susp_point) {
    WUFFS_BASE__COROUTINE_SUSPENSION_POINT_0;

    if ((self->private_impl.f_n_bits >= 8) || ((self->private_impl.f_bits >> (self->private_impl.f_n_bits & 7)) != 0)) {
      status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_n_bits);
      goto exit;
    }
    self->private_impl.f_n_bits = 0;
    self->private_impl.f_bits = 0;
    {
      WUFFS_BASE__COROUTINE_SUSPENSION_POINT(1);
      uint32_t t_0;
      if (WUFFS_BASE__LIKELY(io2_a_src - iop_a_src >= 4)) {
        t_0 = wuffs_base__peek_u32le__no_bounds_check(iop_a_src);
        iop_a_src += 4;
      } else {
        self->private_data.s_decode_uncompressed[0].scratch = 0;
        WUFFS_BASE__COROUTINE_SUSPENSION_POINT(2);
        while (true) {
          if (WUFFS_BASE__UNLIKELY(iop_a_src == io2_a_src)) {
            status = wuffs_base__make_status(wuffs_base__suspension__short_read);
            goto suspend;
          }
          uint64_t* scratch = &self->private_data.s_decode_uncompressed[0].scratch;
          uint32_t num_bits_0 = ((uint32_t)(*scratch >> 56));
          *scratch <<= 8;
          *scratch >>= 8;
          *scratch |= ((uint64_t)(*iop_a_src++)) << num_bits_0;
          if (num_bits_0 == 24) {
            t_0 = ((uint32_t)(*scratch));
            break;
          }
          num_bits_0 += 8;
          *scratch |= ((uint64_t)(num_bits_0)) << 56;
        }
      }
      v_length = t_0;
    }
    if ((((v_length) & 0xFFFF) + ((v_length) >> (32 - (16)))) != 65535) {
      status = wuffs_base__make_status(wuffs_deflate__error__inconsistent_stored_block_length);
      goto exit;
    }
    v_length = ((v_length) & 0xFFFF);
    while (true) {
      v_n_copied = wuffs_base__io_writer__limited_copy_u32_from_reader(
          &iop_a_dst, io2_a_dst,v_length, &iop_a_src, io2_a_src);
      if (v_length <= v_n_copied) {
        status = wuffs_base__make_status(NULL);
        goto ok;
      }
      v_length -= v_n_copied;
      if (((uint64_t)(io2_a_dst - iop_a_dst)) == 0) {
        status = wuffs_base__make_status(wuffs_base__suspension__short_write);
        WUFFS_BASE__COROUTINE_SUSPENSION_POINT_MAYBE_SUSPEND(3);
      } else {
        status = wuffs_base__make_status(wuffs_base__suspension__short_read);
        WUFFS_BASE__COROUTINE_SUSPENSION_POINT_MAYBE_SUSPEND(4);
      }
    }

    ok:
    self->private_impl.p_decode_uncompressed[0] = 0;
    goto exit;
  }

  goto suspend;
  suspend:
  self->private_impl.p_decode_uncompressed[0] = wuffs_base__status__is_suspension(&status) ? coro_susp_point : 0;
  self->private_data.s_decode_uncompressed[0].v_length = v_length;

  goto exit;
  exit:
  if (a_dst && a_dst->data.ptr) {
    a_dst->meta.wi = ((size_t)(iop_a_dst - a_dst->data.ptr));
  }
  if (a_src && a_src->data.ptr) {
    a_src->meta.ri = ((size_t)(iop_a_src - a_src->data.ptr));
  }

  return status;
}

// -------- func deflate.decoder.init_fixed_huffman

static wuffs_base__status
wuffs_deflate__decoder__init_fixed_huffman(
    wuffs_deflate__decoder* self) {
  uint32_t v_i = 0;
  wuffs_base__status v_status = wuffs_base__make_status(NULL);

  while (v_i < 144) {
    self->private_data.f_code_lengths[v_i] = 8;
    v_i += 1;
  }
  while (v_i < 256) {
    self->private_data.f_code_lengths[v_i] = 9;
    v_i += 1;
  }
  while (v_i < 280) {
    self->private_data.f_code_lengths[v_i] = 7;
    v_i += 1;
  }
  while (v_i < 288) {
    self->private_data.f_code_lengths[v_i] = 8;
    v_i += 1;
  }
  while (v_i < 320) {
    self->private_data.f_code_lengths[v_i] = 5;
    v_i += 1;
  }
  v_status = wuffs_deflate__decoder__init_huff(self,
      0,
      0,
      288,
      257);
  if (wuffs_base__status__is_error(&v_status)) {
    return v_status;
  }
  v_status = wuffs_deflate__decoder__init_huff(self,
      1,
      288,
      320,
      0);
  if (wuffs_base__status__is_error(&v_status)) {
    return v_status;
  }
  return wuffs_base__make_status(NULL);
}

// -------- func deflate.decoder.init_dynamic_huffman

static wuffs_base__status
wuffs_deflate__decoder__init_dynamic_huffman(
    wuffs_deflate__decoder* self,
    wuffs_base__io_buffer* a_src) {
  wuffs_base__status status = wuffs_base__make_status(NULL);

  uint32_t v_bits = 0;
  uint32_t v_n_bits = 0;
  uint32_t v_b0 = 0;
  uint32_t v_n_lit = 0;
  uint32_t v_n_dist = 0;
  uint32_t v_n_clen = 0;
  uint32_t v_i = 0;
  uint32_t v_b1 = 0;
  wuffs_base__status v_status = wuffs_base__make_status(NULL);
  uint32_t v_mask = 0;
  uint32_t v_table_entry = 0;
  uint32_t v_table_entry_n_bits = 0;
  uint32_t v_b2 = 0;
  uint32_t v_n_extra_bits = 0;
  uint8_t v_rep_symbol = 0;
  uint32_t v_rep_count = 0;
  uint32_t v_b3 = 0;

  const uint8_t* iop_a_src = NULL;
  const uint8_t* io0_a_src WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  const uint8_t* io1_a_src WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  const uint8_t* io2_a_src WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  if (a_src && a_src->data.ptr) {
    io0_a_src = a_src->data.ptr;
    io1_a_src = io0_a_src + a_src->meta.ri;
    iop_a_src = io1_a_src;
    io2_a_src = io0_a_src + a_src->meta.wi;
  }

  uint32_t coro_susp_point = self->private_impl.p_init_dynamic_huffman[0];
  if (coro_susp_point) {
    v_bits = self->private_data.s_init_dynamic_huffman[0].v_bits;
    v_n_bits = self->private_data.s_init_dynamic_huffman[0].v_n_bits;
    v_n_lit = self->private_data.s_init_dynamic_huffman[0].v_n_lit;
    v_n_dist = self->private_data.s_init_dynamic_huffman[0].v_n_dist;
    v_n_clen = self->private_data.s_init_dynamic_huffman[0].v_n_clen;
    v_i = self->private_data.s_init_dynamic_huffman[0].v_i;
    v_mask = self->private_data.s_init_dynamic_huffman[0].v_mask;
    v_n_extra_bits = self->private_data.s_init_dynamic_huffman[0].v_n_extra_bits;
    v_rep_symbol = self->private_data.s_init_dynamic_huffman[0].v_rep_symbol;
    v_rep_count = self->private_data.s_init_dynamic_huffman[0].v_rep_count;
  }
  switch (coro_susp_point) {
    WUFFS_BASE__COROUTINE_SUSPENSION_POINT_0;

    v_bits = self->private_impl.f_bits;
    v_n_bits = self->private_impl.f_n_bits;
    while (v_n_bits < 14) {
      {
        WUFFS_BASE__COROUTINE_SUSPENSION_POINT(1);
        if (WUFFS_BASE__UNLIKELY(iop_a_src == io2_a_src)) {
          status = wuffs_base__make_status(wuffs_base__suspension__short_read);
          goto suspend;
        }
        uint32_t t_0 = *iop_a_src++;
        v_b0 = t_0;
      }
      v_bits |= (v_b0 << v_n_bits);
      v_n_bits += 8;
    }
    v_n_lit = (((v_bits) & 0x1F) + 257);
    if (v_n_lit > 286) {
      status = wuffs_base__make_status(wuffs_deflate__error__bad_literal_length_code_count);
      goto exit;
    }
    v_bits >>= 5;
    v_n_dist = (((v_bits) & 0x1F) + 1);
    if (v_n_dist > 30) {
      status = wuffs_base__make_status(wuffs_deflate__error__bad_distance_code_count);
      goto exit;
    }
    v_bits >>= 5;
    v_n_clen = (((v_bits) & 0xF) + 4);
    v_bits >>= 4;
    v_n_bits -= 14;
    v_i = 0;
    while (v_i < v_n_clen) {
      while (v_n_bits < 3) {
        {
          WUFFS_BASE__COROUTINE_SUSPENSION_POINT(2);
          if (WUFFS_BASE__UNLIKELY(iop_a_src == io2_a_src)) {
            status = wuffs_base__make_status(wuffs_base__suspension__short_read);
            goto suspend;
          }
          uint32_t t_1 = *iop_a_src++;
          v_b1 = t_1;
        }
        v_bits |= (v_b1 << v_n_bits);
        v_n_bits += 8;
      }
      self->private_data.f_code_lengths[WUFFS_DEFLATE__CODE_ORDER[v_i]] = ((uint8_t)((v_bits & 7)));
      v_bits >>= 3;
      v_n_bits -= 3;
      v_i += 1;
    }
    while (v_i < 19) {
      self->private_data.f_code_lengths[WUFFS_DEFLATE__CODE_ORDER[v_i]] = 0;
      v_i += 1;
    }
    v_status = wuffs_deflate__decoder__init_huff(self,
        0,
        0,
        19,
        4095);
    if (wuffs_base__status__is_error(&v_status)) {
      status = v_status;
      goto exit;
    }
    v_mask = ((((uint32_t)(1)) << self->private_impl.f_n_huffs_bits[0]) - 1);
    v_i = 0;
    label__0__continue:;
    while (v_i < (v_n_lit + v_n_dist)) {
      while (true) {
        v_table_entry = self->private_data.f_huffs[0][(v_bits & v_mask)];
        v_table_entry_n_bits = (v_table_entry & 15);
        if (v_n_bits >= v_table_entry_n_bits) {
          v_bits >>= v_table_entry_n_bits;
          v_n_bits -= v_table_entry_n_bits;
          goto label__1__break;
        }
        {
          WUFFS_BASE__COROUTINE_SUSPENSION_POINT(3);
          if (WUFFS_BASE__UNLIKELY(iop_a_src == io2_a_src)) {
            status = wuffs_base__make_status(wuffs_base__suspension__short_read);
            goto suspend;
          }
          uint32_t t_2 = *iop_a_src++;
          v_b2 = t_2;
        }
        v_bits |= (v_b2 << v_n_bits);
        v_n_bits += 8;
      }
      label__1__break:;
      if ((v_table_entry >> 24) != 128) {
        status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
        goto exit;
      }
      v_table_entry = ((v_table_entry >> 8) & 255);
      if (v_table_entry < 16) {
        self->private_data.f_code_lengths[v_i] = ((uint8_t)(v_table_entry));
        v_i += 1;
        goto label__0__continue;
      }
      v_n_extra_bits = 0;
      v_rep_symbol = 0;
      v_rep_count = 0;
      if (v_table_entry == 16) {
        v_n_extra_bits = 2;
        if (v_i <= 0) {
          status = wuffs_base__make_status(wuffs_deflate__error__bad_huffman_code_length_repetition);
          goto exit;
        }
        v_rep_symbol = (self->private_data.f_code_lengths[(v_i - 1)] & 15);
        v_rep_count = 3;
      } else if (v_table_entry == 17) {
        v_n_extra_bits = 3;
        v_rep_symbol = 0;
        v_rep_count = 3;
      } else if (v_table_entry == 18) {
        v_n_extra_bits = 7;
        v_rep_symbol = 0;
        v_rep_count = 11;
      } else {
        status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
        goto exit;
      }
      while (v_n_bits < v_n_extra_bits) {
        {
          WUFFS_BASE__COROUTINE_SUSPENSION_POINT(4);
          if (WUFFS_BASE__UNLIKELY(iop_a_src == io2_a_src)) {
            status = wuffs_base__make_status(wuffs_base__suspension__short_read);
            goto suspend;
          }
          uint32_t t_3 = *iop_a_src++;
          v_b3 = t_3;
        }
        v_bits |= (v_b3 << v_n_bits);
        v_n_bits += 8;
      }
      v_rep_count += ((v_bits) & WUFFS_BASE__LOW_BITS_MASK__U32(v_n_extra_bits));
      v_bits >>= v_n_extra_bits;
      v_n_bits -= v_n_extra_bits;
      while (v_rep_count > 0) {
        if (v_i >= (v_n_lit + v_n_dist)) {
          status = wuffs_base__make_status(wuffs_deflate__error__bad_huffman_code_length_count);
          goto exit;
        }
        self->private_data.f_code_lengths[v_i] = v_rep_symbol;
        v_i += 1;
        v_rep_count -= 1;
      }
    }
    if (v_i != (v_n_lit + v_n_dist)) {
      status = wuffs_base__make_status(wuffs_deflate__error__bad_huffman_code_length_count);
      goto exit;
    }
    if (self->private_data.f_code_lengths[256] == 0) {
      status = wuffs_base__make_status(wuffs_deflate__error__missing_end_of_block_code);
      goto exit;
    }
    v_status = wuffs_deflate__decoder__init_huff(self,
        0,
        0,
        v_n_lit,
        257);
    if (wuffs_base__status__is_error(&v_status)) {
      status = v_status;
      goto exit;
    }
    v_status = wuffs_deflate__decoder__init_huff(self,
        1,
        v_n_lit,
        (v_n_lit + v_n_dist),
        0);
    if (wuffs_base__status__is_error(&v_status)) {
      status = v_status;
      goto exit;
    }
    self->private_impl.f_bits = v_bits;
    self->private_impl.f_n_bits = v_n_bits;

    goto ok;
    ok:
    self->private_impl.p_init_dynamic_huffman[0] = 0;
    goto exit;
  }

  goto suspend;
  suspend:
  self->private_impl.p_init_dynamic_huffman[0] = wuffs_base__status__is_suspension(&status) ? coro_susp_point : 0;
  self->private_data.s_init_dynamic_huffman[0].v_bits = v_bits;
  self->private_data.s_init_dynamic_huffman[0].v_n_bits = v_n_bits;
  self->private_data.s_init_dynamic_huffman[0].v_n_lit = v_n_lit;
  self->private_data.s_init_dynamic_huffman[0].v_n_dist = v_n_dist;
  self->private_data.s_init_dynamic_huffman[0].v_n_clen = v_n_clen;
  self->private_data.s_init_dynamic_huffman[0].v_i = v_i;
  self->private_data.s_init_dynamic_huffman[0].v_mask = v_mask;
  self->private_data.s_init_dynamic_huffman[0].v_n_extra_bits = v_n_extra_bits;
  self->private_data.s_init_dynamic_huffman[0].v_rep_symbol = v_rep_symbol;
  self->private_data.s_init_dynamic_huffman[0].v_rep_count = v_rep_count;

  goto exit;
  exit:
  if (a_src && a_src->data.ptr) {
    a_src->meta.ri = ((size_t)(iop_a_src - a_src->data.ptr));
  }

  return status;
}

// -------- func deflate.decoder.init_huff

static wuffs_base__status
wuffs_deflate__decoder__init_huff(
    wuffs_deflate__decoder* self,
    uint32_t a_which,
    uint32_t a_n_codes0,
    uint32_t a_n_codes1,
    uint32_t a_base_symbol) {
  uint16_t v_counts[16] = {0};
  uint32_t v_i = 0;
  uint32_t v_remaining = 0;
  uint16_t v_offsets[16] = {0};
  uint32_t v_n_symbols = 0;
  uint32_t v_count = 0;
  uint16_t v_symbols[320] = {0};
  uint32_t v_min_cl = 0;
  uint32_t v_max_cl = 0;
  uint32_t v_initial_high_bits = 0;
  uint32_t v_prev_cl = 0;
  uint32_t v_prev_redirect_key = 0;
  uint32_t v_top = 0;
  uint32_t v_next_top = 0;
  uint32_t v_code = 0;
  uint32_t v_key = 0;
  uint32_t v_value = 0;
  uint32_t v_cl = 0;
  uint32_t v_redirect_key = 0;
  uint32_t v_j = 0;
  uint32_t v_reversed_key = 0;
  uint32_t v_symbol = 0;
  uint32_t v_high_bits = 0;
  uint32_t v_delta = 0;

  v_i = a_n_codes0;
  while (v_i < a_n_codes1) {
    if (v_counts[(self->private_data.f_code_lengths[v_i] & 15)] >= 320) {
      return wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
    }
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#endif
    v_counts[(self->private_data.f_code_lengths[v_i] & 15)] += 1;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    v_i += 1;
  }
  if ((((uint32_t)(v_counts[0])) + a_n_codes0) == a_n_codes1) {
    return wuffs_base__make_status(wuffs_deflate__error__no_huffman_codes);
  }
  v_remaining = 1;
  v_i = 1;
  while (v_i <= 15) {
    if (v_remaining > 1073741824) {
      return wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
    }
    v_remaining <<= 1;
    if (v_remaining < ((uint32_t)(v_counts[v_i]))) {
      return wuffs_base__make_status(wuffs_deflate__error__bad_huffman_code_over_subscribed);
    }
    v_remaining -= ((uint32_t)(v_counts[v_i]));
    v_i += 1;
  }
  if (v_remaining != 0) {
    if ((a_which == 1) && (v_counts[1] == 1) && ((((uint32_t)(v_counts[0])) + a_n_codes0 + 1) == a_n_codes1)) {
      v_i = 0;
      while (v_i <= 29) {
        if (self->private_data.f_code_lengths[(a_n_codes0 + v_i)] == 1) {
          self->private_impl.f_n_huffs_bits[1] = 1;
          self->private_data.f_huffs[1][0] = (WUFFS_DEFLATE__DCODE_MAGIC_NUMBERS[v_i] | 1);
          self->private_data.f_huffs[1][1] = (WUFFS_DEFLATE__DCODE_MAGIC_NUMBERS[31] | 1);
          return wuffs_base__make_status(NULL);
        }
        v_i += 1;
      }
    }
    return wuffs_base__make_status(wuffs_deflate__error__bad_huffman_code_under_subscribed);
  }
  v_i = 1;
  while (v_i <= 15) {
    v_offsets[v_i] = ((uint16_t)(v_n_symbols));
    v_count = ((uint32_t)(v_counts[v_i]));
    if (v_n_symbols > (320 - v_count)) {
      return wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
    }
    v_n_symbols = (v_n_symbols + v_count);
    v_i += 1;
  }
  if (v_n_symbols > 288) {
    return wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
  }
  v_i = a_n_codes0;
  while (v_i < a_n_codes1) {
    if (v_i < a_n_codes0) {
      return wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
    }
    if (self->private_data.f_code_lengths[v_i] != 0) {
      if (v_offsets[(self->private_data.f_code_lengths[v_i] & 15)] >= 320) {
        return wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
      }
      v_symbols[v_offsets[(self->private_data.f_code_lengths[v_i] & 15)]] = ((uint16_t)((v_i - a_n_codes0)));
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#endif
      v_offsets[(self->private_data.f_code_lengths[v_i] & 15)] += 1;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    }
    v_i += 1;
  }
  v_min_cl = 1;
  while (true) {
    if (v_counts[v_min_cl] != 0) {
      goto label__0__break;
    }
    if (v_min_cl >= 9) {
      return wuffs_base__make_status(wuffs_deflate__error__bad_huffman_minimum_code_length);
    }
    v_min_cl += 1;
  }
  label__0__break:;
  v_max_cl = 15;
  while (true) {
    if (v_counts[v_max_cl] != 0) {
      goto label__1__break;
    }
    if (v_max_cl <= 1) {
      return wuffs_base__make_status(wuffs_deflate__error__no_huffman_codes);
    }
    v_max_cl -= 1;
  }
  label__1__break:;
  if (v_max_cl <= 9) {
    self->private_impl.f_n_huffs_bits[a_which] = v_max_cl;
  } else {
    self->private_impl.f_n_huffs_bits[a_which] = 9;
  }
  v_i = 0;
  if ((v_n_symbols != ((uint32_t)(v_offsets[v_max_cl]))) || (v_n_symbols != ((uint32_t)(v_offsets[15])))) {
    return wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
  }
  if ((a_n_codes0 + ((uint32_t)(v_symbols[0]))) >= 320) {
    return wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
  }
  v_initial_high_bits = 512;
  if (v_max_cl < 9) {
    v_initial_high_bits = (((uint32_t)(1)) << v_max_cl);
  }
  v_prev_cl = ((uint32_t)((self->private_data.f_code_lengths[(a_n_codes0 + ((uint32_t)(v_symbols[0])))] & 15)));
  v_prev_redirect_key = 4294967295;
  v_top = 0;
  v_next_top = 512;
  v_code = 0;
  v_key = 0;
  v_value = 0;
  while (true) {
    if ((a_n_codes0 + ((uint32_t)(v_symbols[v_i]))) >= 320) {
      return wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
    }
    v_cl = ((uint32_t)((self->private_data.f_code_lengths[(a_n_codes0 + ((uint32_t)(v_symbols[v_i])))] & 15)));
    if (v_cl > v_prev_cl) {
      v_code <<= (v_cl - v_prev_cl);
      if (v_code >= 32768) {
        return wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
      }
    }
    v_prev_cl = v_cl;
    v_key = v_code;
    if (v_cl > 9) {
      v_cl -= 9;
      v_redirect_key = ((v_key >> v_cl) & 511);
      v_key = ((v_key) & WUFFS_BASE__LOW_BITS_MASK__U32(v_cl));
      if (v_prev_redirect_key != v_redirect_key) {
        v_prev_redirect_key = v_redirect_key;
        v_remaining = (((uint32_t)(1)) << v_cl);
        v_j = v_prev_cl;
        while (v_j <= 15) {
          if (v_remaining <= ((uint32_t)(v_counts[v_j]))) {
            goto label__2__break;
          }
          v_remaining -= ((uint32_t)(v_counts[v_j]));
          if (v_remaining > 1073741824) {
            return wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
          }
          v_remaining <<= 1;
          v_j += 1;
        }
        label__2__break:;
        if ((v_j <= 9) || (15 < v_j)) {
          return wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
        }
        v_j -= 9;
        v_initial_high_bits = (((uint32_t)(1)) << v_j);
        v_top = v_next_top;
        if ((v_top + (((uint32_t)(1)) << v_j)) > 1024) {
          return wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
        }
        v_next_top = (v_top + (((uint32_t)(1)) << v_j));
        v_redirect_key = (((uint32_t)(WUFFS_DEFLATE__REVERSE8[(v_redirect_key >> 1)])) | ((v_redirect_key & 1) << 8));
        self->private_data.f_huffs[a_which][v_redirect_key] = (268435465 | (v_top << 8) | (v_j << 4));
      }
    }
    if ((v_key >= 512) || (v_counts[v_prev_cl] <= 0)) {
      return wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
    }
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#endif
    v_counts[v_prev_cl] -= 1;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    v_reversed_key = (((uint32_t)(WUFFS_DEFLATE__REVERSE8[(v_key >> 1)])) | ((v_key & 1) << 8));
    v_reversed_key >>= (9 - v_cl);
    v_symbol = ((uint32_t)(v_symbols[v_i]));
    if (v_symbol == 256) {
      v_value = (536870912 | v_cl);
    } else if ((v_symbol < 256) && (a_which == 0)) {
      v_value = (2147483648 | (v_symbol << 8) | v_cl);
    } else if (v_symbol >= a_base_symbol) {
      v_symbol -= a_base_symbol;
      if (a_which == 0) {
        v_value = (WUFFS_DEFLATE__LCODE_MAGIC_NUMBERS[(v_symbol & 31)] | v_cl);
      } else {
        v_value = (WUFFS_DEFLATE__DCODE_MAGIC_NUMBERS[(v_symbol & 31)] | v_cl);
      }
    } else {
      return wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
    }
    v_high_bits = v_initial_high_bits;
    v_delta = (((uint32_t)(1)) << v_cl);
    while (v_high_bits >= v_delta) {
      v_high_bits -= v_delta;
      if ((v_top + ((v_high_bits | v_reversed_key) & 511)) >= 1024) {
        return wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
      }
      self->private_data.f_huffs[a_which][(v_top + ((v_high_bits | v_reversed_key) & 511))] = v_value;
    }
    v_i += 1;
    if (v_i >= v_n_symbols) {
      goto label__3__break;
    }
    v_code += 1;
    if (v_code >= 32768) {
      return wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
    }
  }
  label__3__break:;
  return wuffs_base__make_status(NULL);
}

// ‼ WUFFS MULTI-FILE SECTION +x86_bmi2
// -------- func deflate.decoder.decode_huffman_bmi2

#if defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
WUFFS_BASE__MAYBE_ATTRIBUTE_TARGET("bmi2")
static wuffs_base__status
wuffs_deflate__decoder__decode_huffman_bmi2(
    wuffs_deflate__decoder* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__io_buffer* a_src) {
  wuffs_base__status status = wuffs_base__make_status(NULL);

  uint64_t v_bits = 0;
  uint32_t v_n_bits = 0;
  uint32_t v_table_entry = 0;
  uint32_t v_table_entry_n_bits = 0;
  uint64_t v_lmask = 0;
  uint64_t v_dmask = 0;
  uint32_t v_redir_top = 0;
  uint32_t v_redir_mask = 0;
  uint32_t v_length = 0;
  uint32_t v_dist_minus_1 = 0;
  uint32_t v_hlen = 0;
  uint32_t v_hdist = 0;
  uint32_t v_hdist_adjustment = 0;

  uint8_t* iop_a_dst = NULL;
  uint8_t* io0_a_dst WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  uint8_t* io1_a_dst WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  uint8_t* io2_a_dst WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  if (a_dst && a_dst->data.ptr) {
    io0_a_dst = a_dst->data.ptr;
    io1_a_dst = io0_a_dst + a_dst->meta.wi;
    iop_a_dst = io1_a_dst;
    io2_a_dst = io0_a_dst + a_dst->data.len;
    if (a_dst->meta.closed) {
      io2_a_dst = iop_a_dst;
    }
  }
  const uint8_t* iop_a_src = NULL;
  const uint8_t* io0_a_src WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  const uint8_t* io1_a_src WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  const uint8_t* io2_a_src WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  if (a_src && a_src->data.ptr) {
    io0_a_src = a_src->data.ptr;
    io1_a_src = io0_a_src + a_src->meta.ri;
    iop_a_src = io1_a_src;
    io2_a_src = io0_a_src + a_src->meta.wi;
  }

  if ((self->private_impl.f_n_bits >= 8) || ((self->private_impl.f_bits >> (self->private_impl.f_n_bits & 7)) != 0)) {
    status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_n_bits);
    goto exit;
  }
  v_bits = ((uint64_t)(self->private_impl.f_bits));
  v_n_bits = self->private_impl.f_n_bits;
  v_lmask = ((((uint64_t)(1)) << self->private_impl.f_n_huffs_bits[0]) - 1);
  v_dmask = ((((uint64_t)(1)) << self->private_impl.f_n_huffs_bits[1]) - 1);
  if (self->private_impl.f_transformed_history_count < (a_dst ? a_dst->meta.pos : 0)) {
    status = wuffs_base__make_status(wuffs_base__error__bad_i_o_position);
    goto exit;
  }
  v_hdist_adjustment = ((uint32_t)(((self->private_impl.f_transformed_history_count - (a_dst ? a_dst->meta.pos : 0)) & 4294967295)));
  label__loop__continue:;
  while ((((uint64_t)(io2_a_dst - iop_a_dst)) >= 266) && (((uint64_t)(io2_a_src - iop_a_src)) >= 8)) {
    v_bits |= ((uint64_t)(wuffs_base__peek_u64le__no_bounds_check(iop_a_src) << (v_n_bits & 63)));
    iop_a_src += ((63 - (v_n_bits & 63)) >> 3);
    v_n_bits |= 56;
    v_table_entry = self->private_data.f_huffs[0][(v_bits & v_lmask)];
    v_table_entry_n_bits = (v_table_entry & 15);
    v_bits >>= v_table_entry_n_bits;
    v_n_bits -= v_table_entry_n_bits;
    if ((v_table_entry >> 31) != 0) {
      (wuffs_base__poke_u8be__no_bounds_check(iop_a_dst, ((uint8_t)(((v_table_entry >> 8) & 255)))), iop_a_dst += 1);
      goto label__loop__continue;
    } else if ((v_table_entry >> 30) != 0) {
    } else if ((v_table_entry >> 29) != 0) {
      self->private_impl.f_end_of_block = true;
      goto label__loop__break;
    } else if ((v_table_entry >> 28) != 0) {
      v_redir_top = ((v_table_entry >> 8) & 65535);
      v_redir_mask = ((((uint32_t)(1)) << ((v_table_entry >> 4) & 15)) - 1);
      v_table_entry = self->private_data.f_huffs[0][((v_redir_top + (((uint32_t)((v_bits & 4294967295))) & v_redir_mask)) & 1023)];
      v_table_entry_n_bits = (v_table_entry & 15);
      v_bits >>= v_table_entry_n_bits;
      v_n_bits -= v_table_entry_n_bits;
      if ((v_table_entry >> 31) != 0) {
        (wuffs_base__poke_u8be__no_bounds_check(iop_a_dst, ((uint8_t)(((v_table_entry >> 8) & 255)))), iop_a_dst += 1);
        goto label__loop__continue;
      } else if ((v_table_entry >> 30) != 0) {
      } else if ((v_table_entry >> 29) != 0) {
        self->private_impl.f_end_of_block = true;
        goto label__loop__break;
      } else if ((v_table_entry >> 28) != 0) {
        status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
        goto exit;
      } else if ((v_table_entry >> 27) != 0) {
        status = wuffs_base__make_status(wuffs_deflate__error__bad_huffman_code);
        goto exit;
      } else {
        status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
        goto exit;
      }
    } else if ((v_table_entry >> 27) != 0) {
      status = wuffs_base__make_status(wuffs_deflate__error__bad_huffman_code);
      goto exit;
    } else {
      status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
      goto exit;
    }
    v_length = (((v_table_entry >> 8) & 255) + 3);
    v_table_entry_n_bits = ((v_table_entry >> 4) & 15);
    if (v_table_entry_n_bits > 0) {
      v_length = (((v_length + 253 + ((uint32_t)(((v_bits) & WUFFS_BASE__LOW_BITS_MASK__U64(v_table_entry_n_bits))))) & 255) + 3);
      v_bits >>= v_table_entry_n_bits;
      v_n_bits -= v_table_entry_n_bits;
    }
    v_table_entry = self->private_data.f_huffs[1][(v_bits & v_dmask)];
    v_table_entry_n_bits = (v_table_entry & 15);
    v_bits >>= v_table_entry_n_bits;
    v_n_bits -= v_table_entry_n_bits;
    if ((v_table_entry >> 28) == 1) {
      v_redir_top = ((v_table_entry >> 8) & 65535);
      v_redir_mask = ((((uint32_t)(1)) << ((v_table_entry >> 4) & 15)) - 1);
      v_table_entry = self->private_data.f_huffs[1][((v_redir_top + (((uint32_t)((v_bits & 4294967295))) & v_redir_mask)) & 1023)];
      v_table_entry_n_bits = (v_table_entry & 15);
      v_bits >>= v_table_entry_n_bits;
      v_n_bits -= v_table_entry_n_bits;
    }
    if ((v_table_entry >> 24) != 64) {
      if ((v_table_entry >> 24) == 8) {
        status = wuffs_base__make_status(wuffs_deflate__error__bad_huffman_code);
        goto exit;
      }
      status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
      goto exit;
    }
    v_dist_minus_1 = ((v_table_entry >> 8) & 32767);
    v_table_entry_n_bits = ((v_table_entry >> 4) & 15);
    v_dist_minus_1 = ((v_dist_minus_1 + ((uint32_t)(((v_bits) & WUFFS_BASE__LOW_BITS_MASK__U64(v_table_entry_n_bits))))) & 32767);
    v_bits >>= v_table_entry_n_bits;
    v_n_bits -= v_table_entry_n_bits;
    while (true) {
      if (((uint64_t)((v_dist_minus_1 + 1))) > ((uint64_t)(iop_a_dst - io0_a_dst))) {
        v_hlen = 0;
        v_hdist = ((uint32_t)((((uint64_t)((v_dist_minus_1 + 1))) - ((uint64_t)(iop_a_dst - io0_a_dst)))));
        if (v_length > v_hdist) {
          v_length -= v_hdist;
          v_hlen = v_hdist;
        } else {
          v_hlen = v_length;
          v_length = 0;
        }
        v_hdist += v_hdist_adjustment;
        if (self->private_impl.f_history_index < v_hdist) {
          status = wuffs_base__make_status(wuffs_deflate__error__bad_distance);
          goto exit;
        }
        wuffs_base__io_writer__limited_copy_u32_from_slice(
            &iop_a_dst, io2_a_dst,v_hlen, wuffs_base__make_slice_u8_ij(self->private_data.f_history, ((self->private_impl.f_history_index - v_hdist) & 32767), 33025));
        if (v_length == 0) {
          goto label__loop__continue;
        }
        if ((((uint64_t)((v_dist_minus_1 + 1))) > ((uint64_t)(iop_a_dst - io0_a_dst))) || (((uint64_t)(v_length)) > ((uint64_t)(io2_a_dst - iop_a_dst))) || (((uint64_t)((v_length + 8))) > ((uint64_t)(io2_a_dst - iop_a_dst)))) {
          status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_distance);
          goto exit;
        }
      }
      if ((v_dist_minus_1 + 1) >= 8) {
        wuffs_base__io_writer__limited_copy_u32_from_history_8_byte_chunks_fast(
            &iop_a_dst, io0_a_dst, io2_a_dst, v_length, (v_dist_minus_1 + 1));
      } else if ((v_dist_minus_1 + 1) == 1) {
        wuffs_base__io_writer__limited_copy_u32_from_history_8_byte_chunks_distance_1_fast(
            &iop_a_dst, io0_a_dst, io2_a_dst, v_length, (v_dist_minus_1 + 1));
      } else {
        wuffs_base__io_writer__limited_copy_u32_from_history_fast(
            &iop_a_dst, io0_a_dst, io2_a_dst, v_length, (v_dist_minus_1 + 1));
      }
      goto label__0__break;
    }
    label__0__break:;
  }
  label__loop__break:;
  if (v_n_bits > 63) {
    status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_n_bits);
    goto exit;
  }
  while (v_n_bits >= 8) {
    v_n_bits -= 8;
    if (iop_a_src > io1_a_src) {
      iop_a_src--;
    } else {
      status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_i_o);
      goto exit;
    }
  }
  self->private_impl.f_bits = ((uint32_t)((v_bits & ((((uint64_t)(1)) << v_n_bits) - 1))));
  self->private_impl.f_n_bits = v_n_bits;
  if ((self->private_impl.f_n_bits >= 8) || ((self->private_impl.f_bits >> self->private_impl.f_n_bits) != 0)) {
    status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_n_bits);
    goto exit;
  }
  goto exit;
  exit:
  if (a_dst && a_dst->data.ptr) {
    a_dst->meta.wi = ((size_t)(iop_a_dst - a_dst->data.ptr));
  }
  if (a_src && a_src->data.ptr) {
    a_src->meta.ri = ((size_t)(iop_a_src - a_src->data.ptr));
  }

  return status;
}
#endif  // defined(WUFFS_BASE__CPU_ARCH__X86_FAMILY)
// ‼ WUFFS MULTI-FILE SECTION -x86_bmi2

// -------- func deflate.decoder.decode_huffman_fast32

static wuffs_base__status
wuffs_deflate__decoder__decode_huffman_fast32(
    wuffs_deflate__decoder* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__io_buffer* a_src) {
  wuffs_base__status status = wuffs_base__make_status(NULL);

  uint32_t v_bits = 0;
  uint32_t v_n_bits = 0;
  uint32_t v_table_entry = 0;
  uint32_t v_table_entry_n_bits = 0;
  uint32_t v_lmask = 0;
  uint32_t v_dmask = 0;
  uint32_t v_redir_top = 0;
  uint32_t v_redir_mask = 0;
  uint32_t v_length = 0;
  uint32_t v_dist_minus_1 = 0;
  uint32_t v_hlen = 0;
  uint32_t v_hdist = 0;
  uint32_t v_hdist_adjustment = 0;

  uint8_t* iop_a_dst = NULL;
  uint8_t* io0_a_dst WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  uint8_t* io1_a_dst WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  uint8_t* io2_a_dst WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  if (a_dst && a_dst->data.ptr) {
    io0_a_dst = a_dst->data.ptr;
    io1_a_dst = io0_a_dst + a_dst->meta.wi;
    iop_a_dst = io1_a_dst;
    io2_a_dst = io0_a_dst + a_dst->data.len;
    if (a_dst->meta.closed) {
      io2_a_dst = iop_a_dst;
    }
  }
  const uint8_t* iop_a_src = NULL;
  const uint8_t* io0_a_src WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  const uint8_t* io1_a_src WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  const uint8_t* io2_a_src WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  if (a_src && a_src->data.ptr) {
    io0_a_src = a_src->data.ptr;
    io1_a_src = io0_a_src + a_src->meta.ri;
    iop_a_src = io1_a_src;
    io2_a_src = io0_a_src + a_src->meta.wi;
  }

  if ((self->private_impl.f_n_bits >= 8) || ((self->private_impl.f_bits >> (self->private_impl.f_n_bits & 7)) != 0)) {
    status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_n_bits);
    goto exit;
  }
  v_bits = self->private_impl.f_bits;
  v_n_bits = self->private_impl.f_n_bits;
  v_lmask = ((((uint32_t)(1)) << self->private_impl.f_n_huffs_bits[0]) - 1);
  v_dmask = ((((uint32_t)(1)) << self->private_impl.f_n_huffs_bits[1]) - 1);
  if (self->private_impl.f_transformed_history_count < (a_dst ? a_dst->meta.pos : 0)) {
    status = wuffs_base__make_status(wuffs_base__error__bad_i_o_position);
    goto exit;
  }
  v_hdist_adjustment = ((uint32_t)(((self->private_impl.f_transformed_history_count - (a_dst ? a_dst->meta.pos : 0)) & 4294967295)));
  label__loop__continue:;
  while ((((uint64_t)(io2_a_dst - iop_a_dst)) >= 266) && (((uint64_t)(io2_a_src - iop_a_src)) >= 12)) {
    if (v_n_bits < 15) {
      v_bits |= (((uint32_t)(wuffs_base__peek_u8be__no_bounds_check(iop_a_src))) << v_n_bits);
      iop_a_src += 1;
      v_n_bits += 8;
      v_bits |= (((uint32_t)(wuffs_base__peek_u8be__no_bounds_check(iop_a_src))) << v_n_bits);
      iop_a_src += 1;
      v_n_bits += 8;
    } else {
    }
    v_table_entry = self->private_data.f_huffs[0][(v_bits & v_lmask)];
    v_table_entry_n_bits = (v_table_entry & 15);
    v_bits >>= v_table_entry_n_bits;
    v_n_bits -= v_table_entry_n_bits;
    if ((v_table_entry >> 31) != 0) {
      (wuffs_base__poke_u8be__no_bounds_check(iop_a_dst, ((uint8_t)(((v_table_entry >> 8) & 255)))), iop_a_dst += 1);
      goto label__loop__continue;
    } else if ((v_table_entry >> 30) != 0) {
    } else if ((v_table_entry >> 29) != 0) {
      self->private_impl.f_end_of_block = true;
      goto label__loop__break;
    } else if ((v_table_entry >> 28) != 0) {
      if (v_n_bits < 15) {
        v_bits |= (((uint32_t)(wuffs_base__peek_u8be__no_bounds_check(iop_a_src))) << v_n_bits);
        iop_a_src += 1;
        v_n_bits += 8;
        v_bits |= (((uint32_t)(wuffs_base__peek_u8be__no_bounds_check(iop_a_src))) << v_n_bits);
        iop_a_src += 1;
        v_n_bits += 8;
      } else {
      }
      v_redir_top = ((v_table_entry >> 8) & 65535);
      v_redir_mask = ((((uint32_t)(1)) << ((v_table_entry >> 4) & 15)) - 1);
      v_table_entry = self->private_data.f_huffs[0][((v_redir_top + (v_bits & v_redir_mask)) & 1023)];
      v_table_entry_n_bits = (v_table_entry & 15);
      v_bits >>= v_table_entry_n_bits;
      v_n_bits -= v_table_entry_n_bits;
      if ((v_table_entry >> 31) != 0) {
        (wuffs_base__poke_u8be__no_bounds_check(iop_a_dst, ((uint8_t)(((v_table_entry >> 8) & 255)))), iop_a_dst += 1);
        goto label__loop__continue;
      } else if ((v_table_entry >> 30) != 0) {
      } else if ((v_table_entry >> 29) != 0) {
        self->private_impl.f_end_of_block = true;
        goto label__loop__break;
      } else if ((v_table_entry >> 28) != 0) {
        status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
        goto exit;
      } else if ((v_table_entry >> 27) != 0) {
        status = wuffs_base__make_status(wuffs_deflate__error__bad_huffman_code);
        goto exit;
      } else {
        status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
        goto exit;
      }
    } else if ((v_table_entry >> 27) != 0) {
      status = wuffs_base__make_status(wuffs_deflate__error__bad_huffman_code);
      goto exit;
    } else {
      status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
      goto exit;
    }
    v_length = (((v_table_entry >> 8) & 255) + 3);
    v_table_entry_n_bits = ((v_table_entry >> 4) & 15);
    if (v_table_entry_n_bits > 0) {
      if (v_n_bits < 15) {
        v_bits |= (((uint32_t)(wuffs_base__peek_u8be__no_bounds_check(iop_a_src))) << v_n_bits);
        iop_a_src += 1;
        v_n_bits += 8;
        v_bits |= (((uint32_t)(wuffs_base__peek_u8be__no_bounds_check(iop_a_src))) << v_n_bits);
        iop_a_src += 1;
        v_n_bits += 8;
      } else {
      }
      v_length = (((v_length + 253 + ((v_bits) & WUFFS_BASE__LOW_BITS_MASK__U32(v_table_entry_n_bits))) & 255) + 3);
      v_bits >>= v_table_entry_n_bits;
      v_n_bits -= v_table_entry_n_bits;
    } else {
    }
    if (v_n_bits < 15) {
      v_bits |= (((uint32_t)(wuffs_base__peek_u8be__no_bounds_check(iop_a_src))) << v_n_bits);
      iop_a_src += 1;
      v_n_bits += 8;
      v_bits |= (((uint32_t)(wuffs_base__peek_u8be__no_bounds_check(iop_a_src))) << v_n_bits);
      iop_a_src += 1;
      v_n_bits += 8;
    } else {
    }
    v_table_entry = self->private_data.f_huffs[1][(v_bits & v_dmask)];
    v_table_entry_n_bits = (v_table_entry & 15);
    v_bits >>= v_table_entry_n_bits;
    v_n_bits -= v_table_entry_n_bits;
    if ((v_table_entry >> 28) == 1) {
      if (v_n_bits < 15) {
        v_bits |= (((uint32_t)(wuffs_base__peek_u8be__no_bounds_check(iop_a_src))) << v_n_bits);
        iop_a_src += 1;
        v_n_bits += 8;
        v_bits |= (((uint32_t)(wuffs_base__peek_u8be__no_bounds_check(iop_a_src))) << v_n_bits);
        iop_a_src += 1;
        v_n_bits += 8;
      } else {
      }
      v_redir_top = ((v_table_entry >> 8) & 65535);
      v_redir_mask = ((((uint32_t)(1)) << ((v_table_entry >> 4) & 15)) - 1);
      v_table_entry = self->private_data.f_huffs[1][((v_redir_top + (v_bits & v_redir_mask)) & 1023)];
      v_table_entry_n_bits = (v_table_entry & 15);
      v_bits >>= v_table_entry_n_bits;
      v_n_bits -= v_table_entry_n_bits;
    } else {
    }
    if ((v_table_entry >> 24) != 64) {
      if ((v_table_entry >> 24) == 8) {
        status = wuffs_base__make_status(wuffs_deflate__error__bad_huffman_code);
        goto exit;
      }
      status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
      goto exit;
    }
    v_dist_minus_1 = ((v_table_entry >> 8) & 32767);
    v_table_entry_n_bits = ((v_table_entry >> 4) & 15);
    if (v_n_bits < v_table_entry_n_bits) {
      v_bits |= (((uint32_t)(wuffs_base__peek_u8be__no_bounds_check(iop_a_src))) << v_n_bits);
      iop_a_src += 1;
      v_n_bits += 8;
      v_bits |= (((uint32_t)(wuffs_base__peek_u8be__no_bounds_check(iop_a_src))) << v_n_bits);
      iop_a_src += 1;
      v_n_bits += 8;
    }
    v_dist_minus_1 = ((v_dist_minus_1 + ((v_bits) & WUFFS_BASE__LOW_BITS_MASK__U32(v_table_entry_n_bits))) & 32767);
    v_bits >>= v_table_entry_n_bits;
    v_n_bits -= v_table_entry_n_bits;
    while (true) {
      if (((uint64_t)((v_dist_minus_1 + 1))) > ((uint64_t)(iop_a_dst - io0_a_dst))) {
        v_hlen = 0;
        v_hdist = ((uint32_t)((((uint64_t)((v_dist_minus_1 + 1))) - ((uint64_t)(iop_a_dst - io0_a_dst)))));
        if (v_length > v_hdist) {
          v_length -= v_hdist;
          v_hlen = v_hdist;
        } else {
          v_hlen = v_length;
          v_length = 0;
        }
        v_hdist += v_hdist_adjustment;
        if (self->private_impl.f_history_index < v_hdist) {
          status = wuffs_base__make_status(wuffs_deflate__error__bad_distance);
          goto exit;
        }
        wuffs_base__io_writer__limited_copy_u32_from_slice(
            &iop_a_dst, io2_a_dst,v_hlen, wuffs_base__make_slice_u8_ij(self->private_data.f_history, ((self->private_impl.f_history_index - v_hdist) & 32767), 33025));
        if (v_length == 0) {
          goto label__loop__continue;
        }
        if ((((uint64_t)((v_dist_minus_1 + 1))) > ((uint64_t)(iop_a_dst - io0_a_dst))) || (((uint64_t)(v_length)) > ((uint64_t)(io2_a_dst - iop_a_dst))) || (((uint64_t)((v_length + 8))) > ((uint64_t)(io2_a_dst - iop_a_dst)))) {
          status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_distance);
          goto exit;
        }
      }
      if ((v_dist_minus_1 + 1) >= 8) {
        wuffs_base__io_writer__limited_copy_u32_from_history_8_byte_chunks_fast(
            &iop_a_dst, io0_a_dst, io2_a_dst, v_length, (v_dist_minus_1 + 1));
      } else {
        wuffs_base__io_writer__limited_copy_u32_from_history_fast(
            &iop_a_dst, io0_a_dst, io2_a_dst, v_length, (v_dist_minus_1 + 1));
      }
      goto label__0__break;
    }
    label__0__break:;
  }
  label__loop__break:;
  while (v_n_bits >= 8) {
    v_n_bits -= 8;
    if (iop_a_src > io1_a_src) {
      iop_a_src--;
    } else {
      status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_i_o);
      goto exit;
    }
  }
  self->private_impl.f_bits = (v_bits & ((((uint32_t)(1)) << v_n_bits) - 1));
  self->private_impl.f_n_bits = v_n_bits;
  if ((self->private_impl.f_n_bits >= 8) || ((self->private_impl.f_bits >> self->private_impl.f_n_bits) != 0)) {
    status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_n_bits);
    goto exit;
  }
  goto exit;
  exit:
  if (a_dst && a_dst->data.ptr) {
    a_dst->meta.wi = ((size_t)(iop_a_dst - a_dst->data.ptr));
  }
  if (a_src && a_src->data.ptr) {
    a_src->meta.ri = ((size_t)(iop_a_src - a_src->data.ptr));
  }

  return status;
}

// -------- func deflate.decoder.decode_huffman_fast64

static wuffs_base__status
wuffs_deflate__decoder__decode_huffman_fast64(
    wuffs_deflate__decoder* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__io_buffer* a_src) {
  return (*self->private_impl.choosy_decode_huffman_fast64)(self, a_dst, a_src);
}

static wuffs_base__status
wuffs_deflate__decoder__decode_huffman_fast64__choosy_default(
    wuffs_deflate__decoder* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__io_buffer* a_src) {
  wuffs_base__status status = wuffs_base__make_status(NULL);

  uint64_t v_bits = 0;
  uint32_t v_n_bits = 0;
  uint32_t v_table_entry = 0;
  uint32_t v_table_entry_n_bits = 0;
  uint64_t v_lmask = 0;
  uint64_t v_dmask = 0;
  uint32_t v_redir_top = 0;
  uint32_t v_redir_mask = 0;
  uint32_t v_length = 0;
  uint32_t v_dist_minus_1 = 0;
  uint32_t v_hlen = 0;
  uint32_t v_hdist = 0;
  uint32_t v_hdist_adjustment = 0;

  uint8_t* iop_a_dst = NULL;
  uint8_t* io0_a_dst WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  uint8_t* io1_a_dst WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  uint8_t* io2_a_dst WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  if (a_dst && a_dst->data.ptr) {
    io0_a_dst = a_dst->data.ptr;
    io1_a_dst = io0_a_dst + a_dst->meta.wi;
    iop_a_dst = io1_a_dst;
    io2_a_dst = io0_a_dst + a_dst->data.len;
    if (a_dst->meta.closed) {
      io2_a_dst = iop_a_dst;
    }
  }
  const uint8_t* iop_a_src = NULL;
  const uint8_t* io0_a_src WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  const uint8_t* io1_a_src WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  const uint8_t* io2_a_src WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  if (a_src && a_src->data.ptr) {
    io0_a_src = a_src->data.ptr;
    io1_a_src = io0_a_src + a_src->meta.ri;
    iop_a_src = io1_a_src;
    io2_a_src = io0_a_src + a_src->meta.wi;
  }

  if ((self->private_impl.f_n_bits >= 8) || ((self->private_impl.f_bits >> (self->private_impl.f_n_bits & 7)) != 0)) {
    status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_n_bits);
    goto exit;
  }
  v_bits = ((uint64_t)(self->private_impl.f_bits));
  v_n_bits = self->private_impl.f_n_bits;
  v_lmask = ((((uint64_t)(1)) << self->private_impl.f_n_huffs_bits[0]) - 1);
  v_dmask = ((((uint64_t)(1)) << self->private_impl.f_n_huffs_bits[1]) - 1);
  if (self->private_impl.f_transformed_history_count < (a_dst ? a_dst->meta.pos : 0)) {
    status = wuffs_base__make_status(wuffs_base__error__bad_i_o_position);
    goto exit;
  }
  v_hdist_adjustment = ((uint32_t)(((self->private_impl.f_transformed_history_count - (a_dst ? a_dst->meta.pos : 0)) & 4294967295)));
  label__loop__continue:;
  while ((((uint64_t)(io2_a_dst - iop_a_dst)) >= 266) && (((uint64_t)(io2_a_src - iop_a_src)) >= 8)) {
    v_bits |= ((uint64_t)(wuffs_base__peek_u64le__no_bounds_check(iop_a_src) << (v_n_bits & 63)));
    iop_a_src += ((63 - (v_n_bits & 63)) >> 3);
    v_n_bits |= 56;
    v_table_entry = self->private_data.f_huffs[0][(v_bits & v_lmask)];
    v_table_entry_n_bits = (v_table_entry & 15);
    v_bits >>= v_table_entry_n_bits;
    v_n_bits -= v_table_entry_n_bits;
    if ((v_table_entry >> 31) != 0) {
      (wuffs_base__poke_u8be__no_bounds_check(iop_a_dst, ((uint8_t)(((v_table_entry >> 8) & 255)))), iop_a_dst += 1);
      goto label__loop__continue;
    } else if ((v_table_entry >> 30) != 0) {
    } else if ((v_table_entry >> 29) != 0) {
      self->private_impl.f_end_of_block = true;
      goto label__loop__break;
    } else if ((v_table_entry >> 28) != 0) {
      v_redir_top = ((v_table_entry >> 8) & 65535);
      v_redir_mask = ((((uint32_t)(1)) << ((v_table_entry >> 4) & 15)) - 1);
      v_table_entry = self->private_data.f_huffs[0][((v_redir_top + (((uint32_t)((v_bits & 4294967295))) & v_redir_mask)) & 1023)];
      v_table_entry_n_bits = (v_table_entry & 15);
      v_bits >>= v_table_entry_n_bits;
      v_n_bits -= v_table_entry_n_bits;
      if ((v_table_entry >> 31) != 0) {
        (wuffs_base__poke_u8be__no_bounds_check(iop_a_dst, ((uint8_t)(((v_table_entry >> 8) & 255)))), iop_a_dst += 1);
        goto label__loop__continue;
      } else if ((v_table_entry >> 30) != 0) {
      } else if ((v_table_entry >> 29) != 0) {
        self->private_impl.f_end_of_block = true;
        goto label__loop__break;
      } else if ((v_table_entry >> 28) != 0) {
        status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
        goto exit;
      } else if ((v_table_entry >> 27) != 0) {
        status = wuffs_base__make_status(wuffs_deflate__error__bad_huffman_code);
        goto exit;
      } else {
        status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
        goto exit;
      }
    } else if ((v_table_entry >> 27) != 0) {
      status = wuffs_base__make_status(wuffs_deflate__error__bad_huffman_code);
      goto exit;
    } else {
      status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
      goto exit;
    }
    v_length = (((v_table_entry >> 8) & 255) + 3);
    v_table_entry_n_bits = ((v_table_entry >> 4) & 15);
    if (v_table_entry_n_bits > 0) {
      v_length = (((v_length + 253 + ((uint32_t)(((v_bits) & WUFFS_BASE__LOW_BITS_MASK__U64(v_table_entry_n_bits))))) & 255) + 3);
      v_bits >>= v_table_entry_n_bits;
      v_n_bits -= v_table_entry_n_bits;
    }
    v_table_entry = self->private_data.f_huffs[1][(v_bits & v_dmask)];
    v_table_entry_n_bits = (v_table_entry & 15);
    v_bits >>= v_table_entry_n_bits;
    v_n_bits -= v_table_entry_n_bits;
    if ((v_table_entry >> 28) == 1) {
      v_redir_top = ((v_table_entry >> 8) & 65535);
      v_redir_mask = ((((uint32_t)(1)) << ((v_table_entry >> 4) & 15)) - 1);
      v_table_entry = self->private_data.f_huffs[1][((v_redir_top + (((uint32_t)((v_bits & 4294967295))) & v_redir_mask)) & 1023)];
      v_table_entry_n_bits = (v_table_entry & 15);
      v_bits >>= v_table_entry_n_bits;
      v_n_bits -= v_table_entry_n_bits;
    }
    if ((v_table_entry >> 24) != 64) {
      if ((v_table_entry >> 24) == 8) {
        status = wuffs_base__make_status(wuffs_deflate__error__bad_huffman_code);
        goto exit;
      }
      status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
      goto exit;
    }
    v_dist_minus_1 = ((v_table_entry >> 8) & 32767);
    v_table_entry_n_bits = ((v_table_entry >> 4) & 15);
    v_dist_minus_1 = ((v_dist_minus_1 + ((uint32_t)(((v_bits) & WUFFS_BASE__LOW_BITS_MASK__U64(v_table_entry_n_bits))))) & 32767);
    v_bits >>= v_table_entry_n_bits;
    v_n_bits -= v_table_entry_n_bits;
    while (true) {
      if (((uint64_t)((v_dist_minus_1 + 1))) > ((uint64_t)(iop_a_dst - io0_a_dst))) {
        v_hlen = 0;
        v_hdist = ((uint32_t)((((uint64_t)((v_dist_minus_1 + 1))) - ((uint64_t)(iop_a_dst - io0_a_dst)))));
        if (v_length > v_hdist) {
          v_length -= v_hdist;
          v_hlen = v_hdist;
        } else {
          v_hlen = v_length;
          v_length = 0;
        }
        v_hdist += v_hdist_adjustment;
        if (self->private_impl.f_history_index < v_hdist) {
          status = wuffs_base__make_status(wuffs_deflate__error__bad_distance);
          goto exit;
        }
        wuffs_base__io_writer__limited_copy_u32_from_slice(
            &iop_a_dst, io2_a_dst,v_hlen, wuffs_base__make_slice_u8_ij(self->private_data.f_history, ((self->private_impl.f_history_index - v_hdist) & 32767), 33025));
        if (v_length == 0) {
          goto label__loop__continue;
        }
        if ((((uint64_t)((v_dist_minus_1 + 1))) > ((uint64_t)(iop_a_dst - io0_a_dst))) || (((uint64_t)(v_length)) > ((uint64_t)(io2_a_dst - iop_a_dst))) || (((uint64_t)((v_length + 8))) > ((uint64_t)(io2_a_dst - iop_a_dst)))) {
          status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_distance);
          goto exit;
        }
      }
      if ((v_dist_minus_1 + 1) >= 8) {
        wuffs_base__io_writer__limited_copy_u32_from_history_8_byte_chunks_fast(
            &iop_a_dst, io0_a_dst, io2_a_dst, v_length, (v_dist_minus_1 + 1));
      } else if ((v_dist_minus_1 + 1) == 1) {
        wuffs_base__io_writer__limited_copy_u32_from_history_8_byte_chunks_distance_1_fast(
            &iop_a_dst, io0_a_dst, io2_a_dst, v_length, (v_dist_minus_1 + 1));
      } else {
        wuffs_base__io_writer__limited_copy_u32_from_history_fast(
            &iop_a_dst, io0_a_dst, io2_a_dst, v_length, (v_dist_minus_1 + 1));
      }
      goto label__0__break;
    }
    label__0__break:;
  }
  label__loop__break:;
  if (v_n_bits > 63) {
    status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_n_bits);
    goto exit;
  }
  while (v_n_bits >= 8) {
    v_n_bits -= 8;
    if (iop_a_src > io1_a_src) {
      iop_a_src--;
    } else {
      status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_i_o);
      goto exit;
    }
  }
  self->private_impl.f_bits = ((uint32_t)((v_bits & ((((uint64_t)(1)) << v_n_bits) - 1))));
  self->private_impl.f_n_bits = v_n_bits;
  if ((self->private_impl.f_n_bits >= 8) || ((self->private_impl.f_bits >> self->private_impl.f_n_bits) != 0)) {
    status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_n_bits);
    goto exit;
  }
  goto exit;
  exit:
  if (a_dst && a_dst->data.ptr) {
    a_dst->meta.wi = ((size_t)(iop_a_dst - a_dst->data.ptr));
  }
  if (a_src && a_src->data.ptr) {
    a_src->meta.ri = ((size_t)(iop_a_src - a_src->data.ptr));
  }

  return status;
}

// -------- func deflate.decoder.decode_huffman_slow

static wuffs_base__status
wuffs_deflate__decoder__decode_huffman_slow(
    wuffs_deflate__decoder* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__io_buffer* a_src) {
  wuffs_base__status status = wuffs_base__make_status(NULL);

  uint32_t v_bits = 0;
  uint32_t v_n_bits = 0;
  uint32_t v_table_entry = 0;
  uint32_t v_table_entry_n_bits = 0;
  uint32_t v_lmask = 0;
  uint32_t v_dmask = 0;
  uint32_t v_b0 = 0;
  uint32_t v_redir_top = 0;
  uint32_t v_redir_mask = 0;
  uint32_t v_b1 = 0;
  uint32_t v_length = 0;
  uint32_t v_b2 = 0;
  uint32_t v_b3 = 0;
  uint32_t v_b4 = 0;
  uint32_t v_dist_minus_1 = 0;
  uint32_t v_b5 = 0;
  uint32_t v_n_copied = 0;
  uint32_t v_hlen = 0;
  uint32_t v_hdist = 0;

  uint8_t* iop_a_dst = NULL;
  uint8_t* io0_a_dst WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  uint8_t* io1_a_dst WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  uint8_t* io2_a_dst WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  if (a_dst && a_dst->data.ptr) {
    io0_a_dst = a_dst->data.ptr;
    io1_a_dst = io0_a_dst + a_dst->meta.wi;
    iop_a_dst = io1_a_dst;
    io2_a_dst = io0_a_dst + a_dst->data.len;
    if (a_dst->meta.closed) {
      io2_a_dst = iop_a_dst;
    }
  }
  const uint8_t* iop_a_src = NULL;
  const uint8_t* io0_a_src WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  const uint8_t* io1_a_src WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  const uint8_t* io2_a_src WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  if (a_src && a_src->data.ptr) {
    io0_a_src = a_src->data.ptr;
    io1_a_src = io0_a_src + a_src->meta.ri;
    iop_a_src = io1_a_src;
    io2_a_src = io0_a_src + a_src->meta.wi;
  }

  uint32_t coro_susp_point = self->private_impl.p_decode_huffman_slow[0];
  if (coro_susp_point) {
    v_bits = self->private_data.s_decode_huffman_slow[0].v_bits;
    v_n_bits = self->private_data.s_decode_huffman_slow[0].v_n_bits;
    v_table_entry_n_bits = self->private_data.s_decode_huffman_slow[0].v_table_entry_n_bits;
    v_lmask = self->private_data.s_decode_huffman_slow[0].v_lmask;
    v_dmask = self->private_data.s_decode_huffman_slow[0].v_dmask;
    v_redir_top = self->private_data.s_decode_huffman_slow[0].v_redir_top;
    v_redir_mask = self->private_data.s_decode_huffman_slow[0].v_redir_mask;
    v_length = self->private_data.s_decode_huffman_slow[0].v_length;
    v_dist_minus_1 = self->private_data.s_decode_huffman_slow[0].v_dist_minus_1;
  }
  switch (coro_susp_point) {
    WUFFS_BASE__COROUTINE_SUSPENSION_POINT_0;

    if ((self->private_impl.f_n_bits >= 8) || ((self->private_impl.f_bits >> (self->private_impl.f_n_bits & 7)) != 0)) {
      status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_n_bits);
      goto exit;
    }
    v_bits = self->private_impl.f_bits;
    v_n_bits = self->private_impl.f_n_bits;
    v_lmask = ((((uint32_t)(1)) << self->private_impl.f_n_huffs_bits[0]) - 1);
    v_dmask = ((((uint32_t)(1)) << self->private_impl.f_n_huffs_bits[1]) - 1);
    label__loop__continue:;
    while ( ! (self->private_impl.p_decode_huffman_slow[0] != 0)) {
      while (true) {
        v_table_entry = self->private_data.f_huffs[0][(v_bits & v_lmask)];
        v_table_entry_n_bits = (v_table_entry & 15);
        if (v_n_bits >= v_table_entry_n_bits) {
          v_bits >>= v_table_entry_n_bits;
          v_n_bits -= v_table_entry_n_bits;
          goto label__0__break;
        }
        {
          WUFFS_BASE__COROUTINE_SUSPENSION_POINT(1);
          if (WUFFS_BASE__UNLIKELY(iop_a_src == io2_a_src)) {
            status = wuffs_base__make_status(wuffs_base__suspension__short_read);
            goto suspend;
          }
          uint32_t t_0 = *iop_a_src++;
          v_b0 = t_0;
        }
        v_bits |= (v_b0 << v_n_bits);
        v_n_bits += 8;
      }
      label__0__break:;
      if ((v_table_entry >> 31) != 0) {
        self->private_data.s_decode_huffman_slow[0].scratch = ((uint8_t)(((v_table_entry >> 8) & 255)));
        WUFFS_BASE__COROUTINE_SUSPENSION_POINT(2);
        if (iop_a_dst == io2_a_dst) {
          status = wuffs_base__make_status(wuffs_base__suspension__short_write);
          goto suspend;
        }
        *iop_a_dst++ = ((uint8_t)(self->private_data.s_decode_huffman_slow[0].scratch));
        goto label__loop__continue;
      } else if ((v_table_entry >> 30) != 0) {
      } else if ((v_table_entry >> 29) != 0) {
        self->private_impl.f_end_of_block = true;
        goto label__loop__break;
      } else if ((v_table_entry >> 28) != 0) {
        v_redir_top = ((v_table_entry >> 8) & 65535);
        v_redir_mask = ((((uint32_t)(1)) << ((v_table_entry >> 4) & 15)) - 1);
        while (true) {
          v_table_entry = self->private_data.f_huffs[0][((v_redir_top + (v_bits & v_redir_mask)) & 1023)];
          v_table_entry_n_bits = (v_table_entry & 15);
          if (v_n_bits >= v_table_entry_n_bits) {
            v_bits >>= v_table_entry_n_bits;
            v_n_bits -= v_table_entry_n_bits;
            goto label__1__break;
          }
          {
            WUFFS_BASE__COROUTINE_SUSPENSION_POINT(3);
            if (WUFFS_BASE__UNLIKELY(iop_a_src == io2_a_src)) {
              status = wuffs_base__make_status(wuffs_base__suspension__short_read);
              goto suspend;
            }
            uint32_t t_1 = *iop_a_src++;
            v_b1 = t_1;
          }
          v_bits |= (v_b1 << v_n_bits);
          v_n_bits += 8;
        }
        label__1__break:;
        if ((v_table_entry >> 31) != 0) {
          self->private_data.s_decode_huffman_slow[0].scratch = ((uint8_t)(((v_table_entry >> 8) & 255)));
          WUFFS_BASE__COROUTINE_SUSPENSION_POINT(4);
          if (iop_a_dst == io2_a_dst) {
            status = wuffs_base__make_status(wuffs_base__suspension__short_write);
            goto suspend;
          }
          *iop_a_dst++ = ((uint8_t)(self->private_data.s_decode_huffman_slow[0].scratch));
          goto label__loop__continue;
        } else if ((v_table_entry >> 30) != 0) {
        } else if ((v_table_entry >> 29) != 0) {
          self->private_impl.f_end_of_block = true;
          goto label__loop__break;
        } else if ((v_table_entry >> 28) != 0) {
          status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
          goto exit;
        } else if ((v_table_entry >> 27) != 0) {
          status = wuffs_base__make_status(wuffs_deflate__error__bad_huffman_code);
          goto exit;
        } else {
          status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
          goto exit;
        }
      } else if ((v_table_entry >> 27) != 0) {
        status = wuffs_base__make_status(wuffs_deflate__error__bad_huffman_code);
        goto exit;
      } else {
        status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
        goto exit;
      }
      v_length = (((v_table_entry >> 8) & 255) + 3);
      v_table_entry_n_bits = ((v_table_entry >> 4) & 15);
      if (v_table_entry_n_bits > 0) {
        while (v_n_bits < v_table_entry_n_bits) {
          {
            WUFFS_BASE__COROUTINE_SUSPENSION_POINT(5);
            if (WUFFS_BASE__UNLIKELY(iop_a_src == io2_a_src)) {
              status = wuffs_base__make_status(wuffs_base__suspension__short_read);
              goto suspend;
            }
            uint32_t t_2 = *iop_a_src++;
            v_b2 = t_2;
          }
          v_bits |= (v_b2 << v_n_bits);
          v_n_bits += 8;
        }
        v_length = (((v_length + 253 + ((v_bits) & WUFFS_BASE__LOW_BITS_MASK__U32(v_table_entry_n_bits))) & 255) + 3);
        v_bits >>= v_table_entry_n_bits;
        v_n_bits -= v_table_entry_n_bits;
      }
      while (true) {
        v_table_entry = self->private_data.f_huffs[1][(v_bits & v_dmask)];
        v_table_entry_n_bits = (v_table_entry & 15);
        if (v_n_bits >= v_table_entry_n_bits) {
          v_bits >>= v_table_entry_n_bits;
          v_n_bits -= v_table_entry_n_bits;
          goto label__2__break;
        }
        {
          WUFFS_BASE__COROUTINE_SUSPENSION_POINT(6);
          if (WUFFS_BASE__UNLIKELY(iop_a_src == io2_a_src)) {
            status = wuffs_base__make_status(wuffs_base__suspension__short_read);
            goto suspend;
          }
          uint32_t t_3 = *iop_a_src++;
          v_b3 = t_3;
        }
        v_bits |= (v_b3 << v_n_bits);
        v_n_bits += 8;
      }
      label__2__break:;
      if ((v_table_entry >> 28) == 1) {
        v_redir_top = ((v_table_entry >> 8) & 65535);
        v_redir_mask = ((((uint32_t)(1)) << ((v_table_entry >> 4) & 15)) - 1);
        while (true) {
          v_table_entry = self->private_data.f_huffs[1][((v_redir_top + (v_bits & v_redir_mask)) & 1023)];
          v_table_entry_n_bits = (v_table_entry & 15);
          if (v_n_bits >= v_table_entry_n_bits) {
            v_bits >>= v_table_entry_n_bits;
            v_n_bits -= v_table_entry_n_bits;
            goto label__3__break;
          }
          {
            WUFFS_BASE__COROUTINE_SUSPENSION_POINT(7);
            if (WUFFS_BASE__UNLIKELY(iop_a_src == io2_a_src)) {
              status = wuffs_base__make_status(wuffs_base__suspension__short_read);
              goto suspend;
            }
            uint32_t t_4 = *iop_a_src++;
            v_b4 = t_4;
          }
          v_bits |= (v_b4 << v_n_bits);
          v_n_bits += 8;
        }
        label__3__break:;
      }
      if ((v_table_entry >> 24) != 64) {
        if ((v_table_entry >> 24) == 8) {
          status = wuffs_base__make_status(wuffs_deflate__error__bad_huffman_code);
          goto exit;
        }
        status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_huffman_decoder_state);
        goto exit;
      }
      v_dist_minus_1 = ((v_table_entry >> 8) & 32767);
      v_table_entry_n_bits = ((v_table_entry >> 4) & 15);
      if (v_table_entry_n_bits > 0) {
        while (v_n_bits < v_table_entry_n_bits) {
          {
            WUFFS_BASE__COROUTINE_SUSPENSION_POINT(8);
            if (WUFFS_BASE__UNLIKELY(iop_a_src == io2_a_src)) {
              status = wuffs_base__make_status(wuffs_base__suspension__short_read);
              goto suspend;
            }
            uint32_t t_5 = *iop_a_src++;
            v_b5 = t_5;
          }
          v_bits |= (v_b5 << v_n_bits);
          v_n_bits += 8;
        }
        v_dist_minus_1 = ((v_dist_minus_1 + ((v_bits) & WUFFS_BASE__LOW_BITS_MASK__U32(v_table_entry_n_bits))) & 32767);
        v_bits >>= v_table_entry_n_bits;
        v_n_bits -= v_table_entry_n_bits;
      }
      label__inner__continue:;
      while (true) {
        if (((uint64_t)((v_dist_minus_1 + 1))) > ((uint64_t)(iop_a_dst - io0_a_dst))) {
          v_hdist = ((uint32_t)((((uint64_t)((v_dist_minus_1 + 1))) - ((uint64_t)(iop_a_dst - io0_a_dst)))));
          if (v_hdist < v_length) {
            v_hlen = v_hdist;
          } else {
            v_hlen = v_length;
          }
          v_hdist += ((uint32_t)((((uint64_t)(self->private_impl.f_transformed_history_count - (a_dst ? a_dst->meta.pos : 0))) & 4294967295)));
          if (self->private_impl.f_history_index < v_hdist) {
            status = wuffs_base__make_status(wuffs_deflate__error__bad_distance);
            goto exit;
          }
          v_n_copied = wuffs_base__io_writer__limited_copy_u32_from_slice(
              &iop_a_dst, io2_a_dst,v_hlen, wuffs_base__make_slice_u8_ij(self->private_data.f_history, ((self->private_impl.f_history_index - v_hdist) & 32767), 33025));
          if (v_n_copied < v_hlen) {
            v_length -= v_n_copied;
            status = wuffs_base__make_status(wuffs_base__suspension__short_write);
            WUFFS_BASE__COROUTINE_SUSPENSION_POINT_MAYBE_SUSPEND(9);
            goto label__inner__continue;
          }
          v_length -= v_hlen;
          if (v_length == 0) {
            goto label__loop__continue;
          }
        }
        v_n_copied = wuffs_base__io_writer__limited_copy_u32_from_history(
            &iop_a_dst, io0_a_dst, io2_a_dst, v_length, (v_dist_minus_1 + 1));
        if (v_length <= v_n_copied) {
          goto label__loop__continue;
        }
        v_length -= v_n_copied;
        status = wuffs_base__make_status(wuffs_base__suspension__short_write);
        WUFFS_BASE__COROUTINE_SUSPENSION_POINT_MAYBE_SUSPEND(10);
      }
    }
    label__loop__break:;
    self->private_impl.f_bits = v_bits;
    self->private_impl.f_n_bits = v_n_bits;
    if ((self->private_impl.f_n_bits >= 8) || ((self->private_impl.f_bits >> (self->private_impl.f_n_bits & 7)) != 0)) {
      status = wuffs_base__make_status(wuffs_deflate__error__internal_error_inconsistent_n_bits);
      goto exit;
    }

    ok:
    self->private_impl.p_decode_huffman_slow[0] = 0;
    goto exit;
  }

  goto suspend;
  suspend:
  self->private_impl.p_decode_huffman_slow[0] = wuffs_base__status__is_suspension(&status) ? coro_susp_point : 0;
  self->private_data.s_decode_huffman_slow[0].v_bits = v_bits;
  self->private_data.s_decode_huffman_slow[0].v_n_bits = v_n_bits;
  self->private_data.s_decode_huffman_slow[0].v_table_entry_n_bits = v_table_entry_n_bits;
  self->private_data.s_decode_huffman_slow[0].v_lmask = v_lmask;
  self->private_data.s_decode_huffman_slow[0].v_dmask = v_dmask;
  self->private_data.s_decode_huffman_slow[0].v_redir_top = v_redir_top;
  self->private_data.s_decode_huffman_slow[0].v_redir_mask = v_redir_mask;
  self->private_data.s_decode_huffman_slow[0].v_length = v_length;
  self->private_data.s_decode_huffman_slow[0].v_dist_minus_1 = v_dist_minus_1;

  goto exit;
  exit:
  if (a_dst && a_dst->data.ptr) {
    a_dst->meta.wi = ((size_t)(iop_a_dst - a_dst->data.ptr));
  }
  if (a_src && a_src->data.ptr) {
    a_src->meta.ri = ((size_t)(iop_a_src - a_src->data.ptr));
  }

  return status;
}

#endif  // !defined(WUFFS_CONFIG__MODULES) || defined(WUFFS_CONFIG__MODULE__DEFLATE)

#if !defined(WUFFS_CONFIG__MODULES) || defined(WUFFS_CONFIG__MODULE__ZLIB)

// ---------------- Status Codes Implementations

const char wuffs_zlib__note__dictionary_required[] = "@zlib: dictionary required";
const char wuffs_zlib__error__bad_checksum[] = "#zlib: bad checksum";
const char wuffs_zlib__error__bad_compression_method[] = "#zlib: bad compression method";
const char wuffs_zlib__error__bad_compression_window_size[] = "#zlib: bad compression window size";
const char wuffs_zlib__error__bad_parity_check[] = "#zlib: bad parity check";
const char wuffs_zlib__error__incorrect_dictionary[] = "#zlib: incorrect dictionary";
const char wuffs_zlib__error__truncated_input[] = "#zlib: truncated input";

// ---------------- Private Consts

#define WUFFS_ZLIB__QUIRKS_BASE 2113790976

#define WUFFS_ZLIB__QUIRKS_COUNT 1

// ---------------- Private Initializer Prototypes

// ---------------- Private Function Prototypes

static wuffs_base__status
wuffs_zlib__decoder__do_transform_io(
    wuffs_zlib__decoder* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__io_buffer* a_src,
    wuffs_base__slice_u8 a_workbuf);

// ---------------- VTables

const wuffs_base__io_transformer__func_ptrs
wuffs_zlib__decoder__func_ptrs_for__wuffs_base__io_transformer = {
  (wuffs_base__empty_struct(*)(void*,
      uint32_t,
      bool))(&wuffs_zlib__decoder__set_quirk_enabled),
  (wuffs_base__status(*)(void*,
      wuffs_base__io_buffer*,
      wuffs_base__io_buffer*,
      wuffs_base__slice_u8))(&wuffs_zlib__decoder__transform_io),
  (wuffs_base__range_ii_u64(*)(const void*))(&wuffs_zlib__decoder__workbuf_len),
};

// ---------------- Initializer Implementations

wuffs_base__status WUFFS_BASE__WARN_UNUSED_RESULT
wuffs_zlib__decoder__initialize(
    wuffs_zlib__decoder* self,
    size_t sizeof_star_self,
    uint64_t wuffs_version,
    uint32_t options){
  if (!self) {
    return wuffs_base__make_status(wuffs_base__error__bad_receiver);
  }
  if (sizeof(*self) != sizeof_star_self) {
    return wuffs_base__make_status(wuffs_base__error__bad_sizeof_receiver);
  }
  if (((wuffs_version >> 32) != WUFFS_VERSION_MAJOR) ||
      (((wuffs_version >> 16) & 0xFFFF) > WUFFS_VERSION_MINOR)) {
    return wuffs_base__make_status(wuffs_base__error__bad_wuffs_version);
  }

  if ((options & WUFFS_INITIALIZE__ALREADY_ZEROED) != 0) {
    // The whole point of this if-check is to detect an uninitialized *self.
    // We disable the warning on GCC. Clang-5.0 does not have this warning.
#if !defined(__clang__) && defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
    if (self->private_impl.magic != 0) {
      return wuffs_base__make_status(wuffs_base__error__initialize_falsely_claimed_already_zeroed);
    }
#if !defined(__clang__) && defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
  } else {
    if ((options & WUFFS_INITIALIZE__LEAVE_INTERNAL_BUFFERS_UNINITIALIZED) == 0) {
      memset(self, 0, sizeof(*self));
      options |= WUFFS_INITIALIZE__ALREADY_ZEROED;
    } else {
      memset(&(self->private_impl), 0, sizeof(self->private_impl));
    }
  }

  {
    wuffs_base__status z = wuffs_adler32__hasher__initialize(
        &self->private_data.f_checksum, sizeof(self->private_data.f_checksum), WUFFS_VERSION, options);
    if (z.repr) {
      return z;
    }
  }
  {
    wuffs_base__status z = wuffs_adler32__hasher__initialize(
        &self->private_data.f_dict_id_hasher, sizeof(self->private_data.f_dict_id_hasher), WUFFS_VERSION, options);
    if (z.repr) {
      return z;
    }
  }
  {
    wuffs_base__status z = wuffs_deflate__decoder__initialize(
        &self->private_data.f_flate, sizeof(self->private_data.f_flate), WUFFS_VERSION, options);
    if (z.repr) {
      return z;
    }
  }
  self->private_impl.magic = WUFFS_BASE__MAGIC;
  self->private_impl.vtable_for__wuffs_base__io_transformer.vtable_name =
      wuffs_base__io_transformer__vtable_name;
  self->private_impl.vtable_for__wuffs_base__io_transformer.function_pointers =
      (const void*)(&wuffs_zlib__decoder__func_ptrs_for__wuffs_base__io_transformer);
  return wuffs_base__make_status(NULL);
}

wuffs_zlib__decoder*
wuffs_zlib__decoder__alloc() {
  wuffs_zlib__decoder* x =
      (wuffs_zlib__decoder*)(calloc(sizeof(wuffs_zlib__decoder), 1));
  if (!x) {
    return NULL;
  }
  if (wuffs_zlib__decoder__initialize(
      x, sizeof(wuffs_zlib__decoder), WUFFS_VERSION, WUFFS_INITIALIZE__ALREADY_ZEROED).repr) {
    free(x);
    return NULL;
  }
  return x;
}

size_t
sizeof__wuffs_zlib__decoder() {
  return sizeof(wuffs_zlib__decoder);
}

// ---------------- Function Implementations

// -------- func zlib.decoder.dictionary_id

WUFFS_BASE__MAYBE_STATIC uint32_t
wuffs_zlib__decoder__dictionary_id(
    const wuffs_zlib__decoder* self) {
  if (!self) {
    return 0;
  }
  if ((self->private_impl.magic != WUFFS_BASE__MAGIC) &&
      (self->private_impl.magic != WUFFS_BASE__DISABLED)) {
    return 0;
  }

  return self->private_impl.f_dict_id_want;
}

// -------- func zlib.decoder.add_dictionary

WUFFS_BASE__MAYBE_STATIC wuffs_base__empty_struct
wuffs_zlib__decoder__add_dictionary(
    wuffs_zlib__decoder* self,
    wuffs_base__slice_u8 a_dict) {
  if (!self) {
    return wuffs_base__make_empty_struct();
  }
  if (self->private_impl.magic != WUFFS_BASE__MAGIC) {
    return wuffs_base__make_empty_struct();
  }

  if (self->private_impl.f_header_complete) {
    self->private_impl.f_bad_call_sequence = true;
  } else {
    self->private_impl.f_dict_id_got = wuffs_adler32__hasher__update_u32(&self->private_data.f_dict_id_hasher, a_dict);
    wuffs_deflate__decoder__add_history(&self->private_data.f_flate, a_dict);
  }
  self->private_impl.f_got_dictionary = true;
  return wuffs_base__make_empty_struct();
}

// -------- func zlib.decoder.set_quirk_enabled

WUFFS_BASE__MAYBE_STATIC wuffs_base__empty_struct
wuffs_zlib__decoder__set_quirk_enabled(
    wuffs_zlib__decoder* self,
    uint32_t a_quirk,
    bool a_enabled) {
  if (!self) {
    return wuffs_base__make_empty_struct();
  }
  if (self->private_impl.magic != WUFFS_BASE__MAGIC) {
    return wuffs_base__make_empty_struct();
  }

  if (self->private_impl.f_header_complete) {
    self->private_impl.f_bad_call_sequence = true;
  } else if (a_quirk == 1) {
    self->private_impl.f_ignore_checksum = a_enabled;
  } else if (a_quirk >= 2113790976) {
    a_quirk -= 2113790976;
    if (a_quirk < 1) {
      self->private_impl.f_quirks[a_quirk] = a_enabled;
    }
  }
  return wuffs_base__make_empty_struct();
}

// -------- func zlib.decoder.workbuf_len

WUFFS_BASE__MAYBE_STATIC wuffs_base__range_ii_u64
wuffs_zlib__decoder__workbuf_len(
    const wuffs_zlib__decoder* self) {
  if (!self) {
    return wuffs_base__utility__empty_range_ii_u64();
  }
  if ((self->private_impl.magic != WUFFS_BASE__MAGIC) &&
      (self->private_impl.magic != WUFFS_BASE__DISABLED)) {
    return wuffs_base__utility__empty_range_ii_u64();
  }

  return wuffs_base__utility__make_range_ii_u64(1, 1);
}

// -------- func zlib.decoder.transform_io

WUFFS_BASE__MAYBE_STATIC wuffs_base__status
wuffs_zlib__decoder__transform_io(
    wuffs_zlib__decoder* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__io_buffer* a_src,
    wuffs_base__slice_u8 a_workbuf) {
  if (!self) {
    return wuffs_base__make_status(wuffs_base__error__bad_receiver);
  }
  if (self->private_impl.magic != WUFFS_BASE__MAGIC) {
    return wuffs_base__make_status(
        (self->private_impl.magic == WUFFS_BASE__DISABLED)
        ? wuffs_base__error__disabled_by_previous_error
        : wuffs_base__error__initialize_not_called);
  }
  if (!a_dst || !a_src) {
    self->private_impl.magic = WUFFS_BASE__DISABLED;
    return wuffs_base__make_status(wuffs_base__error__bad_argument);
  }
  if ((self->private_impl.active_coroutine != 0) &&
      (self->private_impl.active_coroutine != 1)) {
    self->private_impl.magic = WUFFS_BASE__DISABLED;
    return wuffs_base__make_status(wuffs_base__error__interleaved_coroutine_calls);
  }
  self->private_impl.active_coroutine = 0;
  wuffs_base__status status = wuffs_base__make_status(NULL);

  wuffs_base__status v_status = wuffs_base__make_status(NULL);

  uint32_t coro_susp_point = self->private_impl.p_transform_io[0];
  switch (coro_susp_point) {
    WUFFS_BASE__COROUTINE_SUSPENSION_POINT_0;

    while (true) {
      {
        wuffs_base__status t_0 = wuffs_zlib__decoder__do_transform_io(self, a_dst, a_src, a_workbuf);
        v_status = t_0;
      }
      if ((v_status.repr == wuffs_base__suspension__short_read) && (a_src && a_src->meta.closed)) {
        status = wuffs_base__make_status(wuffs_zlib__error__truncated_input);
        goto exit;
      }
      status = v_status;
      WUFFS_BASE__COROUTINE_SUSPENSION_POINT_MAYBE_SUSPEND(1);
    }

    ok:
    self->private_impl.p_transform_io[0] = 0;
    goto exit;
  }

  goto suspend;
  suspend:
  self->private_impl.p_transform_io[0] = wuffs_base__status__is_suspension(&status) ? coro_susp_point : 0;
  self->private_impl.active_coroutine = wuffs_base__status__is_suspension(&status) ? 1 : 0;

  goto exit;
  exit:
  if (wuffs_base__status__is_error(&status)) {
    self->private_impl.magic = WUFFS_BASE__DISABLED;
  }
  return status;
}

// -------- func zlib.decoder.do_transform_io

static wuffs_base__status
wuffs_zlib__decoder__do_transform_io(
    wuffs_zlib__decoder* self,
    wuffs_base__io_buffer* a_dst,
    wuffs_base__io_buffer* a_src,
    wuffs_base__slice_u8 a_workbuf) {
  wuffs_base__status status = wuffs_base__make_status(NULL);

  uint16_t v_x = 0;
  uint32_t v_checksum_got = 0;
  wuffs_base__status v_status = wuffs_base__make_status(NULL);
  uint32_t v_checksum_want = 0;
  uint64_t v_mark = 0;

  uint8_t* iop_a_dst = NULL;
  uint8_t* io0_a_dst WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  uint8_t* io1_a_dst WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  uint8_t* io2_a_dst WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  if (a_dst && a_dst->data.ptr) {
    io0_a_dst = a_dst->data.ptr;
    io1_a_dst = io0_a_dst + a_dst->meta.wi;
    iop_a_dst = io1_a_dst;
    io2_a_dst = io0_a_dst + a_dst->data.len;
    if (a_dst->meta.closed) {
      io2_a_dst = iop_a_dst;
    }
  }
  const uint8_t* iop_a_src = NULL;
  const uint8_t* io0_a_src WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  const uint8_t* io1_a_src WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  const uint8_t* io2_a_src WUFFS_BASE__POTENTIALLY_UNUSED = NULL;
  if (a_src && a_src->data.ptr) {
    io0_a_src = a_src->data.ptr;
    io1_a_src = io0_a_src + a_src->meta.ri;
    iop_a_src = io1_a_src;
    io2_a_src = io0_a_src + a_src->meta.wi;
  }

  uint32_t coro_susp_point = self->private_impl.p_do_transform_io[0];
  if (coro_susp_point) {
    v_checksum_got = self->private_data.s_do_transform_io[0].v_checksum_got;
  }
  switch (coro_susp_point) {
    WUFFS_BASE__COROUTINE_SUSPENSION_POINT_0;

    if (self->private_impl.f_bad_call_sequence) {
      status = wuffs_base__make_status(wuffs_base__error__bad_call_sequence);
      goto exit;
    } else if (self->private_impl.f_quirks[0]) {
    } else if ( ! self->private_impl.f_want_dictionary) {
      {
        WUFFS_BASE__COROUTINE_SUSPENSION_POINT(1);
        uint16_t t_0;
        if (WUFFS_BASE__LIKELY(io2_a_src - iop_a_src >= 2)) {
          t_0 = wuffs_base__peek_u16be__no_bounds_check(iop_a_src);
          iop_a_src += 2;
        } else {
          self->private_data.s_do_transform_io[0].scratch = 0;
          WUFFS_BASE__COROUTINE_SUSPENSION_POINT(2);
          while (true) {
            if (WUFFS_BASE__UNLIKELY(iop_a_src == io2_a_src)) {
              status = wuffs_base__make_status(wuffs_base__suspension__short_read);
              goto suspend;
            }
            uint64_t* scratch = &self->private_data.s_do_transform_io[0].scratch;
            uint32_t num_bits_0 = ((uint32_t)(*scratch & 0xFF));
            *scratch >>= 8;
            *scratch <<= 8;
            *scratch |= ((uint64_t)(*iop_a_src++)) << (56 - num_bits_0);
            if (num_bits_0 == 8) {
              t_0 = ((uint16_t)(*scratch >> 48));
              break;
            }
            num_bits_0 += 8;
            *scratch |= ((uint64_t)(num_bits_0));
          }
        }
        v_x = t_0;
      }
      if (((v_x >> 8) & 15) != 8) {
        status = wuffs_base__make_status(wuffs_zlib__error__bad_compression_method);
        goto exit;
      }
      if ((v_x >> 12) > 7) {
        status = wuffs_base__make_status(wuffs_zlib__error__bad_compression_window_size);
        goto exit;
      }
      if ((v_x % 31) != 0) {
        status = wuffs_base__make_status(wuffs_zlib__error__bad_parity_check);
        goto exit;
      }
      self->private_impl.f_want_dictionary = ((v_x & 32) != 0);
      if (self->private_impl.f_want_dictionary) {
        self->private_impl.f_dict_id_got = 1;
        {
          WUFFS_BASE__COROUTINE_SUSPENSION_POINT(3);
          uint32_t t_1;
          if (WUFFS_BASE__LIKELY(io2_a_src - iop_a_src >= 4)) {
            t_1 = wuffs_base__peek_u32be__no_bounds_check(iop_a_src);
            iop_a_src += 4;
          } else {
            self->private_data.s_do_transform_io[0].scratch = 0;
            WUFFS_BASE__COROUTINE_SUSPENSION_POINT(4);
            while (true) {
              if (WUFFS_BASE__UNLIKELY(iop_a_src == io2_a_src)) {
                status = wuffs_base__make_status(wuffs_base__suspension__short_read);
                goto suspend;
              }
              uint64_t* scratch = &self->private_data.s_do_transform_io[0].scratch;
              uint32_t num_bits_1 = ((uint32_t)(*scratch & 0xFF));
              *scratch >>= 8;
              *scratch <<= 8;
              *scratch |= ((uint64_t)(*iop_a_src++)) << (56 - num_bits_1);
              if (num_bits_1 == 24) {
                t_1 = ((uint32_t)(*scratch >> 32));
                break;
              }
              num_bits_1 += 8;
              *scratch |= ((uint64_t)(num_bits_1));
            }
          }
          self->private_impl.f_dict_id_want = t_1;
        }
        status = wuffs_base__make_status(wuffs_zlib__note__dictionary_required);
        goto ok;
      } else if (self->private_impl.f_got_dictionary) {
        status = wuffs_base__make_status(wuffs_zlib__error__incorrect_dictionary);
        goto exit;
      }
    } else if (self->private_impl.f_dict_id_got != self->private_impl.f_dict_id_want) {
      if (self->private_impl.f_got_dictionary) {
        status = wuffs_base__make_status(wuffs_zlib__error__incorrect_dictionary);
        goto exit;
      }
      status = wuffs_base__make_status(wuffs_zlib__note__dictionary_required);
      goto ok;
    }
    self->private_impl.f_header_complete = true;
    while (true) {
      v_mark = ((uint64_t)(iop_a_dst - io0_a_dst));
      {
        if (a_dst) {
          a_dst->meta.wi = ((size_t)(iop_a_dst - a_dst->data.ptr));
        }
        if (a_src) {
          a_src->meta.ri = ((size_t)(iop_a_src - a_src->data.ptr));
        }
        wuffs_base__status t_2 = wuffs_deflate__decoder__transform_io(&self->private_data.f_flate, a_dst, a_src, a_workbuf);
        v_status = t_2;
        if (a_dst) {
          iop_a_dst = a_dst->data.ptr + a_dst->meta.wi;
        }
        if (a_src) {
          iop_a_src = a_src->data.ptr + a_src->meta.ri;
        }
      }
      if ( ! self->private_impl.f_ignore_checksum &&  ! self->private_impl.f_quirks[0]) {
        v_checksum_got = wuffs_adler32__hasher__update_u32(&self->private_data.f_checksum, wuffs_base__io__since(v_mark, ((uint64_t)(iop_a_dst - io0_a_dst)), io0_a_dst));
      }
      if (wuffs_base__status__is_ok(&v_status)) {
        goto label__0__break;
      }
      status = v_status;
      WUFFS_BASE__COROUTINE_SUSPENSION_POINT_MAYBE_SUSPEND(5);
    }
    label__0__break:;
    if ( ! self->private_impl.f_quirks[0]) {
      {
        WUFFS_BASE__COROUTINE_SUSPENSION_POINT(6);
        uint32_t t_3;
        if (WUFFS_BASE__LIKELY(io2_a_src - iop_a_src >= 4)) {
          t_3 = wuffs_base__peek_u32be__no_bounds_check(iop_a_src);
          iop_a_src += 4;
        } else {
          self->private_data.s_do_transform_io[0].scratch = 0;
          WUFFS_BASE__COROUTINE_SUSPENSION_POINT(7);
          while (true) {
            if (WUFFS_BASE__UNLIKELY(iop_a_src == io2_a_src)) {
              status = wuffs_base__make_status(wuffs_base__suspension__short_read);
              goto suspend;
            }
            uint64_t* scratch = &self->private_data.s_do_transform_io[0].scratch;
            uint32_t num_bits_3 = ((uint32_t)(*scratch & 0xFF));
            *scratch >>= 8;
            *scratch <<= 8;
            *scratch |= ((uint64_t)(*iop_a_src++)) << (56 - num_bits_3);
            if (num_bits_3 == 24) {
              t_3 = ((uint32_t)(*scratch >> 32));
              break;
            }
            num_bits_3 += 8;
            *scratch |= ((uint64_t)(num_bits_3));
          }
        }
        v_checksum_want = t_3;
      }
      if ( ! self->private_impl.f_ignore_checksum && (v_checksum_got != v_checksum_want)) {
        status = wuffs_base__make_status(wuffs_zlib__error__bad_checksum);
        goto exit;
      }
    }

    ok:
    self->private_impl.p_do_transform_io[0] = 0;
    goto exit;
  }

  goto suspend;
  suspend:
  self->private_impl.p_do_transform_io[0] = wuffs_base__status__is_suspension(&status) ? coro_susp_point : 0;
  self->private_data.s_do_transform_io[0].v_checksum_got = v_checksum_got;

  goto exit;
  exit:
  if (a_dst && a_dst->data.ptr) {
    a_dst->meta.wi = ((size_t)(iop_a_dst - a_dst->data.ptr));
  }
  if (a_src && a_src->data.ptr) {
    a_src->meta.ri = ((size_t)(iop_a_src - a_src->data.ptr));
  }

  return status;
}

#endif  // !defined(WUFFS_CONFIG__MODULES) || defined(WUFFS_CONFIG__MODULE__ZLIB)



#endif  // WUFFS_IMPLEMENTATION

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(__clang__)
#pragma clang diagnostic pop
#endif

#endif  // WUFFS_INCLUDE_GUARD
