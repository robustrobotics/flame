/**
 * This file is part of FLaME.
 * Copyright (C) 2017 W. Nicholas Greene (wng@csail.mit.edu)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see <http://www.gnu.org/licenses/>.
 *
 * @file assert.h
 * @author W. Nicholas Greene
 * @date 2017-08-18 18:22:29 (Fri)
 */

#pragma once

#ifdef FLAME_NO_ASSERT
#define FLAME_ASSERT(x)
#else
#include <cxxabi.h>
#include <unistd.h>
#include <execinfo.h> // for stack trace
#include <cstdlib>   // for abort

#define FLAME_COLOR_RESET   "\033[0m"
#define FLAME_COLOR_RED     "\033[31m"
#define FLAME_COLOR_GREEN   "\033[32m"

namespace flame {

namespace utils {

inline void assert_fail(const char *condition, const char *function,
                        const char *file, int line) {
  fprintf(stderr, FLAME_COLOR_RED "FLAME_ASSERT failed: %s in function %s at %s: %i\n" FLAME_COLOR_RESET,
          condition, function, file, line);

  fprintf(stderr, FLAME_COLOR_RED "Stacktrace:\n" FLAME_COLOR_RESET);

  // Get and print stack trace with demangled names. Taken from:
  // https://panthema.net/2008/0901-stacktrace-demangled
  constexpr int max_frames = 16;
  void* stack_frames[max_frames];
  int num_frames = backtrace(stack_frames, max_frames); // Get stack addresses.

  // Get strings of trace.
  char** symbols = backtrace_symbols(stack_frames, num_frames);

  // Allocate string which will be filled with the demangled function name.
  // allocate string which will be filled with the demangled function name
  size_t funcnamesize = 256;
  char* funcname = static_cast<char*>(malloc(funcnamesize));

  // iterate over the returned symbol lines. skip the first, it is the
  // address of this function.
  for (int i = 1; i < num_frames; i++) {
    char *begin_name = 0, *begin_offset = 0, *end_offset = 0;

    // find parentheses and +address offset surrounding the mangled name:
    // ./module(function+0x15c) [0x8048a6d]
    for (char* p = symbols[i]; *p; ++p) {
      if (*p == '(') {
        begin_name = p;
      } else if (*p == '+') {
        begin_offset = p;
      } else if (*p == ')' && begin_offset) {
        end_offset = p;
        break;
      }
    }

    if (begin_name && begin_offset && end_offset && begin_name < begin_offset) {
      *begin_name++ = '\0';
      *begin_offset++ = '\0';
      *end_offset = '\0';

      // mangled name is now in [begin_name, begin_offset) and caller
      // offset in [begin_offset, end_offset). now apply
      // __cxa_demangle():

      int status;
      char* ret = abi::__cxa_demangle(begin_name, funcname, &funcnamesize,
                                      &status);
      if (status == 0) {
        funcname = ret; // use possibly realloc()-ed string
        fprintf(stderr, FLAME_COLOR_RED "  %s : %s+%s\n" FLAME_COLOR_RESET,
                symbols[i], funcname, begin_offset);
      } else {
        // demangling failed. Output function name as a C function with
        // no arguments.
        fprintf(stderr, FLAME_COLOR_RED "  %s : %s()+%s\n" FLAME_COLOR_RESET,
                symbols[i], begin_name, begin_offset);
      }
    } else {
      // couldn't parse the line? print the whole line.
      fprintf(stderr, FLAME_COLOR_RED "  %s\n" FLAME_COLOR_RESET, symbols[i]);
    }
  }

  free(funcname);
  free(symbols);

  exit(1);
  // abort();
}

}  // namespace utils

}  // namespace flame

#define FLAME_ASSERT(condition) \
  do { \
    if (!(condition)) \
      flame::utils::assert_fail(#condition, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
  } while (false)

#endif
