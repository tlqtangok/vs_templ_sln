#pragma once 

#define FMT_LIB 1

#if FMT_LIB
#include <fmt/ranges.h>
#include <fmt/core.h>
#include <fmt/color.h>
#include <fmt/xchar.h>
#include <fmt/os.h>
#include <fmt/std.h>

#define cout_ fmt::print
#define s_ fmt::format

#endif 