#pragma once

#include <cstdint>
#include <x86intrin.h>

struct Timer {
    uint64_t start;

    void begin() {
        start = __rdtsc();
    }

    uint64_t elapsed_cycles() {
        return __rdtsc() - start;
    }
};
