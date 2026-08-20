#pragma once

#include <array>
#include <atomic>
#include <cstddef>

template <typename T, size_t Capacity>
class SPSCQueue {
private:
    std::array<T, Capacity> queue;
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");

public:
    bool push(const T& item) {
        const size_t h  = head_.load(std::memory_order_relaxed); // own it

        if (h - tail_.load(std::memory_order_acquire) == Capacity) {
            return false;
        }

        queue[h & (Capacity - 1)] = item;
        head_.store(h + 1, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        const size_t t = tail_.load(std::memory_order_relaxed); // own it

        if (head_.load(std::memory_order_acquire) - t == 0) {
            return false;
        }
        item = queue[t & (Capacity - 1)];
        tail_.store(t + 1, std::memory_order_release);
        return true;
    }
};
