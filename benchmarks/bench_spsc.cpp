#include "core/spsc_queue.hpp"
#include <algorithm>
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <assert.h>
#include <iostream>
#include <x86intrin.h>
#include <vector>
#include <cmath>

using ll = long long;

/*
 * acquire only guarantees that the operation is after the previous release on the same atomic
 * so if in a loop iteration 1 releases and between iteration 2 and iteration 1 another thread
 * seeks to acquire it a million times it will always succeed since to it the previous release
 * was by iteration 1
*/

SPSCQueue<ll, 1 << 10> buffer;
std::atomic<bool> ready;
constexpr int epochs = 100000;
constexpr double tsc_ghz = 2.4192; // my machine's TSC is 2419.2 MHz

void pin_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

void producer_ops() {
    pin_to_core(2);

    for (int i = 0; i < epochs; i++) {
        ready.store(false, std::memory_order_relaxed);
        while (!buffer.push(__rdtsc()));
        while (!ready.load(std::memory_order_acquire));
    }
}

void consumer_ops() {
    pin_to_core(4);

    ll total = 0;
    ll min = epochs;
    std::vector<ll> counts;

    for (int i = 0; i < epochs; i++) {
        ll before;
        while (!buffer.pop(before));

        ll count = __rdtsc() - before;
        counts.emplace_back(count);
        total += count;
        min = std::min<ll>(min, count);

        ready.store(true, std::memory_order_release);
    }

    std::sort(counts.begin(), counts.end());

    ll avg_cycles = total / epochs;
    ll p99_cycles = counts[std::floor(epochs * 0.99)];

    std::cout << "Average: " << avg_cycles << " cycles (" << avg_cycles / tsc_ghz << " ns)" << std::endl;
    std::cout << "Min:     " << min << " cycles (" << min / tsc_ghz << " ns)" << std::endl;
    std::cout << "P99:     " << p99_cycles << " cycles (" << p99_cycles / tsc_ghz << " ns)" << std::endl;
}

int main() {
    std::thread producer{producer_ops};
    std::thread consumer{consumer_ops};
    producer.join();
    consumer.join();
}
