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

SPSCQueue<ll, 1 << 10> buffer;
const int epochs = 1000000;

void pin_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

void producer_ops() {
    pin_to_core(2);

    for (int i = 0; i < epochs; i++) {
        while (!buffer.push(__rdtsc()));
    }
}

void consumer_ops() {
    pin_to_core(4);

    ll total = 0;
    ll min = epochs;
    std::vector<int> counts;

    for (int i = 0; i < epochs; i++) {
        ll before;
        while (!buffer.pop(before));

        ll count = __rdtsc() - before;
        counts.emplace_back(count);
        total += count;
        min = std::min<ll>(min, count);
    }

    std::sort(counts.begin(), counts.end());

    std::cout << "Average cycle counts: " << total / epochs << std::endl;
    std::cout << "Min cycle counts: " << min << std::endl;
    std::cout << "99th percentile counts: " << counts[std::floor(epochs * 0.99)] << std::endl;
}

int main() {
    std::thread producer{producer_ops};
    std::thread consumer{consumer_ops};
    producer.join();
    consumer.join();
}
