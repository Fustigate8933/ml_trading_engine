#include "core/spsc_queue.hpp"
#include <thread>
#include <assert.h>
#include <iostream>

SPSCQueue<int, 1 << 10> buffer;

void producer_ops() {
    for (int i = 0; i <= 999999; i++) {
        while (!buffer.push(i));
    }
}

void consumer_ops() {
    for (int i = 0; i <= 999999; i++) {
        int val;
        while (!buffer.pop(val));
        assert(val == i);
    }
    std::cout << "PASS" << std::endl;
}

int main() {
    std::thread producer{producer_ops};
    std::thread consumer{consumer_ops};

    producer.join();
    consumer.join();
}
