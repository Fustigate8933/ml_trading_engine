#include "inference/trt_infer.hpp"
#include <x86intrin.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>

void print_stats(const char* label, std::vector<int64_t>& latencies) {
    std::sort(latencies.begin(), latencies.end());
    int N = latencies.size();
    int64_t sum = 0;
    for (auto l : latencies) sum += l;

    std::cout << label << ":" << std::endl;
    std::cout << "  Min:  " << latencies[0] << " cycles" << std::endl;
    std::cout << "  Avg:  " << sum / N << " cycles" << std::endl;
    std::cout << "  P50:  " << latencies[N / 2] << " cycles" << std::endl;
    std::cout << "  P99:  " << latencies[(int)(N * 0.99)] << " cycles" << std::endl;
    std::cout << "  Max:  " << latencies[N - 1] << " cycles" << std::endl;
    std::cout << std::endl;
}

int main() {
    TRTInfer engine("models/lstm.engine");

    float input[50 * 40] = {};
    float output[3] = {};
    const int N = 10000;

    // === Baseline: regular infer() ===
    for (int i = 0; i < 100; i++) engine.infer(input, output);

    std::vector<int64_t> latencies(N);
    for (int i = 0; i < N; i++) {
        auto start = __rdtsc();
        engine.infer(input, output);
        latencies[i] = __rdtsc() - start;
    }
    print_stats("Regular infer (cudaMemcpy + enqueue + sync)", latencies);

    // === CUDA Graph path ===
    engine.capture_graph();

    // Warmup graph
    for (int i = 0; i < 100; i++) engine.infer_graph();

    for (int i = 0; i < N; i++) {
        // Write input to pinned buffer (simulates real use)
        std::memcpy(engine.pinned_input(), input, sizeof(input));

        auto start = __rdtsc();
        engine.infer_graph();
        latencies[i] = __rdtsc() - start;
    }
    print_stats("CUDA Graph infer (captured pipeline)", latencies);

    // Print output
    float* out = engine.pinned_output();
    std::cout << "Output logits: [" << out[0] << ", " << out[1] << ", " << out[2] << "]" << std::endl;

    return 0;
}
