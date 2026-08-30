#pragma once

#include <cstdint>

constexpr int LEVELS = 10;
constexpr int FEATURES_PER_SNAPSHOT = LEVELS * 4;  // 40
constexpr int WINDOW = 50;

// Flows through Queue A: Feed/Book thread → Inference thread
// One order book snapshot (160 bytes — fits in 3 cache lines)
struct FeatureSnapshot {
    float features[FEATURES_PER_SNAPSHOT];  // 40 floats = 160 bytes
    uint64_t sequence;                       // monotonic counter for ordering
};

// Flows through Queue B: Inference thread → Decision thread
struct InferenceResult {
    float logits[3];        // down/stable/up scores
    uint64_t sequence;      // matches the snapshot that produced this
};
