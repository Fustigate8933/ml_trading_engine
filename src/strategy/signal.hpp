#pragma once

#include <cmath>
enum class Signal {
    Buy,
    Sell,
    Hold
};

void softmax(const float *x, float *res) {
    float exp_sum = 0;
    for (int i = 0; i < 3; i++) {
        exp_sum += std::exp(x[i]);
    }

    for (int i = 0; i < 3; i++) {
        res[i] = exp(x[i]) / exp_sum;
    }
}

Signal interpret_signal(const float* logits, float threshold = 0.6) {
    float res[3];
    softmax(logits, res);

    Signal best;
    float best_prob = -1;
    for (int i = 0; i < 3; i++) {
        Signal sig;
        if (i == 0) {
            sig = Signal::Sell;
        } else if (i == 1) {
            sig = Signal::Hold;
        } else {
            sig = Signal::Buy;
        }

        if (best_prob == -1 or res[i] > best_prob) {
            best = sig;
            best_prob = res[i];
        }
    }

    if (best_prob <= threshold) {
        return Signal::Hold;
    }

    return best;
}
