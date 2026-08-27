#pragma once

#include "strategy/signal.hpp"
#include <cstdint>

struct Position {
    int32_t shares = 0;
    double avg_entry_price = 0.0;
    double realized_pnl = 0.0;
};

class RiskManager {
private:
    int32_t max_position_ = 1000;
    double max_loss_ = -10000.0;
    int32_t max_order_size_ = 100;

public:
    bool can_buy(const Position &pos, int32_t order_size) const {
        return pos.shares + order_size <= max_position_ && pos.realized_pnl > max_loss_;
    }

    bool can_sell(const Position &pos, int32_t order_size) const {
        return pos.shares - order_size >= -max_position_ && pos.realized_pnl > max_loss_;
    }

    int32_t order_size(const Position &pos, Signal signal) const {
        return max_order_size_;
    }
};
