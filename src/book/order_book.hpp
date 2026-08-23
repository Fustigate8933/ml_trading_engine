#pragma once

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <map>

struct Order {
    uint64_t order_ref;
    uint32_t price;
    uint32_t shares;
    char side;
    uint16_t stock_locate;
};


class OrderBook {
private:
    std::unordered_map<uint64_t, Order> orders_; // order ref # : order
    std::map<uint32_t, uint64_t> bid_levels_; // price : shares
    std::map<uint32_t, uint64_t> ask_levels_; // price : shares

    void remove_shares_from_level(const Order &order, uint32_t shares) {
        auto &levels = (order.side == 'B') ? bid_levels_ : ask_levels_;
        auto it = levels.find(order.price);
        if (it == levels.end()) return;

        if (it->second <= shares) {
            levels.erase(it);
        } else {
            it->second -= shares;
        }
    }

public:
    void add_order(uint64_t ref, uint32_t price, uint32_t shares, char side, uint16_t locate) {
        Order order{ref, price, shares, side, locate};
        orders_[ref] = order;

        auto &levels = (side == 'B') ? bid_levels_ : ask_levels_;
        levels[price] += shares;
    }

    void delete_order(uint64_t ref) {
        auto it = orders_.find(ref);
        if (it == orders_.end()) return; // order predates our session

        remove_shares_from_level(it->second, it->second.shares);
        orders_.erase(it);
    }

    void cancel_order(uint64_t ref, uint32_t canceled_shares) {
        auto it = orders_.find(ref);
        if (it == orders_.end()) return;

        Order &order = it->second;
        uint32_t actual = std::min(canceled_shares, order.shares);

        remove_shares_from_level(order, actual);
        order.shares -= actual;

        if (order.shares == 0) {
            orders_.erase(it);
        }
    }

    void execute_order(uint64_t ref, uint32_t executed_shares) {
        cancel_order(ref, executed_shares);
    }

    void replace_order(uint64_t old_ref, uint64_t new_ref, uint32_t new_price, uint32_t new_shares) {
        auto it = orders_.find(old_ref);
        if (it == orders_.end()) return;

        char side = it->second.side;
        uint16_t locate = it->second.stock_locate;

        remove_shares_from_level(it->second, it->second.shares);
        orders_.erase(it);

        add_order(new_ref, new_price, new_shares, side, locate);
    }

    const std::map<uint32_t, uint64_t>& bids() const {
        return bid_levels_;
    }

    const std::map<uint32_t, uint64_t>& asks() const {
        return ask_levels_;
    }

    size_t order_count() const { return orders_.size(); }
};
