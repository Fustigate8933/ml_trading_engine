#pragma once

#include "itch_messages.hpp"
#include <cstdint>
#include <cstring>

inline uint64_t parse_timestamp(const uint8_t *p) {
    uint64_t val = 0;

    for (int i = 0; i < 6; i++) {
        val = (val << 8) | p[i];
    }

    return val;
}

inline uint16_t parse_u16(const uint8_t *p) {
    uint16_t val;
    std::memcpy(&val, p, sizeof(uint16_t));
    val = __builtin_bswap16(val);

    return val;
}

inline uint32_t parse_u32(const uint8_t *p) {
    uint32_t val;
    std::memcpy(&val, p, sizeof(uint32_t));
    val = __builtin_bswap32(val);

    return val;
}

inline uint64_t parse_u64(const uint8_t *p) {
    uint64_t val;
    std::memcpy(&val, p, sizeof(uint64_t));
    val = __builtin_bswap64(val);

    return val;
}

inline OrderAddMessage parse_add_order(const uint8_t *msg) {
    OrderAddMessage m{};
    m.stock_locate = parse_u16(msg);
    m.tracking_number = parse_u16(msg + 2);
    m.timestamp = parse_timestamp(msg + 4);
    m.order_ref = parse_u64(msg + 10);
    m.side = msg[18];
    m.shares = parse_u32(msg + 19);
    std::memcpy(m.stock, msg + 23, 8);
    m.price = parse_u32(msg + 31);
    return m;
}

inline OrderAddMessage parse_add_order_mpid(const uint8_t *msg) {
    return parse_add_order(msg);
}

inline OrderDeleteMessage parse_order_delete(const uint8_t *msg) {
    OrderDeleteMessage m{};
    m.stock_locate = parse_u16(msg);
    m.tracking_number = parse_u16(msg + 2);
    m.timestamp = parse_timestamp(msg + 4);
    m.order_ref = parse_u64(msg + 10);
    return m;
}

inline OrderCancelMessage parse_order_cancel(const uint8_t *msg) {
    OrderCancelMessage m{};
    m.stock_locate = parse_u16(msg);
    m.tracking_number = parse_u16(msg + 2);
    m.timestamp = parse_timestamp(msg + 4);
    m.order_ref = parse_u64(msg + 10);
    m.canceled_shares = parse_u32(msg + 18);
    return m;
}

inline OrderExecutedMessage parse_order_executed(const uint8_t *msg) {
    OrderExecutedMessage m{};
    m.stock_locate = parse_u16(msg);
    m.tracking_number = parse_u16(msg + 2);
    m.timestamp = parse_timestamp(msg + 4);
    m.order_ref = parse_u64(msg + 10);
    m.executed_shares = parse_u32(msg + 18);
    m.match_number = parse_u64(msg + 22);
    return m;
}

inline OrderReplaceMessage parse_order_replace(const uint8_t *msg) {
    OrderReplaceMessage m{};
    m.stock_locate = parse_u16(msg);
    m.tracking_number = parse_u16(msg + 2);
    m.timestamp = parse_timestamp(msg + 4);
    m.original_order_ref = parse_u64(msg + 10);
    m.new_order_ref = parse_u64(msg + 18);
    m.shares = parse_u32(msg + 26);
    m.price = parse_u32(msg + 30);
    return m;
}
