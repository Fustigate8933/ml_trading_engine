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

inline AddOrderMessage parse_add_order(const uint8_t *msg) {
    // assume msg is already at the stock locate (1 byte past message type)

    MessageHeader header;

    header.type = 'A';

    header.stock_locate = parse_u16(msg);
    msg += 2;

    header.tracking_number = parse_u16(msg);
    msg += 2;

    header.timestamp = parse_timestamp(msg);
    msg += 6;

    AddOrderMessage message{header.stock_locate, header.tracking_number, header.timestamp};

    message.order_ref = parse_u64(msg);
    msg += 8;

    std::memcpy(&message.side, msg, 1);
    msg++;

    message.shares = parse_u32(msg);
    msg += 4;

    std::memcpy(&message.stock, msg, 8);
    msg += 8;

    message.price = parse_u32(msg);
    msg += 4;

    return message;
}
