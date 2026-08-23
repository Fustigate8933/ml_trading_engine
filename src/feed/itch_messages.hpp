#pragma once

#include <cstdint>

struct MessageHeader {
    char type;
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t timestamp;
};

struct OrderAddMessage {
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t timestamp;
    uint64_t order_ref;
    char side; // 'B' or 'S'
    uint32_t shares;
    char stock[8];
    uint32_t price;
};

struct OrderDeleteMessage {
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t timestamp;
    uint64_t order_ref;
};

struct OrderCancelMessage {
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t timestamp;
    uint64_t order_ref;
    uint32_t canceled_shares;
};

struct OrderExecutedMessage {
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t timestamp;
    uint64_t order_ref;
    uint32_t executed_shares;
    uint64_t match_number;
};

struct OrderReplaceMessage {
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t timestamp;
    uint64_t original_order_ref;
    uint64_t new_order_ref;
    uint32_t shares;
    uint32_t price;
};

enum class MessageType : char {
    AddOrder = 'A',
    AddOrderMPID = 'F',
    OrderExecuted = 'E',
    OrderExecutedWithPrice = 'C',
    OrderCancel = 'X',
    OrderDelete = 'D',
    OrderReplace = 'U'
};


