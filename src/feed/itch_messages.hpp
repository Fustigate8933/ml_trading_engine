#pragma once

#include <cstdint>

struct AddOrderMessage {
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t timestamp;
    uint64_t order_ref;
    char side; // "B" for buyer, "S" for seller
    uint32_t shares;
    char stock[8];
    uint32_t price;
};

struct MessageHeader {
    char type;
    uint16_t stock_locate;
    uint16_t tracking_number;
    uint64_t timestamp;
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


