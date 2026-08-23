#include "book/order_book.hpp"
#include "feed/itch_messages.hpp"
#include "feed/itch_parser.hpp"
#include "feed/itch_reader.hpp"
#include <cstdint>
#include <iostream>

const char *data_path = "data/sample_100mb.bin";

int main() {
    ITCHReader reader{data_path};
    OrderBook order_book{};

    long processed = 0;

    uint8_t *p = reader.next();
    while (p) {
        char type = *p;
        p++;

        switch (type) {
            case 'A':
            case 'F': {
                OrderAddMessage message = parse_add_order(p);
                order_book.add_order(message.order_ref, message.price, message.shares, message.side, message.stock_locate);
                break;
            }
            case 'D': {
                OrderDeleteMessage message = parse_order_delete(p);
                order_book.delete_order(message.order_ref);
                break;
            }
            case 'X': {
                OrderCancelMessage message = parse_order_cancel(p);
                order_book.cancel_order(message.order_ref, message.canceled_shares);
                break;
            }
            case 'E': {
                OrderExecutedMessage message = parse_order_executed(p);
                order_book.execute_order(message.order_ref, message.executed_shares);
                break;
            }
            case 'U': {
                OrderReplaceMessage message = parse_order_replace(p);
                order_book.replace_order(message.original_order_ref, message.new_order_ref, message.price, message.shares);
                break;
            }
            default:
                break;
        }

        processed++;
        p = reader.next();
    }

    std::cout << "Messages processed: " << processed << std::endl;
    std::cout << "Live orders remaining: " << order_book.order_count() << std::endl;

    std::cout << "\n--- Top 5 Bid Levels (best first) ---" << std::endl;
    int count = 0;
    for (auto it = order_book.bids().rbegin(); it != order_book.bids().rend() && count < 5; ++it, ++count) {
        std::cout << "  $" << it->first / 10000.0 << " : " << it->second << " shares" << std::endl;
    }

    std::cout << "\n--- Top 5 Ask Levels (best first) ---" << std::endl;
    count = 0;
    for (auto it = order_book.asks().begin(); it != order_book.asks().end() && count < 5; ++it, ++count) {
        std::cout << "  $" << it->first / 10000.0 << " : " << it->second << " shares" << std::endl;
    }
}
