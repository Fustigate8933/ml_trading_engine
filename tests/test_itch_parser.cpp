#include "feed/itch_messages.hpp"
#include "feed/itch_parser.hpp"
#include "feed/itch_reader.hpp"
#include <cstdint>
#include <unordered_map>
#include <iostream>

const char *data_path = "data/sample_100mb.bin";

int main() {
    ITCHReader reader{data_path};

    std::unordered_map<char, int> type_count;

    uint8_t *p{reader.next()};
    while (p) {
        char type;
        memcpy(&type, p, 1);
        p++;

        type_count[type]++;

        if (type == 'A' && type_count['A'] <= 5) {
            OrderAddMessage message = parse_add_order(p);

            // stock is not null-terminated
            std::cout << "Stock: ";
            std::cout.write(message.stock, 8);
            std::cout << std::endl;

            std::cout << "Side: " << ('B' == message.side ? "Buyer" : "Seller") << std::endl;
            std::cout << "Shares: " << message.shares << std::endl;
            std::cout << "Price: " << message.price / 10000.0 << std::endl;

            std::cout << std::endl;
        }

        p = reader.next();
    }

    for (const auto &[type, count] : type_count) {
        std::cout << type << ": " << count << std::endl;
    }
}
