#include "book/order_book.hpp"
#include "feed/itch_messages.hpp"
#include "feed/itch_parser.hpp"
#include "feed/itch_reader.hpp"
#include "inference/trt_infer.hpp"
#include "strategy/risk.hpp"
#include "strategy/signal.hpp"
#include <cstring>
#include <iostream>

const char *data_path = "data/sample_100mb.bin";
const char ticker[8] = {'N','V','D','A',' ',' ',' ',' '};  // ITCH format: 8 chars, space-padded

constexpr int LEVELS = 10;
constexpr int FEATURES_PER_SNAPSHOT = LEVELS * 4;  // 40 (price+vol per level per side)
constexpr int WINDOW = 50;

int main() {
    ITCHReader reader{data_path};
    OrderBook ob;
    TRTInfer engine("models/baseline.engine");
    engine.capture_graph();
    RiskManager risk;
    Position pos;

    // Sliding window for last 50 snapshots
    float window_buf[WINDOW][FEATURES_PER_SNAPSHOT] = {};
    int snapshots = 0;
    int signals_generated = 0;

    uint8_t *p = reader.next();
    while (p) {
        char type = *p;
        p++;

        bool book_updated = false;

        switch (type) {
            case 'A':
            case 'F': {
                OrderAddMessage message = parse_add_order(p);
                if (std::memcmp(message.stock, ticker, 8) != 0) break;
                ob.add_order(message.order_ref, message.price, message.shares, message.side, message.stock_locate);
                book_updated = true;
                break;
            }
            case 'E': {
                OrderExecutedMessage message = parse_order_executed(p);
                ob.execute_order(message.order_ref, message.executed_shares);
                book_updated = true;
                break;
            }
            case 'X': {
                OrderCancelMessage message = parse_order_cancel(p);
                ob.cancel_order(message.order_ref, message.canceled_shares);
                book_updated = true;
                break;
            }
            case 'D': {
                OrderDeleteMessage message = parse_order_delete(p);
                ob.delete_order(message.order_ref);
                book_updated = true;
                break;
            }
            case 'U': {
                OrderReplaceMessage message = parse_order_replace(p);
                ob.replace_order(message.original_order_ref, message.new_order_ref, message.price, message.shares);
                book_updated = true;
                break;
            }
            default:
                break;
        }

        if (book_updated) {
            ob.extract_features(window_buf[snapshots % WINDOW], LEVELS);
            snapshots++;

            if (snapshots >= WINDOW && snapshots % 100 == 0) {
                float* input = engine.pinned_input();
                for (int i = 0; i < WINDOW; i++) {
                    int buf_idx = (snapshots - WINDOW + i) % WINDOW;
                    std::memcpy(input + i * FEATURES_PER_SNAPSHOT,
                               window_buf[buf_idx],
                               FEATURES_PER_SNAPSHOT * sizeof(float));
                }

                engine.infer_graph();
                float* logits = engine.pinned_output();
                Signal sig = interpret_signal(logits, 0.5f);

                if (sig == Signal::Buy && risk.can_buy(pos, 100)) {
                    signals_generated++;
                    if (signals_generated <= 10) {
                        std::cout << "[BUY]  snapshot=" << snapshots
                                  << " logits=[" << logits[0] << "," << logits[1] << "," << logits[2] << "]"
                                  << std::endl;
                    }
                } else if (sig == Signal::Sell && risk.can_sell(pos, 100)) {
                    signals_generated++;
                    if (signals_generated <= 10) {
                        std::cout << "[SELL] snapshot=" << snapshots
                                  << " logits=[" << logits[0] << "," << logits[1] << "," << logits[2] << "]"
                                  << std::endl;
                    }
                }
            }
        }

        p = reader.next();
    }

    std::cout << "\n--- Pipeline Summary ---" << std::endl;
    std::cout << "Total book snapshots: " << snapshots << std::endl;
    std::cout << "Inferences run: " << (snapshots >= WINDOW ? (snapshots - WINDOW) / 100 + 1 : 0) << std::endl;
    std::cout << "Signals generated: " << signals_generated << std::endl;
    std::cout << "Live orders: " << ob.order_count() << std::endl;
}
