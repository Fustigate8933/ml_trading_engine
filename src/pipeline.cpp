#include "book/order_book.hpp"
#include "core/timer.hpp"
#include "feed/itch_messages.hpp"
#include "feed/itch_parser.hpp"
#include "feed/itch_reader.hpp"
#include "inference/trt_infer.hpp"
#include "strategy/risk.hpp"
#include "strategy/signal.hpp"
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>

const char ticker[8] = {'N','V','D','A',' ',' ',' ',' '};

constexpr int LEVELS = 10;
constexpr int FEATURES_PER_SNAPSHOT = LEVELS * 4;
constexpr int WINDOW = 50;
constexpr int INPUT_SIZE = WINDOW * FEATURES_PER_SNAPSHOT;

// Collect latency samples for each stage
struct StageTimings {
    std::vector<uint64_t> parse;
    std::vector<uint64_t> book_update;
    std::vector<uint64_t> extract;
    std::vector<uint64_t> flatten;
    std::vector<uint64_t> infer;
    std::vector<uint64_t> signal;
    std::vector<uint64_t> total;  // end-to-end per inference cycle
};

void print_stats(const char* label, std::vector<uint64_t>& v) {
    if (v.empty()) return;
    std::sort(v.begin(), v.end());
    uint64_t sum = 0;
    for (auto x : v) sum += x;
    int n = v.size();
    printf("  %-16s  avg: %7lu  min: %7lu  p50: %7lu  p99: %7lu  (n=%d)\n",
           label, sum/n, v[0], v[n/2], v[(int)(n*0.99)], n);
}

int main() {
    ITCHReader reader{"data/sample_100mb.bin"};
    OrderBook ob;
    TRTInfer engine("models/baseline.engine");
    engine.capture_graph();
    RiskManager risk;
    Position pos;
    Timer timer;

    float window_buf[WINDOW][FEATURES_PER_SNAPSHOT] = {};
    int snapshots = 0;
    int signals_generated = 0;

    StageTimings timings;

    uint8_t *p = reader.next();
    while (p) {
        char type = *p;
        p++;

        bool book_updated = false;
        bool is_target_stock = false;

        // --- PARSE ---
        timer.begin();
        switch (type) {
            case 'A':
            case 'F': {
                OrderAddMessage message = parse_add_order(p);
                if (std::memcmp(message.stock, ticker, 8) != 0) break;
                is_target_stock = true;
                uint64_t parse_t = timer.elapsed_cycles();
                timings.parse.push_back(parse_t);

                // --- BOOK UPDATE ---
                timer.begin();
                ob.add_order(message.order_ref, message.price, message.shares, message.side, message.stock_locate);
                timings.book_update.push_back(timer.elapsed_cycles());
                book_updated = true;
                break;
            }
            case 'E': {
                OrderExecutedMessage message = parse_order_executed(p);
                timer.begin();
                ob.execute_order(message.order_ref, message.executed_shares);
                book_updated = true;
                break;
            }
            case 'X': {
                OrderCancelMessage message = parse_order_cancel(p);
                timer.begin();
                ob.cancel_order(message.order_ref, message.canceled_shares);
                book_updated = true;
                break;
            }
            case 'D': {
                OrderDeleteMessage message = parse_order_delete(p);
                timer.begin();
                ob.delete_order(message.order_ref);
                book_updated = true;
                break;
            }
            case 'U': {
                OrderReplaceMessage message = parse_order_replace(p);
                timer.begin();
                ob.replace_order(message.original_order_ref, message.new_order_ref, message.price, message.shares);
                book_updated = true;
                break;
            }
            default:
                break;
        }

        if (book_updated) {
            // --- EXTRACT FEATURES ---
            timer.begin();
            ob.extract_features(window_buf[snapshots % WINDOW], LEVELS);
            timings.extract.push_back(timer.elapsed_cycles());
            snapshots++;

            if (snapshots >= WINDOW && snapshots % 100 == 0) {
                // --- FLATTEN WINDOW ---
                timer.begin();
                float* input = engine.pinned_input();
                for (int i = 0; i < WINDOW; i++) {
                    int buf_idx = (snapshots - WINDOW + i) % WINDOW;
                    std::memcpy(input + i * FEATURES_PER_SNAPSHOT,
                               window_buf[buf_idx],
                               FEATURES_PER_SNAPSHOT * sizeof(float));
                }
                timings.flatten.push_back(timer.elapsed_cycles());

                // --- GPU INFERENCE ---
                timer.begin();
                engine.infer_graph();
                timings.infer.push_back(timer.elapsed_cycles());

                // --- SIGNAL ---
                timer.begin();
                float* logits = engine.pinned_output();
                Signal sig = interpret_signal(logits, 0.5f);
                timings.signal.push_back(timer.elapsed_cycles());

                if (sig == Signal::Buy && risk.can_buy(pos, 100)) {
                    signals_generated++;
                } else if (sig == Signal::Sell && risk.can_sell(pos, 100)) {
                    signals_generated++;
                }
            }
        }

        p = reader.next();
    }

    // --- Print Results ---
    std::cout << "\n=== Stage Latency Profile (cycles) ===" << std::endl;
    print_stats("Parse (Add)", timings.parse);
    print_stats("Book Update", timings.book_update);
    print_stats("Extract Feat", timings.extract);
    print_stats("Flatten Win", timings.flatten);
    print_stats("GPU Infer", timings.infer);
    print_stats("Signal", timings.signal);

    std::cout << "\n--- Pipeline Summary ---" << std::endl;
    std::cout << "Book snapshots: " << snapshots << std::endl;
    std::cout << "Inferences: " << timings.infer.size() << std::endl;
    std::cout << "Signals: " << signals_generated << std::endl;
}
