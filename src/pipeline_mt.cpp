#include "book/order_book.hpp"
#include "core/messages.hpp"
#include "core/spsc_queue.hpp"
#include "core/timer.hpp"
#include "feed/itch_messages.hpp"
#include "feed/itch_parser.hpp"
#include "feed/itch_reader.hpp"
#include "inference/trt_infer.hpp"
#include "strategy/risk.hpp"
#include "strategy/signal.hpp"

#include <atomic>
#include <cstring>
#include <iostream>
#include <pthread.h>
#include <thread>

// Queues connecting the 3 stages
SPSCQueue<FeatureSnapshot, 1024> feature_queue;   // Book → Inference
SPSCQueue<InferenceResult, 1024> result_queue;    // Inference → Decision

// Signal to stop threads when feed is exhausted
std::atomic<bool> feed_done{false};
std::atomic<bool> inference_done{false};

const char ticker[8] = {'N','V','D','A',' ',' ',' ',' '};

void pin_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

void feed_thread() {
    pin_to_core(2);

    ITCHReader reader{"data/sample_100mb.bin"};
    OrderBook ob;
    uint64_t seq = 0;

    uint8_t *p = reader.next();
    while (p) {
        char type = *p;
        p++;

        bool book_updated = false;

        switch (type) {
            case 'A':
            case 'F': {
                OrderAddMessage msg = parse_add_order(p);
                if (std::memcmp(msg.stock, ticker, 8) != 0) break;
                ob.add_order(msg.order_ref, msg.price, msg.shares, msg.side, msg.stock_locate);
                book_updated = true;
                break;
            }
            case 'E': {
                OrderExecutedMessage msg = parse_order_executed(p);
                ob.execute_order(msg.order_ref, msg.executed_shares);
                book_updated = true;
                break;
            }
            case 'X': {
                OrderCancelMessage msg = parse_order_cancel(p);
                ob.cancel_order(msg.order_ref, msg.canceled_shares);
                book_updated = true;
                break;
            }
            case 'D': {
                OrderDeleteMessage msg = parse_order_delete(p);
                ob.delete_order(msg.order_ref);
                book_updated = true;
                break;
            }
            case 'U': {
                OrderReplaceMessage msg = parse_order_replace(p);
                ob.replace_order(msg.original_order_ref, msg.new_order_ref, msg.price, msg.shares);
                book_updated = true;
                break;
            }
            default:
                break;
        }

        if (book_updated) {
            if (ob.bids().size() >= 10 && ob.asks().size() >= 10) {
                FeatureSnapshot snap;
                ob.extract_features(snap.features, LEVELS);
                snap.sequence = seq++;

                while (!feature_queue.push(snap));
            }
        }

        p = reader.next();
    }

    feed_done.store(true, std::memory_order_release);
    std::cout << "[Feed] Done. Pushed " << seq << " snapshots." << std::endl;
}

void inference_thread() {
    pin_to_core(4);

    TRTInfer engine("models/lstm.engine");
    engine.capture_graph();

    for (int i = 0; i < 50; i++) engine.infer_graph();

    float window_buf[WINDOW][FEATURES_PER_SNAPSHOT] = {};
    int snapshots = 0;
    int inferences = 0;

    FeatureSnapshot snap;

    while (true) {
        if (feature_queue.pop(snap)) {
            std::memcpy(window_buf[snapshots % WINDOW], snap.features, sizeof(snap.features));
            snapshots++;

            if (snapshots >= WINDOW && snapshots % 100 == 0) {
                // Flatten window oldest→newest into pinned input
                float* input = engine.pinned_input();
                for (int i = 0; i < WINDOW; i++) {
                    int buf_idx = (snapshots - WINDOW + i) % WINDOW;
                    std::memcpy(input + i * FEATURES_PER_SNAPSHOT,
                               window_buf[buf_idx],
                               FEATURES_PER_SNAPSHOT * sizeof(float));
                }

                engine.infer_graph();
                inferences++;

                // Push result to decision thread
                InferenceResult result;
                std::memcpy(result.logits, engine.pinned_output(), sizeof(result.logits));
                result.sequence = snap.sequence;

                while (!result_queue.push(result));
            }
        } else if (feed_done.load(std::memory_order_acquire)) {
            // Drain remaining items before exiting
            while (feature_queue.pop(snap)) {
                std::memcpy(window_buf[snapshots % WINDOW], snap.features, sizeof(snap.features));
                snapshots++;
            }
            break;  // Now safe to exit
        }
    }

    inference_done.store(true, std::memory_order_release);
    std::cout << "[Inference] Done. " << snapshots << " snapshots, "
              << inferences << " inferences." << std::endl;
}

void decision_thread() {
    pin_to_core(6);

    RiskManager risk;
    Position pos;
    int buys = 0, sells = 0, holds = 0;

    InferenceResult result;

    while (true) {
        if (result_queue.pop(result)) {
            Signal sig = interpret_signal(result.logits, 0.5f);

            switch (sig) {
                case Signal::Buy:
                    if (risk.can_buy(pos, 100)) buys++;
                    break;
                case Signal::Sell:
                    if (risk.can_sell(pos, 100)) sells++;
                    break;
                case Signal::Hold:
                    holds++;
                    break;
            }
        } else if (inference_done.load(std::memory_order_acquire)) {
            while (result_queue.pop(result)) {
                Signal sig = interpret_signal(result.logits, 0.5f);
                switch (sig) {
                    case Signal::Buy:  if (risk.can_buy(pos, 100)) buys++; break;
                    case Signal::Sell: if (risk.can_sell(pos, 100)) sells++; break;
                    case Signal::Hold: holds++; break;
                }
            }
            break;
        }
    }

    std::cout << "[Decision] Done. Buys: " << buys
              << " Sells: " << sells
              << " Holds: " << holds << std::endl;
}

int main() {
    std::cout << "Starting multi-threaded pipeline..." << std::endl;

    Timer timer;
    timer.begin();

    std::thread t1(feed_thread);
    std::thread t2(inference_thread);
    std::thread t3(decision_thread);

    t1.join();
    t2.join();
    t3.join();

    uint64_t total_cycles = timer.elapsed_cycles();
    std::cout << "\n=== Pipeline Complete ===" << std::endl;
    std::cout << "Total time: " << total_cycles << " cycles (~"
              << total_cycles / 2420000 << " ms)" << std::endl;
}
