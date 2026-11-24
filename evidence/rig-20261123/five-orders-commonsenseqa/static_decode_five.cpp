// Native static-layout five-digit prefill fixture for the pinned llama C ABI.
#include "llama.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef LLAMA_FIXTURE_SLOTS
#define LLAMA_FIXTURE_SLOTS 8
#endif
#ifndef LLAMA_FIXTURE_UBATCH
#define LLAMA_FIXTURE_UBATCH 512
#endif
#ifndef LLAMA_FIXTURE_ITEMS
#define LLAMA_FIXTURE_ITEMS 8
#endif

namespace {
constexpr int slots = LLAMA_FIXTURE_SLOTS;
constexpr int context_per_sequence = 512;
constexpr int case_orders = 120;
constexpr int answer_count = 5;
constexpr int fixture_sequences = LLAMA_FIXTURE_ITEMS * case_orders;
using Sequence = std::vector<llama_token>;
using Digits = std::array<llama_token, answer_count>;

void run_block(llama_context *ctx, const std::vector<Sequence> &seqs, int start, const Digits &digits) {
    std::vector<llama_token> tokens;
    std::vector<llama_pos> positions;
    std::vector<int32_t> n_ids;
    std::vector<llama_seq_id *> ids;
    std::vector<int8_t> output;
    std::array<llama_seq_id, slots> sequence_ids{};
    std::array<int, slots> ends{};

    for (int slot = 0; slot < slots; ++slot) {
        sequence_ids[slot] = slot;
        const auto &sequence = seqs.at(start + slot);
        for (size_t i = 0; i < sequence.size(); ++i) {
            tokens.push_back(sequence[i]);
            positions.push_back(static_cast<llama_pos>(i));
            n_ids.push_back(1);
            ids.push_back(&sequence_ids[slot]);
            output.push_back(i + 1 == sequence.size());
        }
        ends[slot] = static_cast<int>(tokens.size()) - 1;
    }

    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch batch{static_cast<int32_t>(tokens.size()), tokens.data(), nullptr,
                      positions.data(), n_ids.data(), ids.data(), output.data()};
    if (llama_decode(ctx, batch) != 0) {
        throw std::runtime_error("Native decode returned nonzero");
    }
    llama_synchronize(ctx);

    std::cout << "{\"token_count\":" << tokens.size() << ",\"probabilities\":[";
    for (int slot = 0; slot < slots; ++slot) {
        float *logits = llama_get_logits_ith(ctx, ends[slot]);
        if (!logits) {
            throw std::runtime_error("Missing final-token logits");
        }
        std::array<double, answer_count> scores{};
        for (size_t i = 0; i < scores.size(); ++i) {
            scores[i] = logits[digits[i]];
            if (!std::isfinite(scores[i])) {
                throw std::runtime_error("Nonfinite digit logit");
            }
        }
        double max = *std::max_element(scores.begin(), scores.end());
        double mass = 0;
        for (auto &value : scores) {
            value = std::exp(value - max);
            mass += value;
        }
        std::cout << (slot ? ",[" : "[");
        for (size_t i = 0; i < scores.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << scores[i] / mass;
        }
        std::cout << ']';
    }
    std::cout << "]}" << std::endl;
}
}

int main(int argc, char **argv) {
    try {
        if (argc != 4) throw std::runtime_error("Usage: static_decode LIB_DIR MODEL FIXTURE");
        std::ifstream input(argv[3]);
        Digits digits{};
        for (auto &digit : digits) {
            if (!(input >> digit) || digit < 0) throw std::runtime_error("Invalid digit input");
        }
        std::vector<Sequence> sequences(fixture_sequences);
        for (auto &sequence : sequences) {
            int size;
            if (!(input >> size) || size < 1 || size >= context_per_sequence) {
                throw std::runtime_error("Invalid sequence length");
            }
            sequence.resize(size);
            for (auto &token : sequence) {
                if (!(input >> token) || token < 0) throw std::runtime_error("Invalid token");
            }
        }
        std::string extra;
        if (input >> extra) throw std::runtime_error("Unexpected extra token fixture data");

        ggml_backend_load_all_from_path(argv[1]);
        llama_backend_init();
        ggml_backend_dev_t selected = ggml_backend_dev_by_name("CUDA0");
        if (!selected) throw std::runtime_error("Pinned visible CUDA0 is unavailable");
        ggml_backend_dev_t devices[]{selected, nullptr};
        auto model_params = llama_model_default_params();
        model_params.devices = devices;
        model_params.n_gpu_layers = 99;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[2], model_params), llama_model_free);
        if (!model) throw std::runtime_error("Could not load pinned model");
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        if (*std::max_element(digits.begin(), digits.end()) >= vocab) {
            throw std::runtime_error("Digit ID outside vocabulary");
        }

        auto params = llama_context_default_params();
        params.n_ctx = slots * context_per_sequence;
        params.n_batch = params.n_ctx;
        params.n_ubatch = LLAMA_FIXTURE_UBATCH;
        params.n_seq_max = slots;
        params.n_threads = 2;
        params.n_threads_batch = 2;
        params.kv_unified = false;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
            llama_init_from_model(model.get(), params), llama_free);
        if (!ctx || !llama_get_memory(ctx.get())) {
            throw std::runtime_error("Native context has no clearable memory");
        }

        std::cout << std::setprecision(16) << "READY" << std::endl;
        int start;
        while (std::cin >> start) {
            if (start < 0 || start % slots != 0 || start > fixture_sequences - slots) {
                throw std::runtime_error("Block outside frozen fixture");
            }
            run_block(ctx.get(), sequences, start, digits);
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
