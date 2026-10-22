// One CPU model per process. Audio travels through pipes, never through a server.
#include "whisper.h"
#include "json.hpp"
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using json = nlohmann::json;

int main(int argc, char ** argv) {
    if (argc != 4) return 2;
    // ANEForge is enabled by environment variables independently of use_gpu.
    unsetenv("ANEFORGE_ENCODER");
    unsetenv("ANEFORGE_DYLIB");
    auto cp = whisper_context_default_params();
    cp.use_gpu = false;
    auto * ctx = whisper_init_from_file_with_params(argv[1], cp);
    if (!ctx) return 3;
    auto p = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
    p.n_threads = std::stoi(argv[3]);
    p.language = argv[2];
    p.detect_language = false;
    p.translate = false;
    p.no_context = true;
    p.print_progress = p.print_realtime = p.print_timestamps = p.print_special = false;
    p.token_timestamps = true;
    p.suppress_nst = true;
    p.beam_search.beam_size = 5;
    p.temperature = 0.0f;
    p.temperature_inc = 0.0f;
    std::cout << "{\"ready\":true,\"device\":\"cpu\"}" << std::endl;
    std::uint32_t count;
    while (std::cin.read(reinterpret_cast<char *>(&count), sizeof(count))) {
        if (!count) break;
        if (count > 25 * 16000) { whisper_free(ctx); return 4; }
        std::vector<float> audio(count);
        if (!std::cin.read(reinterpret_cast<char *>(audio.data()), count * sizeof(float))) break;
        if (whisper_full(ctx, p, audio.data(), count) != 0) {
            std::cout << "{\"error\":\"whisper_full failed\"}" << std::endl;
            continue;
        }
        json words = json::array();
        std::string text, word;
        double start = 0, end = 0;
        auto flush = [&]() {
            if (word.find_first_not_of(" \n\t\r") != std::string::npos)
                words.push_back({{"text", word}, {"start", start}, {"end", end}});
            word.clear();
        };
        for (int s = 0; s < whisper_full_n_segments(ctx); ++s) {
            text += whisper_full_get_segment_text(ctx, s);
            for (int t = 0; t < whisper_full_n_tokens(ctx, s); ++t) {
                auto d = whisper_full_get_token_data(ctx, s, t);
                if (d.id >= whisper_token_eot(ctx)) continue;
                std::string piece = whisper_full_get_token_text(ctx, s, t);
                if (piece.empty()) continue;
                if (piece.front() == ' ' || piece.front() == '\n') flush();
                if (word.empty()) start = std::max<int64_t>(0, d.t0) / 100.0;
                word += piece;
                end = std::max(start, std::max<int64_t>(0, d.t1) / 100.0);
            }
            flush();
        }
        std::cout << json({{"text", text}, {"words", words}}).dump(-1, ' ', false,
                           json::error_handler_t::replace) << std::endl;
    }
    whisper_free(ctx);
    return 0;
}
