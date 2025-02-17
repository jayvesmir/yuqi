#include "yuqi/nn.hpp"

#include <algorithm>
#include <random>

namespace yuqi::nn {
    void layer::load_random() {
        static auto device = std::random_device{};
        static auto rng = std::mt19937_64{device()};
        static auto dist = std::uniform_real_distribution<float>{0, 1};

        std::ranges::generate(m_biases, [&]() { return dist(rng); });
        std::ranges::generate(m_weights, [&]() { return dist(rng); });
    }
} // namespace yuqi::nn