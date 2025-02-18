#pragma once

#include <vector>

namespace yuqi::nn {
    class layer {
        std::vector<float> m_biases;
        std::vector<float> m_weights;

      public:
        layer(size_t dim = 0) : m_biases(dim), m_weights(dim) { load_random(); };

        auto load_random() -> void;
    };

    using input_layer = std::vector<float>;
    using output_layer = std::vector<float>;
} // namespace yuqi::nn