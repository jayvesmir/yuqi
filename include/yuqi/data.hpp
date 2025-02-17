#pragma once

#include <filesystem>
#include <ranges>
#include <vector>

#include "yuqi/nn.hpp"

namespace yuqi::data {
    constexpr auto DEFAULT_SPLIT_RATIO = 0.9; // 90% training data | 10% test data

    class dataset {
        std::vector<nn::input_layer> m_data;

        std::span<nn::input_layer> m_test;
        std::span<nn::input_layer> m_training;

      public:
        auto split(float ratio = DEFAULT_SPLIT_RATIO) -> void;
    };
} // namespace yuqi::data