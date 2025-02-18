#pragma once

#include <filesystem>
#include <ranges>
#include <vector>

#include "yuqi/data/loaders.hpp"
#include "yuqi/nn.hpp"

namespace yuqi::data {
    constexpr auto DEFAULT_SPLIT_RATIO = 0.9; // 90% training data | 10% test data

    struct data_point {
        nn::input_layer input;
        nn::output_layer output;
    };

    class dataset {
        std::vector<data::data_point> m_data;

        std::span<data::data_point> m_test;
        std::span<data::data_point> m_training;

        float m_last_load_time = 0.0F;

      public:
        // factory function
        static auto load(const loaders::loader& loader) -> dataset;

        // splits the data into m_test and m_training
        auto split(float ratio = DEFAULT_SPLIT_RATIO) -> void;

        constexpr auto data() const { return m_data; }
        constexpr auto test_data() const { return m_test; }
        constexpr auto training_data() const { return m_training; }
        constexpr auto last_load_time() const { return m_last_load_time; }
    };
} // namespace yuqi::data