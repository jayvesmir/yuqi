#include "yuqi/data/data.hpp"

#include <chrono>

namespace yuqi::data {
    auto dataset::load(const loaders::loader& loader) -> dataset {
        auto start = std::chrono::high_resolution_clock::now();

        dataset out;

        auto [data, split] = loader();

        out.m_data = data;
        out.split(std::min(std::max(split, 0.0F), 1.0F));

        auto end = std::chrono::high_resolution_clock::now();

        out.m_last_load_time = (end - start).count();

        return out;
    }
} // namespace yuqi::data