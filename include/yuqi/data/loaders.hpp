#pragma once

#include <filesystem>
#include <functional>

#include "yuqi/nn.hpp"

namespace yuqi::data {
    struct data_point;
}

namespace yuqi::data::loaders {
    // a function that returns the loaded data and a split ratio to split by
    using loader = std::function<std::pair<std::vector<data::data_point>, float>()>;

    auto mnist(const std::filesystem::path& training_images, const std::filesystem::path& training_labels,
               const std::filesystem::path& test_images, const std::filesystem::path& test_labels) -> loader;
} // namespace yuqi::data::loaders