#include "yuqi/yuqi.hpp"

#include <cstdint>
#include <print>

int32_t main(int32_t argc, char** argv) {
    if (argc < 5) {
        std::println("usage:\n{} <mnist_training_images> <mnist_training_labels> <mnist_test_images> <mnist_test_labels>",
                     argv[0]);
        return 0;
    }

    std::string training_images = argv[1];
    std::string training_labels = argv[2];
    std::string test_images = argv[3];
    std::string test_labels = argv[4];

    auto dataset =
        yuqi::data::dataset::load(yuqi::data::loaders::mnist(training_images, training_labels, test_images, test_labels));

    std::println("loaded MNIST ({}, {}) in {} ms", dataset.training_data().size(), dataset.test_data().size(),
                 dataset.last_load_time() * 1e-6);

    return 0;
}