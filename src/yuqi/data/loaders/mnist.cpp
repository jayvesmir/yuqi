#include "yuqi/data/data.hpp"
#include "yuqi/nn.hpp"

#include <bit>
#include <fstream>
#include <vector>

namespace yuqi::data::loaders {
    auto read_swapped(std::fstream& stream, auto& var) {
        stream.read(reinterpret_cast<char*>(&var), sizeof(var));
        var = std::byteswap(var);
    }

    auto load_mnist(const std::filesystem::path& training_images, const std::filesystem::path& training_labels,
                    const std::filesystem::path& test_images, const std::filesystem::path& test_labels)
        -> std::pair<std::vector<data::data_point>, float> {
        std::pair<std::vector<data::data_point>, float> error_out = {{}, 1.0F};

        auto train_img = std::fstream(training_images);
        auto train_label = std::fstream(training_labels);
        auto test_img = std::fstream(test_images);
        auto test_label = std::fstream(test_labels);

        if (!train_img || !train_label || !test_img || !test_label) {
            return error_out;
        }

        std::vector<data::data_point> data_points;

        auto read_images = [&](std::fstream& stream, uint32_t offset) -> size_t {
            int32_t magic = 0;
            read_swapped(stream, magic);

            if (magic != 0x0803) {
                return 0;
            }

            int32_t n_images = 0;
            int32_t dim_x = 0;
            int32_t dim_y = 0;

            read_swapped(stream, n_images);
            read_swapped(stream, dim_x);
            read_swapped(stream, dim_y);

            const auto image_size = dim_x * dim_y;
            std::vector<uint8_t> image_buf;
            image_buf.resize(image_size);

            data_points.resize(offset + n_images);

            for (auto i = 0; i < n_images; i++) {
                data_points[offset + i].input.reserve(image_size);

                stream.read(reinterpret_cast<char*>(image_buf.data()), image_size);
                for (const auto& pixel : image_buf) {
                    data_points[offset + i].input.push_back(pixel / 255.0F);
                }
            }

            return n_images;
        };

        auto read_labels = [&](std::fstream& stream, uint32_t offset) -> size_t {
            int32_t magic = 0;
            read_swapped(stream, magic);

            if (magic != 0x0801) {
                return 0;
            }

            int32_t n_labels = 0;

            read_swapped(stream, n_labels);

            std::vector<uint8_t> label_buf;
            label_buf.resize(n_labels);

            for (auto i = 0; i < n_labels; i++) {
                data_points[offset + i].output.resize(10);

                uint8_t label = 0;
                stream.read(reinterpret_cast<char*>(&label), 1);

                data_points[offset + i].output[label] = 1.0F;
            }

            return n_labels;
        };

        float split_numerator = 0.0F;

        if (auto n_training = read_images(train_img, 0); n_training != 0) {
            split_numerator = n_training;
        } else {
            return error_out;
        }

        if (read_labels(train_label, 0) == 0) {
            return error_out;
        }

        if (read_images(test_img, data_points.size()) == 0) {
            return error_out;
        }

        if (read_labels(test_label, data_points.size()) == 0) {
            return error_out;
        }

        return {data_points, split_numerator / data_points.size()};
    }

    auto mnist(const std::filesystem::path& training_images, const std::filesystem::path& training_labels,
               const std::filesystem::path& test_images, const std::filesystem::path& test_labels) -> loader {
        return [training_images, training_labels, test_images, test_labels]() {
            return load_mnist(training_images, training_labels, test_images, test_labels);
        };
    }
} // namespace yuqi::data::loaders