#include "yuqi/data/data.hpp"

namespace yuqi::data {
    auto dataset::split(float ratio) -> void {
        auto split_idx = m_data.size() * ratio;

        m_test = std::span(m_data.begin() + split_idx, m_data.end());
        m_training = std::span(m_data.begin(), m_data.begin() + split_idx);
    }
} // namespace yuqi::data