// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/opsets/opset13.hpp"

namespace ov::test::npuw {

// A deterministic Kokoro-shaped graph with the same split points and host-side
// duration expansion as the real model, without requiring pretrained weights.
inline std::shared_ptr<ov::Model> make_kokoro_test_model() {
    namespace op = ov::opset13;
    auto ids = std::make_shared<op::Parameter>(ov::element::i64, ov::Shape{1, 8});
    ids->output(0).set_names({"input_ids", "tokens"});
    auto style = std::make_shared<op::Parameter>(ov::element::f32, ov::Shape{1, 1, 1});
    style->output(0).set_names({"style", "style_alias"});

    auto zero = op::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto one = op::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto durations = std::make_shared<op::Add>(ids, one);
    durations->output(0).set_names({"phonemes", "duration_alias"});

    auto mask = std::make_shared<op::Equal>(ids, zero);
    mask->output(0).set_names({"text_mask"});
    auto values = std::make_shared<op::Add>(std::make_shared<op::Convert>(ids, ov::element::f32),
                                            std::make_shared<op::Convert>(mask, ov::element::f32));
    auto axis = op::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto features = std::make_shared<op::Unsqueeze>(values, axis);
    auto weights = op::Constant::create(ov::element::f32, ov::Shape{1, 2, 1}, {0.25f, 0.5f});
    auto en_left = std::make_shared<op::Multiply>(features, weights);
    auto bias = op::Constant::create(ov::element::f32, ov::Shape{1, 2, 1}, {0.125f, 0.25f});
    auto asr_left = std::make_shared<op::Add>(en_left, bias);

    auto sum_axes = op::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
    auto total = std::make_shared<op::ReduceSum>(durations, sum_axes, false);
    auto total_vector =
        std::make_shared<op::Unsqueeze>(total, op::Constant::create(ov::element::i64, ov::Shape{1}, {0}));
    auto alignment_shape = std::make_shared<op::Concat>(
        ov::OutputVector{op::Constant::create(ov::element::i64, ov::Shape{2}, {1, 8}), total_vector},
        0);
    auto alignment =
        std::make_shared<op::Broadcast>(op::Constant::create(ov::element::f32, ov::Shape{}, {1.0f}), alignment_shape);
    auto en = std::make_shared<op::MatMul>(en_left, alignment);
    en->set_friendly_name("aten::matmul/MatMul");
    auto asr = std::make_shared<op::MatMul>(asr_left, alignment);
    asr->set_friendly_name("aten::matmul/MatMul_1");
    auto mixing_weights = op::Constant::create(ov::element::f32, ov::Shape{1, 1, 2}, {1.0001f, 1.0003f});
    auto audio =
        std::make_shared<op::Add>(std::make_shared<op::MatMul>(mixing_weights, std::make_shared<op::Add>(en, asr)),
                                  style);
    audio->output(0).set_names({"audio", "waveform"});

    return std::make_shared<ov::Model>(ov::OutputVector{audio, durations},
                                       ov::ParameterVector{ids, style},
                                       "kokoro_round_trip");
}

}  // namespace ov::test::npuw
