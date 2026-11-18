// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/file_utils.hpp"
#include "intel_npu/config/npuw.hpp"
#include "kokoro_test_helpers.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/util/codec_xor.hpp"
#include "pipelines/kokoro/kokoro_compiled_model.hpp"
#include "serialization.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_iplugin.hpp"

namespace {

class KokoroImportExportTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto prefix = ov::test::utils::generateTestFilePrefix() + "_kokoro";
        m_xml = prefix + ".xml";
        m_bin = prefix + ".bin";
        ov::serialize(ov::test::npuw::make_kokoro_test_model(), m_xml, m_bin);
        m_model = m_core.read_model(m_xml);
    }

    void TearDown() override {
        m_model.reset();
        ov::test::utils::removeIRFiles(m_xml, m_bin);
    }

    ov::CompiledModel compile(ov::CacheMode mode = ov::CacheMode::OPTIMIZE_SPEED,
                              bool encrypted = false,
                              bool weights_as_inputs = false) {
        ov::AnyMap properties = {{"NPU_USE_NPUW", "YES"},
                                 {"NPUW_KOKORO", "YES"},
                                 {"NPUW_DEVICES", "CPU"},
                                 {"NPUW_F16IC", "NO"},
                                 {"NPUW_KOKORO_BLOCK_SIZE", "6"},
                                 {"NPUW_KOKORO_OVERLAP_SIZE", "2"},
                                 {"NPUW_CWAI", weights_as_inputs ? "YES" : "NO"},
                                 {ov::cache_mode.name(), mode}};
        if (encrypted) {
            properties[ov::cache_encryption_callbacks.name()] = callbacks();
        }
        return m_core.compile_model(m_model, "NPU", properties);
    }

    static ov::EncryptionCallbacks callbacks() {
        return {ov::util::codec_xor, ov::util::codec_xor};
    }

    static void expect_ports_equal(const ov::CompiledModel& expected, const ov::CompiledModel& actual) {
        for (const auto& ports : {std::make_pair(expected.inputs(), actual.inputs()),
                                  std::make_pair(expected.outputs(), actual.outputs())}) {
            ASSERT_EQ(ports.first.size(), ports.second.size());
            for (std::size_t i = 0; i < ports.first.size(); ++i) {
                EXPECT_EQ(ports.first[i].get_element_type(), ports.second[i].get_element_type());
                EXPECT_EQ(ports.first[i].get_partial_shape(), ports.second[i].get_partial_shape());
                EXPECT_EQ(ports.first[i].get_names(), ports.second[i].get_names());
            }
        }
    }

    static std::vector<float> infer(ov::InferRequest& request, float style, int64_t token = 2) {
        ov::Tensor ids(ov::element::i64, {1, 8});
        const std::vector<int64_t> tokens{0, 1, token, 0, 0, 0, 0, 0};
        std::copy(tokens.begin(), tokens.end(), ids.data<int64_t>());
        ov::Tensor style_tensor(ov::element::f32, {1, 1, 1});
        style_tensor.data<float>()[0] = style;
        request.set_tensor("tokens", ids);
        request.set_tensor("style_alias", style_tensor);
        request.infer();

        const auto durations = request.get_tensor("duration_alias");
        EXPECT_EQ(durations.get_element_type(), ov::element::i64);
        EXPECT_EQ(durations.get_shape(), (ov::Shape{1, 8}));
        const std::vector<int64_t> expected_durations{1, 2, token + 1, 1, 0, 0, 0, 0};
        EXPECT_EQ(std::vector<int64_t>(durations.data<int64_t>(), durations.data<int64_t>() + durations.get_size()),
                  expected_durations);

        const auto audio = request.get_tensor("waveform");
        EXPECT_EQ(audio.get_element_type(), ov::element::f32);
        EXPECT_EQ(audio.get_shape(), (ov::Shape{1, 1, static_cast<std::size_t>(token + 5)}));
        return {audio.data<float>(), audio.data<float>() + audio.get_size()};
    }

    ov::Core m_core;
    std::shared_ptr<ov::Model> m_model;
    std::string m_xml;
    std::string m_bin;
};

enum class WeightSource { Embedded, EmbeddedWAI, Model, File };

void write_kokoro_meta(std::ostream& stream,
                       const std::shared_ptr<const ov::Model>& model,
                       const ::intel_npu::Config& config) {
    ov::npuw::orc::with_leaf_section(stream, ov::npuw::orc::META_SECTION_TYPE, 0u, [&] {
        auto writer = ov::npuw::orc::Stream::writer(stream);
        writer & model->get_friendly_name() & model->inputs() & model->outputs() &
            const_cast<::intel_npu::Config&>(config);
    });
}

class KokoroRoundTripTest : public KokoroImportExportTest,
                            public ::testing::WithParamInterface<std::tuple<WeightSource, bool>> {};

TEST_P(KokoroRoundTripTest, PreservesPortsConfigurationAndInference) {
    const auto [source, encrypted] = GetParam();
    const auto mode = source == WeightSource::Embedded || source == WeightSource::EmbeddedWAI
                          ? ov::CacheMode::OPTIMIZE_SPEED
                          : ov::CacheMode::OPTIMIZE_SIZE;
    auto compiled = compile(mode, encrypted, source != WeightSource::Embedded);
    auto request = compiled.create_infer_request();
    const auto expected = infer(request, 0.75f);
    const std::vector<float>
        reference{1.1250875f, 2.6254375f, 2.6254375f, 4.1257875f, 4.1257875f, 4.1257875f, 1.1250875f};
    ASSERT_EQ(expected.size(), reference.size());
    for (std::size_t i = 0; i < reference.size(); ++i) {
        EXPECT_NEAR(expected[i], reference[i], 1e-6f);
    }

    std::stringstream blob;
    compiled.export_model(blob);
    const auto start = blob.tellg();
    EXPECT_TRUE(ov::npuw::ICompiledModel::is_npuw_blob(blob));
    EXPECT_EQ(blob.tellg(), start);
    const auto orc_header = ov::npuw::orc::read_file_header(blob);
    EXPECT_EQ(orc_header.schema_uuid, ov::npuw::orc::schema_npuw::NPUW_ORC_KOKORO_SCHEMA);
    blob.seekg(start);
    const auto root = ov::npuw::orc::read_file(blob);
    EXPECT_EQ(root.type, ov::npuw::KokoroCompiledModel::kOrcType);
    ASSERT_GE(root.children.size(), 3u);
    EXPECT_EQ(root.children[0].type, ov::npuw::orc::META_SECTION_TYPE);
    EXPECT_EQ(root.children[1].type,
              static_cast<ov::npuw::orc::TypeId>(ov::npuw::orc::schema_npuw::PartitionedModel::ID));
    EXPECT_EQ(root.children[2].type,
              static_cast<ov::npuw::orc::TypeId>(ov::npuw::orc::schema_npuw::PartitionedModel::ID));
    blob.seekg(start);

    // Import must restore the compiled block geometry, not reinterpret new options.
    ov::AnyMap properties = {{"NPUW_KOKORO_BLOCK_SIZE", "32"}, {"NPUW_KOKORO_OVERLAP_SIZE", "4"}};
    if (source == WeightSource::Model) {
        properties[ov::hint::model.name()] = std::static_pointer_cast<const ov::Model>(m_model);
    } else if (source == WeightSource::File) {
        properties[ov::weights_path.name()] = m_bin;
    }
    if (encrypted) {
        properties[ov::cache_encryption_callbacks.name()] = callbacks();
    }

    ov::CompiledModel imported;
    ASSERT_NO_THROW(imported = m_core.import_model(blob, "NPU", properties));
    EXPECT_EQ(blob.tellg(), static_cast<std::streamoff>(blob.str().size()));
    expect_ports_equal(compiled, imported);
    auto imported_request = imported.create_infer_request();
    EXPECT_EQ(infer(imported_request, 0.75f), expected);
    EXPECT_EQ(infer(imported_request, 1.25f, 3), infer(request, 1.25f, 3));

    blob.seekg(start);
    ov::CompiledModel reimported;
    ASSERT_NO_THROW(reimported = m_core.import_model(blob, "NPU", properties));
    auto reimported_request = reimported.create_infer_request();
    EXPECT_EQ(infer(reimported_request, 0.75f), expected);
}

INSTANTIATE_TEST_SUITE_P(CacheModes,
                         KokoroRoundTripTest,
                         ::testing::Combine(::testing::Values(WeightSource::Embedded,
                                                              WeightSource::EmbeddedWAI,
                                                              WeightSource::Model,
                                                              WeightSource::File),
                                            ::testing::Bool()),
                         ([](const ::testing::TestParamInfo<KokoroRoundTripTest::ParamType>& info) {
                             const auto [source, encrypted] = info.param;
                             const std::string name = source == WeightSource::Embedded      ? "Embedded"
                                                      : source == WeightSource::EmbeddedWAI ? "EmbeddedWAI"
                                                      : source == WeightSource::Model       ? "ModelPtr"
                                                                                            : "WeightsPath";
                             return name + (encrypted ? "Encrypted" : "Plain");
                         }));

TEST_F(KokoroImportExportTest, RejectsWeightlessImportWithoutWeights) {
    auto compiled = compile(ov::CacheMode::OPTIMIZE_SIZE, false, true);
    std::stringstream blob;
    compiled.export_model(blob);
    EXPECT_THROW(m_core.import_model(blob, "NPU"), ov::Exception);
}

TEST_F(KokoroImportExportTest, RejectsEncryptedImportWithoutDecryptCallback) {
    auto compiled = compile(ov::CacheMode::OPTIMIZE_SPEED, true);
    std::stringstream blob;
    compiled.export_model(blob);
    EXPECT_THROW(m_core.import_model(blob, "NPU"), ov::Exception);
}

TEST_F(KokoroImportExportTest, RejectsTruncatedSecondModel) {
    auto compiled = compile();
    std::stringstream blob;
    compiled.export_model(blob);
    auto bytes = blob.str();
    bytes.pop_back();
    std::stringstream truncated(bytes);
    EXPECT_THROW(m_core.import_model(truncated, "NPU"), ov::Exception);
}

TEST(KokoroImportValidationTest, RejectsLegacyIndicatorBlob) {
    std::stringstream blob;
    ov::npuw::s11n::write_header(blob, NPUW_KOKORO_COMPILED_MODEL_INDICATOR);
    try {
        ov::npuw::ICompiledModel::import_model(blob, nullptr, {});
        FAIL() << "Legacy Kokoro blob was accepted";
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string(error.what()).find("Legacy flat NPUW KokoroCompiledModel blobs are no longer supported"),
                  std::string::npos)
            << error.what();
    }
}

TEST(KokoroImportValidationTest, RejectsOtherORCSchema) {
    std::stringstream blob;
    ov::npuw::orc::write_file_header(blob, ov::npuw::orc::schema_npuw::NPUW_ORC_PARTITIONED_SCHEMA);
    ov::npuw::orc::with_section(blob,
                                ov::npuw::KokoroCompiledModel::kOrcType,
                                ov::npuw::KokoroCompiledModel::kOrcVersion,
                                0u,
                                [] {});
    EXPECT_THROW(ov::npuw::KokoroCompiledModel::import_model(blob, nullptr, {}), ov::Exception);
}

TEST(KokoroImportValidationTest, RejectsNewerORCVersion) {
    std::stringstream blob;
    ov::npuw::orc::write_file_header(blob, ov::npuw::orc::schema_npuw::NPUW_ORC_KOKORO_SCHEMA);
    ov::npuw::orc::with_section(blob,
                                ov::npuw::KokoroCompiledModel::kOrcType,
                                ov::npuw::KokoroCompiledModel::kOrcVersion + 1,
                                0u,
                                [] {});
    EXPECT_THROW(ov::npuw::KokoroCompiledModel::import_model(blob, nullptr, {}), ov::Exception);
}

TEST(KokoroImportValidationTest, RejectsORCRootOverrunningStream) {
    std::stringstream blob;
    ov::npuw::orc::write_file_header(blob, ov::npuw::orc::schema_npuw::NPUW_ORC_KOKORO_SCHEMA);
    const auto root_pos = blob.tellp();
    ov::npuw::orc::with_section(blob,
                                ov::npuw::KokoroCompiledModel::kOrcType,
                                ov::npuw::KokoroCompiledModel::kOrcVersion,
                                0u,
                                [] {});
    const std::uint64_t root_size = blob.tellp() - root_pos;
    blob.seekp(root_pos + static_cast<std::streamoff>(sizeof(ov::npuw::orc::TypeId) + sizeof(ov::npuw::orc::Version) +
                                                      sizeof(ov::npuw::orc::SectionFlags)));
    blob.write(reinterpret_cast<const char*>(&root_size), sizeof(root_size));
    EXPECT_THROW(ov::npuw::KokoroCompiledModel::import_model(blob, nullptr, {}), ov::Exception);
}

TEST(KokoroImportValidationTest, RejectsORCMetaOverrunningRoot) {
    std::stringstream blob;
    ov::npuw::orc::write_file_header(blob, ov::npuw::orc::schema_npuw::NPUW_ORC_KOKORO_SCHEMA);
    auto meta_size = std::uint64_t{100};
    ov::npuw::orc::with_section(
        blob,
        ov::npuw::KokoroCompiledModel::kOrcType,
        ov::npuw::KokoroCompiledModel::kOrcVersion,
        0u,
        [&] {
            const auto meta_pos = blob.tellp();
            ov::npuw::orc::with_leaf_section(blob, ov::npuw::orc::META_SECTION_TYPE, 0u, [] {});
            const auto restore = blob.tellp();
            blob.seekp(meta_pos +
                       static_cast<std::streamoff>(sizeof(ov::npuw::orc::TypeId) + sizeof(ov::npuw::orc::Version) +
                                                   sizeof(ov::npuw::orc::SectionFlags)));
            blob.write(reinterpret_cast<const char*>(&meta_size), sizeof(meta_size));
            blob.seekp(restore);
        });
    EXPECT_THROW(ov::npuw::KokoroCompiledModel::import_model(blob, nullptr, {}), ov::Exception);
}

TEST_F(KokoroImportExportTest, RejectsORCChildOverrunningRoot) {
    auto compiled = compile();
    std::stringstream blob;
    compiled.export_model(blob);
    auto bytes = blob.str();
    std::istringstream parser(bytes, std::ios::in | std::ios::binary);
    ov::npuw::orc::read_file_header(parser);
    ov::npuw::orc::ScopedReadSection root(parser);
    ov::npuw::orc::ScopedReadSection meta(parser);
    meta.read_remaining_blob();
    meta.expect_end();
    const auto child_pos = static_cast<std::streamoff>(parser.tellg());
    const auto child_size = std::uint64_t{1000};
    const auto child_size_offset =
        child_pos + static_cast<std::streamoff>(sizeof(ov::npuw::orc::TypeId) + sizeof(ov::npuw::orc::Version) +
                                                sizeof(ov::npuw::orc::SectionFlags));
    ASSERT_LT(static_cast<std::size_t>(child_size_offset) + sizeof(child_size), bytes.size());
    std::memcpy(bytes.data() + static_cast<std::size_t>(child_size_offset), &child_size, sizeof(child_size));
    std::stringstream malformed(bytes);
    EXPECT_THROW(m_core.import_model(malformed, "NPU"), ov::Exception);
}

TEST(KokoroImportValidationTest, RejectsMissingOrNonLeafMetaSection) {
    const std::vector<std::function<void(std::ostream&)>> write_children = {
        [](std::ostream& stream) {
            ov::npuw::orc::with_section(
                stream,
                static_cast<ov::npuw::orc::TypeId>(ov::npuw::orc::schema_npuw::PartitionedModel::ID),
                ov::npuw::CompiledModel::kOrcVersion,
                0u,
                [] {});
        },
        [](std::ostream& stream) {
            ov::npuw::orc::with_section(stream, ov::npuw::orc::META_SECTION_TYPE, 0u, 0u, [] {});
        }};
    for (const auto& write_child : write_children) {
        std::stringstream blob;
        ov::npuw::orc::write_file_header(blob, ov::npuw::orc::schema_npuw::NPUW_ORC_KOKORO_SCHEMA);
        ov::npuw::orc::with_section(blob,
                                    ov::npuw::KokoroCompiledModel::kOrcType,
                                    ov::npuw::KokoroCompiledModel::kOrcVersion,
                                    0u,
                                    [&] {
                                        write_child(blob);
                                    });
        EXPECT_THROW(ov::npuw::KokoroCompiledModel::import_model(blob, nullptr, {}), ov::Exception);
    }
}

TEST(KokoroImportValidationTest, RejectsInvalidBlockGeometryBeforeImportingChildren) {
    auto plugin = std::make_shared<::testing::NiceMock<ov::MockIPlugin>>();
    std::shared_ptr<const ov::Model> model = ov::test::npuw::make_kokoro_test_model();
    for (const auto& sizes : {std::make_pair(0u, 0u), std::make_pair(6u, 6u), std::make_pair(6u, 7u)}) {
        auto options = std::make_shared<::intel_npu::OptionsDesc>();
        ::intel_npu::registerNPUWKokoroOptions(*options);
        ::intel_npu::Config config(options);
        config.update({{"NPUW_KOKORO_BLOCK_SIZE", std::to_string(sizes.first)},
                       {"NPUW_KOKORO_OVERLAP_SIZE", std::to_string(sizes.second)}});
        std::stringstream blob;
        ov::npuw::orc::write_file_header(blob, ov::npuw::orc::schema_npuw::NPUW_ORC_KOKORO_SCHEMA);
        ov::npuw::orc::with_section(blob,
                                    ov::npuw::KokoroCompiledModel::kOrcType,
                                    ov::npuw::KokoroCompiledModel::kOrcVersion,
                                    0u,
                                    [&] {
                                        write_kokoro_meta(blob, model, config);
                                    });
        try {
            ov::npuw::ICompiledModel::import_model(blob, plugin, {});
            FAIL() << "Invalid Kokoro block geometry was accepted";
        } catch (const ov::Exception& error) {
            EXPECT_NE(std::string(error.what()).find("Kokoro blob:"), std::string::npos) << error.what();
        }
    }
}

}  // namespace
