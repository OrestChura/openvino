// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kokoro_compiled_model.hpp"

#include "intel_npu/config/config.hpp"
#include "intel_npu/npuw_private_properties.hpp"
#include "kokoro_infer_request.hpp"
#include "kokoro_model_transforms.hpp"
#include "kokoro_split.hpp"
#include "npuw/logging.hpp"
#include "npuw/serialization.hpp"
#include "openvino/runtime/properties.hpp"
#include "plugin.hpp"

namespace {
void split_kokoro_properties(const ov::AnyMap& properties,
                             ov::AnyMap& other_properties,
                             ov::AnyMap& kokoro_properties) {
    for (auto it = properties.begin(); it != properties.end(); ++it) {
        if (it->first.find("NPUW_KOKORO") != it->first.npos) {
            kokoro_properties.insert(*it);
        } else {
            other_properties.insert(*it);
        }
    }
}

std::map<std::string, std::string> any_copy(const ov::AnyMap& params) {
    std::map<std::string, std::string> result;
    for (auto&& value : params) {
        if (value.second.is<std::string>()) {
            result.emplace(value.first, value.second.as<std::string>());
        } else if (value.second.is<bool>()) {
            result.emplace(value.first, value.second.as<bool>() ? "YES" : "NO");
        } else {
            std::stringstream ss;
            value.second.print(ss);
            result.emplace(value.first, ss.str());
        }
    }
    return result;
}

std::function<std::string(const std::string&)> get_encrypt_callback(const ov::AnyMap& properties) {
    if (const auto it = properties.find(ov::cache_encryption_callbacks.name()); it != properties.end()) {
        return it->second.as<ov::EncryptionCallbacks>().encrypt;
    }
    return {};
}

/**
 * @brief Kokoro's CPU-offloaded subgraphs are precision-sensitive.
 * Setting default CPU inference precision to f32 to ensure better accuracy, if not set by user.
 */
void set_default_cpu_inference_precision(ov::AnyMap& properties) {
    auto device_properties = properties.count(ov::device::properties.name())
                                 ? properties.at(ov::device::properties.name()).as<ov::AnyMap>()
                                 : ov::AnyMap{};
    auto cpu_properties = device_properties.count("CPU") ? device_properties.at("CPU").as<ov::AnyMap>() : ov::AnyMap{};

    if (cpu_properties.count(ov::hint::inference_precision.name())) {
        return;
    }
    cpu_properties[ov::hint::inference_precision.name()] = ov::element::f32;

    device_properties["CPU"] = std::move(cpu_properties);
    properties[ov::device::properties.name()] = std::move(device_properties);
    LOG_DEBUG("Setting default CPU INFERENCE_PRECISION_HINT to f32");
}
}  // namespace

ov::npuw::KokoroCompiledModel::KokoroCompiledModel(const std::shared_ptr<ov::Model>& model,
                                                   const std::shared_ptr<const ov::IPlugin>& plugin)
    : ov::npuw::ICompiledModel(model, plugin),
      m_name(model->get_friendly_name()),
      m_options_desc(std::make_shared<::intel_npu::OptionsDesc>()),
      m_cfg(m_options_desc) {
    ::intel_npu::registerNPUWKokoroOptions(*m_options_desc);
}

ov::npuw::KokoroCompiledModel::KokoroCompiledModel(const std::shared_ptr<ov::Model>& model,
                                                   const std::shared_ptr<const ov::IPlugin>& plugin,
                                                   const ov::AnyMap& properties)
    : KokoroCompiledModel(model, plugin) {
    LOG_DEBUG("Creating KokoroCompiledModel");

    // Split properties to separate Kokoro specific options and NPU plugin options
    ov::AnyMap npuw_kokoro_props;
    ov::AnyMap common_props;

    split_kokoro_properties(properties, common_props, npuw_kokoro_props);
    common_props["NPUW_FALLBACK_EXEC"] = "NO";

    m_cfg.parseEnvVars();
    m_cfg.update(any_copy(npuw_kokoro_props));

    // Get configuration from m_cfg, which now has defaults, env vars and user properties merged
    m_kokoro_cfg.block_size = m_cfg.get<::intel_npu::NPUW_KOKORO_BLOCK_SIZE>();
    m_kokoro_cfg.overlap_size = m_cfg.get<::intel_npu::NPUW_KOKORO_OVERLAP_SIZE>();

    // Decompose kokoro model into two static models
    KokoroSplitResult split_result = KokoroSplit::split_model(model, m_kokoro_cfg);

    // Guard aten::angle Divide(0,0)->NaN in Model B before NPUW partitioning
    ov::npuw::kokoro::guard_angle_divide(split_result.model_b);

    LOG_DEBUG("Compiling kokoro model A...");
    // Model A (BERT + LSTMs + duration predictor) — compile through NPUW
    // so that NPUW_CACHE_DIR caching works for both models.
    // NONE pipeline keeps it as a single subgraph (no partitioning overhead).
    ov::AnyMap properties_model_a = common_props;
    properties_model_a["NPUW_ONLINE_PIPELINE"] = "NONE";
    m_model_a_compiled = std::dynamic_pointer_cast<ov::npuw::CompiledModel>(
        ov::npuw::ICompiledModel::create(split_result.model_a, plugin, properties_model_a));
    OPENVINO_ASSERT(m_model_a_compiled, "Kokoro model A requires an NPUW CompiledModel");

    LOG_DEBUG("Compiling kokoro model B...");
    ov::AnyMap properties_model_b = common_props;

    // Enforce offloading to CPU for non-accurate subgraphs
    if (!properties_model_b.count("NPUW_ONLINE_PIPELINE")) {
        // Long compilation time, but best performance
        properties_model_b["NPUW_ONLINE_PIPELINE"] = "NONE";
    }
    if (!properties_model_b.count("NPUW_ONLINE_AVOID")) {
        properties_model_b["NPUW_ONLINE_AVOID"] = "P:FloorModFP32/NPU,P:CumSumSinGen/NPU,"
                                                  "P:BoxMullerNoise/NPU,P:AngleComplex/NPU";
    }
    set_default_cpu_inference_precision(properties_model_b);
    m_model_b_compiled = std::dynamic_pointer_cast<ov::npuw::CompiledModel>(
        ov::npuw::ICompiledModel::create(split_result.model_b, plugin, properties_model_b));
    OPENVINO_ASSERT(m_model_b_compiled, "Kokoro model B requires an NPUW CompiledModel");
}

void ov::npuw::KokoroCompiledModel::export_model(std::ostream& stream) const {
    LOG_DEBUG("Exporting KokoroCompiledModel");
    ov::npuw::orc::write_file_header(stream, ov::npuw::orc::schema_npuw::NPUW_ORC_KOKORO_SCHEMA);
    ov::npuw::orc::with_section(stream, kOrcType, kOrcVersion, 0u, [&] {
        ov::npuw::orc::with_leaf_section(stream, ov::npuw::orc::META_SECTION_TYPE, 0u, [&] {
            auto meta_stream = ov::npuw::orc::Stream::writer(stream);
            meta_stream & m_name& inputs() & outputs() & m_cfg;
        });
        m_model_a_compiled->serialize_orc_container(stream,
                                                    /* include_weights_bank = */ true,
                                                    get_encrypt_callback(m_model_a_compiled->non_npuw_props()));
        m_model_b_compiled->serialize_orc_container(stream,
                                                    /* include_weights_bank = */ true,
                                                    get_encrypt_callback(m_model_b_compiled->non_npuw_props()));
    });
}

std::shared_ptr<ov::npuw::KokoroCompiledModel> ov::npuw::KokoroCompiledModel::import_model(
    std::istream& stream,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& properties) {
    LOG_DEBUG("Importing KokoroCompiledModel");
    const auto header = ov::npuw::orc::read_file_header(stream);
    OPENVINO_ASSERT(header.schema_uuid == ov::npuw::orc::schema_npuw::NPUW_ORC_KOKORO_SCHEMA,
                    "Unsupported ORC schema for NPUW KokoroCompiledModel");

    ov::npuw::orc::ScopedReadSection root(stream);
    OPENVINO_ASSERT(root.remaining() <= ov::npuw::orc::Stream::reader(stream).remaining(),
                    "Kokoro root section claims ",
                    root.remaining(),
                    " which exceeds the stream size");
    OPENVINO_ASSERT(root.header().type == kOrcType, "Not a Kokoro ORC blob");
    OPENVINO_ASSERT(root.header().version <= kOrcVersion,
                    "Kokoro blob was produced by a newer NPUW (section version ",
                    root.header().version,
                    "; this build supports up to ",
                    kOrcVersion,
                    ")");
    OPENVINO_ASSERT(!ov::npuw::orc::has_flag(root.header().flags, ov::npuw::orc::SectionFlag::LEAF),
                    "Unsupported ORC NPUW Kokoro root section");
    OPENVINO_ASSERT(!root.done(), "Unexpected end of Kokoro section: missing metadata");

    ov::npuw::orc::ScopedReadSection meta(stream);
    OPENVINO_ASSERT(meta.remaining() <= root.remaining(), "Kokoro META section exceeds Kokoro root section");
    OPENVINO_ASSERT(meta.header().type == ov::npuw::orc::META_SECTION_TYPE &&
                        ov::npuw::orc::has_flag(meta.header().flags, ov::npuw::orc::SectionFlag::LEAF),
                    "Expected ORC NPUW metadata section, got type ",
                    meta.header().type);
    const auto meta_blob = meta.read_remaining_blob();
    auto meta_stream = ov::npuw::orc::Stream::memory_reader(meta_blob.data(), meta_blob.size());

    std::string name;
    ov::ParameterVector parameters;
    ov::NodeVector results;
    meta_stream & name & parameters & results;
    auto model = std::make_shared<ov::Model>(ov::as_output_vector(results), parameters, name);
    auto compiled = std::shared_ptr<KokoroCompiledModel>(new KokoroCompiledModel(model, plugin));
    meta_stream & compiled->m_cfg;
    OPENVINO_ASSERT(meta_stream.remaining() == 0, "Unexpected remaining bytes in KokoroCompiledModel META section");
    meta.expect_end();
    compiled->m_kokoro_cfg.block_size = compiled->m_cfg.get<::intel_npu::NPUW_KOKORO_BLOCK_SIZE>();
    compiled->m_kokoro_cfg.overlap_size = compiled->m_cfg.get<::intel_npu::NPUW_KOKORO_OVERLAP_SIZE>();
    OPENVINO_ASSERT(compiled->block_size() > 0, "Kokoro blob: block_size must be > 0");
    OPENVINO_ASSERT(compiled->overlap_size() < compiled->block_size(),
                    "Kokoro blob: overlap_size must be smaller than block_size");

    auto validate_child_size = [&](const char* child_name) {
        const auto body_remaining = root.remaining();
        constexpr std::streamoff section_header_size = sizeof(ov::npuw::orc::TypeId) + sizeof(ov::npuw::orc::Version) +
                                                       sizeof(ov::npuw::orc::SectionFlags) + sizeof(std::uint64_t);
        OPENVINO_ASSERT(body_remaining >= static_cast<std::size_t>(section_header_size),
                        "Kokoro root is missing ",
                        child_name,
                        " CompiledModel section");
        const auto saved = stream.tellg();
        auto child_reader = ov::npuw::orc::Stream::reader(stream);
        ov::npuw::orc::SectionHeader child_header;
        child_reader & child_header;
        stream.seekg(saved);
        OPENVINO_ASSERT(child_header.size <= body_remaining - static_cast<std::size_t>(section_header_size),
                        "Kokoro ",
                        child_name,
                        " CompiledModel section exceeds Kokoro root section");
    };

    validate_child_size("model A");
    compiled->m_model_a_compiled =
        ov::npuw::CompiledModel::deserialize_orc_container(stream, plugin, properties, true, {});
    validate_child_size("model B");
    compiled->m_model_b_compiled =
        ov::npuw::CompiledModel::deserialize_orc_container(stream, plugin, properties, true, {});
    root.expect_end();
    return compiled;
}

std::shared_ptr<const ov::Model> ov::npuw::KokoroCompiledModel::get_runtime_model() const {
    // FIXME Not implemented
    LOG_DEBUG("Getting runtime model from KokoroCompiledModel");
    OPENVINO_NOT_IMPLEMENTED;
}

void ov::npuw::KokoroCompiledModel::set_property(const ov::AnyMap& properties) {
    LOG_DEBUG("Setting properties to KokoroCompiledModel");
    // Both models are NPUW compiled models, they validate which keys are mutable
    std::lock_guard<std::mutex> lock(m_set_property_mutex);
    if (m_model_a_compiled) {
        m_model_a_compiled->set_property(properties);
    }
    if (m_model_b_compiled) {
        m_model_b_compiled->set_property(properties);
    }
}

ov::Any ov::npuw::KokoroCompiledModel::get_property(const std::string& name) const {
    OPENVINO_SUPPRESS_DEPRECATED_START
    if (m_model_b_compiled) {
        return m_model_b_compiled->get_property(name);
    }
    OPENVINO_THROW("Property ", name, " not found");
    OPENVINO_SUPPRESS_DEPRECATED_END
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::KokoroCompiledModel::create_sync_infer_request() const {
    auto* non_const_this = const_cast<ov::npuw::KokoroCompiledModel*>(this);
    return non_const_this->create_kokoro_infer_request();
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::KokoroCompiledModel::create_kokoro_infer_request() {
    auto this_sptr = std::static_pointer_cast<ov::npuw::KokoroCompiledModel>(shared_from_this());
    return std::make_shared<ov::npuw::KokoroInferRequest>(this_sptr);
}
