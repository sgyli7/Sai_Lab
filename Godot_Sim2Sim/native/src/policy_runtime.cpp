#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <onnxruntime_cxx_api.h>

#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace godot {

class MicroDuckPolicy : public RefCounted {
    GDCLASS(MicroDuckPolicy, RefCounted)

    std::unique_ptr<Ort::Session> session;
    PackedByteArray model_bytes;
    std::string input_name;
    std::string output_name;
    Dictionary model_metadata;
    String last_error;
    int64_t last_infer_usec = 0;
    int64_t observation_dim = 61;
    int64_t action_dim = 14;

    static Ort::Env &environment() {
        static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "microduck");
        static const bool telemetry_disabled = [] { env.DisableTelemetryEvents(); return true; }();
        (void)telemetry_disabled;
        return env;
    }

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("load_model", "bytes"), &MicroDuckPolicy::load_model);
        ClassDB::bind_method(D_METHOD("infer", "observation"), &MicroDuckPolicy::infer);
        ClassDB::bind_method(D_METHOD("unload"), &MicroDuckPolicy::unload);
        ClassDB::bind_method(D_METHOD("get_metadata"), &MicroDuckPolicy::get_metadata);
        ClassDB::bind_method(D_METHOD("get_last_error"), &MicroDuckPolicy::get_last_error);
        ClassDB::bind_method(D_METHOD("get_last_infer_usec"), &MicroDuckPolicy::get_last_infer_usec);
        ClassDB::bind_method(D_METHOD("get_runtime_version"), &MicroDuckPolicy::get_runtime_version);
    }

public:
    bool load_model(const PackedByteArray &bytes) {
        unload();
        last_error = String();
        try {
            if (bytes.is_empty()) throw std::runtime_error("Empty ONNX model");
            model_bytes = bytes;
            Ort::SessionOptions options;
            options.SetIntraOpNumThreads(1);
            options.SetInterOpNumThreads(1);
            options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            session = std::make_unique<Ort::Session>(environment(), model_bytes.ptr(),
                                                    model_bytes.size(), options);
            if (session->GetInputCount() != 1 || session->GetOutputCount() != 1)
                throw std::runtime_error("Policy must have one input and one output");
            for (bool input : {true, false}) {
                auto info = input ? session->GetInputTypeInfo(0) : session->GetOutputTypeInfo(0);
                auto tensor = info.GetTensorTypeAndShapeInfo();
                auto shape = tensor.GetShape();
                if (tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                    shape.size() != 2 || (shape[0] != 1 && shape[0] != -1) ||
                    (input ? (shape[1] != 61 && shape[1] != 68 && shape[1] != 82 && shape[1] != 104 && shape[1] != 242 && shape[1] != 243 && shape[1] != 258)
                           : (shape[1] != 14 && shape[1] != 16)))
                    throw std::runtime_error("Expected float32 [1,61|68] -> [1,14] or [1,82|104|242|243|258] -> [1,16] policy");
                if (input) observation_dim = shape[1];
                else action_dim = shape[1];
            }
            if (((observation_dim == 82 || observation_dim == 104 || observation_dim == 242 || observation_dim == 243 || observation_dim == 258)) != (action_dim == 16))
                throw std::runtime_error("Sai observation/action dimensions must be [1,82|104|242|243|258] -> [1,16]");
            Ort::AllocatorWithDefaultOptions allocator;
            input_name = session->GetInputNameAllocated(0, allocator).get();
            output_name = session->GetOutputNameAllocated(0, allocator).get();
            auto metadata = session->GetModelMetadata();
            for (const auto &key : metadata.GetCustomMetadataMapKeysAllocated(allocator)) {
                auto value = metadata.LookupCustomMetadataMapAllocated(key.get(), allocator);
                model_metadata[String::utf8(key.get())] = String::utf8(value.get());
            }
            const String task_mode = model_metadata.get("sim2sim_roller_task_input", "");
            if ((observation_dim == 68 && task_mode != "brake_markov_68_v1") ||
                (observation_dim == 61 && !task_mode.is_empty()))
                throw std::runtime_error("Observation shape and task metadata disagree");
            return true;
        } catch (const std::exception &error) {
            last_error = String::utf8(error.what());
            unload();
            return false;
        }
    }

    PackedFloat32Array infer(const PackedFloat32Array &observation) {
        PackedFloat32Array result;
        last_error = String();
        try {
            if (!session) throw std::runtime_error("No policy loaded");
            if (observation.size() != observation_dim) throw std::runtime_error("Observation dimension mismatch");
            for (int i = 0; i < observation_dim; ++i)
                if (!std::isfinite(observation[i])) throw std::runtime_error("Non-finite observation");
            const auto start = std::chrono::steady_clock::now();
            std::array<int64_t, 2> shape{1, observation_dim};
            auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            auto input = Ort::Value::CreateTensor<float>(memory,
                const_cast<float *>(observation.ptr()), observation_dim, shape.data(), shape.size());
            const char *inputs[] = {input_name.c_str()};
            const char *outputs[] = {output_name.c_str()};
            auto values = session->Run(Ort::RunOptions{nullptr}, inputs, &input, 1, outputs, 1);
            if (!values[0].IsTensor() || values[0].GetTensorTypeAndShapeInfo().GetElementCount() != action_dim)
                throw std::runtime_error("Invalid policy output");
            const float *data = values[0].GetTensorData<float>();
            result.resize(action_dim);
            for (int i = 0; i < action_dim; ++i) {
                if (!std::isfinite(data[i])) throw std::runtime_error("Non-finite action");
                result.set(i, data[i]);
            }
            last_infer_usec = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start).count();
        } catch (const std::exception &error) {
            last_error = String::utf8(error.what());
            result.clear();
        }
        return result;
    }

    void unload() {
        session.reset();
        model_bytes.clear();
        model_metadata.clear();
        input_name.clear();
        output_name.clear();
        last_infer_usec = 0;
        observation_dim = 61;
        action_dim = 14;
    }
    Dictionary get_metadata() const { return model_metadata.duplicate(); }
    String get_last_error() const { return last_error; }
    int64_t get_last_infer_usec() const { return last_infer_usec; }
    String get_runtime_version() const { return String::utf8(OrtGetApiBase()->GetVersionString()); }
};

void initialize_microduck(ModuleInitializationLevel level) {
    if (level == MODULE_INITIALIZATION_LEVEL_SCENE) ClassDB::register_class<MicroDuckPolicy>();
}
void uninitialize_microduck(ModuleInitializationLevel level) {}

} // namespace godot

extern "C" {
GDExtensionBool GDE_EXPORT microduck_library_init(GDExtensionInterfaceGetProcAddress get_proc_address,
    GDExtensionClassLibraryPtr library, GDExtensionInitialization *initialization) {
    godot::GDExtensionBinding::InitObject init(get_proc_address, library, initialization);
    init.register_initializer(godot::initialize_microduck);
    init.register_terminator(godot::uninitialize_microduck);
    init.set_minimum_library_initialization_level(godot::MODULE_INITIALIZATION_LEVEL_SCENE);
    return init.init();
}
}
