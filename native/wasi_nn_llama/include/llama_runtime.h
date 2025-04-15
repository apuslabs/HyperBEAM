#ifndef LLAMA_RUNTIME_H
#define LLAMA_RUNTIME_H

#include <stddef.h> // For size_t
#include <stdbool.h> // For bool

// --- Opaque Handle ---
// The user interacts with this handle, implementation details are hidden.

typedef struct LlamaStateInternal* LlamaHandle;

// --- API Visibility Macros ---
// Define import/export macros for cross-platform compatibility (simplified for Linux/macOS)

#ifdef _WIN32
    #ifdef LLAMA_RUNTIME_BUILD_SHARED // Building the DLL
        #define LLAMA_RUNTIME_API __declspec(dllexport)
    #else // Using the DLL
        #define LLAMA_RUNTIME_API __declspec(dllimport)
    #endif
#else // Non-Windows (Linux, macOS)
    #define LLAMA_RUNTIME_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 
 * Initializes the LLaMA runtime environment.
 *
 * @param model_path (Path to the model file).
 * @param config  (Configuration parameters).
 * @param error_msg_buffer  (Output buffer for error messages).
 * @param error_msg_buffer_size  (Size of the error message buffer).
 * @return LlamaHandle (Returns a valid handle on success, NULL on failure).
 * (If it fails, an error message is written to error_msg_buffer).
 */
LLAMA_RUNTIME_API LlamaHandle initialize_llama_runtime(
    const char* model_path,
    const char* config,
    char* error_msg_buffer,
    size_t error_msg_buffer_size
);

/**
 * @brief 
 * Runs inference using the initialized LLaMA runtime.
 *
 * @param handle (Valid handle obtained from initialize_llama_runtime).
 * @param prompt  (The input prompt).
 * @param result_buffer (Output buffer for the inference result).
 * @param result_buffer_size (Size of the result buffer).
 * @param error_msg_buffer  (Output buffer for error messages).
 * @param error_msg_buffer_size  (Size of the error message buffer).
 * @return bool (If it fails, an error message is written to error_msg_buffer).
 */
LLAMA_RUNTIME_API bool run_inference(
    LlamaHandle handle,
    const char* prompt,
    char* result_buffer,
    size_t result_buffer_size,
    char* error_msg_buffer,
    size_t error_msg_buffer_size
);

/**
 * @brief 
 * Cleans up and releases LLaMA runtime resources.
 *
 * @param handle (The LLaMA handle to release).
 */
LLAMA_RUNTIME_API void cleanup_llama_runtime(LlamaHandle handle);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // LLAMA_RUNTIME_H
