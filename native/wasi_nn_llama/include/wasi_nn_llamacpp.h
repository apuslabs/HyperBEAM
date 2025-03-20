
#include <wasi_nn_types.h>
#include <wasm_c_api.h>
#include <wasm_export.h>
#include <stdint.h>
#include <stdbool.h>



// Public API functions
__attribute__((visibility("default"))) wasi_nn_error init_backend(void **ctx);
// __attribute__((visibility("default"))) wasi_nn_error deinit_backend(void *ctx);
// __attribute__((visibility("default"))) wasi_nn_error load(void *ctx, graph_builder_array *builder, 
//                                                          graph_encoding encoding, execution_target target, 
//                                                          graph *g);
 __attribute__((visibility("default"))) wasi_nn_error load_by_name(void *ctx, const char *filename, 
                                                                  uint32_t filename_len, graph *g);
// __attribute__((visibility("default"))) wasi_nn_error load_by_name_with_config(void *ctx, const char *filename, 
//                                                                              uint32_t filename_len, const char *config, 
//                                                                              uint32_t config_len, graph *g);
// __attribute__((visibility("default"))) wasi_nn_error init_execution_context(void *ctx, graph g, 
//                                                                           graph_execution_context *exec_ctx);
// __attribute__((visibility("default"))) wasi_nn_error set_input(void *ctx, graph_execution_context exec_ctx, 
//                                                               uint32_t index, tensor *input_tensor);
// __attribute__((visibility("default"))) wasi_nn_error compute(void *ctx, graph_execution_context exec_ctx);
// __attribute__((visibility("default"))) wasi_nn_error get_output(void *ctx, graph_execution_context exec_ctx, 
//                                                                uint32_t index, tensor_data output_tensor, 
//                                                                uint32_t *output_tensor_size);
