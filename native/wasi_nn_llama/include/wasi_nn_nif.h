#ifndef WASI_NN_LLAMACPP_H
#define WASI_NN_LLAMACPP_H

#include "wasi_nn_types.h"

wasi_nn_error init_backend(void **ctx);
wasi_nn_error deinit_backend(void *ctx);
wasi_nn_error load_by_name(void *ctx, const char *filename, uint32_t filename_len, graph *g);
// Add other function declarations...

#endif