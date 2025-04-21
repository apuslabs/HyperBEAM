#include "../include/wasi_nn_llamacpp.h"
#include "../include/wasi_nn_logging.h"
#define LIB_PATH "./native/wasi_nn_llama/libwasi_nn_llamacpp.so"
#define MAX_MODEL_PATH 256
#define MAX_INPUT_SIZE 4096
#define MAX_CONFIG_SIZE 1024
#define MAX_OUTPUT_SIZE 8192

typedef struct {
    void* ctx;
    graph g;
    graph_execution_context exec_ctx;
} LlamaContext;

static wasi_nn_backend_api g_wasi_nn_functions = {0};
static ErlNifResourceType* llama_context_resource;

static void llama_context_destructor(ErlNifEnv* env, void* obj)
{

    LlamaContext* ctx = (LlamaContext*)obj;
    if (ctx) {
        // Cleanup backend context
        if (ctx->ctx && g_wasi_nn_functions.deinit_backend) {
            g_wasi_nn_functions.deinit_backend(ctx->ctx);
            ctx->ctx = NULL;
        }
        // No need to cleanup shared library here since it's managed globally
        // Clear the context structure
        memset(ctx, 0, sizeof(LlamaContext));
    }

}

static int load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info)
{
    printf("Load nif start\n");
	
	g_wasi_nn_functions.handle = dlopen(LIB_PATH, RTLD_LAZY);
    if (!g_wasi_nn_functions.handle) {
        printf("Failed to load wasi library: %s\n", dlerror());
        return 1;
    }
	// Load all required functions once
	g_wasi_nn_functions.init_backend = (init_backend_fn)dlsym(g_wasi_nn_functions.handle, "init_backend");
    g_wasi_nn_functions.deinit_backend = (deinit_backend_fn)dlsym(g_wasi_nn_functions.handle, "deinit_backend");
    g_wasi_nn_functions.load_by_name = (load_by_name_fn)dlsym(g_wasi_nn_functions.handle, "load_by_name");
    g_wasi_nn_functions.init_execution_context = (init_execution_context_fn)dlsym(g_wasi_nn_functions.handle, "init_execution_context");
    g_wasi_nn_functions.set_input = (set_input_fn)dlsym(g_wasi_nn_functions.handle, "set_input");
    g_wasi_nn_functions.compute = (compute_fn)dlsym(g_wasi_nn_functions.handle, "compute");
    g_wasi_nn_functions.get_output = (get_output_fn)dlsym(g_wasi_nn_functions.handle, "get_output");
    g_wasi_nn_functions.load_by_name_with_config = (load_by_name_with_config_fn)dlsym(g_wasi_nn_functions.handle, "load_by_name_with_config");
    if (!g_wasi_nn_functions.init_backend ||!g_wasi_nn_functions.deinit_backend ||!g_wasi_nn_functions.load_by_name ||
       !g_wasi_nn_functions.init_execution_context ||!g_wasi_nn_functions.set_input ||!g_wasi_nn_functions.compute ||
       !g_wasi_nn_functions.get_output) {
        dlclose(g_wasi_nn_functions.handle);
        return 1;
    }
	
	llama_context_resource = enif_open_resource_type(env, NULL, "llama_context",
        llama_context_destructor, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);

    return llama_context_resource ? 0 : 1;
}




static ERL_NIF_TERM nif_init_backend(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    LlamaContext* ctx = enif_alloc_resource(llama_context_resource, sizeof(LlamaContext));
    if (!ctx) {
        printf("Failed to allocate LlamaContext resource\n");
        return enif_make_tuple2(env, enif_make_atom(env, "error"), 
                              enif_make_atom(env, "allocation_failed"));
    }
	printf("Initializing backend...\n");
    wasi_nn_error err = g_wasi_nn_functions.init_backend(&ctx->ctx);
    if (err != success) {
        printf("Backend initialization failed with error: %d\n", err);
        enif_release_resource(ctx);
        return enif_make_tuple2(env, enif_make_atom(env, "error"), 
                              enif_make_atom(env, "init_failed"));
    }
	printf("nif_init_backend finished \n");
    ERL_NIF_TERM ctx_term = enif_make_resource(env, ctx);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), ctx_term);
}
static ERL_NIF_TERM nif_load_by_name(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    LlamaContext* ctx;
    char model_path[MAX_MODEL_PATH];
    if (!enif_get_resource(env, argv[0], llama_context_resource, (void**)&ctx) ||
       !enif_get_string(env, argv[1], model_path, sizeof(model_path), ERL_NIF_LATIN1) ) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "invalid_args"));
       }

    if (g_wasi_nn_functions.load_by_name(ctx->ctx, model_path, strlen(model_path), &ctx->g)!= success) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "load_failed"));
    }
    return enif_make_atom(env, "ok");

}
static ERL_NIF_TERM nif_load_by_name_with_config(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    LlamaContext* ctx;
    const char *model_path = (char *)malloc(MAX_MODEL_PATH * sizeof(char));
	const char *config = (char *)malloc(MAX_CONFIG_SIZE * sizeof(char));
	// check context and input

	if(!enif_get_resource(env, argv[0], llama_context_resource, (void**)&ctx))
	{
		printf("Invalid context\n");
		return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "invalid_context"));
	}
    if (!enif_get_string(env, argv[1], model_path, MAX_MODEL_PATH, ERL_NIF_LATIN1)) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"), 
                              enif_make_atom(env, "invalid_model_path"));
    }
	// if conifg is provided, use it
	if (argc > 2 && !enif_get_string(env, argv[2], config, MAX_CONFIG_SIZE, ERL_NIF_LATIN1)) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"),
                              enif_make_atom(env, "invalid_config"));
    }
	printf("Loading model: %s %s\n", model_path, config);
	
    if (g_wasi_nn_functions.load_by_name_with_config(ctx->ctx, model_path, strlen(model_path), 
                                                   config, strlen(config), &ctx->g) != success) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "load_failed"));
    }
    return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM nif_init_execution_context(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    LlamaContext* ctx;
    if (!enif_get_resource(env, argv[0], llama_context_resource, (void**)&ctx)) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "invalid_args_init_execution"));
    }

    if (g_wasi_nn_functions.init_execution_context(ctx->ctx, ctx->g, &ctx->exec_ctx)!= success) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "init_execution_failed"));
    }
	return enif_make_atom(env, "ok");
}
static ERL_NIF_TERM nif_set_input(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    LlamaContext* ctx;
	const char *input = (char *)malloc(MAX_INPUT_SIZE * sizeof(char));
    if (!enif_get_resource(env, argv[0], llama_context_resource, (void**)&ctx)) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "invalid_args"));
    }
	// get input from argcs
	if (argc < 2) {
		printf("Invalid args\n");
		return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "invalid_args"));
	}
	if (!enif_get_string(env, argv[1], input, MAX_MODEL_PATH, ERL_NIF_LATIN1)) {
		printf("Invalid input\n");
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "invalid_input"));
    }
	printf("Loading input : %s\n", input);
	
	tensor_dimensions *dims = (tensor_dimensions *)malloc(sizeof(tensor_dimensions));
    if (dims == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
	dims ->size = 1;
	dims -> buf = (uint32_t *)malloc(dims->size * sizeof(uint32_t));
	dims->buf[0] = 1;

    tensor input_tensor = {
		.dimensions = dims,
		.type = u8,
        .data = (uint8_t *)input,
    };
    if (g_wasi_nn_functions.set_input(ctx->ctx, ctx->exec_ctx, 0, &input_tensor)!= success) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "set_input_failed"));
    }
	return enif_make_atom(env, "ok");
}
static ERL_NIF_TERM nif_compute(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    LlamaContext* ctx;
	
    if (!enif_get_resource(env, argv[0], llama_context_resource, (void**)&ctx)) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "invalid_args"));
    }
    if (g_wasi_nn_functions.compute(ctx->ctx, ctx->exec_ctx)!= success) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "compute_failed"));
    }
	return enif_make_atom(env, "ok");
}
static ERL_NIF_TERM nif_get_output(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    LlamaContext* ctx;
    if (!enif_get_resource(env, argv[0], llama_context_resource, (void**)&ctx)) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "invalid_args"));
    }
    uint8_t *output_buffer = (uint8_t *)malloc(MAX_OUTPUT_SIZE * sizeof(uint8_t));
	if (output_buffer == NULL) {
		fprintf(stderr, "OutputBuffer allocation failed\n");
        free(output_buffer);
        return 1;
	}
    uint32_t output_size;
    
    if (g_wasi_nn_functions.get_output(ctx->ctx, ctx->exec_ctx, 0, output_buffer, &output_size) != success) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "get_output_failed"));
    }

	// Create a new binary term in Erlang
    ERL_NIF_TERM result_bin;
    unsigned char* bin_data = enif_make_new_binary(env, output_size, &result_bin);
    if (!bin_data) {
        free(output_buffer);
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "binary_creation_failed"));
    }

    // Copy the output_buffer into the Erlang binary
    memcpy(bin_data, output_buffer, output_size);

    // Free the output_buffer as it's no longer needed
    free(output_buffer);

    return enif_make_tuple2(env,
        enif_make_atom(env, "ok"),
		result_bin);
}
static ERL_NIF_TERM nif_deinit_backend(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    LlamaContext* ctx;
    if (!enif_get_resource(env, argv[0], llama_context_resource, (void**)&ctx)) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "invalid_args"));
    }
    if (g_wasi_nn_functions.deinit_backend(ctx->ctx)!= success) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "deinit_failed"));
    }
    return enif_make_atom(env, "ok");
}
static ErlNifFunc nif_funcs[] = {
    {"init_backend", 0, nif_init_backend},
    {"load_by_name_with_config", 3, nif_load_by_name_with_config},
    {"init_execution_context", 1, nif_init_execution_context},
    {"set_input", 2, nif_set_input},
    {"compute", 1, nif_compute},
    {"get_output", 1, nif_get_output},
	{"deinit_backend", 1, nif_deinit_backend}
};


static void unload(ErlNifEnv* env, void* priv_data)
{
	// The resource destructor will be called automatically for any remaining resources

}
ERL_NIF_INIT(dev_wasi_nn_nif, nif_funcs, load, NULL, NULL, unload)