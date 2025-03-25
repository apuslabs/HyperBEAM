#include "../include/wasi_nn_llamacpp.h"
#include "../include/wasi_nn_logging.h"
#define LIB_PATH "./native/wasi_nn_llama/libwasi_nn_llamacpp.so"

typedef struct {
    void* ctx;
    graph g;
    graph_execution_context exec_ctx;
    WasiNnFunctions fns;
} LlamaContext;

static ErlNifResourceType* llama_context_resource;

static void llama_context_destructor(ErlNifEnv* env, void* obj)
{
    LlamaContext* ctx = (LlamaContext*)obj;
    if (ctx) {
        if (ctx->ctx && ctx->fns.deinit_backend) {
            ctx->fns.deinit_backend(ctx->ctx);
        }
        if (ctx->fns.handle) {
            dlclose(ctx->fns.handle);
        }
    }
}

static int load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info)
{
    printf("Load nif\n");
    llama_context_resource = enif_open_resource_type(env, NULL, "llama_context",
        llama_context_destructor, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);
    return llama_context_resource ? 0 : 1;
}

static ERL_NIF_TERM nif_load_model(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    char model_path[256];
    LlamaContext* ctx = enif_alloc_resource(llama_context_resource, sizeof(LlamaContext));
    if (!ctx) {
        return enif_make_tuple2(env,
            enif_make_atom(env, "error"),
            enif_make_atom(env, "allocation_failed"));
    }

    // Load the shared library
    ctx->fns.handle = dlopen(LIB_PATH, RTLD_LAZY);
    if (!ctx->fns.handle) {
        enif_release_resource(ctx);
        return enif_make_tuple2(env,
            enif_make_atom(env, "error"),
            enif_make_atom(env, "library_load_failed"));
    }

    // Load all required functions
    ctx->fns.init_backend = (init_backend_fn)dlsym(ctx->fns.handle, "init_backend");
    ctx->fns.deinit_backend = (deinit_backend_fn)dlsym(ctx->fns.handle, "deinit_backend");
    ctx->fns.load_by_name = (load_by_name_fn)dlsym(ctx->fns.handle, "load_by_name");
    ctx->fns.init_execution_context = (init_execution_context_fn)dlsym(ctx->fns.handle, "init_execution_context");
    ctx->fns.set_input = (set_input_fn)dlsym(ctx->fns.handle, "set_input");
    ctx->fns.compute = (compute_fn)dlsym(ctx->fns.handle, "compute");
    ctx->fns.get_output = (get_output_fn)dlsym(ctx->fns.handle, "get_output");
	ctx->fns.load_by_name_with_config = (load_by_name_with_config_fn)dlsym(ctx->fns.handle, "load_by_name_with_config");
    // Verify all functions were loaded
    if (!ctx->fns.init_backend || !ctx->fns.deinit_backend || !ctx->fns.load_by_name ||
        !ctx->fns.init_execution_context || !ctx->fns.set_input || !ctx->fns.compute || 
        !ctx->fns.get_output) {
        dlclose(ctx->fns.handle);
        enif_release_resource(ctx);
        return enif_make_tuple2(env,
            enif_make_atom(env, "error"),
            enif_make_atom(env, "function_load_failed"));
    }

    // Initialize the backend
    if (ctx->fns.init_backend(&ctx->ctx) != success) {
        dlclose(ctx->fns.handle);
        enif_release_resource(ctx);
        return enif_make_tuple2(env,
            enif_make_atom(env, "error"),
            enif_make_atom(env, "init_failed"));
    }

    // Get model path from arguments
    if (!enif_get_string(env, argv[0], model_path, sizeof(model_path), ERL_NIF_LATIN1)) {
        ctx->fns.deinit_backend(ctx->ctx);
        dlclose(ctx->fns.handle);
        enif_release_resource(ctx);
        return enif_make_tuple2(env,
            enif_make_atom(env, "error"),
            enif_make_atom(env, "invalid_path"));
    }
	printf("Model path: %s\n", model_path);
    // Load the model
	// Replace the model loading section
	const char* config = "{"
	"\"enable_log\": true,"
	"\"enable_debug_log\": true,"
	"\"n_gpu_layers\": 20,"
	"\"ctx_size\": 1024"
	"}";
    printf("Loading model with config: %s\n", config);
    if (ctx->fns.load_by_name_with_config(ctx->ctx, model_path, strlen(model_path), 
                                         config, strlen(config), &ctx->g) != success) {
        ctx->fns.deinit_backend(ctx->ctx);
        dlclose(ctx->fns.handle);
        enif_release_resource(ctx);
        return enif_make_tuple2(env,
            enif_make_atom(env, "error"),
            enif_make_atom(env, "load_failed"));
    }
	printf("Initialize process: %s\n", model_path);
    // Initialize execution context
    if (ctx->fns.init_execution_context(ctx->ctx, ctx->g, &ctx->exec_ctx) != success) {
        ctx->fns.deinit_backend(ctx->ctx);
        dlclose(ctx->fns.handle);
        enif_release_resource(ctx);
        return enif_make_tuple2(env,
            enif_make_atom(env, "error"),
            enif_make_atom(env, "context_init_failed"));
    }

    // Set input
	const char* input_text = "<|system|> You are a  assistant.";
    tensor input_tensor = {
        .data = (uint8_t*)input_text,
		
    };

    printf("Setting input: %s\n", input_text);
    if (ctx->fns.set_input(ctx->ctx, ctx->exec_ctx, 0, &input_tensor) != success) {
        ctx->fns.deinit_backend(ctx->ctx);
        dlclose(ctx->fns.handle);
        enif_release_resource(ctx);
        return enif_make_tuple2(env,
            enif_make_atom(env, "error"),
            enif_make_atom(env, "set_input_failed"));
    }

    printf("Computing response...\n");
    if (ctx->fns.compute(ctx->ctx, ctx->exec_ctx) != success) {
        ctx->fns.deinit_backend(ctx->ctx);
        dlclose(ctx->fns.handle);
        enif_release_resource(ctx);
        return enif_make_tuple2(env,
            enif_make_atom(env, "error"),
            enif_make_atom(env, "compute_failed"));
    }
    // Get output
    uint8_t output_buffer[8192];  // Increased buffer size
    uint32_t output_size = sizeof(output_buffer);
    printf("Getting output...\n");
    if (ctx->fns.get_output(ctx->ctx, ctx->exec_ctx, 0, output_buffer, &output_size) != success) {
        ctx->fns.deinit_backend(ctx->ctx);
        dlclose(ctx->fns.handle);
        enif_release_resource(ctx);
        return enif_make_tuple2(env,
            enif_make_atom(env, "error"),
            enif_make_atom(env, "get_output_failed"));
    }
	printf("ctx freed\n");
    printf("Response: %s\n", output_buffer);
    // Create reference and return
    ERL_NIF_TERM result = enif_make_resource(env, ctx);
    enif_release_resource(ctx);
	printf("end\n");
    return enif_make_tuple2(env,
        enif_make_atom(env, "ok"),
        result);

	
}

static ErlNifFunc nif_funcs[] = {
    {"load_model", 1, nif_load_model},
};

ERL_NIF_INIT(dev_wasi_nn_nif, nif_funcs, load, NULL, NULL, NULL)