#include "../include/wasi_nn_llamacpp.h"
#include "../include/wasi_nn_logging.h"
#define LIB_PATH "./native/wasi_nn_llama/libwasi_nn_llamacpp.so"
#define MAX_MODEL_PATH 256
#define MAX_CONFIG_SIZE 1024
#define MAX_OUTPUT_SIZE 8192
#include <sys/resource.h>
#include <unistd.h>
#include <malloc.h>
typedef struct {
    void* ctx;
    graph g;
    graph_execution_context exec_ctx;
} LlamaContext;

static wasi_nn_backend_api g_wasi_nn_functions = {0};
static ErlNifResourceType* llama_context_resource;
static void print_memory_usage(const char* tag) {
	struct mallinfo mi = mallinfo();
	
    printf("  Total allocated space: %d bytes\n", mi.uordblks);
    printf("  Total free space: %d bytes\n", mi.fordblks);
    printf("  Top-most memory: %d bytes\n", mi.keepcost);
	struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
	printf("[Memory %s] MaxRSS: %ld KB\n", tag, r_usage.ru_maxrss);
}
static void llama_context_destructor(ErlNifEnv* env, void* obj)
{
	print_memory_usage("Destructor before");
    LlamaContext* ctx = (LlamaContext*)obj;
    if (ctx) {
        // Cleanup backend context
        if (ctx->ctx && ctx->fns.deinit_backend) {
            ctx->fns.deinit_backend(ctx->ctx);
            ctx->ctx = NULL;
        }
        // Cleanup shared library
        if (ctx->fns.handle) {
            dlclose(ctx->fns.handle);
            ctx->fns.handle = NULL;
        }
        // Clear all function pointers
        memset(&ctx->fns, 0, sizeof(WasiNnFunctions));
    }
	print_memory_usage("Destructor end");
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
	print_memory_usage("Load End");
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
    ERL_NIF_TERM ctx_term = enif_make_resource(env, ctx);
    return enif_make_tuple2(env, enif_make_atom(env, "ok"), ctx_term);
}

static ERL_NIF_TERM nif_load_by_name_with_config(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    LlamaContext* ctx;
    char model_path[MAX_MODEL_PATH];
	char config[MAX_CONFIG_SIZE] = "{"
	"\"enable_log\": true,"
	"\"enable_debug_log\": true,"
	"\"n_gpu_layers\": 20,"
	"\"ctx_size\": 1024"
	"}";
	// check context and input
    if (!enif_get_resource(env, argv[0], llama_context_resource, (void**)&ctx) ||
        !enif_get_string(env, argv[1], model_path, sizeof(model_path), ERL_NIF_LATIN1) ) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "invalid_args"));
    }
	// if conifg is provided, use it
	if (argc > 2 && !enif_get_string(env, argv[2], config, sizeof(config), ERL_NIF_LATIN1)) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"),
                              enif_make_atom(env, "invalid_config"));
    }
	printf("Loading model: %s\n", model_path);
    printf("Loading model with config: %s\n", config);
	
    if (g_wasi_nn_functions.load_by_name_with_config(ctx->ctx, model_path, strlen(model_path), 
                                                   config, strlen(config), &ctx->g) != success) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "load_failed"));
    }

    return enif_make_atom(env, "ok");
}

static ERL_NIF_TERM nif_init_execution_context(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    LlamaContext* ctx;
    if (!enif_get_resource(env, argv[0], llama_context_resource, (void**)&ctx) ||
       !enif_get_uint(env, argv[1], &ctx->g)) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "invalid_args"));
    }
    if (g_wasi_nn_functions.init_execution_context(ctx->ctx, ctx->g, &ctx->exec_ctx)!= success) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "init_execution_failed"));
    }
	return enif_make_atom(env, "ok");
}
static ERL_NIF_TERM nif_set_input(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    LlamaContext* ctx;
    if (!enif_get_resource(env, argv[0], llama_context_resource, (void**)&ctx)) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "invalid_args"));
    }
	const char* input_text = "<|system|> You are a  assistant.";
    tensor input_tensor = {
        .data = (uint8_t*)input_text,
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
    printf("Computing response...\n");
    if (g_wasi_nn_functions.compute(ctx->ctx, ctx->exec_ctx)!= success) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "compute_failed"));
    }
	return enif_make_atom(env, "ok");
}
static ERL_NIF_TERM nif_get_output(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    LlamaContext* ctx;
    if (!enif_get_resource(env, argv[0], llama_context_resource, (void**)&ctx)) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "invalid_context"));
    }
    uint8_t output_buffer[8192];  // Increased buffer size
    uint32_t output_size = sizeof(output_buffer);
    if (g_wasi_nn_functions.get_output(ctx->ctx, ctx->exec_ctx, 0, output_buffer, &output_size)!= success) {
        return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_atom(env, "get_output_failed"));
    }
    printf("output: %s\n", output_buffer);
    // Create reference and return
    ERL_NIF_TERM ctx_term = enif_make_resource(env, ctx);
    enif_release_resource(ctx);
    return enif_make_tuple3(env,
        enif_make_atom(env, "ok"),
        ctx_term,
        enif_make_binary(env, output_buffer, output_size));
}

static ErlNifFunc nif_funcs[] = {
    {"init_backend", 0, nif_init_backend},
	{"load_model", 1, nif_load_model},
    {"load_by_name_with_config", 2, nif_load_by_name_with_config},
    {"init_execution_context", 0, nif_init_execution_context},
    {"set_input", 1, nif_set_input},
    {"compute", 0, nif_compute},
    {"get_output", 0, nif_get_output},
};


static void unload(ErlNifEnv* env, void* priv_data)
{
	// The resource destructor will be called automatically for any remaining resources
	print_memory_usage("Unload nif");
}
ERL_NIF_INIT(dev_wasi_nn_nif, nif_funcs, load, NULL, NULL, unload)