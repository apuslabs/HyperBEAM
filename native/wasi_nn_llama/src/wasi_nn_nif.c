#include "../include/wasi_nn_llamacpp.h"
#include "../include/wasi_nn_logging.h"
#define LIB_PATH "./native/wasi_nn_llama/libwasi_nn_llamacpp.so"
#include <sys/resource.h>
#include <unistd.h>
#include <malloc.h>
typedef struct {
    void* ctx;
    graph g;
    graph_execution_context exec_ctx;
    WasiNnFunctions fns;
} LlamaContext;

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
        // First cleanup execution context if it exists

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
        printf("Destructor: all resources cleaned up\n");
    }
	print_memory_usage("Destructor end");
    printf("Destructor final\n");
}

static int load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info)
{
	print_memory_usage("Load Start");
    printf("Load nif\n");
    llama_context_resource = enif_open_resource_type(env, NULL, "llama_context",
        llama_context_destructor, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);
	print_memory_usage("Load End");
    return llama_context_resource ? 0 : 1;
}

static void unload(ErlNifEnv* env, void* priv_data)
{
	print_memory_usage("Unload Start");
    printf("Unload nif\n");
    // Any global cleanup can be done here
    // The resource destructor will be called automatically for any remaining resources
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
	memset(ctx, 0, sizeof(LlamaContext));

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
    if (ctx->fns.load_by_name_with_config(ctx->ctx, model_path, strlen(model_path),config,strlen(config), 
                                         &ctx->g) != success) {
        ctx->fns.deinit_backend(ctx->ctx);
        dlclose(ctx->fns.handle);
        enif_release_resource(ctx);
        return enif_make_tuple2(env,
            enif_make_atom(env, "error"),
            enif_make_atom(env, "load_failed"));
    }
	print_memory_usage("Loading model");
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
	print_memory_usage("Initialize execution");
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
	print_memory_usage("set_input");
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

	ERL_NIF_TERM output_term;
	unsigned char* output_data = enif_make_new_binary(env, output_size, &output_term);
    memcpy(output_data, output_buffer, output_size);

    printf("Getting output...\n");
    if (ctx->fns.get_output(ctx->ctx, ctx->exec_ctx, 0, output_buffer, &output_size) != success) {
        ctx->fns.deinit_backend(ctx->ctx);
        dlclose(ctx->fns.handle);
        enif_release_resource(ctx);
        return enif_make_tuple2(env,
            enif_make_atom(env, "error"),
            enif_make_atom(env, "get_output_failed"));
    }
    printf("output: %s\n", output_data);
    // Create reference and return
    ERL_NIF_TERM ctx_term = enif_make_resource(env, ctx);
    enif_release_resource(ctx);
	print_memory_usage("Load compute end");
	printf("end\n");
    return enif_make_tuple3(env,
        enif_make_atom(env, "ok"),
        ctx_term,
        output_term);

	
}

static ErlNifFunc nif_funcs[] = {
    {"load_model", 1, nif_load_model},
};

ERL_NIF_INIT(dev_wasi_nn_nif, nif_funcs, load, NULL, NULL, unload)