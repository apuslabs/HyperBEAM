-module(dev_wasi_nn_nif).
-include("include/hb.hrl").
-include_lib("eunit/include/eunit.hrl").
-on_load(init/0).
-export([
    init_backend/0,
    load_by_name_with_config/3,
    init_execution_context/1,
    deinit_backend/1,
    run_inference/2
]).
-export([load_by_name_with_config_once/3, init_execution_context_once/1]).
-export([start_cache_owner/0, cache_owner_loop/0]).  % Export cache owner process functions

%% Module-level cache
-define(CACHE_TAB, wasi_nn_cache).
-define(SINGLETON_KEY, global_cache).
-define(CACHE_OWNER_NAME, wasi_nn_cache_owner).  % Registered name for cache owner process

%% Function to start the dedicated ETS table owner process
start_cache_owner() ->
    case whereis(?CACHE_OWNER_NAME) of
        undefined ->
            % No owner process exists, create one
            Pid = spawn(fun() -> 
                % Create the table if it doesn't exist
                case ets:info(?CACHE_TAB) of
                    undefined ->
                        io:format("Cache owner creating table ~p~n", [?CACHE_TAB]),
                        ets:new(?CACHE_TAB, [set, named_table, public]);
                    _ ->
                        io:format("Cache table ~p already exists, taking ownership~n", [?CACHE_TAB])
                end,
                % Register the process with a name for easy lookup
                register(?CACHE_OWNER_NAME, self()),
                cache_owner_loop()
            end),
            {ok, Pid};
        Pid ->
            % Owner process already exists
            {ok, Pid}
    end.

%% Loop function for the cache owner process - keeps the process alive
cache_owner_loop() ->
    receive
        stop -> 
            io:format("Cache owner stopping~n"),
            ok;
        {From, ping} ->
            From ! {self(), pong},
            cache_owner_loop();
        _ -> 
            cache_owner_loop()
    after 
        3600000 -> % Stay alive for a long time (1 hour), then check again
            cache_owner_loop()
    end.

%% Create ETS table in a persistent process if it doesn't exist
init() ->
    PrivDir = code:priv_dir(hb),
    Path = filename:join(PrivDir, "wasi_nn"),
    io:format("Loading NIF from: ~p~n", [Path]),
    
    % Start the dedicated cache owner process
    start_cache_owner(),
    
    % No need to create the table here, the owner process handles this
    case erlang:load_nif(Path, 0) of
        ok ->
            io:format("NIF loaded successfully~n"),
            ok;
        {error, {load_failed, Reason}} ->
            io:format("Failed to load NIF: ~p~n", [Reason]),
            exit({load_failed, {load_failed, Reason}});
        {error, Reason} ->
            io:format("Failed to load NIF with error: ~p~n", [Reason]),
            exit({load_failed, Reason})
    end.

init_backend() ->
    erlang:nif_error("NIF library not loaded").

load_by_name_with_config(_Context, _Path, _Config) ->
    erlang:nif_error("NIF library not loaded").

init_execution_context(_Context) ->
    erlang:nif_error("NIF library not loaded").

% set_input(_Context, _Prompt) ->
%     erlang:nif_error("NIF library not loaded").

% compute(_Context) ->
%     erlang:nif_error("NIF library not loaded").
% get_output(_Context) ->
%     erlang:nif_error("NIF library not loaded").
deinit_backend(_Context) ->
    erlang:nif_error("NIF library not loaded").
run_inference(_Context, _Prompt) ->
    erlang:nif_error("NIF library not loaded").

%% Helper function to safely access the ETS table
ensure_cache_table() ->
    case ets:info(?CACHE_TAB) of
        undefined ->
            % Start the cache owner which will create the table
            io:format("Table doesn't exist, starting cache owner process~n"),
            start_cache_owner();
        _ ->
            % Table exists, ensure owner process is running
            case whereis(?CACHE_OWNER_NAME) of
                undefined ->
                    % Strange case: table exists but no owner - restart owner
                    io:format("Table exists but no owner, restarting owner process~n"),
                    start_cache_owner();
                _ ->
                    % All good, table exists and owner is running
                    ok
            end
    end.

%% Function to ensure model is only loaded once globally
load_by_name_with_config_once(_Context, Path, Config) ->
    ensure_cache_table(),
    
    % Check if this model is already loaded globally
    case ets:lookup(?CACHE_TAB, {?SINGLETON_KEY, model_loaded}) of
        [{_, {ok, StoredContext}}] ->
            io:format("Model already loaded globally~n"),
            {ok, StoredContext};
        [] ->
            InitResult =
                case ets:lookup(?CACHE_TAB, {?SINGLETON_KEY, backend_init}) of
                    [{_, {ok, ExistingContext}}] ->
                        {ok, ExistingContext};
                    [] ->
                        try
                            BackendResult = init_backend(),
                            case BackendResult of
                                {ok, _NewContext} ->
                                    ets:insert(?CACHE_TAB, {
                                        {?SINGLETON_KEY, backend_init}, BackendResult
                                    }),
                                    BackendResult;
                                Error ->
                                    Error
                            end
                        catch
                            _:UnknownError ->
                                {error, {backend_init_failed, UnknownError}}
                        end
                end,

            case InitResult of
                {ok, FinalContext} ->
                    ?event({load_model, FinalContext, Path, Config}),
                    try
                        LoadResult = load_by_name_with_config(FinalContext, Path, Config),
                        case LoadResult of
                            ok ->
                                ets:insert(?CACHE_TAB, {
                                    {?SINGLETON_KEY, model_loaded}, {ok, FinalContext}
                                }),
								?event({model_load_cached, FinalContext}),
                                {ok, FinalContext};
                            LoadError ->
                                ets:delete(?CACHE_TAB, {?SINGLETON_KEY, backend_init}),
                                {error, {model_load_failed, LoadError}}
                        end
                    catch
                        _:UnknownError2 ->
                            ets:delete(?CACHE_TAB, {?SINGLETON_KEY, backend_init}),
                            {error, {model_load_failed, UnknownError2}}
                    end;
                InitError ->
                    InitError
            end
    end.

%% Function to ensure execution context is only initialized once globally
init_execution_context_once(Context) ->
    ensure_cache_table(),
    case ets:lookup(?CACHE_TAB, {?SINGLETON_KEY, context_initialized}) of
        [{_, ok}] ->
            io:format("Execution context already initialized globally~n"),
            ok;
        [] ->
            ModelContext =
                case ets:lookup(?CACHE_TAB, {?SINGLETON_KEY, model_loaded}) of
                    [{_, {ok, StoredContext}}] ->
                        StoredContext;
                    [] ->
                        % If we don't have a cached context, use the provided one
                        % This is a fallback but ideally load_by_name_with_config_once should be called first
                        Context
                end,

            Result = init_execution_context(ModelContext),
            case Result of
                ok ->
                    ets:insert(?CACHE_TAB, {{?SINGLETON_KEY, context_initialized}, ok}),
                    ok;
                Error ->
                    Error
            end
    end.

run_inference_test() ->
    Path = "test/qwen2.5-14b-instruct-q2_k.gguf",
    Config =
        "{\"n_gpu_layers\":98,\"ctx_size\":2048,\"stream-stdout\":true,\"enable_debug_log\":true}",
    Prompt1 = "What is the meaning of life",
    {ok, Context} = init_backend(),
    ?event({load_model, Context, Path, Config}),
    load_by_name_with_config(Context, Path, Config),
    init_execution_context(Context),
    {ok, Output1} = run_inference(Context, Prompt1),
	?event({run_inference, Context, Prompt1, Output1}),
    ?assertNotEqual(Output1, ""),
    Prompt2 = "Who are you",
    {ok, Output2} = run_inference(Context, Prompt2),
	?event({run_inference, Context, Prompt2, Output2}),
    ?assertNotEqual(Output2, ""),
    deinit_backend(Context).
