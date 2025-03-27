-module(dev_wasi_nn_nif).
-include("include/hb.hrl").
-include_lib("eunit/include/eunit.hrl").
-hb_debug(print).
-on_load(init/0).

-export([load_model/1, load_model_with_config/2, generate/2, unload_model/1]).


cleanup() ->
    io:format("Cleanup called~n"),
    % Force garbage collection
    erlang:garbage_collect(),
    % Print process info
    io:format("Process memory: ~p~n", [erlang:process_info(self(), memory)]),
    ok.

init() ->
    PrivDir = code:priv_dir(hb),
    Path = filename:join(PrivDir, "wasi_nn"),
    io:format("Loading NIF from: ~p~n", [Path]),
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

load_model(_Path) ->
    erlang:nif_error("NIF library not loaded").

load_model_with_config(_Path, _Config) ->
    erlang:nif_error("NIF library not loaded").

generate(_Context, _Prompt) ->
    erlang:nif_error("NIF library not loaded").

unload_model(_Context) ->
    erlang:nif_error("NIF library not loaded").

load_model_test() ->

	process_flag(trap_exit, true),  % Trap exits to handle crashes
    Self = self(),
    
    % Start a monitor process
    spawn_link(fun() ->
        monitor(process, Self),
        receive
			{'DOWN', _, process, _, _} ->
				cleanup(),
				% Force NIF cleanup
				code:purge(?MODULE),
				code:delete(?MODULE),
				io:format("Test process terminated~n")
        end
    end),
	% Skip test if model doesn't exist
	ModelPath = "test/qwen1_5-0_5b-chat-q2_k.gguf",
	case filelib:is_regular(ModelPath) of
		true ->
			?event(ModelPath),
			% Load the model
			{ok, Context} = load_model(ModelPath),
			try
				?assertNotEqual(undefined, Context),

				% Test simple generation
				Prompt = "Once upon a time",
				{ok, Generated} = generate(Context, Prompt),
				?assertNotEqual("", Generated)
			after
				cleanup(),
				% Clean up will happen even if test fails
				ok = unload_model(Context)
			end;
		false ->
			?debugMsg("Skipping test - model file not found")
	end.