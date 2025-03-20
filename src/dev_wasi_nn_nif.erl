-module(dev_wasi_nn_nif).
-include("include/hb.hrl").
-include_lib("eunit/include/eunit.hrl").
-hb_debug(print).
-on_load(init/0).

-export([load_model/1, load_model_with_config/2, generate/2, unload_model/1]).


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
	% Skip test if model doesn't exist
	ModelPath = "test/test.wasm", % Use a path that exists or can be created
	case filelib:is_regular(ModelPath) of
		true ->
			?event(ModelPath),
			% Load the model
			{ok, Context} = load_model(ModelPath),
			?assertNotEqual(undefined, Context),

			% Test simple generation
			Prompt = "Once upon a time",
			{ok, Generated} = generate(Context, Prompt),
			?assertNotEqual("", Generated),

			% Clean up
			ok = unload_model(Context);
		false ->
			?debugMsg("Skipping test - model file not found")
	end.