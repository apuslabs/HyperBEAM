-module(dev_wasi_nn_nif).
-include("include/hb.hrl").
-include_lib("eunit/include/eunit.hrl").
-hb_debug(print).
-on_load(init/0).

-export([load_model/1, load_model_with_config/2, generate/2, unload_model/1]).


init() ->
    PrivDir = code:priv_dir(hyperbeam),
    ok = erlang:load_nif(filename:join(PrivDir, "wasi_nn_nif"), 0).

load_model(_Path) ->
    erlang:nif_error("NIF library not loaded").

load_model_with_config(_Path, _Config) ->
    erlang:nif_error("NIF library not loaded").

generate(_Context, _Prompt) ->
    erlang:nif_error("NIF library not loaded").

unload_model(_Context) ->
    erlang:nif_error("NIF library not loaded").

load_model_test() ->
	% Initialize the model path - adjust this to your model location
	ModelPath = "/path/to/your/llama/model.gguf",
	?event(ModelPath),
	% Load the model
	{ok, Context} = load_model(ModelPath),
	?assertNotEqual(undefined, Context),

	% Test simple generation
	Prompt = "Once upon a time",
	{ok, Generated} = generate(Context, Prompt),
	?assertNotEqual("", Generated),

	% Clean up
	ok = unload_model(Context).