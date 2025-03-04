%%% @doc A wasi-nn implementation device.

-module(dev_wasi_nn).
-export([init/3, compute/1]).
% -export([set_input/3, get_output/3, load_by_name/3, load_by_name_with_config/3, init_execution_context/3, compute/3]).
-include("include/hb.hrl").
-include_lib("eunit/include/eunit.hrl").
-hb_debug(print).

init(M1, _M2, Opts) ->
    ?event(running_init),
    MsgWithLib =
        hb_converge:set(
            M1,
            #{
                <<"wasm/stdlib/wasi_ephemeral_nn">> =>
                    #{ <<"device">> => <<"WASI-NN/1.0">>}
            },
            Opts
        ),
    {ok, MsgWithLib}.

compute(Msg1) ->
    {ok, Msg1}.

% set_input(Msg1, Msg2, Opts) ->
% 	{ok, #{ state => hb_converge:get(<<"State">>, Msg1, Opts), wasm_response => [0] }}.

% get_output(Msg1, Msg2, Opts) ->
% 	{ok, #{ state => hb_converge:get(<<"State">>, Msg1, Opts), wasm_response => [0] }}.

% load_by_name(Msg1, Msg2, Opts) ->
% 	{ok, #{ state => hb_converge:get(<<"State">>, Msg1, Opts), wasm_response => [0] }}.

% load_by_name_with_config(Msg1, Msg2, Opts) ->
% 	{ok, #{ state => hb_converge:get(<<"State">>, Msg1, Opts), wasm_response => [0] }}.

% init_execution_context(Msg1, Msg2, Opts) ->
% 	{ok, #{ state => hb_converge:get(<<"State">>, Msg1, Opts), wasm_response => [0] }}.

% compute(Msg1, Msg2, Opts) ->
% 	{ok, #{ state => hb_converge:get(<<"State">>, Msg1, Opts), wasm_response => [0] }}.

%%% Tests

init() ->
    application:ensure_all_started(hb).

generate_wasi_nn_stack(File, Func, Params) ->
    init(),
    Msg0 = dev_wasm:cache_wasm_image(File),
    Msg1 = Msg0#{
        <<"device">> => <<"Stack@1.0">>,
        <<"device-stack">> => [<<"WASI@1.0">>, <<"WASM-64@1.0">>],
        <<"output-prefixes">> => [<<"wasm">>, <<"wasm">>],
        <<"stack-keys">> => [<<"init">>, <<"compute">>],
        <<"wasm-function">> => Func,
        <<"wasm-params">> => Params
    },
    {ok, Msg2} = hb_converge:resolve(Msg1, <<"init">>, #{}),
    Msg2.

qwen_test() ->
	Init = generate_wasi_nn_stack("test/wasmedge-ggml-qwen.wasm", <<"run_inference">>, []),
	Instance = hb_private:get(<<"wasm/instance">>, Init, #{}),
	% Prompt = <<"Hello, im Jax">>,
    % {ok, Ptr1} = hb_beamr_io:malloc(Instance, byte_size(Prompt)),
    % ?assertNotEqual(0, Ptr1),
    % hb_beamr_io:write(Instance, Ptr1, Prompt),
	% Ready = Init#{ <<"wasm-params">> => [Ptr1] },
	Ready = Init,
    {ok, StateRes} = hb_converge:resolve(Ready, <<"compute">>, #{}),
    [Ptr] = hb_converge:get(<<"results/wasm/output">>, StateRes),
    {ok, Output} = hb_beamr_io:read_string(Instance, Ptr),
    ?event({got_output, Output}).

