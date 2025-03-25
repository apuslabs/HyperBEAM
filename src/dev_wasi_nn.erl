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
	Ready = Init,
    {ok, StateRes} = hb_converge:resolve(Ready, <<"compute">>, #{}),
    [Ptr] = hb_converge:get(<<"results/wasm/output">>, StateRes),
    {ok, Output} = hb_beamr_io:read_string(Instance, Ptr),
    ?event({got_output, Output}).

aos_wasi_nn_exec_test() ->
	Init = generate_wasi_nn_stack("test/aos-wasi-nn.wasm", <<"lib_main">>, []),
	Instance = hb_private:get(<<"wasm/instance">>, Init, #{}),
	Ready = Init,
    {ok, StateRes} = hb_converge:resolve(Ready, <<"compute">>, #{}),
    [Ptr] = hb_converge:get(<<"results/wasm/output">>, StateRes),
    {ok, Output} = hb_beamr_io:read_string(Instance, Ptr),
    ?event({got_output, Output}).

% aos_wasi_nn_test() ->
%     Init = generate_wasi_nn_stack("test/aos-wasi-nn.wasm", <<"handle">>, []),
%     Msg = gen_test_aos_msg("local wasinn = require(\"wasi-nn\"); wasinn.run_inference(\"Hello\");"),
%     Env = gen_test_env(),
%     Instance = hb_private:get(<<"wasm/instance">>, Init, #{}),
%     {ok, Ptr1} = hb_beamr_io:malloc(Instance, byte_size(Msg)),
%     ?assertNotEqual(0, Ptr1),
%     hb_beamr_io:write(Instance, Ptr1, Msg),
%     {ok, Ptr2} = hb_beamr_io:malloc(Instance, byte_size(Env)),
%     ?assertNotEqual(0, Ptr2),
%     hb_beamr_io:write(Instance, Ptr2, Env),
%     % Read the strings to validate they are correctly passed
%     {ok, MsgBin} = hb_beamr_io:read(Instance, Ptr1, byte_size(Msg)),
%     {ok, EnvBin} = hb_beamr_io:read(Instance, Ptr2, byte_size(Env)),
%     ?assertEqual(Env, EnvBin),
%     ?assertEqual(Msg, MsgBin),
%     Ready = Init#{ <<"wasm-params">> => [Ptr1, Ptr2] },
%     {ok, StateRes} = hb_converge:resolve(Ready, <<"compute">>, #{}),
%     [Ptr] = hb_converge:get(<<"results/wasm/output">>, StateRes),
%     {ok, Output} = hb_beamr_io:read_string(Instance, Ptr),
%     ?event({got_output, Output}),
%     #{ <<"response">> := #{ <<"Output">> := #{ <<"data">> := Data }} }
%         = jiffy:decode(Output, [return_maps]),
%     ?assertEqual(<<"2">>, Data).


%%% Test Helpers
gen_test_env() ->
    <<"{\"Process\":{\"Id\":\"AOS\",\"Owner\":\"FOOBAR\",\"Tags\":[{\"name\":\"Name\",\"value\":\"Thomas\"}, {\"name\":\"Authority\",\"value\":\"FOOBAR\"}]}}\0">>.

gen_test_aos_msg(Command) ->
    <<"{\"From\":\"FOOBAR\",\"Block-Height\":\"1\",\"Target\":\"AOS\",\"Owner\":\"FOOBAR\",\"Id\":\"1\",\"Module\":\"W\",\"Tags\":[{\"name\":\"Action\",\"value\":\"Eval\"}],\"Data\":\"", (list_to_binary(Command))/binary, "\"}\0">>.