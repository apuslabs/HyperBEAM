

# Module dev_meta #
* [Description](#description)

The hyperbeam meta device, which is the default entry point
for all messages processed by the machine.

<a name="description"></a>

## Description ##
This device executes a
AO-Core singleton request, after first applying the node's
pre-processor, if set. The pre-processor can halt the request by
returning an error, or return a modified version if it deems necessary --
the result of the pre-processor is used as the request for the AO-Core
resolver. Additionally, a post-processor can be set, which is executed after
the AO-Core resolver has returned a result.<a name="index"></a>

## Function Index ##


<table width="100%" border="1" cellspacing="0" cellpadding="2" summary="function index"><tr><td valign="top"><a href="#add_dynamic_keys-1">add_dynamic_keys/1*</a></td><td>Add dynamic keys to the node message.</td></tr><tr><td valign="top"><a href="#adopt_node_message-2">adopt_node_message/2</a></td><td>Attempt to adopt changes to a node message.</td></tr><tr><td valign="top"><a href="#authorized_set_node_msg_succeeds_test-0">authorized_set_node_msg_succeeds_test/0*</a></td><td>Test that we can set the node message if the request is signed by the
owner of the node.</td></tr><tr><td valign="top"><a href="#claim_node_test-0">claim_node_test/0*</a></td><td>Test that we can claim the node correctly and set the node message after.</td></tr><tr><td valign="top"><a href="#config_test-0">config_test/0*</a></td><td>Test that we can get the node message.</td></tr><tr><td valign="top"><a href="#embed_status-1">embed_status/1*</a></td><td>Wrap the result of a device call in a status.</td></tr><tr><td valign="top"><a href="#filter_node_msg-1">filter_node_msg/1*</a></td><td>Remove items from the node message that are not encodable into a
message.</td></tr><tr><td valign="top"><a href="#halt_request_test-0">halt_request_test/0*</a></td><td>Test that we can halt a request if the preprocessor returns an error.</td></tr><tr><td valign="top"><a href="#handle-2">handle/2</a></td><td>Normalize and route messages downstream based on their path.</td></tr><tr><td valign="top"><a href="#handle_initialize-2">handle_initialize/2*</a></td><td></td></tr><tr><td valign="top"><a href="#handle_resolve-3">handle_resolve/3*</a></td><td>Handle an AO-Core request, which is a list of messages.</td></tr><tr><td valign="top"><a href="#info-1">info/1</a></td><td>Ensure that the helper function <code>adopt_node_message/2</code> is not exported.</td></tr><tr><td valign="top"><a href="#info-3">info/3</a></td><td>Get/set the node message.</td></tr><tr><td valign="top"><a href="#maybe_sign-2">maybe_sign/2*</a></td><td>Sign the result of a device call if the node is configured to do so.</td></tr><tr><td valign="top"><a href="#message_to_status-1">message_to_status/1*</a></td><td>Get the HTTP status code from a transaction (if it exists).</td></tr><tr><td valign="top"><a href="#modify_request_test-0">modify_request_test/0*</a></td><td>Test that a preprocessor can modify a request.</td></tr><tr><td valign="top"><a href="#permanent_node_message_test-0">permanent_node_message_test/0*</a></td><td>Test that a permanent node message cannot be changed.</td></tr><tr><td valign="top"><a href="#priv_inaccessible_test-0">priv_inaccessible_test/0*</a></td><td>Test that we can't get the node message if the requested key is private.</td></tr><tr><td valign="top"><a href="#resolve_processor-5">resolve_processor/5*</a></td><td>Execute a message from the node message upon the user's request.</td></tr><tr><td valign="top"><a href="#status_code-1">status_code/1*</a></td><td>Calculate the appropriate HTTP status code for an AO-Core result.</td></tr><tr><td valign="top"><a href="#unauthorized_set_node_msg_fails_test-0">unauthorized_set_node_msg_fails_test/0*</a></td><td>Test that we can't set the node message if the request is not signed by
the owner of the node.</td></tr><tr><td valign="top"><a href="#uninitialized_node_test-0">uninitialized_node_test/0*</a></td><td>Test that an uninitialized node will not run computation.</td></tr><tr><td valign="top"><a href="#update_node_message-2">update_node_message/2*</a></td><td>Validate that the request is signed by the operator of the node, then
allow them to update the node message.</td></tr></table>


<a name="functions"></a>

## Function Details ##

<a name="add_dynamic_keys-1"></a>

### add_dynamic_keys/1 * ###

`add_dynamic_keys(NodeMsg) -> any()`

Add dynamic keys to the node message.

<a name="adopt_node_message-2"></a>

### adopt_node_message/2 ###

`adopt_node_message(Request, NodeMsg) -> any()`

Attempt to adopt changes to a node message.

<a name="authorized_set_node_msg_succeeds_test-0"></a>

### authorized_set_node_msg_succeeds_test/0 * ###

`authorized_set_node_msg_succeeds_test() -> any()`

Test that we can set the node message if the request is signed by the
owner of the node.

<a name="claim_node_test-0"></a>

### claim_node_test/0 * ###

`claim_node_test() -> any()`

Test that we can claim the node correctly and set the node message after.

<a name="config_test-0"></a>

### config_test/0 * ###

`config_test() -> any()`

Test that we can get the node message.

<a name="embed_status-1"></a>

### embed_status/1 * ###

`embed_status(X1) -> any()`

Wrap the result of a device call in a status.

<a name="filter_node_msg-1"></a>

### filter_node_msg/1 * ###

`filter_node_msg(Msg) -> any()`

Remove items from the node message that are not encodable into a
message.

<a name="halt_request_test-0"></a>

### halt_request_test/0 * ###

`halt_request_test() -> any()`

Test that we can halt a request if the preprocessor returns an error.

<a name="handle-2"></a>

### handle/2 ###

`handle(NodeMsg, RawRequest) -> any()`

Normalize and route messages downstream based on their path. Messages
with a `Meta` key are routed to the `handle_meta/2` function, while all
other messages are routed to the `handle_resolve/2` function.

<a name="handle_initialize-2"></a>

### handle_initialize/2 * ###

`handle_initialize(Rest, NodeMsg) -> any()`

<a name="handle_resolve-3"></a>

### handle_resolve/3 * ###

`handle_resolve(Req, Msgs, NodeMsg) -> any()`

Handle an AO-Core request, which is a list of messages. We apply
the node's pre-processor to the request first, and then resolve the request
using the node's AO-Core implementation if its response was `ok`.
After execution, we run the node's `postprocessor` message on the result of
the request before returning the result it grants back to the user.

<a name="info-1"></a>

### info/1 ###

`info(X1) -> any()`

Ensure that the helper function `adopt_node_message/2` is not exported.
The naming of this method carefully avoids a clash with the exported `info/3`
function. We would like the node information to be easily accessible via the
`info` endpoint, but AO-Core also uses `info` as the name of the function
that grants device information. The device call takes two or fewer arguments,
so we are safe to use the name for both purposes in this case, as the user
info call will match the three-argument version of the function. If in the
future the `request` is added as an argument to AO-Core's internal `info`
function, we will need to find a different approach.

<a name="info-3"></a>

### info/3 ###

`info(X1, Request, NodeMsg) -> any()`

Get/set the node message. If the request is a `POST`, we check that the
request is signed by the owner of the node. If not, we return the node message
as-is, aside all keys that are private (according to `hb_private`).

<a name="maybe_sign-2"></a>

### maybe_sign/2 * ###

`maybe_sign(Res, NodeMsg) -> any()`

Sign the result of a device call if the node is configured to do so.

<a name="message_to_status-1"></a>

### message_to_status/1 * ###

`message_to_status(Item) -> any()`

Get the HTTP status code from a transaction (if it exists).

<a name="modify_request_test-0"></a>

### modify_request_test/0 * ###

`modify_request_test() -> any()`

Test that a preprocessor can modify a request.

<a name="permanent_node_message_test-0"></a>

### permanent_node_message_test/0 * ###

`permanent_node_message_test() -> any()`

Test that a permanent node message cannot be changed.

<a name="priv_inaccessible_test-0"></a>

### priv_inaccessible_test/0 * ###

`priv_inaccessible_test() -> any()`

Test that we can't get the node message if the requested key is private.

<a name="resolve_processor-5"></a>

### resolve_processor/5 * ###

`resolve_processor(PathKey, Processor, Req, Query, NodeMsg) -> any()`

Execute a message from the node message upon the user's request. The
invocation of the processor provides a request of the following form:

```

       /path => preprocess | postprocess
       /request => the original request singleton
       /body => list of messages the user wishes to process
```

<a name="status_code-1"></a>

### status_code/1 * ###

`status_code(X1) -> any()`

Calculate the appropriate HTTP status code for an AO-Core result.
The order of precedence is:
1. The status code from the message.
2. The HTTP representation of the status code.
3. The default status code.

<a name="unauthorized_set_node_msg_fails_test-0"></a>

### unauthorized_set_node_msg_fails_test/0 * ###

`unauthorized_set_node_msg_fails_test() -> any()`

Test that we can't set the node message if the request is not signed by
the owner of the node.

<a name="uninitialized_node_test-0"></a>

### uninitialized_node_test/0 * ###

`uninitialized_node_test() -> any()`

Test that an uninitialized node will not run computation.

<a name="update_node_message-2"></a>

### update_node_message/2 * ###

`update_node_message(Request, NodeMsg) -> any()`

Validate that the request is signed by the operator of the node, then
allow them to update the node message.

