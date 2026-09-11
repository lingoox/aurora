package official

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestNewChatCompletionWithMetadata(t *testing.T) {
	response := NewChatCompletionWithMetadata(
		"hello",
		1,
		2,
		"gpt-4o",
		"conv-xxx",
		[]map[string]interface{}{
			{
				"event":      "artifact",
				"kind":       "generated_image",
				"slot_index": 1,
				"url":        "http://example.test/image.png",
			},
			{
				"event":      "artifact_slot_final",
				"kind":       "generated_image",
				"slot_index": 1,
			},
		},
	)

	data, err := json.Marshal(response)
	if err != nil {
		t.Fatalf("marshal response: %v", err)
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}

	if raw["conversation_id"] != "conv-xxx" {
		t.Fatalf("conversation_id = %#v, want conv-xxx", raw["conversation_id"])
	}
	choices := raw["choices"].([]interface{})
	message := choices[0].(map[string]interface{})["message"].(map[string]interface{})
	if message["content"] != "hello" {
		t.Fatalf("message content = %#v, want hello", message["content"])
	}
	sentinel := raw["sentinel"].([]interface{})
	if len(sentinel) != 2 {
		t.Fatalf("sentinel count = %d, want 2", len(sentinel))
	}
	if sentinel[0].(map[string]interface{})["event"] != "artifact" {
		t.Fatalf("first sentinel = %#v, want artifact", sentinel[0])
	}
}

func TestNewResponsesResponseWithReasoning(t *testing.T) {
	resp := NewResponsesResponse("hello", "thinking...", 100, 50, 30, 80, 20, "auto")
	if resp.Object != "response" {
		t.Fatalf("object = %q, want response", resp.Object)
	}
	if resp.OutputText != "hello" {
		t.Fatalf("output_text = %q, want hello", resp.OutputText)
	}
	b, _ := json.Marshal(resp)
	s := string(b)
	for _, want := range []string{
		`"input_tokens":100`,
		`"cached_tokens":80`,
		`"cache_write_tokens":20`,
		`"reasoning_tokens":30`,
		`"type":"reasoning"`,
		`"reasoning_text"`,
		`"reasoning_content":"thinking..."`,
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("missing %q in %s", want, s)
		}
	}
}

func TestNewResponsesResponseWithoutReasoning(t *testing.T) {
	resp := NewResponsesResponse("hi", "", 10, 5, 0, 0, 0, "auto")
	if len(resp.Output) != 1 {
		t.Fatalf("output len = %d, want 1", len(resp.Output))
	}
	if resp.Output[0].Type != "message" {
		t.Fatalf("first output type = %q, want message", resp.Output[0].Type)
	}
}

// 验证 Responses API 的 tools/tool_choice 解析与 function_call item 映射。
func TestResponsesAPIRequestToolsMapping(t *testing.T) {
	raw := `{
		"model": "gpt-4o-mini",
		"instructions": "You are a helpful agent.",
		"tools": [
			{"type": "function", "name": "get_weather", "description": "查询天气", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}
		],
		"tool_choice": "required",
		"input": [
			{"role": "user", "content": "北京天气怎么样"},
			{"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"北京\"}"},
			{"type": "function_call_output", "call_id": "call_1", "output": "22度,晴"}
		]
	}`
	var req ResponsesAPIRequest
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	api, err := req.ToAPIRequest()
	if err != nil {
		t.Fatalf("ToAPIRequest: %v", err)
	}
	// tools 平铺 → 嵌套
	if len(api.Tools) != 1 || api.Tools[0].Function.Name != "get_weather" {
		t.Fatalf("tools mapping wrong: %+v", api.Tools)
	}
	if api.ToolChoice == nil || api.ToolChoice.Type != "required" {
		t.Fatalf("tool_choice wrong: %+v", api.ToolChoice)
	}
	// instructions → system 消息在最前
	if api.Messages[0].Role != "system" {
		t.Fatalf("first msg role = %q", api.Messages[0].Role)
	}
	// user 消息
	if api.Messages[1].Role != "user" {
		t.Fatalf("msg[1] role = %q", api.Messages[1].Role)
	}
	// function_call → assistant + ToolCalls
	m2 := api.Messages[2]
	if m2.Role != "assistant" || len(m2.ToolCalls) != 1 {
		t.Fatalf("function_call mapping wrong: %+v", m2)
	}
	if m2.ToolCalls[0].ID != "call_1" || m2.ToolCalls[0].Function.Name != "get_weather" {
		t.Fatalf("call ref wrong: %+v", m2.ToolCalls[0])
	}
	// function_call_output → role=tool
	m3 := api.Messages[3]
	if m3.Role != "tool" || m3.ToolCallID != "call_1" {
		t.Fatalf("function_call_output mapping wrong: %+v", m3)
	}
	if m3.Content.Text() != "22度,晴" {
		t.Fatalf("output text wrong: %q", m3.Content.Text())
	}
}
