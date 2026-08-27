//! A JSON-RPC 2.0 stdio client for the four MCP tools agentbbs pins.
//!
//! agentbbs's own MCP transport (`agentbbs-mcp/src/transport.rs::serve_stdio`)
//! frames JSON-RPC 2.0 requests/responses newline-delimited over stdio, and
//! its own client (`agentbbs-mcp/src/client.rs`) is `tokio`-async. This
//! module transcribes the same JSON-RPC request/response shapes
//! (`agentbbs-mcp/src/jsonrpc.rs`) but implements the client synchronously,
//! generic over any [`std::io::Read`] + [`std::io::Write`] pair — a
//! `std::process::Child`'s stdio for a real `agentbbs mcp` subprocess in
//! production, or an in-memory `Cursor`/pipe pair in tests, with no async
//! runtime dependency either way.
//!
//! Method surface: `initialize` (handshake, `protocolVersion: "2024-11-05"`
//! per `agentbbs-mcp/src/server.rs:30`), `tools/list`, and `tools/call` for
//! the four pinned tools — `list_boards`, `read_board`, `post_message`,
//! `search_memory` (`server.rs:132,141,154,168`).

use std::io::{BufRead, BufReader, Write};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::wire::MCP_PROTOCOL_VERSION;

/// `agentbbs-mcp/src/jsonrpc.rs:12` — the JSON-RPC protocol version string.
pub const JSONRPC_VERSION: &str = "2.0";

/// `agentbbs-mcp/src/jsonrpc.rs:15-28` — standard JSON-RPC error codes (the
/// subset agentbbs defines).
pub mod codes {
    pub const PARSE_ERROR: i64 = -32700;
    pub const INVALID_REQUEST: i64 = -32600;
    pub const METHOD_NOT_FOUND: i64 = -32601;
    pub const INVALID_PARAMS: i64 = -32602;
    pub const INTERNAL_ERROR: i64 = -32603;
    pub const APPLICATION_ERROR: i64 = -32000;
}

/// `agentbbs-mcp/src/jsonrpc.rs:31-44` — a JSON-RPC 2.0 request.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Request {
    #[serde(default = "default_version")]
    pub jsonrpc: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

fn default_version() -> String {
    JSONRPC_VERSION.to_string()
}

impl Request {
    pub fn new(id: impl Into<Value>, method: impl Into<String>, params: Value) -> Self {
        Request {
            jsonrpc: JSONRPC_VERSION.to_string(),
            id: Some(id.into()),
            method: method.into(),
            params,
        }
    }
}

/// `agentbbs-mcp/src/jsonrpc.rs:68-97` — a JSON-RPC 2.0 error object.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// `agentbbs-mcp/src/jsonrpc.rs:100-134` — a JSON-RPC 2.0 response. Exactly
/// one of `result` / `error` is set.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Response {
    pub jsonrpc: String,
    pub id: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

/// Errors from the stdio MCP client.
#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    /// The peer closed the stream before a response line arrived.
    #[error("connection closed")]
    Closed,
    /// The peer returned a JSON-RPC error object.
    #[error("rpc error {}: {}", .0.code, .0.message)]
    Rpc(RpcError),
}

/// A synchronous, newline-delimited JSON-RPC 2.0 client over any
/// [`std::io::Read`] + [`std::io::Write`] pair. `R`/`W` are split so the
/// client works directly against a `std::process::Child`'s
/// `(ChildStdout, ChildStdin)` without needing a combined duplex stream.
pub struct McpStdioClient<R: std::io::Read, W: Write> {
    reader: BufReader<R>,
    writer: W,
    next_id: i64,
}

impl<R: std::io::Read, W: Write> McpStdioClient<R, W> {
    pub fn new(reader: R, writer: W) -> Self {
        McpStdioClient {
            reader: BufReader::new(reader),
            writer,
            next_id: 1,
        }
    }

    fn alloc_id(&mut self) -> i64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Send one request and block for its matching response line.
    fn request(&mut self, method: &str, params: Value) -> Result<Value, ClientError> {
        let id = self.alloc_id();
        let req = Request::new(id, method, params);
        let mut line = serde_json::to_vec(&req)?;
        line.push(b'\n');
        self.writer.write_all(&line)?;
        self.writer.flush()?;

        let mut resp_line = String::new();
        let read = self.reader.read_line(&mut resp_line)?;
        if read == 0 {
            return Err(ClientError::Closed);
        }
        let resp: Response = serde_json::from_str(resp_line.trim_end())?;
        if let Some(err) = resp.error {
            return Err(ClientError::Rpc(err));
        }
        Ok(resp.result.unwrap_or(Value::Null))
    }

    /// `agentbbs-mcp/src/client.rs:103-114` — the `initialize` handshake.
    pub fn initialize(&mut self) -> Result<Value, ClientError> {
        self.request(
            "initialize",
            json!({
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "clientInfo": {
                    "name": "latentmesh-agentbbs-bridge",
                    "version": env!("CARGO_PKG_VERSION"),
                },
                "capabilities": {}
            }),
        )
    }

    /// `agentbbs-mcp/src/client.rs:116-119`.
    pub fn list_tools(&mut self) -> Result<Value, ClientError> {
        self.request("tools/list", Value::Null)
    }

    fn call_tool(&mut self, name: &str, args: Value) -> Result<Value, ClientError> {
        self.request("tools/call", json!({ "name": name, "arguments": args }))
    }

    /// `agentbbs-mcp/src/server.rs:132-139` — `list_boards`, no arguments.
    pub fn list_boards(&mut self) -> Result<Value, ClientError> {
        self.call_tool("list_boards", json!({}))
    }

    /// `agentbbs-mcp/src/server.rs:140-152` — `read_board { board, limit? }`.
    pub fn read_board(&mut self, board: &str, limit: Option<u64>) -> Result<Value, ClientError> {
        let mut args = json!({ "board": board });
        if let Some(limit) = limit {
            args["limit"] = json!(limit);
        }
        self.call_tool("read_board", args)
    }

    /// `agentbbs-mcp/src/server.rs:153-166` — `post_message { board, subject?, text }`.
    /// This is ADR-020's "simplest, human-board-facing" publish path.
    pub fn post_message(
        &mut self,
        board: &str,
        subject: &str,
        text: &str,
    ) -> Result<Value, ClientError> {
        self.call_tool(
            "post_message",
            json!({ "board": board, "subject": subject, "text": text }),
        )
    }

    /// `agentbbs-mcp/src/server.rs:167-183` — `search_memory { query, top_k? }`.
    pub fn search_memory(
        &mut self,
        query: &[f32],
        top_k: Option<u64>,
    ) -> Result<Value, ClientError> {
        let mut args = json!({ "query": query });
        if let Some(top_k) = top_k {
            args["top_k"] = json!(top_k);
        }
        self.call_tool("search_memory", args)
    }
}

/// Extract the text of a tool result's first content block —
/// `agentbbs-mcp/src/server.rs:220-230`'s
/// `{"content":[{"type":"text","text":...}],"isError":false}` shape.
pub fn extract_tool_text(result: &Value) -> Option<&str> {
    result.get("content")?.get(0)?.get("text")?.as_str()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// A client wired to a canned response line, so a test can drive one
    /// request/response round trip and inspect exactly what request bytes
    /// were sent (`client.writer`).
    fn client_for(reply: &str) -> McpStdioClient<Cursor<Vec<u8>>, Vec<u8>> {
        let mut body = reply.as_bytes().to_vec();
        body.push(b'\n');
        McpStdioClient::new(Cursor::new(body), Vec::new())
    }

    #[test]
    fn initialize_sends_well_formed_request_and_parses_result() {
        let mut client =
            client_for(r#"{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05"}}"#);
        let result = client.initialize().unwrap();
        assert_eq!(result["protocolVersion"], "2024-11-05");

        let sent = String::from_utf8(client.writer.clone()).unwrap();
        let sent_req: Request = serde_json::from_str(sent.trim_end()).unwrap();
        assert_eq!(sent_req.method, "initialize");
        assert_eq!(sent_req.jsonrpc, "2.0");
        assert_eq!(sent_req.id, Some(json!(1)));
        assert_eq!(sent_req.params["protocolVersion"], MCP_PROTOCOL_VERSION);
    }

    #[test]
    fn post_message_call_tool_framing() {
        let mut client = client_for(
            r#"{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"posted"}],"isError":false}}"#,
        );
        let result = client
            .post_message("air-relay", "subj", "body text")
            .unwrap();
        assert_eq!(extract_tool_text(&result), Some("posted"));

        let sent = String::from_utf8(client.writer.clone()).unwrap();
        let sent_req: Request = serde_json::from_str(sent.trim_end()).unwrap();
        assert_eq!(sent_req.method, "tools/call");
        assert_eq!(sent_req.params["name"], "post_message");
        assert_eq!(sent_req.params["arguments"]["board"], "air-relay");
        assert_eq!(sent_req.params["arguments"]["subject"], "subj");
        assert_eq!(sent_req.params["arguments"]["text"], "body text");
    }

    #[test]
    fn rpc_error_surfaces_as_client_error() {
        let mut client = client_for(
            r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"unknown tool: bogus"}}"#,
        );
        let err = client.list_boards().unwrap_err();
        match err {
            ClientError::Rpc(e) => {
                assert_eq!(e.code, codes::METHOD_NOT_FOUND);
                assert_eq!(e.message, "unknown tool: bogus");
            }
            other => panic!("expected Rpc error, got {other:?}"),
        }
    }

    #[test]
    fn closed_stream_before_a_response_is_reported() {
        let mut client = McpStdioClient::new(Cursor::new(Vec::new()), Vec::new());
        assert!(matches!(client.list_boards(), Err(ClientError::Closed)));
    }

    #[test]
    fn ids_increment_across_calls() {
        let mut body = Vec::new();
        for _ in 0..2 {
            body.extend_from_slice(br#"{"jsonrpc":"2.0","id":1,"result":{}}"#);
            body.push(b'\n');
        }
        let mut client = McpStdioClient::new(Cursor::new(body), Vec::new());
        client.list_boards().unwrap();
        client.list_boards().unwrap();
        let sent = String::from_utf8(client.writer.clone()).unwrap();
        let ids: Vec<Value> = sent
            .lines()
            .map(|line| serde_json::from_str::<Request>(line).unwrap().id.unwrap())
            .collect();
        assert_eq!(ids, vec![json!(1), json!(2)]);
    }
}
