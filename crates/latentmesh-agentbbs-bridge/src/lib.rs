//! Store-and-forward bridge from LatentMesh Air to agentbbs (ADR-020).
//!
//! A bridge/gateway pattern, not a tunnel: an Air-connected node with
//! internet access decodes received `SemanticEnvelope`/`SemanticDelta`
//! messages ([`latentmesh_air_core`]) and republishes their content into
//! agentbbs. It does not carry agentbbs's JSON wire format over the radio
//! link, and it does not carry LMS1/LMAD bytes through agentbbs unmodified —
//! the two protocols meet at exactly one decode/re-encode boundary per
//! direction, implemented in [`bridge`].
//!
//! **agentbbs is not a Cargo dependency of this workspace.** It is a
//! separate Rust workspace (`ruvnet/agentbbs`, canonical name AgentBBS) with
//! its own independently-versioned crate tree; vendoring it as a path/git
//! dependency would break the hermetic/offline `cargo test --workspace`
//! guarantee this repository relies on elsewhere. Instead this crate speaks
//! agentbbs's contract as a **wire boundary**:
//!
//! - [`wire`] — hand-rolled `serde` types transcribed field-for-field from
//!   agentbbs's federation crate: `FederationEnvelope`/`FederationPayload`
//!   (four of the seven upstream variants — `AnnounceBoard`,
//!   `ReplicateMessage`, `PeerHello`, `Ack`), `Board`, `Message`, and the
//!   Ed25519 identity/signature types they're built from.
//! - [`bridge`] — the pure mapping functions: decoded Air content in, BBS
//!   payload/message content out. No I/O.
//! - [`mcp`] — a synchronous JSON-RPC 2.0 stdio client for the four MCP
//!   tools agentbbs pins (`list_boards`, `read_board`, `post_message`,
//!   `search_memory`), generic over `Read`/`Write` so it is testable
//!   against canned byte streams without spawning the real `agentbbs mcp`
//!   binary.
//! - [`transport`] — a bridge-side simulated in-memory sink for exercising
//!   the federation (`FederationPayload`) publish path end to end. It is
//!   explicitly not agentbbs's own `Transport` trait/`LoopbackTransport`
//!   (see that module's doc comment for why).
//!
//! Every public item that touches agentbbs's data shapes documents whether
//! it is asserted against agentbbs's real source (loopback/simulation,
//! tested here) or would require a live agentbbs peer (not implemented,
//! not claimed) — see ADR-020's "What is simulated / what is
//! hardware-pending" table.

pub mod bridge;
pub mod mcp;
pub mod transport;
pub mod wire;

pub use bridge::{delta_subject, render_delta_text, BridgeError};
pub use mcp::{ClientError, McpStdioClient};
pub use transport::{EnvelopeSink, FederationPublisher, InMemoryPeer};
pub use wire::{
    AgentId, Board, FederationEnvelope, FederationPayload, Identity, Message, MessageBody,
    MessageId, MessageKind, SignatureBytes, WireError, MCP_PROTOCOL_VERSION, PROTOCOL_VERSION,
};
