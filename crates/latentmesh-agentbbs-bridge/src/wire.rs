//! agentbbs's federation wire contract, transcribed field-for-field.
//!
//! **agentbbs is not a Cargo dependency of this workspace** (ADR-020's
//! Decision section). Everything in this module is a hand-rolled `serde`
//! re-implementation of the types agentbbs actually defines, so that this
//! crate can produce and parse byte-identical JSON without pulling in
//! agentbbs's own (much larger, independently-versioned) crate tree. Every
//! type below cites the exact agentbbs source file/lines it was transcribed
//! from, as read from a shallow clone of `github.com/ruvnet/agentbbs` at
//! ADR-020's authoring time.
//!
//! Only the four `FederationPayload` variants the pinned MCP-tool/federation
//! contract cites are implemented here (`AnnounceBoard`, `ReplicateMessage`,
//! `PeerHello`, `Ack`) — agentbbs defines three more (`BoardSnapshot`,
//! `BoardDigest`, `PeerExchange`) that are out of scope for this bridge.
//!
//! The Ed25519 identity/signature types (`AgentId`, `SignatureBytes`,
//! `Identity`) and this module's shared [`WireError`] live in the
//! [`identity`] submodule, split out to keep both files under this repo's
//! file-size norm.

mod identity;

pub use identity::{AgentId, Identity, SignatureBytes, WireError};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// `agentbbs-core/src/lib.rs:78` — `pub const PROTOCOL_VERSION: &str = "agentbbs/0.1";`
pub const PROTOCOL_VERSION: &str = "agentbbs/0.1";

/// `agentbbs-mcp/src/server.rs:30` — `pub const PROTOCOL_VERSION: &str = "2024-11-05";`
///
/// Deliberately a *different* constant from [`PROTOCOL_VERSION`] above:
/// agentbbs itself defines two protocol-version strings — one for the
/// federation wire (`agentbbs/0.1`) and one for the MCP handshake
/// (`2024-11-05`, the MCP spec's own dated version tag). Naming mirrors the
/// distinction agentbbs makes.
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// `agentbbs-core/src/board.rs:15-37` — a content-addressed message id: the
/// BLAKE3 hash of the message's canonical signing bytes, rendered as hex.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MessageId(pub String);

impl std::fmt::Display for MessageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::fmt::Debug for MessageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MessageId({})", &self.0[..self.0.len().min(8)])
    }
}

/// `agentbbs-core/src/board.rs:40-56` — a board (message base / conference).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Board {
    pub slug: String,
    pub title: String,
    pub description: String,
    pub locked: bool,
    pub founder: AgentId,
    pub created_at: DateTime<Utc>,
    pub federated: bool,
}

/// `agentbbs-core/src/board.rs:81-102` — how a message participates in a
/// long-running agent process (upstream ADR-0052). `Post` reproduces the
/// exact pre-ADR-0052 v1 signing bytes.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum MessageKind {
    #[default]
    Post,
    Milestone,
    Step,
}

impl MessageKind {
    /// `board.rs:95-101` — stable discriminant folded into v2 signing bytes.
    fn discriminant(self) -> &'static str {
        match self {
            MessageKind::Post => "post",
            MessageKind::Milestone => "milestone",
            MessageKind::Step => "step",
        }
    }
}

/// `agentbbs-core/src/board.rs:104-181` — the author-supplied, pre-signature
/// body of a message.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MessageBody {
    pub board: String,
    pub parent: Option<MessageId>,
    pub subject: String,
    pub body: String,
    pub author: AgentId,
    pub handle: String,
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub kind: MessageKind,
}

impl MessageBody {
    /// `board.rs:132-174` — canonical, deterministic signing bytes,
    /// transcribed byte-for-byte including the v1/v2 branch.
    pub fn signing_bytes(&self) -> Vec<u8> {
        let is_v2 = self.kind != MessageKind::Post;
        let mut out = Vec::with_capacity(self.body.len() + 128);
        out.extend_from_slice(if is_v2 {
            b"agentbbs.msg.v2\n"
        } else {
            b"agentbbs.msg.v1\n"
        });
        out.extend_from_slice(self.board.as_bytes());
        out.push(b'\n');
        out.extend_from_slice(
            self.parent
                .as_ref()
                .map(|p| p.0.as_str())
                .unwrap_or("-")
                .as_bytes(),
        );
        out.push(b'\n');
        if is_v2 {
            out.extend_from_slice(self.kind.discriminant().as_bytes());
            out.push(b'\n');
        }
        out.extend_from_slice(self.subject.as_bytes());
        out.push(b'\n');
        out.extend_from_slice(self.author.to_hex().as_bytes());
        out.push(b'\n');
        out.extend_from_slice(self.handle.as_bytes());
        out.push(b'\n');
        out.extend_from_slice(self.created_at.to_rfc3339().as_bytes());
        out.push(b'\n');
        out.extend_from_slice(format!("{}:", self.body.len()).as_bytes());
        out.extend_from_slice(self.body.as_bytes());
        out
    }

    /// `board.rs:176-180` — the content-addressed id for this body.
    pub fn id(&self) -> MessageId {
        let hash = blake3::hash(&self.signing_bytes());
        MessageId(hash.to_hex().to_string())
    }
}

/// `agentbbs-core/src/board.rs:184-223` — a fully-formed, signed,
/// content-addressed message.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Message {
    pub id: MessageId,
    pub body: MessageBody,
    pub signature: SignatureBytes,
}

impl Message {
    /// `board.rs:197-211` — sign `body` with `identity`. The identity must
    /// match `body.author`.
    pub fn sign(identity: &Identity, body: MessageBody) -> Result<Self, WireError> {
        if identity.id() != body.author {
            return Err(WireError::Malformed {
                field: "message",
                reason: "signing identity does not match author".into(),
            });
        }
        let bytes = body.signing_bytes();
        let signature = identity.sign(&bytes);
        Ok(Message {
            id: body.id(),
            body,
            signature,
        })
    }

    /// `board.rs:214-222` — verify the message: id must match content, and
    /// the signature must validate under the author's key.
    pub fn verify(&self) -> Result<(), WireError> {
        let bytes = self.body.signing_bytes();
        let recomputed = MessageId(blake3::hash(&bytes).to_hex().to_string());
        if recomputed != self.id {
            return Err(WireError::Malformed {
                field: "message",
                reason: "id does not match content".into(),
            });
        }
        self.body.author.verify(&bytes, &self.signature)
    }
}

/// `agentbbs-federation/src/envelope.rs:18-63` — what a node is telling its
/// peers. Only the four variants this bridge's contract cites are
/// implemented (out of seven total upstream); see the module doc.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FederationPayload {
    /// `envelope.rs:20-22` — "this board exists and is federated; mirror its
    /// metadata."
    AnnounceBoard(Board),
    /// `envelope.rs:22-24` — "here is a verified, content-addressed message;
    /// store it idempotently."
    ReplicateMessage(Message),
    /// `envelope.rs:51-57` — a peer introducing itself on link-up.
    PeerHello { node: AgentId, protocol: String },
    /// `envelope.rs:58-62` — acknowledgement of a previously-seen
    /// envelope/message id.
    Ack { id: String },
}

/// `agentbbs-federation/src/envelope.rs:71-81` — a signed, replayable
/// federation message.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FederationEnvelope {
    pub node: AgentId,
    pub seq: u64,
    pub payload: FederationPayload,
    pub signature: SignatureBytes,
}

impl FederationEnvelope {
    /// `envelope.rs:90-102` — canonical, deterministic bytes the node signs.
    fn compose_signing_bytes(node: &AgentId, seq: u64, payload_json: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(payload_json.len() + 128);
        out.extend_from_slice(b"agentbbs.fed.v1\n");
        out.extend_from_slice(PROTOCOL_VERSION.as_bytes());
        out.push(b'\n');
        out.extend_from_slice(node.to_hex().as_bytes());
        out.push(b'\n');
        out.extend_from_slice(seq.to_string().as_bytes());
        out.push(b'\n');
        out.extend_from_slice(format!("{}:", payload_json.len()).as_bytes());
        out.extend_from_slice(payload_json);
        out
    }

    /// `envelope.rs:104-112`.
    pub fn signing_bytes(&self) -> Result<Vec<u8>, WireError> {
        let payload_json = serde_json::to_vec(&self.payload)?;
        Ok(Self::compose_signing_bytes(
            &self.node,
            self.seq,
            &payload_json,
        ))
    }

    /// `envelope.rs:114-127` — seal `payload` under `identity` at sequence
    /// `seq`, producing a signed envelope whose `node` is the signer's id.
    pub fn seal(
        identity: &Identity,
        payload: FederationPayload,
        seq: u64,
    ) -> Result<Self, WireError> {
        let node = identity.id();
        let payload_json = serde_json::to_vec(&payload)?;
        let bytes = Self::compose_signing_bytes(&node, seq, &payload_json);
        let signature = identity.sign(&bytes);
        Ok(FederationEnvelope {
            node,
            seq,
            payload,
            signature,
        })
    }

    /// `envelope.rs:129-138` — verify the node signature and return the
    /// inner payload.
    pub fn open(&self) -> Result<&FederationPayload, WireError> {
        let bytes = self.signing_bytes()?;
        self.node.verify(&bytes, &self.signature)?;
        Ok(&self.payload)
    }

    /// `envelope.rs:140-143`.
    pub fn to_bytes(&self) -> Result<Vec<u8>, WireError> {
        Ok(serde_json::to_vec(self)?)
    }

    /// `envelope.rs:145-148` — parse wire bytes (does NOT verify; call
    /// [`Self::open`]).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, WireError> {
        Ok(serde_json::from_slice(bytes)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(author: AgentId) -> MessageBody {
        MessageBody {
            board: "general".into(),
            parent: None,
            subject: "hello".into(),
            body: "first post from an agent".into(),
            author,
            handle: "wildcat".into(),
            created_at: Utc::now(),
            kind: MessageKind::Post,
        }
    }

    #[test]
    fn message_sign_and_verify() {
        let id = Identity::generate();
        let msg = Message::sign(&id, body(id.id())).unwrap();
        assert!(msg.verify().is_ok());
    }

    #[test]
    fn message_tampered_body_detected() {
        let id = Identity::generate();
        let mut msg = Message::sign(&id, body(id.id())).unwrap();
        msg.body.body = "edited after signing".into();
        assert!(msg.verify().is_err());
    }

    #[test]
    fn envelope_seal_and_open_roundtrip() {
        let id = Identity::generate();
        let payload = FederationPayload::Ack { id: "msg-1".into() };
        let envelope = FederationEnvelope::seal(&id, payload.clone(), 1).unwrap();
        let opened = envelope.open().unwrap();
        assert_eq!(opened, &payload);
    }

    #[test]
    fn envelope_tampered_seq_rejected() {
        let id = Identity::generate();
        let payload = FederationPayload::Ack { id: "msg-1".into() };
        let mut envelope = FederationEnvelope::seal(&id, payload, 1).unwrap();
        envelope.seq = 2;
        assert!(matches!(envelope.open(), Err(WireError::BadSignature)));
    }

    #[test]
    fn envelope_bytes_roundtrip() {
        let id = Identity::generate();
        let payload = FederationPayload::PeerHello {
            node: id.id(),
            protocol: PROTOCOL_VERSION.into(),
        };
        let envelope = FederationEnvelope::seal(&id, payload, 5).unwrap();
        let bytes = envelope.to_bytes().unwrap();
        let parsed = FederationEnvelope::from_bytes(&bytes).unwrap();
        assert!(parsed.open().is_ok());
        assert_eq!(parsed, envelope);
    }
}
