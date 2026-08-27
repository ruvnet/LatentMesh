//! A bridge-side simulated transport for exercising the federation path.
//!
//! **This is not agentbbs's own `Transport` trait.** agentbbs is not a
//! Cargo dependency of this workspace (ADR-020's Decision section), so this
//! crate cannot implement or be tested against
//! `agentbbs_federation::Transport` (`async fn send(&self, peer: &Peer,
//! bytes: Vec<u8>)`, byte-opaque per `agentbbs-federation/src/transport.rs`)
//! or its shipped `LoopbackTransport`. [`EnvelopeSink`] is a minimal,
//! synchronous stand-in that exercises the same thing an agentbbs
//! `Transport` implementation would receive: signed [`FederationEnvelope`]
//! bytes. Any test built on it is labeled a simulation, never a claim about
//! live agentbbs federation.

use crate::wire::{FederationEnvelope, FederationPayload, Identity, WireError};

/// Something that accepts sealed federation envelope bytes. Analogous in
/// shape to agentbbs's `Transport::send`, but synchronous and without the
/// `Peer` addressing agentbbs's real trait carries — this bridge's job ends
/// at "produce correct bytes," not "route them to a specific peer."
pub trait EnvelopeSink {
    fn send(&mut self, bytes: Vec<u8>);
}

/// An in-process sink that simply records every envelope it receives, in
/// order. Stands in for agentbbs's `LoopbackTransport` in this crate's
/// hermetic tests.
#[derive(Default)]
pub struct InMemoryPeer {
    pub received: Vec<Vec<u8>>,
}

impl InMemoryPeer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse and open (verify) every received envelope, in receipt order.
    /// Fails on the first envelope that doesn't parse or verify.
    pub fn opened_payloads(&self) -> Result<Vec<FederationPayload>, WireError> {
        self.received
            .iter()
            .map(|bytes| {
                let envelope = FederationEnvelope::from_bytes(bytes)?;
                envelope.open().cloned()
            })
            .collect()
    }
}

impl EnvelopeSink for InMemoryPeer {
    fn send(&mut self, bytes: Vec<u8>) {
        self.received.push(bytes);
    }
}

/// Seals `payload` under `identity` at the next sequence number and hands
/// the signed wire bytes to `sink` — the publish-side half of the
/// federation path ADR-020 names (`FederationPayload::ReplicateMessage` /
/// `AnnounceBoard` / `PeerHello` / `Ack` all flow through this one call).
pub struct FederationPublisher<'a, S: EnvelopeSink> {
    identity: &'a Identity,
    sink: S,
    seq: u64,
}

impl<'a, S: EnvelopeSink> FederationPublisher<'a, S> {
    pub fn new(identity: &'a Identity, sink: S) -> Self {
        FederationPublisher {
            identity,
            sink,
            seq: 0,
        }
    }

    /// The current sequence counter (the value used by the *next* publish).
    pub fn next_seq(&self) -> u64 {
        self.seq + 1
    }

    pub fn sink(&self) -> &S {
        &self.sink
    }

    /// Reclaim ownership of the sink, consuming the publisher.
    pub fn into_sink(self) -> S {
        self.sink
    }

    pub fn publish(&mut self, payload: FederationPayload) -> Result<FederationEnvelope, WireError> {
        self.seq += 1;
        let envelope = FederationEnvelope::seal(self.identity, payload, self.seq)?;
        let bytes = envelope.to_bytes()?;
        self.sink.send(bytes);
        Ok(envelope)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge;

    #[test]
    fn publisher_increments_sequence_and_sink_receives_verifiable_bytes() {
        let identity = Identity::generate();
        let mut publisher = FederationPublisher::new(&identity, InMemoryPeer::new());

        let first = publisher.publish(bridge::ack("a")).unwrap();
        let second = publisher.publish(bridge::ack("b")).unwrap();
        assert_eq!(first.seq, 1);
        assert_eq!(second.seq, 2);

        let opened = publisher.sink().opened_payloads().unwrap();
        assert_eq!(opened.len(), 2);
        assert_eq!(opened[0], FederationPayload::Ack { id: "a".into() });
        assert_eq!(opened[1], FederationPayload::Ack { id: "b".into() });
    }

    #[test]
    fn tampered_wire_bytes_fail_to_open() {
        let identity = Identity::generate();
        let mut publisher = FederationPublisher::new(&identity, InMemoryPeer::new());
        publisher.publish(bridge::ack("a")).unwrap();
        let mut sink = publisher.into_sink();

        // Flip a byte inside the JSON payload region (well past the header
        // fields) to simulate corruption/tampering in flight.
        let mutated = sink.received[0].len() / 2;
        sink.received[0][mutated] ^= 0xFF;
        assert!(sink.opened_payloads().is_err());
    }
}
