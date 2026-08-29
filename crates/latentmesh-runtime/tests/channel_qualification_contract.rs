#[path = "../examples/common/channel_qualification.rs"]
mod channel_qualification;

use std::path::{Path, PathBuf};

const REGISTRATION_SHA256: &str =
    "ebe4e76947fdd514d3759c4b02e8c9189696e635cd14a8c03c3bf8488d445915";

fn crate_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("runtime crate must remain inside the repository")
        .to_path_buf()
}

#[test]
fn committed_registration_and_sources_are_byte_exact() {
    let validated = channel_qualification::load_and_validate(
        &crate_path("receipts/run2-m35-channel-preregistration.json"),
        REGISTRATION_SHA256,
        "qwen2.5-1.5b-exact-channel",
    )
    .expect("the committed registration must remain frozen and valid");
    assert_eq!(
        validated.path,
        crate_path("receipts/run2-m35-channel-preregistration.json")
    );
    channel_qualification::profile(&validated.registration, "qwen2.5-3b-scale-oracle")
        .expect("the receiver-scale oracle must remain registered");

    let root = repo_root();
    let adr = &validated.registration.frozen_source.adr_024;
    let adr_bytes = std::fs::read(root.join(&adr.path)).expect("frozen ADR-024 must exist");
    assert_eq!(
        channel_qualification::sha256_hex(&adr_bytes),
        adr.registered_sha256().unwrap()
    );

    let s1a = &validated.registration.frozen_source.s1a_receipt;
    let s1a_bytes = std::fs::read(root.join(&s1a.path)).expect("frozen S1a receipt must exist");
    channel_qualification::validate_s1a_bytes(&s1a_bytes, &validated.registration)
        .expect("the S1a source receipt must match the registered experiment");
}
