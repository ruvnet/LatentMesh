//! memmap2 zero-copy loader over the M2 per-token paired dump (ADR-024
//! § Data pipeline: ragged concatenated `[T_i × dim]` f32 blocks per layer
//! plus one shared offsets index).
//!
//! Integrity discipline: every bin file's sha256 AND the index file's sha256
//! are verified against the committed M2 receipt
//! (`run2-pertoken-dump-receipt.json`) BEFORE any training reads a byte.

// The raw dumps are little-endian f32; the zero-copy reinterpret below is
// only valid on a little-endian host.
#[cfg(not(target_endian = "little"))]
compile_error!("latentmesh-train's f32bin zero-copy loader requires a little-endian host");

use memmap2::Mmap;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

/// One layer file's entry in the shared index / M2 receipt.
#[derive(Debug, Clone, Deserialize)]
pub struct FileEntry {
    pub bytes: u64,
    pub dim: usize,
    pub sha256: String,
    pub tokens: u64,
}

/// The shared per-token offsets index (`run2-pertoken-index.json`).
#[derive(Debug, Deserialize)]
pub struct PerTokenIndex {
    pub files: BTreeMap<String, FileEntry>,
    pub n_items: usize,
    pub item_indices: Vec<usize>,
    pub gen_len: Vec<usize>,
    pub prompt_len: Vec<usize>,
    /// Cumulative token offsets, length `n_items + 1`.
    pub token_offsets: Vec<u64>,
    pub total_tokens: u64,
}

/// sha256 of a file, streamed (the bins are up to 5.9 GB).
pub fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// One memory-mapped ragged layer file, validated against the index.
pub struct LayerMap {
    pub name: String,
    pub dim: usize,
    map: Mmap,
}

impl LayerMap {
    /// All f32 values, zero-copy. Little-endian host assumed (x86_64) —
    /// asserted at open time.
    fn floats(&self) -> &[f32] {
        // Safety: the mmap'd file length was asserted == tokens*dim*4 and
        // the base pointer of a page-aligned mmap is 4-byte aligned.
        unsafe { std::slice::from_raw_parts(self.map.as_ptr() as *const f32, self.map.len() / 4) }
    }

    /// Row `t` (global token index) as a `dim`-length f32 slice.
    pub fn row(&self, t: usize) -> &[f32] {
        &self.floats()[t * self.dim..(t + 1) * self.dim]
    }
}

/// The verified, memory-mapped per-token dataset for one (sender, receiver)
/// layer pair.
pub struct PairedDataset {
    pub index: PerTokenIndex,
    pub sender: LayerMap,
    pub receiver: LayerMap,
    /// sha256 of the index file itself, as verified.
    pub index_sha256: String,
    /// Verified sha256 per bin file name (all four, not just the pair used).
    pub verified_bin_sha256: BTreeMap<String, String>,
}

/// Verification outcome for the receipt.
#[derive(Debug, serde::Serialize)]
pub struct VerifiedFile {
    pub file: String,
    pub sha256: String,
    pub expected: String,
    pub bytes: u64,
    pub pass: bool,
}

/// Open the dump directory, verifying the index sha256 and ALL FOUR bin
/// sha256s against the expectations taken from the committed M2 receipt,
/// then mmap the requested sender/receiver layer files.
///
/// `expected` maps file name → sha256 (plus `"run2-pertoken-index.json"` →
/// its sha256), read by the caller from `run2-pertoken-dump-receipt.json`.
pub fn open_verified(
    dir: &Path,
    sender_file: &str,
    receiver_file: &str,
    expected: &BTreeMap<String, String>,
) -> anyhow::Result<(PairedDataset, Vec<VerifiedFile>)> {
    let index_path = dir.join("run2-pertoken-index.json");
    let index_sha = sha256_file(&index_path)?;
    let expect_index = expected
        .get("run2-pertoken-index.json")
        .ok_or_else(|| anyhow::anyhow!("no pinned index sha256"))?;
    anyhow::ensure!(
        &index_sha == expect_index,
        "index sha256 {index_sha} != receipt-pinned {expect_index}"
    );
    let index: PerTokenIndex = serde_json::from_slice(&std::fs::read(&index_path)?)?;
    anyhow::ensure!(index.token_offsets.len() == index.n_items + 1);
    anyhow::ensure!(index.gen_len.len() == index.n_items);
    anyhow::ensure!(index.item_indices.len() == index.n_items);
    anyhow::ensure!(*index.token_offsets.last().unwrap() == index.total_tokens);

    let mut verified = Vec::new();
    for (name, entry) in &index.files {
        let path = dir.join(name);
        let sha = sha256_file(&path)?;
        let expect = expected
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("no pinned sha256 for {name}"))?;
        let bytes = std::fs::metadata(&path)?.len();
        let pass = &sha == expect && sha == entry.sha256 && bytes == entry.bytes;
        verified.push(VerifiedFile {
            file: name.clone(),
            sha256: sha,
            expected: expect.clone(),
            bytes,
            pass,
        });
        anyhow::ensure!(
            pass,
            "{name}: sha256/bytes mismatch against the M2 receipt (measured {bytes} bytes)"
        );
    }

    let mmap_layer = |name: &str| -> anyhow::Result<LayerMap> {
        let entry = index
            .files
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("{name} not in index"))?;
        let f = std::fs::File::open(dir.join(name))?;
        let map = unsafe { Mmap::map(&f)? };
        anyhow::ensure!(
            map.len() as u64 == index.total_tokens * entry.dim as u64 * 4,
            "{name}: mapped {} bytes != tokens x dim x 4",
            map.len()
        );
        Ok(LayerMap {
            name: name.to_string(),
            dim: entry.dim,
            map,
        })
    };
    let sender = mmap_layer(sender_file)?;
    let receiver = mmap_layer(receiver_file)?;
    let verified_bin_sha256 = verified
        .iter()
        .map(|v| (v.file.clone(), v.sha256.clone()))
        .collect();
    Ok((
        PairedDataset {
            index,
            sender,
            receiver,
            index_sha256: index_sha,
            verified_bin_sha256,
        },
        verified,
    ))
}

impl PairedDataset {
    /// Global token indices for a set of item rows (the split sides are
    /// row/item-level; training consumes token-level pairs within them).
    pub fn token_indices_for_rows(&self, rows: &[usize]) -> Vec<u32> {
        let mut out = Vec::new();
        for &r in rows {
            let start = self.index.token_offsets[r] as u32;
            let end = self.index.token_offsets[r + 1] as u32;
            out.extend(start..end);
        }
        out
    }
}

/// Pinned-expectation map builder from the committed M2 receipt JSON.
pub fn expected_from_receipt(
    receipt_path: &Path,
) -> anyhow::Result<(BTreeMap<String, String>, PathBuf)> {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(receipt_path)?)?;
    let mut expected = BTreeMap::new();
    let files = v["files"]
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("receipt has no files map"))?;
    for (name, entry) in files {
        expected.insert(
            name.clone(),
            entry["sha256"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("{name}: no sha256 in receipt"))?
                .to_string(),
        );
    }
    expected.insert(
        v["index"]["file"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("receipt has no index.file"))?
            .to_string(),
        v["index"]["sha256"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("receipt has no index.sha256"))?
            .to_string(),
    );
    let run_dir = PathBuf::from(
        v["run_dir"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("receipt has no run_dir"))?,
    );
    Ok((expected, run_dir))
}
