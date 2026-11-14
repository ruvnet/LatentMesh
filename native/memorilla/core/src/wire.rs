//! Export candidates using the existing LatentFrame JSON vocabulary. No authority is granted.
use crate::{ReadResult, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub fn latent_frame(
    result: &ReadResult,
    receiver_model: &str,
    sequence: u64,
    timestamp: u64,
) -> Result<Value> {
    if result.shape.len() != 3
        || result.shape[0] != 1
        || result.shape.contains(&0)
        || receiver_model.is_empty()
    {
        candle_core::bail!("a frame requires one query and an explicit receiver model identity");
    }
    let bytes: Vec<u8> = result.values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let context = json!({"schema":"memorilla.context.v1","shape":result.shape,"scope":result.scope_hash,
        "collection":result.collection_hash,"query":result.query_hash,"checkpoint":result.checkpoint_hash,
        "receiver_model":receiver_model});
    let context_hash = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&context).map_err(candle_core::Error::wrap)?)
    );
    let frame = json!({"id":format!("memorilla:{}:{sequence}",context_hash),
        "sender_model":format!("memorilla:{}",result.checkpoint_hash),"receiver_space":receiver_model,
        "transform_hash":result.checkpoint_hash,"sequence":sequence,
        "payload":{"encoding":"f32","dim":result.values.len(),"bytes":bytes,"int8_params":null},
        "confidence":0.0,"provenance":{"sender_model":format!("memorilla:{}",result.checkpoint_hash),
            "context_hash":context_hash,"parents":[]},"authority":"observe_only","timestamp":timestamp});
    Ok(json!({"schema":"memorilla.frame.v1","context":context,"frame":frame}))
}
