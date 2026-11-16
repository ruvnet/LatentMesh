//! ADR-049 section 4 canonical JSON: object keys sorted bytewise, no
//! whitespace, integers only, printable-ASCII strings, `"` and `\` escaped.

use serde_json::Value;
use sha2::{Digest, Sha256};

/// Canonical text of `v`, or an error naming the first offending value.
pub fn canonical_json(v: &Value) -> Result<String, String> {
    let mut out = String::new();
    write(v, &mut out)?;
    Ok(out)
}

pub fn sha256_hex(text: &str) -> String {
    hex(&Sha256::digest(text.as_bytes()))
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn write_string(s: &str, out: &mut String) -> Result<(), String> {
    out.push('"');
    for c in s.chars() {
        if !(' '..='~').contains(&c) {
            return Err(format!("canonical JSON: non-printable-ASCII char {c:?}"));
        }
        if c == '"' || c == '\\' {
            out.push('\\');
        }
        out.push(c);
    }
    out.push('"');
    Ok(())
}

fn write(v: &Value, out: &mut String) -> Result<(), String> {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) if n.is_i64() || n.is_u64() => out.push_str(&n.to_string()),
        Value::Number(n) => return Err(format!("canonical JSON: non-integer number {n}")),
        Value::String(s) => write_string(s, out)?,
        Value::Array(a) => write_array(a, out)?,
        Value::Object(m) => write_object(m, out)?,
    }
    Ok(())
}

fn write_array(a: &[Value], out: &mut String) -> Result<(), String> {
    out.push('[');
    for (i, item) in a.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write(item, out)?;
    }
    out.push(']');
    Ok(())
}

fn write_object(m: &serde_json::Map<String, Value>, out: &mut String) -> Result<(), String> {
    let mut keys: Vec<&String> = m.keys().collect();
    keys.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
    out.push('{');
    for (i, k) in keys.into_iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_string(k, out)?;
        out.push(':');
        write(&m[k.as_str()], out)?;
    }
    out.push('}');
    Ok(())
}
