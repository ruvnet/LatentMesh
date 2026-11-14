use latentmesh_memorilla::{self as core, MemoryModule, MemoryService};
use napi::{bindgen_prelude::*, Env, Task};
use napi_derive::napi;
use std::{
    path::Path,
    sync::{Arc, Mutex, MutexGuard},
};

fn err(e: impl std::fmt::Display) -> Error {
    Error::from_reason(e.to_string())
}
fn cache_budget(value: f64) -> Result<usize> {
    if !value.is_finite()
        || value <= 0.0
        || value.fract() != 0.0
        || value > 9_007_199_254_740_991.0
        || value > usize::MAX as f64
    {
        return Err(err("cache budget must be a positive safe integer"));
    }
    Ok(value as usize)
}
fn lock<T>(v: &Mutex<T>) -> Result<MutexGuard<'_, T>> {
    v.lock().map_err(|_| err("native engine lock poisoned"))
}

#[napi(object)]
pub struct MemoryTokens {
    pub values: Float32Array,
    pub shape: Vec<u32>,
    pub checkpoint_hash: String,
    pub collection_hash: String,
    pub query_hash: String,
    pub scope_hash: String,
}
impl From<core::ReadResult> for MemoryTokens {
    fn from(v: core::ReadResult) -> Self {
        Self {
            values: v.values.into(),
            shape: v.shape.into_iter().map(|x| x as u32).collect(),
            checkpoint_hash: v.checkpoint_hash,
            collection_hash: v.collection_hash,
            query_hash: v.query_hash,
            scope_hash: v.scope_hash,
        }
    }
}

#[napi]
pub struct Memorilla {
    inner: Arc<Mutex<MemoryService>>,
}
#[napi]
impl Memorilla {
    /// Float32 inputs are copied before scheduling, so JS mutations cannot race Rust.
    #[napi(ts_return_type = "Promise<number>")]
    pub fn write(&self, id: String, documents: Float32Array) -> AsyncTask<WriteMemory> {
        AsyncTask::new(WriteMemory {
            inner: self.inner.clone(),
            id,
            documents: documents.to_vec(),
        })
    }
    #[napi(ts_return_type = "Promise<MemoryTokens>")]
    pub fn read(&self, id: String, questions: Float32Array) -> AsyncTask<ReadMemory> {
        AsyncTask::new(ReadMemory {
            inner: self.inner.clone(),
            id,
            questions: questions.to_vec(),
        })
    }
    #[napi(ts_return_type = "Promise<boolean>")]
    pub fn delete(&self, id: String) -> AsyncTask<DeleteMemory> {
        AsyncTask::new(DeleteMemory {
            inner: self.inner.clone(),
            id,
        })
    }
    #[napi(ts_return_type = "Promise<MemoryStats>")]
    pub fn stats(&self) -> AsyncTask<StatsMemory> {
        AsyncTask::new(StatsMemory {
            inner: self.inner.clone(),
        })
    }
}

/// Load an upstream checkpoint. CPU/f32 is the portable default.
#[napi(ts_return_type = "Promise<Memorilla>")]
pub fn load_memory(
    directory: String,
    scope: String,
    cache_budget_bytes: f64,
    device: Option<String>,
    dtype: Option<String>,
) -> AsyncTask<LoadMemory> {
    AsyncTask::new(LoadMemory {
        directory,
        scope,
        budget: cache_budget_bytes,
        device: device.unwrap_or("cpu".into()),
        dtype: dtype.unwrap_or("f32".into()),
    })
}

pub struct LoadMemory {
    directory: String,
    scope: String,
    budget: f64,
    device: String,
    dtype: String,
}
impl Task for LoadMemory {
    type Output = MemoryService;
    type JsValue = Memorilla;
    fn compute(&mut self) -> Result<Self::Output> {
        let device = core::device(&self.device).map_err(err)?;
        let dtype = core::dtype(&self.dtype).map_err(err)?;
        MemoryService::new(
            MemoryModule::load(&self.directory, &device, dtype).map_err(err)?,
            &self.scope,
            cache_budget(self.budget)?,
        )
        .map_err(err)
    }
    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(Memorilla {
            inner: Arc::new(Mutex::new(output)),
        })
    }
}
pub struct WriteMemory {
    inner: Arc<Mutex<MemoryService>>,
    id: String,
    documents: Vec<f32>,
}
impl Task for WriteMemory {
    type Output = f64;
    type JsValue = f64;
    fn compute(&mut self) -> Result<f64> {
        Ok(lock(&self.inner)?
            .write(&self.id, &self.documents)
            .map_err(err)? as f64)
    }
    fn resolve(&mut self, _env: Env, v: f64) -> Result<f64> {
        Ok(v)
    }
}
pub struct ReadMemory {
    inner: Arc<Mutex<MemoryService>>,
    id: String,
    questions: Vec<f32>,
}
impl Task for ReadMemory {
    type Output = core::ReadResult;
    type JsValue = MemoryTokens;
    fn compute(&mut self) -> Result<Self::Output> {
        lock(&self.inner)?
            .read(&self.id, &self.questions)
            .map_err(err)
    }
    fn resolve(&mut self, _env: Env, v: Self::Output) -> Result<MemoryTokens> {
        Ok(v.into())
    }
}
pub struct DeleteMemory {
    inner: Arc<Mutex<MemoryService>>,
    id: String,
}
impl Task for DeleteMemory {
    type Output = bool;
    type JsValue = bool;
    fn compute(&mut self) -> Result<bool> {
        Ok(lock(&self.inner)?.delete(&self.id))
    }
    fn resolve(&mut self, _env: Env, v: bool) -> Result<bool> {
        Ok(v)
    }
}
#[napi(object)]
pub struct MemoryStats {
    pub collections: u32,
    pub cache_bytes: f64,
    pub config_json: String,
    pub checkpoint_hash: String,
}
pub struct StatsMemory {
    inner: Arc<Mutex<MemoryService>>,
}
impl Task for StatsMemory {
    type Output = MemoryStats;
    type JsValue = MemoryStats;
    fn compute(&mut self) -> Result<MemoryStats> {
        let engine = lock(&self.inner)?;
        Ok(MemoryStats {
            collections: engine.collection_count() as u32,
            cache_bytes: engine.cache_bytes() as f64,
            config_json: serde_json::to_string(engine.model.config()).map_err(err)?,
            checkpoint_hash: engine.model.checkpoint_hash().to_owned(),
        })
    }
    fn resolve(&mut self, _env: Env, v: MemoryStats) -> Result<MemoryStats> {
        Ok(v)
    }
}

/// Metadata and f32 bytes use LatentMesh's existing LatentFrame schema.
/// Export is always ObserveOnly; callers must qualify the edge independently.
#[napi]
pub fn export_frame(
    tokens: MemoryTokens,
    receiver_model: String,
    sequence: u32,
    timestamp: u32,
) -> Result<String> {
    let v = core::ReadResult {
        values: tokens.values.to_vec(),
        shape: tokens.shape.into_iter().map(|n| n as usize).collect(),
        checkpoint_hash: tokens.checkpoint_hash,
        collection_hash: tokens.collection_hash,
        query_hash: tokens.query_hash,
        scope_hash: tokens.scope_hash,
    };
    core::service::finite(&v.values).map_err(err)?;
    let expected = v
        .shape
        .iter()
        .try_fold(1usize, |a, b| a.checked_mul(*b))
        .ok_or_else(|| err("shape overflow"))?;
    if expected != v.values.len() {
        return Err(err("shape does not match values"));
    }
    serde_json::to_string(
        &core::wire::latent_frame(&v, &receiver_model, sequence as u64, timestamp as u64)
            .map_err(err)?,
    )
    .map_err(err)
}

#[napi]
pub struct MemorillaAgent {
    inner: Arc<Mutex<core::pipeline::MemorillaAgent>>,
}
#[napi]
impl MemorillaAgent {
    #[napi(ts_return_type = "Promise<number>")]
    pub fn write(&self, id: String, documents: Vec<String>) -> AsyncTask<WriteAgent> {
        AsyncTask::new(WriteAgent {
            inner: self.inner.clone(),
            id,
            documents,
        })
    }
    #[napi(ts_return_type = "Promise<string>")]
    pub fn answer(
        &self,
        id: String,
        question: String,
        system_prompt: String,
        max_tokens: u32,
    ) -> AsyncTask<AnswerAgent> {
        AsyncTask::new(AnswerAgent {
            inner: self.inner.clone(),
            id,
            question,
            system_prompt,
            max_tokens: max_tokens as usize,
        })
    }
    #[napi(ts_return_type = "Promise<boolean>")]
    pub fn delete(&self, id: String) -> AsyncTask<DeleteAgent> {
        AsyncTask::new(DeleteAgent {
            inner: self.inner.clone(),
            id,
        })
    }
}
#[napi(ts_return_type = "Promise<MemorillaAgent>")]
pub fn load_agent(
    memory_directory: String,
    encoder_directory: String,
    decoder_directory: String,
    scope: String,
    cache_budget_bytes: f64,
    device: Option<String>,
    dtype: Option<String>,
) -> AsyncTask<LoadAgent> {
    AsyncTask::new(LoadAgent {
        memory_directory,
        encoder_directory,
        decoder_directory,
        scope,
        budget: cache_budget_bytes,
        device: device.unwrap_or("cpu".into()),
        dtype: dtype.unwrap_or("f32".into()),
    })
}
pub struct LoadAgent {
    memory_directory: String,
    encoder_directory: String,
    decoder_directory: String,
    scope: String,
    budget: f64,
    device: String,
    dtype: String,
}
impl Task for LoadAgent {
    type Output = core::pipeline::MemorillaAgent;
    type JsValue = MemorillaAgent;
    fn compute(&mut self) -> Result<Self::Output> {
        let device = core::device(&self.device).map_err(err)?;
        let dtype = core::dtype(&self.dtype).map_err(err)?;
        core::pipeline::MemorillaAgent::load(
            Path::new(&self.memory_directory),
            Path::new(&self.encoder_directory),
            Path::new(&self.decoder_directory),
            &self.scope,
            cache_budget(self.budget)?,
            &device,
            dtype,
        )
        .map_err(err)
    }
    fn resolve(&mut self, _env: Env, v: Self::Output) -> Result<MemorillaAgent> {
        Ok(MemorillaAgent {
            inner: Arc::new(Mutex::new(v)),
        })
    }
}
pub struct WriteAgent {
    inner: Arc<Mutex<core::pipeline::MemorillaAgent>>,
    id: String,
    documents: Vec<String>,
}
impl Task for WriteAgent {
    type Output = f64;
    type JsValue = f64;
    fn compute(&mut self) -> Result<f64> {
        Ok(lock(&self.inner)?
            .write(&self.id, &self.documents)
            .map_err(err)? as f64)
    }
    fn resolve(&mut self, _env: Env, v: f64) -> Result<f64> {
        Ok(v)
    }
}
pub struct AnswerAgent {
    inner: Arc<Mutex<core::pipeline::MemorillaAgent>>,
    id: String,
    question: String,
    system_prompt: String,
    max_tokens: usize,
}
impl Task for AnswerAgent {
    type Output = String;
    type JsValue = String;
    fn compute(&mut self) -> Result<String> {
        lock(&self.inner)?
            .answer(
                &self.id,
                &self.question,
                &self.system_prompt,
                self.max_tokens,
            )
            .map_err(err)
    }
    fn resolve(&mut self, _env: Env, v: String) -> Result<String> {
        Ok(v)
    }
}
pub struct DeleteAgent {
    inner: Arc<Mutex<core::pipeline::MemorillaAgent>>,
    id: String,
}
impl Task for DeleteAgent {
    type Output = bool;
    type JsValue = bool;
    fn compute(&mut self) -> Result<bool> {
        Ok(lock(&self.inner)?.memory.delete(&self.id))
    }
    fn resolve(&mut self, _env: Env, v: bool) -> Result<bool> {
        Ok(v)
    }
}
