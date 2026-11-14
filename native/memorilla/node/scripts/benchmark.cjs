// Released checkpoint benchmark. Inputs are deterministic synthetic embeddings.
// Run on an idle machine, one backend/process at a time. No quality claim.
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const os = require('node:os')
const crypto = require('node:crypto')
const { performance } = require('node:perf_hooks')
const { execFileSync } = require('node:child_process')

const percentile = (a, p) => [...a].sort((x, y) => x - y)[Math.ceil(a.length * p) - 1]
const hash = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex')
function embeddings(rows, width, seed) {
  const out = new Float32Array(rows * width)
  let s = seed >>> 0
  for (let row = 0; row < rows; row++) {
    let norm = 0
    for (let j = 0; j < width; j++) {
      s ^= s << 13; s ^= s >>> 17; s ^= s << 5
      const v = ((s >>> 0) / 0x100000000) * 2 - 1
      out[row * width + j] = v
      norm += v * v
    }
    for (let j = 0; j < width; j++) out[row * width + j] /= Math.sqrt(norm)
  }
  return out
}
async function main() {
  const [checkpoint, binary, device, report, baseline] = process.argv.slice(2)
  if (!report) throw new Error('Usage: node benchmark.cjs CHECKPOINT BINARY cpu|metal REPORT [BASELINE_REPORT]')
  const { loadMemory } = require(path.resolve(binary))
  const config = JSON.parse(fs.readFileSync(path.join(checkpoint, 'config.json')))
  const warmups = Number(process.env.BENCH_WARMUPS || 20)
  const samples = Number(process.env.BENCH_SAMPLES || 100)
  assert.ok(samples >= 20 && warmups >= 5)
  const start = performance.now()
  const model = await loadMemory(checkpoint, 'benchmark', 1024 ** 3, device, 'f32')
  const loadMs = performance.now() - start
  const cases = []
  fs.mkdirSync(path.dirname(report), { recursive: true })
  for (const [documents, batch] of [[8, 1], [128, 1], [128, 8], [1024, 1], [1024, 8]]) {
    const id = `n${documents}-b${batch}`
    const docs = embeddings(documents, config.embedding_dim, 314159)
    const query = embeddings(batch, config.embedding_dim, 271828)
    const writeStart = performance.now()
    const cacheBytes = await model.write(id, docs)
    const writeMs = performance.now() - writeStart
    for (let i = 0; i < warmups; i++) await model.read(id, query)
    const times = []
    let result
    for (let i = 0; i < samples; i++) {
      const before = performance.now()
      result = await model.read(id, query)
      times.push(performance.now() - before)
    }
    assert.equal(result.values.length, batch * config.num_memories * config.output_dim)
    const output = Buffer.from(result.values.buffer, result.values.byteOffset, result.values.byteLength)
    fs.writeFileSync(`${report}.${id}.bin`, output)
    let maxAbsoluteError = null
    let rmse = null
    if (baseline) {
      const reference = fs.readFileSync(`${baseline}.${id}.bin`)
      assert.equal(reference.length, output.length)
      let max = 0, sum = 0
      for (let i = 0; i < result.values.length; i++) {
        const error = Math.abs(result.values[i] - reference.readFloatLE(i * 4))
        assert.ok(Number.isFinite(error))
        max = Math.max(max, error); sum += error * error
      }
      maxAbsoluteError = max; rmse = Math.sqrt(sum / result.values.length)
      assert.ok(max < 0.0003, `${id}: baseline parity failed: ${max}`)
    } else assert.ok(result.values.every(Number.isFinite))
    const p50Ms = percentile(times, 0.5)
    const c = { id, documents, batch, cacheBytes, writeMs, p50Ms,
      p95Ms: percentile(times, 0.95), queriesPerSecond: batch * 1000 / p50Ms,
      maxAbsoluteError, rmse, checkpointHash: result.checkpointHash, samplesMs: times }
    cases.push(c)
    process.stdout.write(JSON.stringify({ id, device, p50Ms, p95Ms: c.p95Ms, maxAbsoluteError }) + '\n')
    await model.delete(id)
  }
  const receipt = {
    schema: 'memorilla.benchmark.v1', timestamp: new Date().toISOString(),
    sourceCommit: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
    sourceDirty: execFileSync('git', ['status', '--porcelain'], { encoding: 'utf8' }).trim().length > 0,
    binarySha256: hash(binary), node: process.version, platform: process.platform, arch: process.arch,
    cpu: os.cpus()[0].model, systemMemoryBytes: os.totalmem(), maxRssKiB: process.resourceUsage().maxRSS,
    device, dtype: 'f32', warmups, samples, loadMs, cases,
    checkpointRevision: 'f6b15af56aa687861c8d2d03b1cd8f7979881361',
    seeds: { documents: 314159, questions: 271828 }, fullModelQualityEvaluated: false,
    notes: 'Warm cached reads include native worker scheduling, validation, hashing and host output copy. Write is a single cold observation. Encoder and decoder excluded. Synthetic normalized embeddings. CPU and Metal run sequentially.'
  }
  fs.writeFileSync(report, JSON.stringify(receipt, null, 2) + '\n')
}
main().catch(e => { console.error(e); process.exitCode = 1 })
