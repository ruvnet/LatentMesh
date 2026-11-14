const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const os = require('node:os')
const { performance } = require('node:perf_hooks')
const { loadMemory } = require('..')

async function main() {
  const [checkpoint, oracle, device = 'cpu', report] = process.argv.slice(2)
  if (!checkpoint || !oracle) throw new Error('Usage: node scripts/check-released.cjs CHECKPOINT ORACLE [cpu|metal|cuda] [REPORT]')
  const data = JSON.parse(fs.readFileSync(oracle))
  const loadStart = performance.now()
  const model = await loadMemory(checkpoint, 'public-oracle-fixture', 128 * 1024 * 1024, device, 'f32')
  const loadMs = performance.now() - loadStart
  const docs = data.documents[0].filter((_, i) => !data.mask[0][i]).flat()
  const query = Float32Array.from(data.questions[0])
  const writeStart = performance.now()
  const cacheBytes = await model.write('history', Float32Array.from(docs))
  const writeMs = performance.now() - writeStart
  await model.read('history', query)
  const readMs = []
  let result
  for (let i = 0; i < 5; i++) {
    const start = performance.now()
    result = await model.read('history', query)
    readMs.push(performance.now() - start)
  }
  const expected = data.expected[0].flat()
  let maxAbsoluteError = 0
  for (let i = 0; i < expected.length; i++) maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(result.values[i] - expected[i]))
  assert.equal(result.values.length, expected.length)
  assert.ok(maxAbsoluteError < 0.0002, `oracle mismatch ${maxAbsoluteError}`)
  const receipt = {
    schema: 'memorilla.native.validation.v1', timestamp: new Date().toISOString(),
    upstreamCommit: data.upstream, checkpointRevision: 'f6b15af56aa687861c8d2d03b1cd8f7979881361',
    checkpointHash: result.checkpointHash, platform: process.platform, arch: process.arch, cpu: os.cpus()[0].model,
    node: process.version, device, dtype: 'f32', documents: docs.length / data.config.embedding_dim,
    outputShape: result.shape, cacheBytes, loadMs, writeMs, readMs,
    medianReadMs: [...readMs].sort((a, b) => a - b)[2], maxAbsoluteError,
    fullModelQualityEvaluated: false,
    notes: 'Released memory weights with synthetic embeddings. Read timing includes NAPI and host output copy; excludes text encoder and decoder.'
  }
  const json = JSON.stringify(receipt, null, 2) + '\n'
  if (report) { fs.mkdirSync(path.dirname(report), { recursive: true }); fs.writeFileSync(report, json) }
  process.stdout.write(json)
}
main().catch(e => { console.error(e); process.exitCode = 1 })
