const { test } = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')
const fs = require('node:fs')
const { loadMemory, loadAgent, exportFrame } = require('..')
const device = process.env.MEMORILLA_TEST_DEVICE || 'cpu'
const open = (directory, scope, budget) => loadMemory(directory, scope, budget, device, 'f32')
const fixture = path.resolve(__dirname, '../../fixtures/learned')
const golden = JSON.parse(fs.readFileSync(path.join(fixture, 'golden.json')))

test('native cache agrees with the upstream PyTorch oracle and owns inputs', async () => {
  const model = await open(fixture, 'tenant-a', 1_000_000)
  // Remove masked documents to test the public dense collection interface.
  const docs = golden.documents[0].filter((_, i) => !golden.mask[0][i]).flat()
  const input = Float32Array.from(docs)
  const writing = model.write('history', input)
  input.fill(NaN)
  await writing
  const query = Float32Array.from(golden.questions[0])
  const reading = model.read('history', query)
  query.fill(NaN)
  const result = await reading
  assert.deepEqual(result.shape, [1, 3, 12])
  const expected = golden.expected[0].flat()
  const error = Math.max(...result.values.map((v, i) => Math.abs(v - expected[i])))
  assert.ok(error < 0.00002, `max absolute error ${error}`)
  const packet = JSON.parse(exportFrame(result, 'test-qwen3-revision', 1, 0))
  assert.equal(packet.frame.authority, 'observe_only')
  assert.equal(packet.frame.confidence, 0)
  assert.equal(packet.frame.payload.bytes.length, 3 * 12 * 4)
  assert.equal(packet.frame.payload.dim, 36)
  for (let i = 0; i < result.values.length; i++) {
    assert.equal(Buffer.from(packet.frame.payload.bytes).readFloatLE(4 * i), result.values[i])
  }
})

test('scope separation, finite values, cache budgets and deletion', async () => {
  const a = await open(fixture, 'tenant-a', 128)
  const b = await open(fixture, 'tenant-b', 128)
  await a.write('same-id', new Float32Array(8).fill(0.1))
  await assert.rejects(b.read('same-id', new Float32Array(8)), /unknown collection/)
  await assert.rejects(a.write('same-id', new Float32Array(16)), /budget/)
  await assert.rejects(a.read('same-id', new Float32Array(7)), /dimension/)
  await assert.rejects(a.read('same-id', new Float32Array(8).fill(Infinity)), /nonfinite/)
  assert.equal((await a.stats()).cacheBytes, 128)
  assert.equal(await a.delete('same-id'), true)
  assert.equal((await a.stats()).cacheBytes, 0)
  await assert.rejects(a.read('same-id', new Float32Array(8)), /unknown collection/)
  await assert.rejects(loadMemory(fixture, 'tenant-a', 100, 'unavailable'), /unavailable/)
  await assert.rejects(loadMemory(fixture, 'tenant-a', -1), /safe integer/)
})

test('native work yields to the Node event loop', async () => {
  const model = await open(fixture, 'tenant', 32_000_000)
  await model.write('history', new Float32Array(8 * 512).fill(0.1))
  let heartbeats = 0
  const timer = setInterval(() => heartbeats++, 1)
  try {
    const result = await model.read('history', new Float32Array(8 * 512).fill(0.2))
    assert.equal(result.shape[0], 512)
    assert.ok(heartbeats > 0, 'main thread never advanced during native inference')
  } finally { clearInterval(timer) }
})

test('native text encoder, memory and decoder agree with Transformers', async () => {
  const root = path.resolve(__dirname, '../../fixtures/pipeline')
  const data = JSON.parse(fs.readFileSync(path.join(root, 'golden.json')))
  const agent = await loadAgent(fixture, path.join(root, 'encoder'), path.join(root, 'decoder'), 'tenant', 1_000_000, device, 'f32')
  await agent.write('history', data.documents)
  assert.equal(await agent.answer('history', data.question, data.system, 5), data.expected)
  assert.equal(await agent.answer('history', data.question, data.system, 5), data.expected)
  await assert.rejects(agent.answer('history', '<|memory|>', data.system, 5), /reserved/)
  await agent.delete('history')
  await assert.rejects(agent.answer('history', data.question, data.system, 5), /unknown collection/)
})
