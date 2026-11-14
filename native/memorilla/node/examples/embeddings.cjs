// Runs entirely locally using the small committed test checkpoint.
const path = require('node:path')
const { loadMemory } = require('..')
async function main() {
  const memory = await loadMemory(path.resolve(__dirname, '../../core/fixtures/learned'), 'example', 1_000_000)
  await memory.write('history', new Float32Array(8 * 3).fill(0.1))
  const result = await memory.read('history', new Float32Array(8).fill(0.2))
  console.log({ shape: result.shape, firstValues: [...result.values.slice(0, 4)] })
  await memory.delete('history')
}
main().catch(e => { console.error(e); process.exitCode = 1 })
