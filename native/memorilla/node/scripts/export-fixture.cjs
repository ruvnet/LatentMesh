const path = require('node:path')
const fs = require('node:fs')
const { loadMemory, exportFrame } = require('..')
async function main() {
  const root = path.resolve(__dirname, '../../fixtures')
  const model = await loadMemory(path.join(root, 'learned'), 'interop-test-only', 1_000_000)
  await model.write('history', new Float32Array(8).fill(0.1))
  const result = await model.read('history', new Float32Array(8).fill(0.2))
  fs.writeFileSync(path.join(root, 'frame.json'), exportFrame(result, 'fixture-qwen3-12', 1, 0) + '\n')
}
main().catch(e => { console.error(e); process.exitCode = 1 })
