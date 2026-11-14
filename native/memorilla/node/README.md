Native Rust Memorilla with asynchronous napi bindings. Inference runs locally in Rust using Candle. Python is only used by the source repository's test fixture generators.

Build locally with `npm ci && npm run build`, then run `npm test`.

CPU is the default backend. The macOS ARM64 release binary also supports Metal. Linux x64 GNU uses CPU. CUDA requires a source build and separate validation.

```js
const { loadMemory } = require('@latentmesh/memorilla')
const model = await loadMemory('/models/memorilla/stage3', 'tenant:acme', 1_073_741_824, 'cpu', 'f32')
await model.write('history', documentEmbeddings)
const tokens = await model.read('history', questionEmbeddings)
await model.delete('history')
```

Document embeddings and question embeddings are Float32Array values, with 2560 values per document or question for the released checkpoint. Use its matching Qwen3 Embedding 4B encoder. The output has shape [batch, 16, 4096] for the Qwen3 8B decoder. The scope is a namespace, not an authentication mechanism. The application must authorize access before sharing a handle.

`loadAgent(memoryDirectory, encoderDirectory, decoderDirectory, scope, budget, device, dtype)` loads a complete native Qwen3 pipeline from local Hugging Face safetensors directories. Its `write(id, documents)` takes strings; `answer(id, question, systemPrompt, maxTokens)` returns generated text. Full pretrained task accuracy remains unmeasured. Reduced precision and CUDA require deployment validation.

`exportFrame(tokens, receiverModel, sequence, timestamp)` emits LatentMesh JSON at ObserveOnly authority. It does not grant generation or action rights to a remote receiver.

Release packages contain macOS ARM64 and Linux x64 GNU binaries. Other platforms require their own build. The complete source guide and measured receipts are in `native/memorilla/README.md` and `native/memorilla/validation/` in the source checkout.
