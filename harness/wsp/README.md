# Latent transfer into a receiver weight patch

The sender can project one of its own hidden layers onto a named direction.
It signs one quantized coordinate per shared probe and sends the packet through
LatentMesh Air. The receiver uses its own hidden activations to fit one final
linear output and bias with ruv-FANN weight space programming. Source and receiver
may have different depths and hidden widths. They must agree on ordered input
features, preprocessing, probe values, output meaning and units. Choosing an
appropriate sender direction requires task or model knowledge.

The target fit uses a bounded, closed form ridge solve. It returns a hash bound
patch for evaluation in a local WSP workspace. The harness also implements
sender output deltas and a packed teacher label as comparison arms. A hidden
feature fixture verifies that a nonconstant feature can transfer even when
the sender outputs the same constant for every probe.

## Language encoders

A [second experiment on pretrained DistilBERT and RoBERTa](language/README.md) transfers an additional anger response with a 13 parameter adapter. All controls use the same 218 byte Air frame.

## Reproduce

Use sibling checkouts, Node 20.9 or newer, npm, and a C11 compiler. The compiler
builds the existing portable C transmitter and receiver in a temporary directory.

```sh
git clone https://github.com/ruvnet/LatentMesh.git
git -C LatentMesh fetch origin feat/latent-wsp-transfer
git -C LatentMesh switch --detach FETCH_HEAD
git clone https://github.com/ruvnet/ruv-FANN.git
git -C ruv-FANN fetch origin 1abb28d3d23f00c04810f59ff7b3797b57bdc24c
git -C ruv-FANN switch --detach 1abb28d3d23f00c04810f59ff7b3797b57bdc24c
(cd ruv-FANN/packages/wsp && npm install --ignore-scripts --no-package-lock --workspaces=false)
cd LatentMesh
node --test harness/wsp/transfer.test.mjs
node harness/wsp/benchmark.mjs > /tmp/latentmesh-readout.json
cmp harness/wsp/receipt.json /tmp/latentmesh-readout.json
node harness/wsp/summarize.mjs /tmp/latentmesh-readout.json
```

Set `RUV_FANN_ROOT` to use a different checkout location. The pinned checkout
contains the development WSP API and the hash verified UCI datasets. The
runtime is the package's existing scalar WASM dependency.

## Results

Mean held out classification accuracy across five fixed splits per dataset.

| Receiver | Iris | Wine |
| --- | ---: | ---: |
| Damaged baseline | 69.43% | 76.41% |
| Norm matched random | 74.98% | 76.92% |
| Shuffled probe order | 90.99% | 91.79% |
| One bit teacher membership | 93.77% | 96.92% |
| Signed 8 bit hidden coordinate | 95.15% | 96.92% |
| Full precision sender change | 95.15% | 96.92% |
| Original intact receiver | 95.84% | 97.44% |

In each run six receiver parameters changed. Sibling outputs matched exactly.
Class decisions can still change through competition with the repaired output.
On Iris, retained class accuracy went from 93.79% in the intact receiver to
92.79% after repair. On Wine it held at 97.14%.

The q8 arm used one C loopback frame of 198 to 204 bytes per repair. One bit
membership used 170 to 171 bytes. Packet sizes include the application
signature, Air envelope and frame checksum. Shared probes, manifest and trust
key provisioning, Meshtastic protobuf and physical radio overhead are outside
these counts.

**Decision.** The q8 advantage over teacher membership was 1.38 points on Iris
and 0.00 on Wine. The registered 2 point threshold on both datasets was missed.
This is controlled recovery of an erased output on two small classifiers. The
two source modes gave bit identical receiver patches in this setup because the
erased sender row was precisely a linear function of its hidden layer. The
hidden path was checked as an algebraic implementation control after scoring.

[Experiment protocol](protocol.md) · [Machine readable receipt](receipt.json) · [WSP API and tests](https://github.com/ruvnet/ruv-FANN/pull/232)

## Packet and authority

A locally provisioned manifest binds the exact source revision and direction
or checkpoint pair, target revision, skill identifier, output mapping and probe
hash. The wire payload carries the manifest digest. The decoder checks the
expected encoding and a locally trusted Ed25519 public key before interpreting
the signal. The target compiler checks the fresh checkpoint and ordered probes.
Recompiling an old packet against a changed receiver fails. A local workspace
owns explicit evaluation, revision checks and rollback. The same packet can be
reexamined against the original base after a deliberate rollback.

| Wire field | Bytes |
| --- | ---: |
| Magic, encoding, count | 6 |
| SHA256 agreement digest | 32 |
| Q8 scale and coordinates | 4 plus probe count |
| Alternative teacher membership bits | ceiling of probe count divided by 8 |
| Ed25519 application signature | 64 |

The hidden direction includes a layer, up to 128 coefficients and a bias. The
packet supports up to 64 probes. A signed application body travels in an Air
diagnostic envelope. This experiment has no firmware listener or production
promotion path. The existing causal routing gates continue to govern admission.

The published [LS Merge architecture](https://ls-merge.github.io/) trains a
weight VAE. This prototype uses a functional projection and measures task
fidelity directly. A VAE, discovery of useful directions, general language
model transfer and physical radio delivery are open experiments.
