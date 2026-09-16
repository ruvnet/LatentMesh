import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash, generateKeyPairSync} from 'node:crypto';
import {detectMode, signMessage, ModeReceiver, loadAdapter, steerResidual} from '../lib/control.mjs';
const keys = generateKeyPairSync('ed25519');
const h = 'a'.repeat(64);
const payload = {version:1,sender:'agent-a',session:'run-1',sequence:0,expiresAt:2000,
  targetModel:'local/model',targetRevision:'rev-1',adapterHash:h,mode:'challenge',probability:.9,evidenceHash:h};
const policy = {...payload,publicKey:keys.publicKey,minProbability:.8,maxTtlMs:2000};
const wire = p => signMessage(p, keys.privateKey);
const probe = {model:'local/model',revision:'rev-1',indices:[0,2],weights:[[2,0],[0,2],[-2,-2]],bias:[0,0,0],mean:[0,0],scale:[1,1],maxZ:5,minProbability:.8};

test('sparse probe: three modes, ambiguity, OOD and nonfinite abstention',()=>{
  for (const [a,m] of [[[2,9,0],'explore'],[[0,9,2],'converge'],[[-2,9,-2],'challenge']]) assert.equal(detectMode(a,probe).mode,m);
  for (const a of [[0,0,0],[6,0,0],[NaN,0,0],[]]) assert.equal(detectMode(a,probe).mode,'unknown');
  assert.throws(()=>detectMode([1,2,3],{...probe,scale:[0,1]}));
  assert.throws(()=>detectMode([1,2,3],{...probe,indices:[0,0]}));
});
test('authenticated round trip and replay high-water restoration',()=>{
  const r = new ModeReceiver(policy); const result = r.receive(wire(payload),1000);
  assert.equal(result.mode,'challenge'); assert.equal(result.advisoryOnly,true);
  assert.throws(()=>r.receive(wire(payload),1000),/replay/);
  assert.throws(()=>new ModeReceiver(policy,r.lastSequence).receive(wire(payload),1000),/replay/);
  assert.equal(r.receive(wire({...payload,sequence:1,probability:.1}),1000).mode,'unknown');
});
test('tamper, wrong signer, target, session, artifact, expiry and oversized envelope fail closed',()=>{
  const bad = JSON.parse(wire(payload)); bad.payload.mode='explore';
  assert.throws(()=>new ModeReceiver(policy).receive(JSON.stringify(bad),1000),/signature/);
  const other = generateKeyPairSync('ed25519');
  assert.throws(()=>new ModeReceiver(policy).receive(signMessage(payload,other.privateKey),1000));
  for (const patch of [{targetModel:'wrong'},{targetRevision:'wrong'},{session:'wrong'},{sender:'wrong'},
    {adapterHash:'b'.repeat(64)},{expiresAt:1000},{expiresAt:4000}]) {
    const r = new ModeReceiver(policy); assert.throws(()=>r.receive(wire({...payload,...patch}),1000));
    assert.equal(r.lastSequence,-1);
  }
  assert.throws(()=>wire({...payload,capabilities:['write']}));
  assert.throws(()=>wire({...payload,probability:NaN}));
  assert.throws(()=>new ModeReceiver(policy).receive(' '.repeat(4097),1000));
});
const bytes = JSON.stringify({model:'local/model',revision:'rev-1',width:2,layers:[
  {index:2,directions:{explore:[.1,0],converge:[0,.1],challenge:[-.1,-.1]}}]});
const adapterHash=createHash('sha256').update(bytes).digest('hex');
const adapter=loadAdapter(bytes,adapterHash);
const options={adapter,model:adapter.model,revision:adapter.revision,layer:2,mode:'explore',gain:1,maxDeltaNorm:.2};
test('offline steering is default off, bounded, immutable and replayable',()=>{
  const input=[1,2];assert.deepEqual(steerResidual(input,options).residual,input);
  const a=steerResidual(input,{...options,researchEnabled:true});
  assert.deepEqual(a.residual,[1.1,2]);assert.deepEqual(input,[1,2]);
  assert.deepEqual(a,steerResidual(input,{...options,researchEnabled:true}));
  assert.throws(()=>adapter.layers[0].directions.explore[0]=100);
  assert.throws(()=>loadAdapter(bytes+' ',adapterHash));
  for(const patch of [{model:'other'},{revision:'other'},{gain:NaN},{gain:-1},{maxDeltaNorm:.01}])
    assert.throws(()=>steerResidual(input,{...options,researchEnabled:true,...patch}));
  assert.equal(steerResidual(input,{...options,researchEnabled:true,layer:4}).applied,false);
  assert.equal(steerResidual(input,{...options,researchEnabled:true,mode:'unknown'}).applied,false);
});
test('sender probe to signed transport to independently local adapter',()=>{
  const observation=detectMode([2,0,0],probe);
  const message={...payload,adapterHash,mode:observation.mode,probability:observation.probability};
  const receiver=new ModeReceiver({...policy,adapterHash});
  const accepted=receiver.receive(wire(message),1000);
  const out=steerResidual([0,0],{...options,mode:accepted.mode,researchEnabled:true});
  assert.deepEqual(out.residual,[.1,0]);
  assert.equal(out.receipt.adapterHash,adapterHash);
});
