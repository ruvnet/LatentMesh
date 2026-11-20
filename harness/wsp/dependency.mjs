import { resolve, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

export const fannRoot = process.env.RUV_FANN_ROOT
  ? resolve(process.env.RUV_FANN_ROOT)
  : fileURLToPath(new URL('../../../ruv-FANN/', import.meta.url));
export const wsp = await import(pathToFileURL(join(fannRoot, 'packages/wsp/src/index.js')));
if (typeof wsp.proposeReadoutDelta !== 'function')
  throw new Error('Use the ruv-FANN revision specified in harness/wsp/README.md');
