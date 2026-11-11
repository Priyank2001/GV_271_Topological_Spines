// Lets Node resolve the bare specifier "three" to the vendored copy, as the
// importmap in index.html does in the browser. Used via `node --import`.
import { register } from 'node:module';
register('./three-hook.mjs', import.meta.url);
