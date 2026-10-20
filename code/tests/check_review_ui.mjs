// Regression check for the generated standalone page and CSV download logic.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const html = fs.readFileSync(process.argv[2], 'utf8');
const script = html.split('<script>')[1].split('</script>')[0];
const elements = {filter:{value:'all'}, cases:{innerHTML:''}, progress:{textContent:''}};
const memory = new Map();
let downloadedBlob;
const context = {
  document: {getElementById: id => elements[id], createElement: () => ({click(){}})},
  localStorage: {getItem: k => memory.get(k), setItem: (k,v) => memory.set(k,v)},
  URL: {createObjectURL: blob => {downloadedBlob=blob;return 'blob:test';}, revokeObjectURL(){}},
  Blob
};
vm.createContext(context);
vm.runInContext(script, context);
const total = (elements.cases.innerHTML.match(/<article>/g)||[]).length;
assert(total > 0);
assert.equal(elements.progress.textContent, `0 / ${total} categorized`);
vm.runInContext('download()', context);
const csv = await downloadedBlob.text();
assert.equal(csv.split('\n').filter(Boolean).length, total+1);
assert.equal(csv.split('\n')[0], 'claim_id,final_category,reviewer_note');
assert(csv.split('\n').slice(1).filter(Boolean).every(line => line.includes(',"",""')));
console.log(`Rendered ${total} cases; downloadable CSV retains blank categories.`);
