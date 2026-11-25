'use strict';
const $ = id => document.getElementById(id);
let documentData = null, group = null, selected = null, recordIndex = 0, requestId = 0, renderedNodes = null;
let view = {x:0,y:0,scale:1}, bounds = {width:500,height:300};
const svgNS = 'http://www.w3.org/2000/svg';
const detailOrderKey = 'rollout-visualizer.detail-order.v1';
let detailOrder = [];
try {
  const saved = JSON.parse(localStorage.getItem(detailOrderKey));
  if (Array.isArray(saved)) detailOrder = [...new Set(saved.filter(key => typeof key === 'string'))];
} catch (_) { /* Keep reordering available when browser storage is unavailable. */ }
function moveDetailToTop(event) {
  const row = event.target.closest('tr[data-detail-field]');
  if (!row) return;
  const keyboard = event.type === 'keydown';
  if (keyboard && (event.target.tagName !== 'TH' || event.repeat || !['Enter', ' '].includes(event.key))) return;
  event.preventDefault();
  const key = row.dataset.detailField;
  detailOrder = [key, ...detailOrder.filter(name => name !== key)];
  try {
    localStorage.setItem(detailOrderKey, JSON.stringify(detailOrder));
  } catch (_) {
    $('detail-order-hint').textContent = 'Order kept for this session; browser storage is unavailable.';
  }
  row.parentElement.prepend(row);
  if (keyboard) row.querySelector('th').focus({preventScroll:true});
  $('details').scrollTop = 0;
}
function svg(tag, attrs = {}, text) {
  const el = document.createElementNS(svgNS, tag);
  for (const [k,v] of Object.entries(attrs)) el.setAttribute(k,v);
  if (text !== undefined) el.textContent = text;
  return el;
}
function options(el, values) {
  el.replaceChildren(...values.map(([value,label]) => new Option(label,value)));
  el.disabled = !values.length;
}
async function get(url) {
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}
function clear(message) {
  group = null; selected = null; renderedNodes = null;
  $('scene').replaceChildren(); $('details').replaceChildren(); $('text').textContent = '';
  $('stats').textContent = ''; $('notice').textContent = ''; $('record-label').hidden = true;
  $('empty').textContent = message;
  showDecisionResults();
}
function updateColors() {
  const mode = $('color-mode').value;
  const advantageMode = ['reasoning', 'igpo-turn', 'gigpo-turn'].includes(mode);
  $('structure-legend').hidden = mode !== 'structure';
  $('score-legend').hidden = mode !== 'score';
  $('reward-legend').hidden = mode !== 'reward';
  $('entropy-legend').hidden = mode !== 'answer-entropy';
  $('reasoning-legend').hidden = !advantageMode;
  if (!renderedNodes) return;
  let maxAdvantage = 0;
  const fields = {'gigpo-turn': 'gigpo_turn_advantage', 'igpo-turn': 'igpo_turn_advantage'};
  for (const n of renderedNodes.values()) {
    n.colorAdvantage = fields[mode] ? meanNumeric(n.records, fields[mode]) : DecisionModelView.nodeAdvantage(n.records);
    if (Number.isFinite(n.colorAdvantage)) maxAdvantage = Math.max(maxAdvantage, Math.abs(n.colorAdvantage));
  }
  $('advantage-negative-label').textContent = `−${maxAdvantage.toPrecision(3)}`;
  $('advantage-positive-label').textContent = `+${maxAdvantage.toPrecision(3)}`;
  let minReward = Infinity, maxReward = -Infinity;
  for (const n of renderedNodes.values()) if (Number.isFinite(n.turnReward)) {
    minReward = Math.min(minReward, n.turnReward);
    maxReward = Math.max(maxReward, n.turnReward);
  }
  $('reward-low-label').textContent = minReward === Infinity ? 'Lower reward' : `Reward ${minReward}`;
  $('reward-high-label').textContent = maxReward === -Infinity ? 'Higher reward' : `Reward ${maxReward}`;
  let minEntropy = Infinity, maxEntropy = -Infinity;
  for (const n of renderedNodes.values()) {
    n.answerEntropy = DecisionModelView.nodeAnswerEntropy(n.records);
    if (n.kind !== 'root' && Number.isFinite(n.answerEntropy)) {
      minEntropy = Math.min(minEntropy, n.answerEntropy);
      maxEntropy = Math.max(maxEntropy, n.answerEntropy);
    }
  }
  $('entropy-low-label').textContent = minEntropy === Infinity ? 'Lower entropy' : `${minEntropy.toPrecision(3)} nats/token`;
  $('entropy-high-label').textContent = maxEntropy === -Infinity ? 'Higher entropy' : `${maxEntropy.toPrecision(3)} nats/token`;
  for (const n of renderedNodes.values()) {
    let fill, textFill = '#172033';
    if (advantageMode && n.kind !== 'root') {
      const value = n.colorAdvantage;
      if (Number.isFinite(value)) {
        const lightness = 96 - 56 * (maxAdvantage ? Math.abs(value) / maxAdvantage : 0);
        fill = `hsl(${value < 0 ? 25 : 210} 65% ${lightness}%)`;
        if (lightness <= 55) textFill = 'white';
      } else fill = '#e2e8f0';
    } else if (mode !== 'structure' && n.kind !== 'root') {
      const value = mode === 'answer-entropy' ? n.answerEntropy : mode === 'score' ? n.score : n.turnReward;
      if (Number.isFinite(value)) {
        const low = mode === 'answer-entropy' ? minEntropy : minReward;
        const high = mode === 'answer-entropy' ? maxEntropy : maxReward;
        const position = mode === 'score' ? Math.max(0, Math.min(1, value)) : low === high ? .5 : (value - low) / (high - low);
        const lightness = 85 - 55 * position;
        fill = `hsl(210 65% ${lightness}%)`;
        if (lightness <= 55) textFill = 'white';
      } else fill = '#e2e8f0';
    } else {
      fill = n.kind === 'root' ? '#56b4e9' : n.children.length ? '#e69f00' : '#009e73';
      if (n.kind === 'saved' && !n.children.length) textFill = 'white';
    }
    n.element.querySelector('rect').setAttribute('fill', fill);
    n.element.querySelectorAll('text').forEach(el => el.setAttribute('fill', textFill));
    const value = advantageMode ? n.colorAdvantage : mode === 'answer-entropy' ? n.answerEntropy : mode === 'score' ? n.score : n.turnReward;
    const metric = mode === 'answer-entropy' ? 'answer entropy (nats/token; mean across records)' : mode === 'igpo-turn' ? 'IGPO turn advantage (before reasoning adjustment; mean across records)' : mode === 'gigpo-turn' ? 'GiGPO turn advantage (before weighting and reasoning adjustment; mean across records)' : mode === 'reasoning' ? 'reasoning advantage (mean across steps; mean across records)' : `mean ${mode === 'score' ? 'score' : 'turn reward'}`;
    n.element.querySelector('title').textContent = mode !== 'structure' && Number.isFinite(value) ? `${n.label} · ${metric}: ${value}` : n.label;
  }
}
function meanNumeric(records, field) {
  const values = records.map(r => r[field]).filter(v => typeof v === 'number' && Number.isFinite(v));
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}
async function refresh() {
  const ticket = ++requestId;
  $('error').textContent = '';
  try {
    const data = await get('/api/files');
    if (ticket !== requestId) return;
    $('directory').textContent = data.directory; $('directory').title = data.directory;
    const previous = $('step').value;
    options($('step'),data.files.map(name => [name,`Step ${name.replace(/\.jsonl$/, '')}`]));
    $('step').value = data.files.includes(previous) ? previous : data.files.at(-1) || '';
    await loadFile();
  } catch(e) { if(ticket === requestId) {clear('Unable to load files'); $('error').textContent = e.message;} }
}
async function loadFile() {
  const ticket = ++requestId;
  const previousGroup = $('group').value;
  clear('Loading…'); options($('group'),[]); $('error').textContent = '';
  if (!$('step').value) {clear('No JSONL files found. Refresh after logs are saved.'); return;}
  try {
    const data = await get(`/api/file?name=${encodeURIComponent($('step').value)}`);
    if(ticket !== requestId) return;
    documentData = data;
    options($('group'), data.groups.map((g,i) => [i,`Group ${i+1} · ${g.grouping} · ${g.record_count} records`]));
    if (data.groups[Number(previousGroup)]) $('group').value = previousGroup || '0';
    showGroup();
  } catch(e) {if(ticket === requestId) {clear('Unable to load step'); $('error').textContent = e.message;}}
}
function showGroup() {
  clear('No records in this file.');
  group = documentData.groups[Number($('group').value)];
  $('notice').textContent = [documentData.grouping,...documentData.warnings,...(group?.warnings || [])].join(' ');
  if (!group) return;
  $('empty').textContent = '';
  $('stats').textContent = `${group.record_count} records · ${group.mode}`;
  const nodes = new Map(group.nodes.map(n => [n.id,{...n,children:[]}]));
  for(const n of nodes.values()) if(n.parent) nodes.get(n.parent).children.push(n);
  for (const n of nodes.values()) {
    n.score = meanNumeric(n.records, 'score');
    n.turnReward = meanNumeric(n.records, 'turn_reward');
  }
  let leaves = 0, maxDepth = 0;
  function layout(n,depth) {
    n.y = 50 + depth * 100; maxDepth = Math.max(maxDepth,depth);
    for(const c of n.children) layout(c,depth+1);
    n.x = n.children.length ? (n.children[0].x + n.children.at(-1).x)/2 : 110 + leaves++ * 210;
  }
  layout(nodes.get('root'),0);
  bounds = {width:Math.max(leaves*210,220),height: maxDepth*100+110};
  const scene = $('scene');
  for(const n of nodes.values()) if(n.parent) {
    const p = nodes.get(n.parent), middle = (p.y+n.y)/2;
    scene.append(svg('path',{class:'edge',d:`M${p.x},${p.y+24} C${p.x},${middle} ${n.x},${middle} ${n.x},${n.y-24}`}));
  }
  for(const n of nodes.values()) {
    const el = svg('g',{class:'node',transform:`translate(${n.x},${n.y})`,tabindex:0,role:'button','aria-label':n.label});
    el.append(svg('rect',{x:-92,y:-24,width:184,height:48,rx:7}));
    el.append(svg('text',{'text-anchor':'middle',y:-2},n.label.length>25 ? n.label.slice(0,24)+'…' : n.label));
    el.append(svg('text',{'text-anchor':'middle',y:15},n.records.length ? `${n.records.length} record${n.records.length===1?'':'s'}` : 'Search question'));
    el.append(svg('title',{},n.label));
    n.element = el;
    const select = () => {selected=n; recordIndex=0; scene.querySelectorAll('.selected').forEach(e=>e.classList.remove('selected')); el.classList.add('selected'); showDetails();};
    el.addEventListener('click',select);
    el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}});
    scene.append(el);
  }
  renderedNodes = nodes; updateColors();
  selected=nodes.get('root'); scene.querySelector('.node').classList.add('selected'); showDetails(); fit();
}
function selectResultTab(name) {
  for (const tab of ['saved', 'decision']) {
    const active = name === tab;
    $(tab + '-tab').setAttribute('aria-selected', String(active));
    $(tab + '-tab').tabIndex = active ? 0 : -1;
    $(tab + '-panel').hidden = !active;
  }
}
function showDecisionResults() {
  const record = selected?.records[recordIndex];
  const trace = DecisionModelView.trace(record);
  const results = record?.decision_model_probabilities;
  const rows = new Map(trace.steps.map(step => [step.number, step.text]));
  // Keep returned answers visible even when an older log lacks reasoning text.
  for (const key of [...Object.keys(results || {}), ...Object.keys(record?.reasoning_step_advantages || {}), ...Object.keys(record?.reasoning_step_entropies || {})]) {
    if (/^step_\d+$/.test(key) && !rows.has(Number(key.slice(5)))) rows.set(Number(key.slice(5)), 'Reasoning text is not available in this log.');
  }
  $('decision-action').textContent = trace.action || 'No saved action.';
  $('decision-answer-entropy').textContent = `Answer ${DecisionModelView.entropyText(record?.answer_entropy).toLowerCase()}`;
  $('decision-answer-entropy').title = record?.entropy_alignment_status ? `Token alignment: ${record.entropy_alignment_status}` : 'Entropy tracking was not saved for this record.';
  const option = $('decision-highlight').value;
  const input = $('decision-threshold');
  input.disabled = option === 'off';
  const threshold = input.value === '' ? NaN : Number(input.value);
  let count = 0;
  const elements = [...rows].sort(([a], [b]) => a - b).map(([number, text]) => {
    const values = DecisionModelView.probabilities(results?.[`step_${number}`]);
    const highlighted = DecisionModelView.highlighted(values, option, threshold);
    const row = document.createElement('tr'), summary = document.createElement('th'), content = document.createElement('td');
    summary.scope = 'row';
    summary.textContent = `[${number}]`;
    const probabilities = document.createElement('span');
    probabilities.className = 'decision-probabilities';
    probabilities.textContent = DecisionModelView.probabilityText(values);
    summary.append(probabilities);
    const advantage = document.createElement('span');
    advantage.className = 'decision-advantage';
    const entry = record?.reasoning_step_advantages?.[`step_${number}`];
    advantage.textContent = `Advantage: ${Number.isFinite(entry?.advantage) ? entry.advantage.toPrecision(5) : '—'}`;
    if (entry) advantage.title = `${entry.token_count} trainable tokens; original mean: ${entry.original_advantage}; final range: ${entry.min} to ${entry.max}`;
    summary.append(advantage);
    const entropy = document.createElement('span');
    entropy.className = 'decision-entropy';
    entropy.textContent = DecisionModelView.entropyText(record?.reasoning_step_entropies?.[`step_${number}`]);
    summary.append(entropy);
    if (highlighted) {
      count++;
      row.className = 'decision-highlighted';
      const label = document.createElement('span');
      label.className = 'decision-highlight-label';
      label.textContent = 'Above threshold';
      summary.append(label);
    }
    content.textContent = text;
    row.append(summary, content);
    return row;
  });
  $('decision-rows').replaceChildren(...elements);
  $('decision-results').hidden = !elements.length;
  const validThreshold = Number.isFinite(threshold) && threshold >= 0 && threshold <= 100;
  $('decision-highlight-count').textContent = option === 'off' ? '' : !validThreshold ? 'Enter a threshold from 0 to 100.' : `${count} of ${elements.length} steps highlighted`;
  $('decision-status').textContent = !record ? 'Select a node with a saved record to inspect its decision model results.'
    : !results || !Object.keys(results).length ? 'No decision model probabilities saved for this record.'
    : !trace.steps.length ? 'No reasoning text saved for this record.' : '';
}
function showDetails() {
  const records = selected.records;
  options($('record'), records.map((r,i)=>[i,`Record ${i+1} of ${records.length} · line ${r.line}`]));
  $('record').value=String(recordIndex); $('record-label').hidden=!records.length;
  const record=records[recordIndex];
  const values = record ? Object.fromEntries(Object.entries(record).filter(([k])=>!['input','output','reasoning_steps','reasoning_trace','decision_model_probabilities','reasoning_step_advantages','reasoning_step_entropies'].includes(k))) : {node:selected.label,source:'Shared initial context'};
  const table=document.createElement('table');
  const keys = [...detailOrder.filter(key => Object.hasOwn(values, key)), ...Object.keys(values).filter(key => !detailOrder.includes(key))];
  for(const key of keys) {
    const value = values[key];
    const tr=document.createElement('tr'), th=document.createElement('th'), td=document.createElement('td'), pre=document.createElement('pre');
    tr.dataset.detailField=key;
    th.textContent=key;
    th.tabIndex=0; th.setAttribute('role','button');
    th.title='Double-click to move to top, or press Enter or Space';
    th.setAttribute('aria-label',`Move ${th.textContent} to top`);
    pre.textContent=typeof value==='string'?value:JSON.stringify(value,null,2);
    td.append(pre);tr.append(th,td);table.append(tr);
  }
  $('details').replaceChildren(table); showText(); showDecisionResults();
  $('decision-content').scrollTop = 0;
}
function searchQuestion(prompt) {
  return prompt.match(/Your question:\s*([\s\S]*?)(?:\r?\n\s*\r?\n|$)/i)?.[1]?.trim() || 'No search question found in the saved prompt.';
}
function generatedActions(record) {
  if (selected.kind === 'root') return 'No actions generated yet.';
  let actions;
  if (record && Array.isArray(record.previous_step_ids) && typeof record.id === 'string') {
    const records = new Map(group.nodes.flatMap(n => n.records).map(r => [r.id, r]));
    actions = [...record.previous_step_ids, record.id].map(id => {
      const saved = id === record.id ? record : records.get(id);
      if (!saved) return `[Missing saved action: ${id}]`;
      return DecisionModelView.trace(saved).action || '[No action tag]';
    });
  } else {
    // Older logs may contain only an independent saved action.
    actions = [];
    for (let node = selected; node && node.kind !== 'root'; node = renderedNodes.get(node.parent)) actions.push(node.label);
    actions.reverse();
  }
  return actions.map((action, i) => `${i + 1}. ${action}`).join('\n');
}
function showText() {
  if(!selected) return;
  const record=selected.records[recordIndex], field=$('field').value;
  $('text').textContent=field==='reasoning_steps' ? record?.reasoning_steps ?? 'No saved reasoning steps for this node (missing thinking, invalid action, or older log).' : field==='problem' ? searchQuestion(record?.input ?? group.context) : field==='actions' ? generatedActions(record) : field==='context' || selected.kind==='root' ? group.context : !record ? 'No saved record for this node.' : field==='raw' ? JSON.stringify(record,null,2) : record[field] ?? '';
  $('text').scrollTop=0;
}
function transform() {$('scene').setAttribute('transform',`translate(${view.x},${view.y}) scale(${view.scale})`);}
function fit() {
  const box=$('graph').getBoundingClientRect();
  view.scale=Math.min((box.width-40)/bounds.width,(box.height-40)/bounds.height,1.5);
  view.scale=Math.max(.02,view.scale); view.x=(box.width-bounds.width*view.scale)/2; view.y=(box.height-bounds.height*view.scale)/2; transform();
}
function zoom(factor,x,y) {
  const next=Math.max(.02,Math.min(4,view.scale*factor)),ratio=next/view.scale;
  view.x=x-(x-view.x)*ratio; view.y=y-(y-view.y)*ratio; view.scale=next;transform();
}
$('graph').addEventListener('wheel',e=>{e.preventDefault();const b=$('graph').getBoundingClientRect();zoom(Math.exp(-e.deltaY*.001),e.clientX-b.left,e.clientY-b.top);},{passive:false});
let drag=null;
$('graph').addEventListener('pointerdown',e=>{if(e.target.closest('.node'))return;drag={x:e.clientX,y:e.clientY};$('graph').setPointerCapture(e.pointerId);});
$('graph').addEventListener('pointermove',e=>{if(!drag)return;view.x+=e.clientX-drag.x;view.y+=e.clientY-drag.y;drag={x:e.clientX,y:e.clientY};transform();});
for(const event of ['pointerup','pointercancel']) $('graph').addEventListener(event,()=>drag=null);
$('step').onchange=loadFile; $('group').onchange=showGroup; $('refresh').onclick=refresh; $('fit').onclick=fit;
$('color-mode').onchange=updateColors;
$('record').onchange=()=>{recordIndex=Number($('record').value);showDetails();}; $('field').onchange=showText;
for (const event of ['dblclick','keydown']) $('details').addEventListener(event,moveDetailToTop);
for(const [id,factor] of [['zoom-in',1.25],['zoom-out',.8]]) $(id).onclick=()=>{const b=$('graph').getBoundingClientRect();zoom(factor,b.width/2,b.height/2);};
for (const name of ['saved', 'decision']) {
  $(name + '-tab').onclick = () => selectResultTab(name);
  $(name + '-tab').onkeydown = event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 'saved' : event.key === 'End' ? 'decision' : name === 'saved' ? 'decision' : 'saved';
    selectResultTab(next);
    $(next + '-tab').focus();
  };
}
$('decision-highlight').onchange = showDecisionResults;
$('decision-threshold').oninput = showDecisionResults;
refresh();
