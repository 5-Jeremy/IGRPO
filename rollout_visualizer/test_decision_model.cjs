const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const {createContext, runInContext, runInNewContext} = require('node:vm');
const {test} = require('node:test');
const view = runInNewContext(readFileSync(join(__dirname, 'static/decision_model.js'), 'utf8') + '\nDecisionModelView;');
const plain = value => JSON.parse(JSON.stringify(value));

test('probabilities use semantic option order regardless of API key order', () => {
  const values = view.probabilities({[view.choices[2]]: .15, [view.choices[0]]: .8, [view.choices[1]]: .05});
  assert.deepEqual(plain(values), [.8, .05, .15]);
  assert.equal(view.probabilityText(values), '(80.00%, 5.00%, 15.00%)');
  assert.equal(view.probabilityText(view.probabilities(null)), '(—, —, —)');
  assert.deepEqual(plain(view.probabilities({[view.choices[0]]: '0.8', [view.choices[1]]: NaN, [view.choices[2]]: 2})), [null, null, null]);
});

test('highlighting selects an option and uses a strict percentage threshold', () => {
  const values = [.8, .05, .15];
  assert.equal(view.highlighted(values, 'off', 0), false);
  assert.equal(view.highlighted(values, 'any', 80), false);
  assert.equal(view.highlighted(values, 'any', 79), true);
  assert.equal(view.highlighted(values, '0', 79), true);
  assert.equal(view.highlighted(values, '1', 79), false);
  assert.equal(view.highlighted(values, '2', 14), true);
  for (const threshold of [NaN, -1, 101]) assert.equal(view.highlighted(values, 'any', threshold), false);
  assert.equal(view.highlighted([null, null, null], 'any', 0), false);
});

test('structured traces preserve embedded numbering and multiline text exactly', () => {
  const steps = ['A numbered example:\n\n[2] not a new reasoning step.', 'Wait, <script>literal text</script>'];
  assert.deepEqual(plain(view.trace({reasoning_trace:{steps, action:'Lima, Peru'}, reasoning_steps:'ignored'})), {
    action:'Lima, Peru', steps: steps.map((text, index) => ({number:index + 1, text})),
  });
});

test('older formatted logs recover steps, inline references, and action', () => {
  assert.deepEqual(plain(view.trace({reasoning_steps:'[1] See [2] below.\nContinue here.\n\n[2] Wait, reconsider.\n\n<action>population of Lima</action>'})), {
    action:'population of Lima', steps:[{number:1,text:'See [2] below.\nContinue here.'}, {number:2,text:'Wait, reconsider.'}],
  });
  assert.deepEqual(plain(view.trace(null)), {action:'', steps:[]});
  assert.deepEqual(plain(view.trace({output:'<action>Lima population</action>'})), {action:'Lima population', steps:[]});
});

test('node advantages average steps per record, preserve zero, and omit missing values', () => {
  const records = [
    {reasoning_step_advantages:{step_1:{advantage:-2}, step_2:{advantage:0}}},
    {reasoning_step_advantages:{step_1:{advantage:4}, step_2:{advantage:null}, step_3:{advantage:6}}},
    {},
  ];
  assert.equal(view.advantage(records[0]), -1);
  assert.equal(view.advantage({reasoning_step_advantages:{step_1:{advantage:0}}}), 0);
  assert.equal(view.nodeAdvantage(records), 2); // Mean of -1 and 5, not token-weighted.
  assert.equal(view.nodeAdvantage([{}]), null);
});

test('tab controls and record changes render aligned results and clear stale values', () => {
  // A minimal DOM exercises the actual event handlers without browser dependencies.
  class Element {
    constructor() { this.children = []; this.dataset = {}; this.attributes = {}; this.value = ''; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    setAttribute(key, value) { this.attributes[key] = value; }
    addEventListener() {}
    focus() { this.focused = true; }
  }
  const html = readFileSync(join(__dirname, 'static/index.html'), 'utf8');
  assert.match(html, /<option value="gigpo-turn">GiGPO turn advantage<\/option>/);
  const elements = Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(match => [match[1], new Element()]));
  const context = createContext({
    document: {getElementById: id => elements[id], createElement: () => new Element()},
    localStorage: {getItem: () => null},
    Option: class extends Element {},
    DecisionModelView: view,
  });
  const app = readFileSync(join(__dirname, 'static/app.js'), 'utf8').replace(/refresh\(\);\s*$/, '');
  runInContext(app, context);
  elements['decision-highlight'].value = '0';
  elements['decision-threshold'].value = '79';
  context.records = [
    {id:'first', reasoning_trace:{steps:['<script>Literal text</script>', 'Wait, reconsider.'], action:'Lima, Peru'},
      reasoning_step_advantages:{step_1:{advantage:-.25, original_advantage:-.25, min:-.25, max:-.25, token_count:4}},
      reasoning_step_entropies:{step_1:{mean:0, sum:0, token_count:4}, step_2:{mean:2, sum:6, token_count:3}},
      answer_entropy:{mean:1.5, sum:6, token_count:4}, entropy_alignment_status:'aligned',
      decision_model_probabilities:{step_1:{[view.choices[1]]:.05, [view.choices[2]]:.15, [view.choices[0]]:.8}}},
    {id:'second', reasoning_trace:{steps:['Different record.'], action:'Lima population'}, decision_model_probabilities:null},
  ];
  runInContext("selected = {kind:'saved', records}; showDetails();", context);
  assert.equal(elements['decision-action'].textContent, 'Lima, Peru');
  assert.equal(elements['decision-answer-entropy'].textContent, 'Answer entropy: 1.5000 nats/token · total 6.0000 nats · 4 tokens');
  const rows = elements['decision-rows'].children;
  assert.equal(rows.length, 2);
  assert.equal(rows[0].children[1].textContent, '<script>Literal text</script>');
  assert.equal(rows[0].children[0].children[0].textContent, '(80.00%, 5.00%, 15.00%)');
  assert.equal(rows[0].children[0].children[1].textContent, 'Advantage: -0.25000');
  assert.equal(rows[0].children[0].children[2].textContent, 'Entropy: 0.0000 nats/token · total 0.0000 nats · 4 tokens');
  assert.equal(rows[1].children[0].children[2].textContent, 'Entropy: 2.0000 nats/token · total 6.0000 nats · 3 tokens');
  assert.equal(rows[0].className, 'decision-highlighted');
  assert.equal(rows[1].children[0].children[0].textContent, '(—, —, —)');
  assert.equal(elements.details.children[0].children.some(row => row.dataset.detailField === 'decision_model_probabilities'), false);
  elements['decision-tab'].onclick();
  assert.equal(elements['saved-panel'].hidden, true);
  assert.equal(elements['decision-panel'].hidden, false);
  assert.equal(elements['decision-tab'].attributes['aria-selected'], 'true');
  elements['decision-tab'].onkeydown({key:'Home', preventDefault(){}});
  assert.equal(elements['saved-panel'].hidden, false);
  assert.equal(elements['saved-tab'].focused, true);
  elements.record.value = '1';
  elements.record.onchange();
  assert.equal(elements['decision-action'].textContent, 'Lima population');
  assert.equal(elements['decision-answer-entropy'].textContent, 'Answer entropy: —');
  assert.equal(elements['decision-rows'].children[0].children[0].children[2].textContent, 'Entropy: —');
  assert.equal(elements['decision-rows'].children.length, 1);
  assert.equal(elements['decision-highlight-count'].textContent, '0 of 1 steps highlighted');
  assert.match(elements['decision-status'].textContent, /No decision model probabilities/);
  elements['decision-threshold'].value = '';
  elements['decision-threshold'].oninput();
  assert.match(elements['decision-highlight-count'].textContent, /Enter a threshold/);
  runInContext("clear('Loading');", context);
  assert.equal(elements['decision-rows'].children.length, 0);
  assert.equal(elements['decision-action'].textContent, 'No saved action.');

  // Exercise graph coloring with positive, negative, zero, and unavailable data.
  function node(value) {
    const rect = new Element(), title = new Element(), text = new Element();
    return {kind:'saved', label:'node', records:[{reasoning_step_advantages:{step_1:{advantage:value}}}],
      element:{querySelector: tag => tag === 'rect' ? rect : title, querySelectorAll: () => [text]}, rect, title};
  }
  const nodes = [node(-2), node(2), node(0), node(null)];
  context.nodes = nodes;
  elements['color-mode'].value = 'reasoning';
  runInContext('renderedNodes = new Map(nodes.map((node, i) => [i, node])); updateColors();', context);
  assert.equal(nodes[0].rect.attributes.fill, 'hsl(25 65% 40%)');
  assert.equal(nodes[1].rect.attributes.fill, 'hsl(210 65% 40%)');
  assert.equal(nodes[2].rect.attributes.fill, 'hsl(210 65% 96%)');
  assert.equal(nodes[3].rect.attributes.fill, '#e2e8f0');
  assert.match(nodes[0].title.textContent, /mean across steps; mean across records/);
  assert.equal(elements['reasoning-legend'].hidden, false);

  // GiGPO uses the saved turn component, independently of reasoning advantages.
  nodes[0].records = [{gigpo_turn_advantage:-1, gigpo_weighted_turn_advantage:0}];
  nodes[1].records = [{gigpo_turn_advantage:1}, {gigpo_turn_advantage:3}, {gigpo_turn_advantage:null}];
  nodes[2].records = [{gigpo_turn_advantage:0}];
  nodes[3].records = [{}, {gigpo_turn_advantage:null}, {gigpo_turn_advantage:'2'}, {gigpo_turn_advantage:NaN}];
  elements['color-mode'].value = 'gigpo-turn';
  runInContext('updateColors();', context);
  assert.equal(nodes[0].rect.attributes.fill, 'hsl(25 65% 68%)');
  assert.equal(nodes[1].rect.attributes.fill, 'hsl(210 65% 40%)');
  assert.equal(nodes[2].rect.attributes.fill, 'hsl(210 65% 96%)');
  assert.equal(nodes[3].rect.attributes.fill, '#e2e8f0');
  assert.match(nodes[0].title.textContent, /GiGPO turn advantage .*before weighting and reasoning adjustment/);
  assert.match(nodes[1].title.textContent, /mean across records\): 2$/);
  assert.equal(elements['reasoning-legend'].hidden, false);
  assert.equal(elements['advantage-negative-label'].textContent, '−2.00');
  assert.equal(elements['advantage-positive-label'].textContent, '+2.00');
  // Changing modes must restore the reasoning scale rather than reusing GiGPO.
  elements['color-mode'].value = 'reasoning';
  runInContext('updateColors();', context);
  assert.equal(nodes[0].rect.attributes.fill, '#e2e8f0');
  // All-zero GiGPO values are neutral, and the context root retains its color.
  nodes[0].records = [{gigpo_turn_advantage:0}];
  nodes[1].records = [];
  nodes[3].kind = 'root';
  nodes[3].children = [];
  elements['color-mode'].value = 'gigpo-turn';
  runInContext('updateColors();', context);
  assert.equal(nodes[0].rect.attributes.fill, 'hsl(210 65% 96%)');
  assert.equal(nodes[3].rect.attributes.fill, '#56b4e9');
  // Entropy uses a sequential scale, preserves zero, averages records, and clears missing values.
  nodes[0].records = [{answer_entropy:{mean:0}}];
  nodes[1].records = [{answer_entropy:{mean:1}}, {answer_entropy:{mean:3}}, {}];
  nodes[2].records = [{answer_entropy:{mean:null}}, {answer_entropy:{mean:NaN}}, {answer_entropy:{mean:'2'}}];
  elements['color-mode'].value = 'answer-entropy';
  runInContext('updateColors();', context);
  assert.equal(nodes[0].rect.attributes.fill, 'hsl(210 65% 85%)');
  assert.equal(nodes[1].rect.attributes.fill, 'hsl(210 65% 30%)');
  assert.equal(nodes[2].rect.attributes.fill, '#e2e8f0');
  assert.equal(nodes[3].rect.attributes.fill, '#56b4e9');
  assert.match(nodes[1].title.textContent, /answer entropy .*mean across records\): 2$/);
  assert.equal(elements['entropy-legend'].hidden, false);
  assert.equal(elements['reasoning-legend'].hidden, true);
  assert.equal(elements['entropy-low-label'].textContent, '0.00 nats/token');
  assert.equal(elements['entropy-high-label'].textContent, '2.00 nats/token');
  nodes[1].records = [{answer_entropy:{mean:0}}];
  runInContext('updateColors();', context);
  assert.equal(nodes[0].rect.attributes.fill, 'hsl(210 65% 57.5%)');
  // Node details retain the exact per-record scalar alongside graph averages.
  context.records[0].gigpo_turn_advantage = -0.125;
  runInContext("selected = {kind:'saved', records}; recordIndex = 0; showDetails();", context);
  const detail = elements.details.children[0].children.find(row => row.dataset.detailField === 'gigpo_turn_advantage');
  assert.equal(detail.children[1].children[0].textContent, '-0.125');

  assert.equal(runInContext("searchQuestion('Your question: Where is Lima?\\n\\nHistory: ...')", context), 'Where is Lima?');

  // The action panel follows saved ancestry, not the prompt's recent history.
  context.turns = Array.from({length:50}, (_, i) => ({id:`t-${i}`,
    previous_step_ids:Array.from({length:i}, (_, j) => `t-${j}`),
    input:'Only the last two turns are visible here.', output:`<Action>look ${i}</Action>`}));
  context.turns[20].output = 'Invalid untagged output';
  runInContext("group = {nodes:[...turns].reverse().map(r => ({records:[r]}))}; selected = {kind:'saved'};", context);
  const actions = runInContext('generatedActions(turns[49])', context).split('\n');
  assert.equal(actions.length, 50);
  assert.equal(actions[0], '1. look 0');
  assert.equal(actions[20], '21. [No action tag]');
  assert.equal(actions[49], '50. look 49');
  runInContext("group.nodes = group.nodes.filter(n => n.records[0].id !== 't-10');", context);
  assert.match(runInContext('generatedActions(turns[49])', context), /11\. \[Missing saved action: t-10\]/);
});
