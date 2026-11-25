'use strict';
const DecisionModelView = (() => {
  const choices = [
    'The reasoning step supports the chosen action',
    'The reasoning step contradicts the chosen action',
    'The reasoning step makes no claim on what the correct action is',
  ];
  const validProbability = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1;
  function probabilities(values) {
    return choices.map(choice => validProbability(values?.[choice]) ? values[choice] : null);
  }
  function probabilityText(values) {
    return `(${values.map(value => validProbability(value) ? `${(value * 100).toFixed(2)}%` : '—').join(', ')})`;
  }
  function highlighted(values, option, threshold) {
    if (option === 'off' || !Number.isFinite(threshold) || threshold < 0 || threshold > 100) return false;
    return values.some((value, index) => validProbability(value) && (option === 'any' || option === String(index)) && value * 100 > threshold);
  }
  function trace(record) {
    const structured = record?.reasoning_trace;
    if (Array.isArray(structured?.steps) && structured.steps.every(step => typeof step === 'string') && typeof structured.action === 'string') {
      return {action: structured.action, steps: structured.steps.map((text, index) => ({number:index + 1, text}))};
    }
    // Older logs saved only the formatted trace. Match numbered paragraph starts,
    // never inline references such as "consider [2]".
    const saved = typeof record?.reasoning_steps === 'string' ? record.reasoning_steps : '';
    const action = /<action>([\s\S]*?)<\/action>\s*$/i.exec(saved);
    const reasoning = action ? saved.slice(0, action.index).trim() : saved.trim();
    const markers = [...reasoning.matchAll(/(?:^|\n\s*\n)\[(\d+)\][ \t]/g)];
    const steps = markers.map((match, index) => ({
      number: Number(match[1]),
      text: reasoning.slice(match.index + match[0].length, markers[index + 1]?.index ?? reasoning.length).trim(),
    }));
    const output = typeof record?.output === 'string' ? record.output.split(/<\/think>/i).at(-1) : '';
    const matches = [...output.matchAll(/<(search|answer|action)>([\s\S]*?)<\/\1>/gi)];
    const label = matches.map(m => m[1].toLowerCase() === 'action' ? m[2].trim() || '[Empty action]' : m[0]).join(' | ');
    return {action: action?.[1]?.trim() ?? label, steps};
  }
  function advantage(record) {
    const entries = record?.reasoning_step_advantages;
    if (!entries) return null;
    const values = Object.values(entries)
      .map(entry => entry?.advantage).filter(value => typeof value === 'number' && Number.isFinite(value));
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  }
  function nodeAdvantage(records) {
    const values = records.map(record => advantage(record)).filter(value => value !== null);
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  }
  const validEntropy = value => typeof value === 'number' && Number.isFinite(value) && value >= 0;
  function entropyText(entry) {
    if (!validEntropy(entry?.mean)) return 'Entropy: —';
    const total = validEntropy(entry.sum) ? entry.sum.toPrecision(5) : '—';
    return `Entropy: ${entry.mean.toPrecision(5)} nats/token · total ${total} nats · ${entry.token_count} tokens`;
  }
  function nodeAnswerEntropy(records) {
    const values = records.map(record => record?.answer_entropy?.mean).filter(validEntropy);
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  }
  return {choices, probabilities, probabilityText, highlighted, trace, advantage, nodeAdvantage, entropyText, nodeAnswerEntropy};
})();
