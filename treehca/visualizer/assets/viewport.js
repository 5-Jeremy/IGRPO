window.dash_clientside = Object.assign({}, window.dash_clientside, {
  rolloutViewport: {
    fit: function (wholeClicks, subtreeClicks, elements, selection) {
      const no = window.dash_clientside.no_update;
      const host = document.getElementById("graph");
      if (!host || !elements || !elements.length) return [no, no];
      const trigger = window.dash_clientside.callback_context.triggered_id;
      const selected = selection && selection.node;
      let ids = null;
      if (trigger === "fit-subtree") {
        if (!selected) return [no, no];
        const children = new Map();
        for (const element of elements) {
          const data = element.data;
          if (data.source !== undefined) {
            if (!children.has(data.source)) children.set(data.source, []);
            children.get(data.source).push(data.target);
          }
        }
        ids = new Set([selected]);
        const pending = [selected];
        while (pending.length) {
          for (const child of children.get(pending.pop()) || []) {
            if (!ids.has(child)) { ids.add(child); pending.push(child); }
          }
        }
      }
      const nodes = elements.filter(e => e.position && (!ids || ids.has(e.data.id)));
      if (!nodes.length) return [no, no];
      // Include full node boxes and selection borders, not just their centers.
      const left = Math.min(...nodes.map(e => e.position.x)) - 90;
      const right = Math.max(...nodes.map(e => e.position.x)) + 90;
      const top = Math.min(...nodes.map(e => e.position.y)) - 29;
      const bottom = Math.max(...nodes.map(e => e.position.y)) + 29;
      const width = host.clientWidth, height = host.clientHeight;
      const zoom = Math.max(0.01, Math.min(4, (width - 72) / (right - left), (height - 72) / (bottom - top)));
      return [zoom, {x: width / 2 - zoom * (left + right) / 2, y: height / 2 - zoom * (top + bottom) / 2}];
    }
  }
});
