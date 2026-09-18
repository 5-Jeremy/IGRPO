window.dash_clientside = Object.assign({}, window.dash_clientside, {
  treeVisualizer: {
    scrollText: function (request) {
      if (!request) return window.dash_clientside.no_update;
      // Wait for React to commit the text and highlighted match from the callback.
      window.requestAnimationFrame(function () {
        window.requestAnimationFrame(function () {
          const text = document.getElementById("node-text");
          if (!text) return;
          if (request.kind === "end") {
            text.scrollTop = text.scrollHeight;
          } else if (request.kind === "reset") {
            text.scrollTop = 0;
            text.scrollLeft = 0;
          } else {
            const match = document.getElementById("current-search-match");
            if (!match) return;
            // Scroll only the text pane, keeping the graph and page in place.
            const bounds = text.getBoundingClientRect();
            const target = match.getBoundingClientRect();
            text.scrollTop += target.top - bounds.top - text.clientHeight / 2 + target.height / 2;
          }
        });
      });
      return {handled: request.nonce};
    }
  }
});
