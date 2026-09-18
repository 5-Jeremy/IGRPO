// Delegation survives Dash replacing rows when the selected node changes.
(function () {
  let pinned = [];
  function toggle(event) {
    const field = event.target.closest("#details th[data-pin-field]");
    if (!field) return;
    if (event.type === "keydown" && (event.repeat || !["Enter", " "].includes(event.key))) return;
    event.preventDefault();
    const name = field.getAttribute("data-pin-field");
    pinned = pinned.includes(name) ? pinned.filter(value => value !== name) : [...pinned, name];
    window.dash_clientside.set_props("pinned-fields", {data: pinned});
  }
  document.addEventListener("dblclick", toggle);
  document.addEventListener("keydown", toggle);
})();
