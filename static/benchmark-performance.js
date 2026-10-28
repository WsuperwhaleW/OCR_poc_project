/* Bounded, conditional cache. Failures never replace the last successful entry. */
function createEvidenceCache({fetch: request, now = Date.now, ttl = 5000, maxEntries = 20, maxBytes = 16 * 1024 * 1024}) {
  const entries = new Map(), pending = new Map();
  let bytes = 0;
  return async function load(url, revision = '') {
    const key = url + '|' + revision;
    if (pending.has(key)) return pending.get(key);
    const old = entries.get(key);
    if (old && now() - old.at < ttl) {
      entries.delete(key); entries.set(key, old); return old.data;
    }
    const task = (async () => {
      const response = await request(url, {headers: old?.etag ? {'If-None-Match': old.etag} : {}});
      if (response.status === 304 && old) { old.at = now(); return old.data; }
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'This page is no longer kept.');
      const size = JSON.stringify(data).length * 2;
      if (old) { bytes -= old.size; entries.delete(key); }
      if (size <= maxBytes) {
        entries.set(key, {data, size, at: now(), etag: response.headers?.get('ETag')}); bytes += size;
        while (entries.size > maxEntries || bytes > maxBytes) {
          const first = entries.keys().next().value;
          bytes -= entries.get(first).size; entries.delete(first);
        }
      }
      return data;
    })();
    pending.set(key, task);
    try { return await task; } finally { pending.delete(key); }
  };
}

/* Keep existing rows, focus, scroll positions and open details while applying
   new report markup. Alert rows have stable IDs even when inserted at the top. */
function patchStressReport(host, html) {
  const template = document.createElement('template'); template.innerHTML = html;
  const key = node => node.nodeType === 1
    ? node.id || (node.hasAttribute('data-alert-id') ? 'alert:' + node.dataset.alertId : '') : '';
  function patch(parent, fresh) {
    const keyed = new Map([...parent.childNodes].filter(key).map(node => [key(node), node]));
    let cursor = parent.firstChild;
    for (const next of [...fresh.childNodes]) {
      const id = key(next);
      let node = id ? keyed.get(id) : cursor;
      if (!id && node && key(node)) node = null;
      if (!node || node.nodeType !== next.nodeType || node.nodeName !== next.nodeName) {
        node = next.cloneNode(true); parent.insertBefore(node, cursor);
      } else {
        if (node !== cursor) parent.insertBefore(node, cursor);
        if (node.nodeType === 3 || node.nodeType === 8) {
          if (node.nodeValue !== next.nodeValue) node.nodeValue = next.nodeValue;
        } else if (!node.isEqualNode(next)) {
          const selected = node.tagName === 'SELECT' ? node.value : null;
          const preserve = name => (node.tagName === 'DETAILS' && name === 'open');
          for (const attr of [...node.attributes]) if (!preserve(attr.name) && !next.hasAttribute(attr.name)) node.removeAttribute(attr.name);
          for (const attr of [...next.attributes]) if (!preserve(attr.name) && node.getAttribute(attr.name) !== attr.value) node.setAttribute(attr.name, attr.value);
          // Job detail rows are populated separately by stWireChatContext.
          if (!node.hasAttribute('data-chat-rows')) patch(node, next);
          if (selected !== null && [...node.options].some(option => option.value === selected)) node.value = selected;
        }
      }
      cursor = node.nextSibling;
    }
    while (cursor) { const next = cursor.nextSibling; cursor.remove(); cursor = next; }
  }
  patch(host, template.content);
}
