const q = document.getElementById('q');
const method = document.getElementById('method');
const go = document.getElementById('go');
const answer = document.getElementById('answer');
const citations = document.getElementById('citations');

function highlight(text, query) {
  const terms = query.toLowerCase().split(/[^a-z0-9]+/).filter(t => t.length > 2);
  let out = text.replace(/&/g,'&amp;').replace(/</g,'&lt;');
  for (const t of terms) out = out.replace(new RegExp(`\\b(${t})\\b`, 'gi'), '<mark>$1</mark>');
  return out;
}

go.onclick = async () => {
  if (!q.value.trim()) return;
  go.disabled = true; answer.textContent = 'searching 7,000 plots…'; answer.className = 'quiet';
  try {
    const r = await fetch('/api/ask', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question: q.value, method: method.value})});
    const data = await r.json();
    answer.textContent = data.answer; answer.className = '';
    citations.innerHTML = '';
    for (const c of data.citations) {
      const div = document.createElement('div');
      div.className = 'cite';
      div.innerHTML = `<div class="meta">[${c.n}] ${c.title} (${c.year})<span class="score">${c.method || ''} score ${c.score}</span></div>
                       <div>${highlight(c.evidence, q.value)}</div>`;
      citations.appendChild(div);
    }
  } finally { go.disabled = false; }
};
q.addEventListener('keydown', e => { if (e.key === 'Enter') go.click(); });
