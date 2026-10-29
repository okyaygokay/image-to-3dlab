// The GPU VM, from the page: run Python on the rented Colab machine without a terminal.
// Everything else in the studio is a worked-out button; this is the box for the questions
// nobody drew one for — how much disk is left, is Drive mounted, what does the mesh look
// like from here. Code goes to the lab's shared VM exactly the way job legs go (one
// bounded colab exec, output streamed back), so the answer lands beside the question.
// Google Drive is a button because mounting may open a consent tab only the person at
// the machine can click; mounted Drive is for keeping finished models, not installs.

import { sheet, duration } from './pages.js';

const TIMEOUTS = [
  [30, '30 s'], [120, '2 min'], [600, '10 min'], [1800, '30 min'],
];

function statusLine(answer) {
  if (!answer?.live) return `VM asleep or unreachable — ${answer?.note || 'no answer'}`;
  let text = `VM live — ${(answer.free_gb ?? '?')} GB free of ${answer.total_gb ?? '?'} GB`;
  const drive = answer.drive;
  text += drive ? ` · Google Drive mounted at ${drive.mount}` : ' · Google Drive not mounted';
  return text;
}

async function poll(el, id, onDone, watch) {
  // One run, watched until it ends: the tail replaces the pane each time (the server
  // keeps the last 64 KB), so the pane never grows without bound on a chatty run.
  // `watch` sees each tail as it arrives — the mount uses it to surface Google's
  // consent URL the moment the CLI prints it.
  for (;;) {
    let answer;
    try { answer = await (await fetch(`/api/colab/console/${id}`)).json(); }
    catch { await new Promise((r) => setTimeout(r, 1500)); continue; }
    if (typeof answer.output === 'string') el.querySelector('.console-out').textContent = answer.output;
    el.querySelector('.console-out').hidden = false;
    watch?.(answer.output || '');
    if (answer.state !== 'running') {
      el.querySelector('.console-meta').textContent =
        answer.exit === 0 ? `done in ${duration(answer.seconds)}`
          : `failed in ${duration(answer.seconds)} (exit ${answer.exit}) — see the output`;
      el.querySelector('.console-consent').hidden = true;
      onDone?.(answer);
      return;
    }
    await new Promise((r) => setTimeout(r, 1500));
  }
}

async function refreshStatus(el) {
  const line = el.querySelector('.console-status');
  line.textContent = 'Checking the VM…';
  try { line.textContent = statusLine(await (await fetch('/api/colab/vm')).json()); }
  catch { line.textContent = 'Could not reach the studio server for the VM status.'; }
}

export async function openConsole(stage) {
  const el = sheet(stage, `<div class="sheet-card console-card">
    <div class="sheet-top"><h2>GPU VM (Colab)</h2><button type="button" class="ghost" data-close>Close</button></div>
    <p class="meta">Run Python on the rented GPU machine (session <code>i2l-lab</code>) — the same VM the
      steps use. Nothing model-related runs on this computer. A run waits behind whatever
      the VM is already doing, its own clock ticking.</p>
    <p class="console-status meta">Checking the VM…</p>
    <div class="console-tools">
      <button type="button" class="ghost small" data-refresh>Refresh status</button>
      <button type="button" class="ghost small" data-mount>Mount Google Drive</button>
    </div>
    <div class="console-consent" hidden>
      <a target="_blank" rel="noopener">Open the Google consent page</a>
      <button type="button" class="primary small" data-consent>I clicked Allow — continue</button>
      <span class="meta">about a 2-minute limit once it appears</span>
    </div>
    <textarea class="console-code" spellcheck="false" aria-label="Python to run on the VM"
      placeholder='import shutil&#10;total, used, free = shutil.disk_usage("/content")&#10;print(f"free {free/1e9:.0f} of {total/1e9:.0f} GB")'></textarea>
    <div class="console-run">
      <select class="console-timeout" aria-label="Run timeout">${TIMEOUTS.map(([v, label]) => `<option value="${v}">${label}</option>`).join('')}</select>
      <button type="button" class="primary" data-run>Run on the VM</button>
      <span class="console-meta meta"></span>
    </div>
    <pre class="console-out" hidden></pre>
  </div>`);

  const code = el.querySelector('.console-code');
  const meta = el.querySelector('.console-meta');
  const out = el.querySelector('.console-out');
  let current = null;                              // the run a continue click belongs to

  // The consent link arrives in the mount's own output; the moment it does, it becomes
  // a clickable link beside the button that types the Enter the CLI is waiting for.
  const watchConsent = (text) => {
    if (el.dataset.consentShown) return;
    const match = /https:\/\/accounts\.google\.com\/\S+/.exec(text);
    if (!match) return;
    el.dataset.consentShown = '1';
    const box = el.querySelector('.console-consent');
    box.querySelector('a').href = match[0];
    box.hidden = false;
    box.querySelector('[data-consent]').focus();
  };

  el.querySelector('[data-consent]').addEventListener('click', async () => {
    if (!current) return;
    try {
      const answer = await (await fetch('/api/colab/drive/continue', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: current }),
      })).json();
      el.querySelector('.console-meta').textContent = answer?.sent
        ? 'continuing the mount…' : 'no mount is waiting for that — run Mount again';
    } catch { el.querySelector('.console-meta').textContent = 'could not reach the studio server'; }
  });

  const run = async (kind) => {
    const body = kind === 'exec'
      ? { code: code.value, timeout: Number(el.querySelector('.console-timeout').value) }
      : {};
    let answer;
    try { answer = await (await fetch(kind === 'exec' ? '/api/colab/console' : '/api/colab/drive/mount',
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })).json(); }
    catch { meta.textContent = 'could not reach the studio server'; return; }
    if (answer?.error) { meta.textContent = answer.error; return; }
    current = answer.id;
    delete el.dataset.consentShown;
    el.querySelector('.console-consent').hidden = true;
    meta.textContent = kind === 'exec' ? 'running…' : 'mounting — a consent link will appear here…';
    out.hidden = false; out.textContent = '';
    poll(el, answer.id, () => refreshStatus(el), kind === 'mount' ? watchConsent : null);
  };

  el.querySelector('[data-run]').addEventListener('click', () => run('exec'));
  el.querySelector('[data-mount]').addEventListener('click', () => run('mount'));
  el.querySelector('[data-refresh]').addEventListener('click', () => refreshStatus(el));
  code.addEventListener('keydown', (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') run('exec');
  });
  refreshStatus(el);
  code.focus();
}
