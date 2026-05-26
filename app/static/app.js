const userTag = document.getElementById('user-tag');
const switchUserButton = document.getElementById('switch-user-button');
const newSessionButton = document.getElementById('new-session-button');
const sessionList = document.getElementById('session-list');
const uploadButton = document.getElementById('upload-button');
const videoFile = document.getElementById('video-file');
const uploadProgress = document.getElementById('upload-progress');
const uploadStatus = document.getElementById('upload-status');
const messages = document.getElementById('messages');
const chatForm = document.getElementById('chat-form');
const questionInput = document.getElementById('question-input');
const sendButton = document.getElementById('send-button');

const USER_KEY = 'mbe_user';
const SESSION_KEY = 'mbe_session_id';

let currentUser = null;
let currentSessionId = localStorage.getItem(SESSION_KEY);
let currentVideoId = null;
let currentFrames = [];
let chatSource = null;
let progressSource = null;

init();

async function init() {
  currentUser = await ensureUser();
  userTag.textContent = currentUser.username;
  await refreshSessions();
  if (currentSessionId) {
    await restoreSession(currentSessionId);
  }
  questionInput.focus();
}

switchUserButton.addEventListener('click', () => {
  localStorage.removeItem(USER_KEY);
  localStorage.removeItem(SESSION_KEY);
  window.location.reload();
});

newSessionButton.addEventListener('click', async () => {
  await createSession(null, true);
});

uploadButton.addEventListener('click', () => {
  const file = videoFile.files[0];
  if (!file) {
    setUploadStatus('Choose a video first.');
    return;
  }

  uploadButton.disabled = true;
  uploadProgress.value = 0;
  setUploadStatus('Uploading...');

  const form = new FormData();
  form.append('file', file);

  const xhr = new XMLHttpRequest();
  const userParam = currentUser ? `?user_id=${encodeURIComponent(currentUser.user_id)}` : '';
  xhr.open('POST', `/upload${userParam}`);
  xhr.upload.onprogress = (event) => {
    if (event.lengthComputable) {
      uploadProgress.value = Math.round((event.loaded / event.total) * 35);
    }
  };
  xhr.onload = async () => {
    uploadButton.disabled = false;
    if (xhr.status >= 400) {
      setUploadStatus(readError(xhr.responseText));
      return;
    }
    const payload = JSON.parse(xhr.responseText);
    currentVideoId = payload.video_id;
    if (!currentSessionId) {
      await createSession(currentVideoId, false);
    } else {
      await updateSessionVideo(currentSessionId, currentVideoId);
    }
    if (payload.status === 'done') {
      onVideoReady();
    } else {
      subscribePreprocess(payload.stream_url || `/api/preprocess_stream/${payload.video_id}`);
    }
  };
  xhr.onerror = () => {
    uploadButton.disabled = false;
    setUploadStatus('Upload failed.');
  };
  xhr.send(form);
});

chatForm.addEventListener('submit', (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question || !currentUser) return;

  questionInput.value = '';
  appendMessage('user', question);
  const assistant = appendMessage('assistant', '', []);
  sendButton.disabled = true;

  const params = new URLSearchParams({
    user_id: currentUser.user_id,
    question,
  });
  if (currentSessionId) params.set('session_id', currentSessionId);
  if (currentVideoId) params.set('video_id', currentVideoId);

  closeChatSource();
  chatSource = new EventSource(`/api/chat_stream?${params.toString()}`);

  chatSource.addEventListener('frames', (event) => {
    const payload = JSON.parse(event.data);
    if (payload.session_id) {
      currentSessionId = payload.session_id;
      localStorage.setItem(SESSION_KEY, currentSessionId);
    }
    currentFrames = payload.frames || [];
    renderGallery(assistant.article, currentFrames);
    refreshSessions();
  });

  chatSource.addEventListener('token', (event) => {
    const payload = JSON.parse(event.data);
    assistant.text += payload.text || '';
    renderAnswer(assistant.bubble, assistant.text, currentFrames);
    scrollMessages();
  });

  chatSource.addEventListener('done', (event) => {
    const payload = JSON.parse(event.data);
    if (payload.session_id) {
      currentSessionId = payload.session_id;
      localStorage.setItem(SESSION_KEY, currentSessionId);
    }
    sendButton.disabled = false;
    closeChatSource();
    refreshSessions();
    questionInput.focus();
  });

  chatSource.addEventListener('error', (event) => {
    let detail = 'Chat failed.';
    if (event.data) {
      detail = JSON.parse(event.data).detail || detail;
    }
    assistant.bubble.textContent = detail;
    sendButton.disabled = false;
    closeChatSource();
  });
});

async function ensureUser() {
  const cached = localStorage.getItem(USER_KEY);
  if (cached) {
    try {
      return JSON.parse(cached);
    } catch {
      localStorage.removeItem(USER_KEY);
    }
  }
  const response = await fetch('/api/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: null }),
  });
  if (!response.ok) throw new Error('Login failed.');
  const user = await response.json();
  localStorage.setItem(USER_KEY, JSON.stringify(user));
  return user;
}

async function createSession(videoId = null, clear = true) {
  const response = await fetch('/api/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: currentUser.user_id,
      video_id: videoId,
      title: null,
    }),
  });
  if (!response.ok) throw new Error('Could not create session.');
  const payload = await response.json();
  currentSessionId = payload.session_id;
  currentVideoId = videoId;
  currentFrames = [];
  localStorage.setItem(SESSION_KEY, currentSessionId);
  if (clear) messages.textContent = '';
  await refreshSessions();
}

async function updateSessionVideo(sessionId, videoId) {
  await fetch(`/api/sessions/${sessionId}?user_id=${encodeURIComponent(currentUser.user_id)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video_id: videoId }),
  });
  await refreshSessions();
}

async function refreshSessions() {
  if (!currentUser) return;
  const response = await fetch(`/api/sessions?user_id=${encodeURIComponent(currentUser.user_id)}`);
  if (!response.ok) return;
  const sessions = await response.json();
  renderSessions(sessions);
}

function renderSessions(sessions) {
  sessionList.textContent = '';
  if (!sessions.length) {
    const empty = document.createElement('div');
    empty.className = 'empty-state';
    empty.textContent = 'No sessions yet.';
    sessionList.appendChild(empty);
    return;
  }

  let currentDate = '';
  for (const session of sessions) {
    const date = new Date(session.updated_at.replace(' ', 'T')).toLocaleDateString();
    if (date !== currentDate) {
      currentDate = date;
      const group = document.createElement('div');
      group.className = 'session-date';
      group.textContent = date;
      sessionList.appendChild(group);
    }
    const item = document.createElement('button');
    item.type = 'button';
    item.className = `session-item ${session.session_id === currentSessionId ? 'active' : ''}`;
    item.innerHTML = `
      <span>${escapeHtml(session.title || 'New session')}</span>
      <small>${escapeHtml(session.video_filename || session.video_id || 'No video')}</small>
    `;
    item.addEventListener('click', () => restoreSession(session.session_id));
    sessionList.appendChild(item);
  }
}

async function restoreSession(sessionId) {
  const response = await fetch(
    `/api/sessions/${sessionId}/messages?user_id=${encodeURIComponent(currentUser.user_id)}`,
  );
  if (!response.ok) {
    localStorage.removeItem(SESSION_KEY);
    return;
  }
  const payload = await response.json();
  currentSessionId = sessionId;
  currentVideoId = payload.video_id || null;
  currentFrames = [];
  localStorage.setItem(SESSION_KEY, sessionId);
  messages.textContent = '';
  for (const message of payload.messages || []) {
    appendMessage(message.role === 'assistant' ? 'assistant' : 'user', message.content);
  }
  await refreshSessions();
}

function subscribePreprocess(url) {
  closeProgressSource();
  setUploadStatus('Preprocessing...');
  progressSource = new EventSource(url);
  progressSource.addEventListener('stage', (event) => {
    const payload = JSON.parse(event.data);
    const progress = payload.progress == null ? uploadProgress.value / 100 : payload.progress;
    uploadProgress.value = Math.max(uploadProgress.value, Math.round(progress * 100));
    setUploadStatus(payload.label);
  });
  progressSource.addEventListener('done', () => {
    onVideoReady();
    closeProgressSource();
  });
  progressSource.addEventListener('error', (event) => {
    let detail = 'Preprocessing failed.';
    if (event.data) {
      detail = JSON.parse(event.data).detail || detail;
    }
    setUploadStatus(detail);
    closeProgressSource();
  });
}

function onVideoReady() {
  uploadProgress.value = 100;
  setUploadStatus('Video ready.');
  questionInput.focus();
}

function appendMessage(role, text, frames = []) {
  const article = document.createElement('article');
  article.className = `message ${role}`;

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  if (role === 'assistant') {
    renderAnswer(bubble, text, frames);
  } else {
    bubble.textContent = text;
  }
  article.appendChild(bubble);
  messages.appendChild(article);

  if (frames.length) {
    renderGallery(article, frames);
  }
  scrollMessages();
  return { article, bubble, text };
}

const FRAME_PLACEHOLDER_PREFIX = '@@MBE_FRAME_';
const FRAME_PLACEHOLDER_SUFFIX = '@@';

function renderAnswer(container, text, frames) {
  container.textContent = '';
  const marker = /\[FRAME:t=([0-9]+(?:\.[0-9]+)?)\]/g;
  const seenFrames = [];
  const masked = text.replace(marker, (_m, ts) => {
    const idx = seenFrames.length;
    seenFrames.push(closestFrame(Number(ts), frames));
    return `${FRAME_PLACEHOLDER_PREFIX}${idx}${FRAME_PLACEHOLDER_SUFFIX}`;
  });

  if (typeof window.marked === 'undefined' || typeof window.DOMPurify === 'undefined') {
    // CDN libs not loaded (network blocked / slow). Fall back to plain text but
    // keep newlines visible so the answer is still readable.
    container.classList.add('plain-text-fallback');
    if (!window.__mbeMarkdownWarned) {
      window.__mbeMarkdownWarned = true;
      console.warn('[mbe] marked/DOMPurify not loaded — markdown rendering disabled');
    }
    appendMaskedTextFallback(container, masked, seenFrames);
    return;
  }
  container.classList.remove('plain-text-fallback');

  const rawHtml = window.marked.parse(masked, { breaks: true, gfm: true });
  container.innerHTML = window.DOMPurify.sanitize(rawHtml);

  if (typeof window.renderMathInElement === 'function') {
    try {
      window.renderMathInElement(container, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false },
          { left: '\\(', right: '\\)', display: false },
          { left: '\\[', right: '\\]', display: true },
        ],
        throwOnError: false,
        ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
      });
    } catch (_) {
      // never let a bad LaTeX expression kill the whole render
    }
  }

  swapFramePlaceholders(container, seenFrames);
}

function appendMaskedTextFallback(container, masked, frames) {
  const re = new RegExp(`${FRAME_PLACEHOLDER_PREFIX}(\\d+)${FRAME_PLACEHOLDER_SUFFIX}`, 'g');
  let lastIndex = 0;
  let match;
  while ((match = re.exec(masked)) !== null) {
    appendText(container, masked.slice(lastIndex, match.index));
    const frame = frames[Number(match[1])];
    if (frame) container.appendChild(inlineFrame(frame));
    lastIndex = re.lastIndex;
  }
  appendText(container, masked.slice(lastIndex));
}

function swapFramePlaceholders(container, frames) {
  const re = new RegExp(`${FRAME_PLACEHOLDER_PREFIX}(\\d+)${FRAME_PLACEHOLDER_SUFFIX}`);
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null);
  const targets = [];
  let node;
  while ((node = walker.nextNode())) {
    if (re.test(node.nodeValue)) targets.push(node);
  }
  for (const textNode of targets) {
    const parent = textNode.parentNode;
    if (!parent) continue;
    const reGlobal = new RegExp(`${FRAME_PLACEHOLDER_PREFIX}(\\d+)${FRAME_PLACEHOLDER_SUFFIX}`, 'g');
    const value = textNode.nodeValue;
    let cursor = 0;
    let m;
    const fragment = document.createDocumentFragment();
    while ((m = reGlobal.exec(value)) !== null) {
      if (m.index > cursor) {
        fragment.appendChild(document.createTextNode(value.slice(cursor, m.index)));
      }
      const frame = frames[Number(m[1])];
      if (frame) {
        fragment.appendChild(inlineFrame(frame));
      } else {
        fragment.appendChild(document.createTextNode(m[0]));
      }
      cursor = reGlobal.lastIndex;
    }
    if (cursor < value.length) {
      fragment.appendChild(document.createTextNode(value.slice(cursor)));
    }
    parent.replaceChild(fragment, textNode);
  }
}

function appendText(container, text) {
  if (!text) return;
  container.appendChild(document.createTextNode(text));
}

function inlineFrame(frame) {
  const wrapper = document.createElement('span');
  wrapper.className = 'inline-frame';
  const img = document.createElement('img');
  img.src = `data:image/jpeg;base64,${frame.image_b64}`;
  img.alt = `${frame.timestamp.toFixed(1)}s`;
  const label = document.createElement('span');
  label.textContent = `${frame.timestamp.toFixed(1)}s`;
  wrapper.appendChild(img);
  wrapper.appendChild(label);
  return wrapper;
}

function renderGallery(article, frames) {
  const existing = article.querySelector('.gallery');
  if (existing) existing.remove();
  if (!frames.length) return;

  const gallery = document.createElement('div');
  gallery.className = 'gallery';
  for (const frame of frames) {
    const item = document.createElement('figure');
    const img = document.createElement('img');
    const caption = document.createElement('figcaption');
    img.src = `data:image/jpeg;base64,${frame.image_b64}`;
    img.alt = `${frame.timestamp.toFixed(1)}s`;
    caption.textContent = `${frame.timestamp.toFixed(1)}s`;
    item.appendChild(img);
    item.appendChild(caption);
    gallery.appendChild(item);
  }
  article.appendChild(gallery);
  scrollMessages();
}

function closestFrame(timestamp, frames) {
  if (!frames.length || Number.isNaN(timestamp)) return null;
  return frames.reduce((best, frame) => {
    if (!best) return frame;
    return Math.abs(frame.timestamp - timestamp) < Math.abs(best.timestamp - timestamp)
      ? frame
      : best;
  }, null);
}

function setUploadStatus(text) {
  uploadStatus.textContent = text;
}

function scrollMessages() {
  messages.scrollTop = messages.scrollHeight;
}

function closeChatSource() {
  if (chatSource) {
    chatSource.close();
    chatSource = null;
  }
}

function closeProgressSource() {
  if (progressSource) {
    progressSource.close();
    progressSource = null;
  }
}

function readError(text) {
  try {
    const payload = JSON.parse(text);
    return payload.detail || text;
  } catch {
    return text || 'Request failed.';
  }
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => {
    const map = {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;',
    };
    return map[char];
  });
}
