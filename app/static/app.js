const userTag = document.getElementById('user-tag');
const uploadSection = document.getElementById('upload-section');
const chatSection = document.getElementById('chat-section');
const uploadButton = document.getElementById('upload-button');
const videoFile = document.getElementById('video-file');
const uploadProgress = document.getElementById('upload-progress');
const uploadStatus = document.getElementById('upload-status');
const messages = document.getElementById('messages');
const chatForm = document.getElementById('chat-form');
const questionInput = document.getElementById('question-input');
const sendButton = document.getElementById('send-button');

let userId = localStorage.getItem('mbe_userId');
if (!userId) {
  userId = crypto.randomUUID().slice(0, 8);
  localStorage.setItem('mbe_userId', userId);
}
userTag.textContent = userId;

let currentVideoId = null;
const history = [];
let pollTimer = null;

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
  xhr.open('POST', '/upload');
  xhr.upload.onprogress = (event) => {
    if (event.lengthComputable) {
      uploadProgress.value = Math.round((event.loaded / event.total) * 100);
    }
  };
  xhr.onload = () => {
    uploadButton.disabled = false;
    if (xhr.status >= 400) {
      setUploadStatus(readError(xhr.responseText));
      return;
    }
    const payload = JSON.parse(xhr.responseText);
    currentVideoId = payload.video_id;
    if (payload.status === 'done') {
      onVideoReady();
    } else {
      setUploadStatus('Preprocessing...');
      pollStatus(payload.video_id);
    }
  };
  xhr.onerror = () => {
    uploadButton.disabled = false;
    setUploadStatus('Upload failed.');
  };
  xhr.send(form);
});

chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question || !currentVideoId) return;

  questionInput.value = '';
  appendMessage('user', question);
  history.push({ role: 'user', content: question });
  sendButton.disabled = true;

  try {
    const response = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        video_id: currentVideoId,
        question,
        history: history.slice(-8),
      }),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || 'Chat failed.');
    }
    const payload = await response.json();
    appendMessage('assistant', payload.answer, payload.frames || []);
    history.push({ role: 'assistant', content: payload.answer });
  } catch (error) {
    appendMessage('assistant', error.message);
  } finally {
    sendButton.disabled = false;
    questionInput.focus();
  }
});

function pollStatus(videoId) {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const response = await fetch(`/status/${videoId}`);
      const payload = await response.json();
      if (payload.status === 'done') {
        clearInterval(pollTimer);
        onVideoReady();
      } else if (payload.status.startsWith('failed:')) {
        clearInterval(pollTimer);
        setUploadStatus(payload.status);
      } else {
        setUploadStatus('Preprocessing...');
      }
    } catch {
      clearInterval(pollTimer);
      setUploadStatus('Status check failed.');
    }
  }, 2000);
}

function onVideoReady() {
  setUploadStatus('Video ready.');
  uploadSection.classList.add('hidden');
  chatSection.classList.remove('hidden');
  questionInput.focus();
}

function appendMessage(role, text, frames = []) {
  const article = document.createElement('article');
  article.className = `message ${role}`;

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  article.appendChild(bubble);

  if (frames.length) {
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
  }

  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
}

function setUploadStatus(text) {
  uploadStatus.textContent = text;
}

function readError(text) {
  try {
    const payload = JSON.parse(text);
    return payload.detail || text;
  } catch {
    return text || 'Request failed.';
  }
}
