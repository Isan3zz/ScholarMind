// frontend/src/services/api.js

const API_BASE = "http://localhost:8000/api";
const SESSION_THREAD_ID_KEY = "scholarmind_session_thread_id";

function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
  });
}

function getSessionStorage() {
  return typeof sessionStorage === "undefined" ? null : sessionStorage;
}

export function getSessionThreadId() {
  const storage = getSessionStorage();
  const existing = storage?.getItem(SESSION_THREAD_ID_KEY);
  if (existing) return existing;

  const next = generateUUID();
  storage?.setItem(SESSION_THREAD_ID_KEY, next);
  return next;
}

export function resetSessionThreadId() {
  const next = generateUUID();
  getSessionStorage()?.setItem(SESSION_THREAD_ID_KEY, next);
  return next;
}

export function createChatPayload(query, search_mode, intent, threadId = getSessionThreadId()) {
  const payload = {
      query: query,
      search_mode: search_mode,
      thread_id: threadId
  };
  if (intent) {
      payload.intent = intent;
  }
  return payload;
}

/**
 * Batch upload files.
 * @param {Array<File>} files - File objects.
 */
export async function uploadFiles(files) {
    const formData = new FormData();
    files.forEach(file => {
        formData.append('files', file);
    });

    const response = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData
    });

    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Upload failed");
    }

    return await response.json();
}

export async function clearContext() {
  const response = await fetch(`${API_BASE}/clear`, {
      method: "POST"
  });
  if (!response.ok) throw new Error("Failed to clear context");
  return await response.json();
}

/**
 * Stream chat events.
 * @param {string} query - User question.
 * @param {string} searchMode - 'hybrid' | 'document'.
 * @param {function} onMessage - Receive message callback.
 * @param {function} onDone - Completion callback.
 * @param {function} onError - Error callback.
 */
export async function streamChat(query, search_mode, intent, onData, onDone, onError) {
  try {
      const response = await fetch(`${API_BASE}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(createChatPayload(query, search_mode, intent)),
      });

      if (!response.ok) throw new Error('Network error');

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');

      while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');
          for (const line of lines) {
              if (line.startsWith('data: ')) {
                  const dataStr = line.replace('data: ', '').trim();
                  if (dataStr === '[DONE]') {
                      onDone(); return;
                  }
                  try { onData(JSON.parse(dataStr)); } catch(e){}
              }
          }
      }
  } catch (error) { onError(error); }
}
