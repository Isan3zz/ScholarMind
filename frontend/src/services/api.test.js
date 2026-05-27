import assert from 'node:assert/strict';
import test from 'node:test';

import { createChatPayload, getSessionThreadId, resetSessionThreadId } from './api.js';

test('chat payload supports new topic intent', () => {
    const payload = createChatPayload('Analyze a new paper', 'hybrid', 'new_topic', 'thread-1');

    assert.equal(payload.query, 'Analyze a new paper');
    assert.equal(payload.search_mode, 'hybrid');
    assert.equal(payload.intent, 'new_topic');
    assert.equal(payload.thread_id, 'thread-1');
});

test('chat payload supports edit report intent', () => {
    const payload = createChatPayload('Make it more concise', 'document', 'edit_report', 'thread-1');

    assert.equal(payload.intent, 'edit_report');
    assert.equal(payload.thread_id, 'thread-1');
});

test('chat payload supports augment report intent', () => {
    const payload = createChatPayload('Add experiment citations', 'document', 'augment_report', 'thread-1');

    assert.equal(payload.intent, 'augment_report');
    assert.equal(payload.thread_id, 'thread-1');
});

test('chat payload keeps intent optional for compatibility', () => {
    const payload = createChatPayload('Analyze a paper', 'hybrid', undefined, 'thread-1');

    assert.equal(Object.hasOwn(payload, 'intent'), false);
});

test('session thread id is reused from sessionStorage', () => {
    const storage = new Map();
    globalThis.sessionStorage = {
        getItem: (key) => storage.get(key) ?? null,
        setItem: (key, value) => storage.set(key, value),
    };

    const first = getSessionThreadId();
    const second = getSessionThreadId();
    const payload = createChatPayload('Continue analysis', 'hybrid');

    assert.equal(first, second);
    assert.equal(payload.thread_id, first);
});

test('resetSessionThreadId creates a new session thread', () => {
    const storage = new Map();
    globalThis.sessionStorage = {
        getItem: (key) => storage.get(key) ?? null,
        setItem: (key, value) => storage.set(key, value),
    };

    const first = getSessionThreadId();
    const next = resetSessionThreadId();

    assert.notEqual(next, first);
    assert.equal(getSessionThreadId(), next);
});
