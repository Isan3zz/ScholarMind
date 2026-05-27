import assert from 'node:assert/strict';
import test from 'node:test';

import { createReportReviewGate } from './reportReviewGate.js';

test('holds writer draft until reviewer passes it', () => {
    const gate = createReportReviewGate();

    const writerResult = gate.holdDraft('unreviewed draft');
    const passResult = gate.review('PASS');

    assert.deepEqual(writerResult, { action: 'hold' });
    assert.deepEqual(passResult, { action: 'publish', report: 'unreviewed draft' });
});

test('discards failed writer draft instead of publishing it', () => {
    const gate = createReportReviewGate();

    gate.holdDraft('failed draft');
    const failResult = gate.review('FAIL');
    const passResult = gate.review('PASS');

    assert.deepEqual(failResult, { action: 'discard' });
    assert.deepEqual(passResult, { action: 'none' });
});
