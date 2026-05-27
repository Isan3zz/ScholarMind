export function createReportReviewGate() {
    let pendingReport = '';

    return {
        reset() {
            pendingReport = '';
        },

        holdDraft(report) {
            pendingReport = report || '';
            return { action: 'hold' };
        },

        // 取当前暂存的报告（不消费），用于达到重试上限仍展示草稿
        peekDraft() {
            return pendingReport;
        },

        review(status, { forcePublish = false } = {}) {
            if (status === 'FAIL') {
                if (forcePublish && pendingReport) {
                    const report = pendingReport;
                    pendingReport = '';
                    return { action: 'publish', report };
                }
                pendingReport = '';
                return { action: 'discard' };
            }

            if (status === 'PASS' && pendingReport) {
                const report = pendingReport;
                pendingReport = '';
                return { action: 'publish', report };
            }

            return { action: 'none' };
        },
    };
}
