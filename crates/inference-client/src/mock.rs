//! A scripted [`InferenceTransport`] for tests. This is what drives every
//! escalation test — the escalation logic is covered with **zero network
//! and zero Python**.
//!
//! Responses are queued per task and popped in order, so a test can script a
//! multi-layer sequence (classifier abstains, judge resolves) without any
//! notion of time or concurrency.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;

use crate::contract::{InferRequest, InferResult, MlDecision, ModelInfo};
use crate::transport::{InferError, InferenceTransport};

/// What the mock should do for the next call to a given task.
pub enum Scripted {
    Ok(InferResult),
    Err(InferError),
}

#[derive(Default)]
struct Calls {
    /// Every (task, text) the transport was asked for, in order — so a test
    /// can assert *which view* was scored, not just what came back.
    seen: Vec<(String, String)>,
}

#[derive(Default)]
pub struct MockTransport {
    scripted: Mutex<HashMap<String, Vec<Scripted>>>,
    models: Mutex<Vec<ModelInfo>>,
    calls: Mutex<Calls>,
    /// Popped when a task's queue is empty. `None` means the task is
    /// unknown, which surfaces as [`InferError::UnknownTask`] — the default,
    /// so a test that forgets to script a task fails loudly rather than
    /// silently getting an `Allow`.
    default: Mutex<Option<InferResult>>,
}

impl MockTransport {
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue one response for `task`. Call repeatedly to script a sequence.
    pub fn push(&self, task: &str, response: Scripted) -> &Self {
        self.scripted
            .lock()
            .unwrap()
            .entry(task.to_string())
            .or_default()
            .push(response);
        self
    }

    pub fn push_ok(&self, task: &str, result: InferResult) -> &Self {
        self.push(task, Scripted::Ok(result))
    }

    pub fn push_err(&self, task: &str, error: InferError) -> &Self {
        self.push(task, Scripted::Err(error))
    }

    /// Answer every unscripted call with this result instead of erroring.
    pub fn with_default(&self, result: InferResult) -> &Self {
        *self.default.lock().unwrap() = Some(result);
        self
    }

    pub fn with_models(&self, models: Vec<ModelInfo>) -> &Self {
        *self.models.lock().unwrap() = models;
        self
    }

    /// Every (task, text) scored so far, in call order.
    pub fn calls(&self) -> Vec<(String, String)> {
        self.calls.lock().unwrap().seen.clone()
    }

    pub fn call_count(&self) -> usize {
        self.calls.lock().unwrap().seen.len()
    }
}

/// Convenience constructor for the common "a classifier answered" case.
pub fn result(decision: MlDecision, risk_score: u8, confidence: Option<f32>) -> InferResult {
    InferResult {
        decision,
        risk_score,
        confidence,
        label_scores: None,
        calibrated_score: None,
        threshold: None,
        model_version: "mock-model@v0".to_string(),
    }
}

#[async_trait]
impl InferenceTransport for MockTransport {
    async fn infer(&self, task: &str, req: InferRequest<'_>) -> Result<InferResult, InferError> {
        let scored = req
            .text
            .map(str::to_string)
            .or_else(|| req.texts.map(|t| t.join("\u{1f}")))
            .unwrap_or_default();
        self.calls
            .lock()
            .unwrap()
            .seen
            .push((task.to_string(), scored));

        let next = self
            .scripted
            .lock()
            .unwrap()
            .get_mut(task)
            .and_then(|queue| {
                if queue.is_empty() {
                    None
                } else {
                    Some(queue.remove(0))
                }
            });

        match next {
            Some(Scripted::Ok(r)) => Ok(r),
            Some(Scripted::Err(e)) => Err(e),
            None => match self.default.lock().unwrap().clone() {
                Some(r) => Ok(r),
                None => Err(InferError::UnknownTask(task.to_string())),
            },
        }
    }

    async fn models(&self) -> Result<Vec<ModelInfo>, InferError> {
        Ok(self.models.lock().unwrap().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn scripted_responses_pop_in_order() {
        let mock = MockTransport::new();
        mock.push_ok("prompt_injection", result(MlDecision::Allow, 0, Some(0.9)))
            .push_ok(
                "prompt_injection",
                result(MlDecision::Block, 90, Some(0.95)),
            );

        let first = mock
            .infer("prompt_injection", InferRequest::text("a"))
            .await
            .unwrap();
        let second = mock
            .infer("prompt_injection", InferRequest::text("b"))
            .await
            .unwrap();

        assert_eq!(first.decision, MlDecision::Allow);
        assert_eq!(second.decision, MlDecision::Block);
        assert_eq!(
            mock.calls(),
            vec![
                ("prompt_injection".to_string(), "a".to_string()),
                ("prompt_injection".to_string(), "b".to_string()),
            ]
        );
    }

    #[tokio::test]
    async fn unscripted_task_errors_rather_than_allowing() {
        let mock = MockTransport::new();
        let err = mock
            .infer("ner", InferRequest::text("a"))
            .await
            .unwrap_err();
        assert!(matches!(err, InferError::UnknownTask(t) if t == "ner"));
    }

    #[tokio::test]
    async fn scripted_error_surfaces_to_the_caller() {
        let mock = MockTransport::new();
        mock.push_err("judge", InferError::Timeout { elapsed_ms: 50 });
        let err = mock
            .infer("judge", InferRequest::text("a"))
            .await
            .unwrap_err();
        assert!(matches!(err, InferError::Timeout { .. }));
        assert!(
            !err.is_breaker_signal(),
            "a timeout must not trip the breaker"
        );
    }
}
