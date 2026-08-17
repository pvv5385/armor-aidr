"""Runners: one per model kind.

`base` is dependency-free and always importable. Every other module in this
package may import torch/onnxruntime/tokenizers — but only inside
`Runner.load()`, never at module scope, so the registry can decide a task is
unavailable without the import cost or the crash.
"""
