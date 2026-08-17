//! Best-effort local host hardware inventory (CPU, RAM, GPU) for
//! `GET /api/v1/hardware`.
//!
//! Core and the inference sidecar are commonly deployed on different
//! hardware (a CPU-only host running `armor-api` next to a GPU box running
//! `armor-inference`, or vice versa), so this reports *this process's* host
//! only — a sibling to `armor_inference.hardware.get_hardware_info()`, which
//! does the same for the sidecar's own host. `control_plane.rs` combines
//! both into one response so the UI can show them side by side rather than
//! one number implying they share a machine.
//!
//! Standard library plus a `nvidia-smi` shell-out only — no new crate for a
//! nice-to-have, matching this service's dependency posture elsewhere.

use std::{collections::HashSet, fs, process::Command};

use serde::Serialize;

#[derive(Serialize)]
pub struct CpuInfo {
    pub model: Option<String>,
    pub architecture: String,
    pub logical_cores: Option<usize>,
    pub physical_cores: Option<usize>,
}

#[derive(Serialize)]
pub struct MemoryInfo {
    pub total_bytes: Option<u64>,
}

#[derive(Serialize)]
pub struct GpuInfo {
    pub name: String,
    pub memory_total_mb: Option<u64>,
    pub driver_version: Option<String>,
}

#[derive(Serialize)]
pub struct HardwareInfo {
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub gpus: Vec<GpuInfo>,
    pub os: String,
}

/// `(model name, physical core count)` from `/proc/cpuinfo`. Physical cores
/// is the count of distinct (physical id, core id) pairs across logical
/// processors — `available_parallelism()` counts hyperthreads too, which
/// overstates cores on any SMT-enabled host.
fn linux_cpuinfo() -> (Option<String>, Option<usize>) {
    let Ok(text) = fs::read_to_string("/proc/cpuinfo") else {
        return (None, None);
    };

    let mut model = None;
    let mut pairs: HashSet<(String, String)> = HashSet::new();
    let mut phys_id: Option<String> = None;
    let mut core_id: Option<String> = None;
    for line in text.lines() {
        if line.trim().is_empty() {
            if let (Some(p), Some(c)) = (phys_id.take(), core_id.take()) {
                pairs.insert((p, c));
            }
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim().to_string();
        match key.trim().to_lowercase().as_str() {
            "model name" if model.is_none() => model = Some(value),
            "physical id" => phys_id = Some(value),
            "core id" => core_id = Some(value),
            _ => {}
        }
    }
    if let (Some(p), Some(c)) = (phys_id, core_id) {
        pairs.insert((p, c));
    }
    (model, (!pairs.is_empty()).then_some(pairs.len()))
}

fn linux_mem_total_bytes() -> Option<u64> {
    let text = fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

fn os_description() -> String {
    if let Ok(text) = fs::read_to_string("/etc/os-release") {
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("PRETTY_NAME=") {
                return rest.trim().trim_matches('"').to_string();
            }
        }
    }
    std::env::consts::OS.to_string()
}

/// Queries `nvidia-smi` if it's on `PATH`. Absent binary, no GPU, or a
/// failed/non-zero call all just mean "no GPUs to report" — this must never
/// turn a CPU-only host into a request failure.
fn nvidia_gpus() -> Vec<GpuInfo> {
    let Ok(output) = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let mut gpus = Vec::new();
    for line in text.lines() {
        let parts: Vec<&str> = line.split(',').map(str::trim).collect();
        let [name, mem, ..] = parts.as_slice() else {
            continue;
        };
        if name.is_empty() {
            continue;
        }
        gpus.push(GpuInfo {
            name: name.to_string(),
            memory_total_mb: mem.parse::<f64>().ok().map(|v| v as u64),
            driver_version: parts
                .get(2)
                .filter(|v| !v.is_empty())
                .map(|v| v.to_string()),
        });
    }
    gpus
}

/// This process's host: the machine `armor-api` itself is running on.
pub fn local_hardware_info() -> HardwareInfo {
    let (model, physical_cores) = linux_cpuinfo();
    let logical_cores = std::thread::available_parallelism().ok().map(|n| n.get());
    let mem_total = linux_mem_total_bytes();

    HardwareInfo {
        cpu: CpuInfo {
            model,
            architecture: std::env::consts::ARCH.to_string(),
            logical_cores,
            physical_cores,
        },
        memory: MemoryInfo {
            total_bytes: mem_total,
        },
        gpus: nvidia_gpus(),
        os: os_description(),
    }
}
