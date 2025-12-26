
# GPU PCIe Diagnostic & Bandwidth Analysis

A deterministic command-line tool for validating GPU PCIe link health, bandwidth, and real-world PCIe utilization using only observable hardware data.

This tool answers one question reliably:

> **Is my GPU’s PCIe link behaving as it should, and can I prove it?**

No registry hacks. 
No BIOS assumptions. 
No “magic” optimizations. 

Only measurable link state, copy throughput, and hardware counters.


## What This Tool Does

This tool performs hardware-observable PCIe diagnostics and reports factual results with deterministic verdicts.

It measures and reports directly from GPU hardware:

- PCIe **current and maximum** link generation and width (via NVML)
- **Peak Host→Device and Device→Host copy bandwidth** using CUDA memcpy timing
- **Sustained PCIe utilization under load** using NVML TX/RX counters
- Efficiency relative to theoretical PCIe payload bandwidth
- Clear VERDICT from observable conditions only

The tool does not attempt to tune, fix, or modify system configuration.


## Verdict Semantics

- **OK** — The negotiated PCIe link and measured throughput are consistent with expected behavior.
- **DEGRADED** — The GPU is operating below its maximum supported PCIe generation or width.
- **UNDERPERFORMING** — The full link is negotiated, but sustained bandwidth is significantly lower than expected.

Verdicts are rule-based and derived only from measured data.


## Why This Tool Exists

Modern systems frequently exhibit PCIe issues that are difficult to diagnose:

- GPUs negotiating **x8 / x4 / x1** instead of **x16**
- PCIe generation downgrades after BIOS or firmware updates
- Slot bifurcation, riser cable, or motherboard lane-sharing issues
- Reduced PCIe bandwidth occurring while system status is reported as normal
- Confusion between PCIe transport limits and workload bottlenecks

This tool exists to:

1. **Reproducible PCIe diagnostic baseline**
2. **Hardware-level proof** of PCIe behavior 
3. **Isolate link negotiation** from kernel/workload effects


## Example Output
```text
GPU PCIe Diagnostic & Bandwidth Analysis v2.7.4
GPU:   NVIDIA GeForce GTX 1080
BDF:   00000000:01:00.0
UUID:  GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx (redacted)

PCIe Link
  Current: Gen3 x16
  Max Cap: Gen3 x16
  Theoretical (payload): 15.76 GB/s
  Transfer Size: 1024 MiB

Peak Copy Bandwidth
  Host → Device: 12.5 GB/s
  Device → Host: 12.7 GB/s

Telemetry (NVML)
  Window:   5.0 s (50 samples @ 100 ms)
  TX avg:   7.6 GB/s
  RX avg:   7.1 GB/s
  Combined: 14.7 GB/s

Verdict
  State:      OK
  Reason:     Throughput and link state are consistent with a healthy PCIe path
  Efficiency: 93.5%

System Signals (informational)
  MaxReadReq: 512 bytes
  Persistence Mode: Disabled
  ASPM Policy (sysfs string): [default] performance powersave powersupersave
  IOMMU: Platform default (no explicit flags)
```
## Requirements

- NVIDIA GPU with a supported driver
- CUDA Toolkit (for `nvcc`)
- NVML development library (`-lnvidia-ml`)

## Platform Compatibility Note

- Linux operating system
- Tested on **Ubuntu 24.04.3 LTS**


## Permissions & Logging Notes

On some Linux systems, PCIe and NVML diagnostics require elevated privileges due to kernel and driver access controls.
If log files were previously created using `sudo`, the results directory may become root-owned. In that case, subsequent runs may prompt for a password when appending logs.

To restore normal user access to the results directory:

```bash
sudo chown -R $USER:$USER results/
```


## Build

Using the provided Makefile:

```bash
make
```


Or manually, with nvcc:

```bash
nvcc -O3 -Xnvlink=-w pcie_diagnostic_pro.cu -o pcie_diag -lnvidia-ml -lpthread
```


## Usage

Basic diagnostic (1 GiB test):

./pcie_diag 1024


Add logging (CSV, JSON, or both):

./pcie_diag 1024 --log --csv 
./pcie_diag 1024 --log --json 
./pcie_diag 1024 --log --csv --json 

Logs are written to:

- `results/csv/pcie_log.csv`
- `results/json/pcie_sessions.json`


## Extended Telemetry Window

./pcie_diag 1024 --duration-ms 8000 
- improves measurement stability


## Optional Integrity Counters

./pcie_diag 1024 --integrity 
- Enables read-only inspection of PCIe Advanced Error Reporting (AER) counters via Linux sysfs, if exposed by the platform.
- If counters are unavailable on the platform, integrity checks are automatically skipped with clear reporting.


## Multi-GPU Logging Behavior

When running in multi-GPU mode (`--all-gpus`), each detected GPU is evaluated independently.

- One result row (CSV) or object (JSON) is emitted per GPU per run.
- Each entry includes device UUID and PCIe BDF for unambiguous attribution.
- Multi-GPU configurations have not been exhaustively validated on all platforms. 
- Users are encouraged to verify results on their specific hardware.

Example:

```bash
./pcie_diag 1024 --all-gpus --log --csv
./pcie_diag 1024 --all-gpus --log --json 
./pcie_diag 1024 --gpu-index 1     # Target single GPU by index
```  


## Logging & Reproducibility

- CSV and JSON logs include stable device identifiers 
- Device UUIDs are reported at runtime via NVML for consistent identification across runs 
- UUIDs shown in documentation are intentionally **redacted** 
- Logs are append-friendly for time-series analysis and automated monitoring 


## Scope & Limitations

- This tool evaluates PCIe transport behavior only 
- It does not measure kernel performance or application-level efficiency 
- It does not modify BIOS, firmware, registry, or PCIe configuration 
- It reports observable facts only and never infers beyond available data 


## Validation

- Memcpy timing and PCIe behavior were cross-validated during development using Nsight Systems. 
- Nsight is not required to use this tool and is referenced only as an external correctness check. 


## Author
 
Author: Joe McLaren (Human–AI collaborative engineering) 
https://github.com/parallelArchitect


## License

MIT License 

Copyright (c) 2025 Joe McLaren

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions: 

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software. 

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.


## References

- **NVIDIA PCIe Logging & Counters** 
  https://docs.nvidia.com/networking/display/bfswtroubleshooting/pcie#src-4103229342_PCIe-LoggingandCounters

- **Linux PCIe AER Documentation** 
  https://docs.kernel.org/PCI/pcieaer-howto.html

- **Oracle Linux PCIe AER Overview** 
  https://blogs.oracle.com/linux/pci-express-advanced-error-reporting
