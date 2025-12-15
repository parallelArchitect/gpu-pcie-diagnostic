/*
 * GPU PCIe Diagnostic & Bandwidth Analysis
 *  Version : 2.7.4
 *
 * Author: Joe McLaren (Human–AI collaborative engineering)
 * Repository: https://github.com/parallelArchitect
 *
 * Purpose
 *   Tier-1 (always enabled):
 *     - Query PCIe link capability and negotiated state (Gen, width) via NVML
 *     - Measure peak copy bandwidth (Host→Device, Device→Host) using CUDA memcpy timing
 *     - Measure sustained PCIe utilization under a defined load window via NVML TX/RX counters
 *     - Compute efficiency versus theoretical PCIe payload bandwidth
 *     - Emit deterministic verdicts for:
 *         • Link downgrade (Gen / width below capability)
 *         • Underperformance relative to negotiated link
 *
 *   Tier-2 (optional, read-only; enabled with --integrity):
 *     - Attempt to read PCIe Advanced Error Reporting (AER) device counters via sysfs
 *     - If counters are not exported by the platform, explicitly report “cannot access”
 *       and do not infer signal integrity or link health
 *
 * Design Principles
 *   - Observe, do not modify
 *   - No guessing, no hardcoded thresholds beyond PCIe specification math
 *   - No system tuning, firmware changes, or power policy manipulation
 *   - Conclusions are based solely on observable hardware signals
 *
 * References (design grounding)
 *   NVIDIA PCIe Logging & Counters:
 *     https://docs.nvidia.com/networking/display/bfswtroubleshooting/pcie#src-4103229342_PCIe-LoggingandCounters
 *
 *   Oracle Linux PCIe Advanced Error Reporting (AER):
 *     https://blogs.oracle.com/linux/pci-express-advanced-error-reporting
 *
 *   Linux PCIe AER How-To:
 *     https://docs.kernel.org/PCI/pcieaer-howto.html
 *
 * Build
 *   nvcc -O3 pcie_diagnostic_pro.cu -lnvidia-ml -Xcompiler -pthread -o pcie_diag
 *
 * Examples
 *   ./pcie_diag 1024
 *   ./pcie_diag 1024 --log --csv
 *   ./pcie_diag 1024 --log --json
 *   ./pcie_diag 1024 --duration-ms 8000
 *   ./pcie_diag 1024 --all-gpus --log --csv --json
 *   ./pcie_diag 1024 --integrity
 *
 * Notes
 *   - This tool does not attempt to “fix” PCIe behavior, MaxReadReq, ASPM, or BIOS settings
 *   - It reports measurable facts and rule-based advisories only
 *   - If the link is healthy, the correct outcome is “no action required”
 */

#include <cuda_runtime.h>
#include <nvml.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <atomic>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// -----------------------------
// CUDA / NVML error helpers
// -----------------------------
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t _e = (call);                                                 \
        if (_e != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                         \
                    cudaGetErrorString(_e), __FILE__, __LINE__);                 \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

#define NVML_CHECK(call)                                                         \
    do {                                                                         \
        nvmlReturn_t _e = (call);                                                \
        if (_e != NVML_SUCCESS) {                                                \
            fprintf(stderr, "NVML error: %s at %s:%d\n",                         \
                    nvmlErrorString(_e), __FILE__, __LINE__);                    \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

// -----------------------------
// Small filesystem helpers
// -----------------------------
static bool ensure_dir(const std::string &path) {
    struct stat st{};
    if (stat(path.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    return mkdir(path.c_str(), 0755) == 0;
}

static std::string join_path(const std::string &a, const std::string &b) {
    if (a.empty()) return b;
    if (!a.empty() && a.back() == '/') return a + b;
    return a + "/" + b;
}

static std::string read_first_line(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open()) return "";
    std::string line;
    std::getline(f, line);
    return line;
}

static std::string now_utc_iso8601() {
    std::time_t t = std::time(nullptr);
    std::tm tm_utc{};
    gmtime_r(&t, &tm_utc);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    return std::string(buf);
}


static bool read_ll(const std::string &path, long long &out) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    long long v = 0;
    f >> v;
    if (!f.good()) return false;
    out = v;
    return true;
}

// -----------------------------
// MaxReadReq detection (lspci)
// -----------------------------
// Returns MaxReadReq in bytes, or -1 if unavailable.
static int detect_max_read_req_bytes(const std::string &nvmlBusId) {
    // Extract short BDF: "00000000:01:00.0" -> "01:00.0"
    std::string b = nvmlBusId;
    size_t last_colon = b.rfind(':');
    if (last_colon != std::string::npos && last_colon > 8) {
        size_t second_colon = b.rfind(':', last_colon - 1);
        if (second_colon != std::string::npos) {
            b = b.substr(second_colon + 1);
        }
    }
    
    // Try sudo first (often needed for -vvv details)
    std::string cmd = "sudo /usr/bin/lspci -vvv -s " + b + " 2>/dev/null";
    FILE *fp = popen(cmd.c_str(), "r");
    if (!fp) return -1;
    
    int val = -1;
    char line[512];
    
    while (fgets(line, sizeof(line), fp)) {
        const char *p = std::strstr(line, "MaxReadReq");
        if (!p) continue;
        
        p += std::strlen("MaxReadReq");
        while (*p == ' ' || *p == '\t') ++p;
        
        int tmp = 0;
        if (std::sscanf(p, "%d", &tmp) == 1 && tmp > 0) {
            val = tmp;
            break;
        }
    }
    
    pclose(fp);
    return val;
}
static double pcie_payload_gbps_per_lane(unsigned int gen) {
    // Gen1 2.5GT/s 8b/10b  -> ~0.250 GB/s payload per lane
    // Gen2 5.0GT/s 8b/10b  -> ~0.500 GB/s payload per lane
    // Gen3 8.0GT/s 128b/130b -> ~0.985 GB/s payload per lane
    // Gen4 16GT/s 128b/130b -> ~1.969 GB/s payload per lane
    // Gen5 32GT/s 128b/130b -> ~3.938 GB/s payload per lane
    switch (gen) {
        case 1: return 0.250;
        case 2: return 0.500;
        case 3: return 0.985;
        case 4: return 1.969;
        case 5: return 3.938;
        default: return 0.0;
    }
}

static double pcie_theoretical_gbps(unsigned int gen, unsigned int width) {
    double per = pcie_payload_gbps_per_lane(gen);
    if (per <= 0.0 || width == 0) return 0.0;
    return per * static_cast<double>(width);
}

// -----------------------------
// Tier-2: AER counters snapshot
// -----------------------------
struct AerSnapshot {
    long long correctable = 0;
    long long nonfatal = 0;
    long long fatal = 0;
    bool valid = false;
    std::string source; // which sysfs names were used, for debug/logging
};

static std::string normalize_sysfs_bdf(const std::string &nvmlBusId) {
    // NVML: "00000000:01:00.0" -> sysfs: "0000:01:00.0"
    if (nvmlBusId.size() == 13 && nvmlBusId.rfind("00000000:", 0) == 0) {
        return "0000:" + nvmlBusId.substr(9);
    }
    if (nvmlBusId.size() == 12) return "0" + nvmlBusId;
    return nvmlBusId;
}

static AerSnapshot read_aer_snapshot(const std::string &nvmlBusId) {
    AerSnapshot s{};
    const std::string bdf = normalize_sysfs_bdf(nvmlBusId);
    const std::string base = "/sys/bus/pci/devices/" + bdf;

    // Different kernels expose different filenames. We probe a small set.
    const char* corr_candidates[] = {
        "aer_dev_correctable",
        "aer_correctable",
        "aer_correctable_errors",
    };
    const char* nonfatal_candidates[] = {
        "aer_dev_nonfatal",
        "aer_nonfatal",
        "aer_nonfatal_errors",
    };
    const char* fatal_candidates[] = {
        "aer_dev_fatal",
        "aer_fatal",
        "aer_fatal_errors",
    };

    auto try_read_any = [&](const char* const* names, size_t n, long long &out, std::string &used) -> bool {
        for (size_t i = 0; i < n; ++i) {
            std::string path = base + "/" + names[i];
            long long v = 0;
            if (read_ll(path, v)) {
                out = v;
                used = names[i];
                return true;
            }
        }
        return false;
    };

    std::string used_c, used_n, used_f;
    bool ok_c = try_read_any(corr_candidates, sizeof(corr_candidates)/sizeof(corr_candidates[0]), s.correctable, used_c);
    bool ok_n = try_read_any(nonfatal_candidates, sizeof(nonfatal_candidates)/sizeof(nonfatal_candidates[0]), s.nonfatal, used_n);
    bool ok_f = try_read_any(fatal_candidates, sizeof(fatal_candidates)/sizeof(fatal_candidates[0]), s.fatal, used_f);

    if (ok_c && ok_n && ok_f) {
        s.valid = true;
        s.source = used_c + std::string(",") + used_n + std::string(",") + used_f;
    } else {
        s.valid = false;
    }
    return s;
}

// -----------------------------
// PCIe load generator + NVML sampling
// -----------------------------
__global__ void noop_kernel(float *d, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d[idx] = d[idx] * 1.0000001f;
}

struct NvmlTelemetry {
    double tx_avg_gbps = 0.0;
    double rx_avg_gbps = 0.0;
    int valid_samples = 0;
};

static NvmlTelemetry sample_nvml_pcie(nvmlDevice_t dev, int samples, int interval_ms) {
    NvmlTelemetry t{};
    double tx_sum = 0.0, rx_sum = 0.0;

    for (int i = 0; i < samples; ++i) {
        unsigned int txKBs = 0;
        unsigned int rxKBs = 0;

        nvmlReturn_t r1 = nvmlDeviceGetPcieThroughput(dev, NVML_PCIE_UTIL_TX_BYTES, &txKBs);
        nvmlReturn_t r2 = nvmlDeviceGetPcieThroughput(dev, NVML_PCIE_UTIL_RX_BYTES, &rxKBs);

        if (r1 == NVML_SUCCESS && r2 == NVML_SUCCESS) {
            // NVML returns KB/s. Convert: KB/s * 1024 / 1e9 -> GB/s (decimal).
            tx_sum += (static_cast<double>(txKBs) * 1024.0) / 1e9;
            rx_sum += (static_cast<double>(rxKBs) * 1024.0) / 1e9;
            t.valid_samples++;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }

    if (t.valid_samples > 0) {
        t.tx_avg_gbps = tx_sum / t.valid_samples;
        t.rx_avg_gbps = rx_sum / t.valid_samples;
    }
    return t;
}

static void pcie_load_thread(size_t bytes, std::atomic<bool> &runFlag) {
    const size_t n = bytes / sizeof(float);

    float *h = nullptr;
    float *d = nullptr;

    CUDA_CHECK(cudaMallocHost(&h, bytes));
    CUDA_CHECK(cudaMalloc(&d, bytes));

    for (size_t i = 0; i < n; ++i) h[i] = 1.0f;

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);

    while (runFlag.load(std::memory_order_relaxed)) {
        CUDA_CHECK(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s));
        noop_kernel<<<blocks, threads, 0, s>>>(d, n);
        CUDA_CHECK(cudaMemcpyAsync(h, d, bytes, cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
    }

    CUDA_CHECK(cudaStreamDestroy(s));
    CUDA_CHECK(cudaFree(d));
    CUDA_CHECK(cudaFreeHost(h));
}

// -----------------------------
// Tier-1 peak memcpy measurement (H2D, D2H)
// -----------------------------
static double measure_memcpy_gbps(size_t bytes, cudaMemcpyKind kind, int iters = 10) {
    float *h = nullptr;
    float *d = nullptr;

    CUDA_CHECK(cudaMallocHost(&h, bytes));
    CUDA_CHECK(cudaMalloc(&d, bytes));
    CUDA_CHECK(cudaMemset(d, 0, bytes));
    std::memset(h, 0, bytes);

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaMemcpyAsync((kind == cudaMemcpyHostToDevice) ? d : h,
                                   (kind == cudaMemcpyHostToDevice) ? h : d,
                                   bytes, kind, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
    }

    CUDA_CHECK(cudaEventRecord(start, s));
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaMemcpyAsync((kind == cudaMemcpyHostToDevice) ? d : h,
                                   (kind == cudaMemcpyHostToDevice) ? h : d,
                                   bytes, kind, s));
    }
    CUDA_CHECK(cudaEventRecord(stop, s));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(s));
    CUDA_CHECK(cudaFree(d));
    CUDA_CHECK(cudaFreeHost(h));

    double seconds = (ms / 1000.0);
    if (seconds <= 0.0) return 0.0;

    double total_bytes = static_cast<double>(bytes) * static_cast<double>(iters);
    return (total_bytes / seconds) / 1e9; // GB/s decimal
}

// -----------------------------
// Verdict engine (Tier-1)
// -----------------------------
struct Verdict {
    std::string state;   // e.g., OK / LINK_DEGRADED / UNDERPERFORMING
    std::string reason;  // short reason string
    std::string advisory; // optional; empty when not needed
};

static Verdict compute_verdict(unsigned int curGen, unsigned int curWidth,
                               unsigned int maxGen, unsigned int maxWidth,
                               double efficiency_pct) {
    Verdict v{};
    v.state = "OK";
    v.reason = "Link operating at expected generation and width";
    v.advisory = "";

    const bool link_degraded = (curGen < maxGen) || (curWidth < maxWidth);
    if (link_degraded) {
        v.state = "LINK_DEGRADED";
        std::ostringstream oss;
        oss << "Current link Gen" << curGen << " x" << curWidth
            << " below capability Gen" << maxGen << " x" << maxWidth;
        v.reason = oss.str();

        // Deterministic, non-claiming advisory (BIOS/slot/bifurcation are common causes).
        v.advisory =
            "Advisory: Check BIOS PCIe settings, slot bifurcation, shared M.2 lane sharing, riser/cable seating, and chipset/BIOS updates.";
        return v;
    }

    // Only advise on efficiency when clearly low.
    if (efficiency_pct < 65.0) {
        v.state = "UNDERPERFORMING";
        v.reason = "Measured throughput is significantly below theoretical for negotiated link";
        v.advisory =
            "Advisory: Re-run with a larger transfer (e.g., 1024MB), minimize background PCIe activity, and consider testing with persistence mode enabled and ASPM minimized for diagnosis.";
        return v;
    }

    // No nags in normal range.
    if (efficiency_pct < 80.0) {
        v.state = "NOMINAL";
        v.reason = "Throughput is below peak but within common host↔device memcpy variance";
        return v;
    }

    v.state = "OK";
    v.reason = "Throughput and link state are consistent with a healthy PCIe path";
    return v;
}

// -----------------------------
// Logging: CSV + JSON array
// -----------------------------
static void append_csv_row(const std::string &csv_path,
                           const std::string &header,
                           const std::string &row) {
    bool need_header = false;
    {
        std::ifstream f(csv_path);
        if (!f.good()) need_header = true;
    }
    std::ofstream out(csv_path, std::ios::app);
    if (!out.good()) return;
    if (need_header) out << header << "\n";
    out << row << "\n";
}

// Append an object to a JSON array file.
// Creates a new array if file does not exist.
// Keeps the output as valid JSON.
static void append_json_array_object(const std::string &path, const std::string &obj_json) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) {
        std::ofstream out(path, std::ios::binary);
        out << "[\n" << obj_json << "\n]\n";
        return;
    }

    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    // Find last non-whitespace character
    size_t i = content.size();
    while (i > 0 && (content[i - 1] == ' ' || content[i - 1] == '\n' || content[i - 1] == '\r' || content[i - 1] == '\t')) {
        --i;
    }
    if (i == 0) {
        std::ofstream out(path, std::ios::binary);
        out << "[\n" << obj_json << "\n]\n";
        return;
    }

    // Expect content to end with ']'. If not, fall back to JSONL-style append to avoid corrupting.
    if (content[i - 1] != ']') {
        std::ofstream out(path, std::ios::app);
        out << obj_json << "\n";
        return;
    }

    // Determine if array currently empty: look for '[' followed by optional ws then ']'
    bool empty_array = false;
    {
        size_t j = 0;
        while (j < content.size() && (content[j] == ' ' || content[j] == '\n' || content[j] == '\r' || content[j] == '\t')) ++j;
        if (j < content.size() && content[j] == '[') {
            ++j;
            while (j < content.size() && (content[j] == ' ' || content[j] == '\n' || content[j] == '\r' || content[j] == '\t')) ++j;
            if (j < content.size() && content[j] == ']') empty_array = true;
        }
    }

    std::string updated = content.substr(0, i - 1); // up to before ']'
    if (!empty_array) {
        // If last meaningful char before ']' is not '[', we need a comma separator.
        // We search backwards for the last non-ws before ']'.
        size_t k = updated.size();
        while (k > 0 && (updated[k - 1] == ' ' || updated[k - 1] == '\n' || updated[k - 1] == '\r' || updated[k - 1] == '\t')) --k;
        if (k > 0 && updated[k - 1] != '[') {
            updated += ",\n";
        }
    }
    updated += obj_json;
    updated += "\n]\n";

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << updated;
}

// -----------------------------
// CLI
// -----------------------------
struct Options {
    size_t transferMB = 1024;

    bool log = false;
    bool csv = false;
    bool json = false;

    bool all_gpus = false;
    int gpu_index = 0;

    bool integrity = false;

    int duration_ms = 5000;  // NVML window length
    int interval_ms = 100;   // NVML sampling interval

    std::string out_root = "results";
};

static void print_usage(const char *argv0) {
    std::printf(
        "Usage:\n"
        "  %s <transfer_mb> [--log] [--csv] [--json] [--duration-ms N] [--interval-ms N]\n"
        "                  [--gpu-index N | --all-gpus] [--integrity] [--out-dir PATH]\n\n"
        "Examples:\n"
        "  %s 1024\n"
        "  %s 1024 --log --csv\n"
        "  %s 1024 --log --json\n"
        "  %s 1024 --duration-ms 8000\n"
        "  %s 1024 --all-gpus --log --csv --json\n"
        "  %s 1024 --integrity\n",
        argv0, argv0, argv0, argv0, argv0, argv0, argv0
    );
}

static bool parse_args(int argc, char **argv, Options &opt) {
    if (argc < 2) return false;

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);

        if (a == "--help" || a == "-h") {
            return false;
        } else if (a == "--log") {
            opt.log = true;
        } else if (a == "--csv") {
            opt.csv = true;
        } else if (a == "--json") {
            opt.json = true;
        } else if (a == "--all-gpus") {
            opt.all_gpus = true;
        } else if (a == "--integrity") {
            opt.integrity = true;
        } else if (a == "--gpu-index" && i + 1 < argc) {
            opt.gpu_index = std::atoi(argv[++i]);
        } else if (a == "--duration-ms" && i + 1 < argc) {
            opt.duration_ms = std::max(500, std::atoi(argv[++i]));
        } else if (a == "--interval-ms" && i + 1 < argc) {
            opt.interval_ms = std::max(10, std::atoi(argv[++i]));
        } else if (a == "--out-dir" && i + 1 < argc) {
            opt.out_root = argv[++i];
        } else if (!a.empty() && a[0] != '-') {
            opt.transferMB = static_cast<size_t>(std::strtoull(a.c_str(), nullptr, 10));
        } else {
            // unknown flag -> fail
            return false;
        }
    }

    if (opt.transferMB == 0) opt.transferMB = 1024;
    return true;
}

// -----------------------------
// Per-GPU run
// -----------------------------
struct RunResult {
    // identity
    std::string timestamp_utc;
    std::string gpu_name;
    std::string gpu_uuid;
    std::string bus_id;

    unsigned int cur_gen = 0, cur_width = 0;
    unsigned int max_gen = 0, max_width = 0;

    int max_read_req_bytes = -1;

    // measurements
    size_t transfer_mb = 0;
    double theoretical_gbps = 0.0;

    double h2d_peak_gbps = 0.0;
    double d2h_peak_gbps = 0.0;

    double tx_avg_gbps = 0.0;
    double rx_avg_gbps = 0.0;
    double combined_gbps = 0.0;

    double efficiency_pct = 0.0;

    // tier-2
    bool aer_available = false;
    long long aer_corr_delta = 0;
    long long aer_nonfatal_delta = 0;
    long long aer_fatal_delta = 0;
    std::string integrity_note; // empty when not requested, otherwise short note

    // verdict
    Verdict verdict;

    // environment signals (informational only)
    std::string aspm_policy;
    std::string iommu_status;
    bool persistence_enabled = false;

    // telemetry window
    int window_ms = 0;
    int interval_ms = 0;
    int valid_samples = 0;
};

static RunResult run_one_gpu(int gpu_index, const Options &opt) {
    RunResult r{};
    r.timestamp_utc = now_utc_iso8601();
    r.transfer_mb = opt.transferMB;

    CUDA_CHECK(cudaSetDevice(gpu_index));

    nvmlDevice_t dev{};
    NVML_CHECK(nvmlDeviceGetHandleByIndex(gpu_index, &dev));

    char name[128] = {0};
    NVML_CHECK(nvmlDeviceGetName(dev, name, sizeof(name)));
    r.gpu_name = name;

    char uuid[96] = {0};
    NVML_CHECK(nvmlDeviceGetUUID(dev, uuid, sizeof(uuid)));
    r.gpu_uuid = uuid;

    nvmlPciInfo_t pci{};
    NVML_CHECK(nvmlDeviceGetPciInfo(dev, &pci));
    r.bus_id = pci.busId;

    NVML_CHECK(nvmlDeviceGetMaxPcieLinkGeneration(dev, &r.max_gen));
    NVML_CHECK(nvmlDeviceGetMaxPcieLinkWidth(dev, &r.max_width));
    NVML_CHECK(nvmlDeviceGetCurrPcieLinkGeneration(dev, &r.cur_gen));
    NVML_CHECK(nvmlDeviceGetCurrPcieLinkWidth(dev, &r.cur_width));

    // MaxReadReq (best-effort; does not affect verdict)
    r.max_read_req_bytes = detect_max_read_req_bytes(r.bus_id);

    // env signals (informational only)
    {
        std::string policy = read_first_line("/sys/module/pcie_aspm/parameters/policy");
        r.aspm_policy = policy.empty() ? "default (kernel-managed)" : policy;

        std::string cmdline = read_first_line("/proc/cmdline");
        if (cmdline.find("intel_iommu=on") != std::string::npos ||
            cmdline.find("amd_iommu=on") != std::string::npos) {
            r.iommu_status = "Enabled (kernel cmdline)";
        } else {
            r.iommu_status = "Platform default (no explicit flags)";
        }

        nvmlEnableState_t pm{};
        if (nvmlDeviceGetPersistenceMode(dev, &pm) == NVML_SUCCESS) {
            r.persistence_enabled = (pm == NVML_FEATURE_ENABLED);
        }
    }

    // Theoretical uses CURRENT negotiated link (what you actually have), not max.
    r.theoretical_gbps = pcie_theoretical_gbps(r.cur_gen, r.cur_width);

    const size_t bytes = opt.transferMB * 1024ull * 1024ull;

    // Tier-1 peak memcpy
    r.h2d_peak_gbps = measure_memcpy_gbps(bytes, cudaMemcpyHostToDevice, 10);
    r.d2h_peak_gbps = measure_memcpy_gbps(bytes, cudaMemcpyDeviceToHost, 10);

    // Tier-2 snapshots only if requested
    AerSnapshot aer_before{}, aer_after{};
    if (opt.integrity) {
        aer_before = read_aer_snapshot(r.bus_id);
    }

    // Sustained load + NVML sampling window
    std::atomic<bool> runFlag{true};
    std::thread loadThread(pcie_load_thread, bytes, std::ref(runFlag));

    // Give the load loop a brief start so NVML sees activity.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    const int samples = std::max(1, opt.duration_ms / opt.interval_ms);
    NvmlTelemetry t = sample_nvml_pcie(dev, samples, opt.interval_ms);

    runFlag.store(false, std::memory_order_relaxed);
    loadThread.join();

    r.window_ms = samples * opt.interval_ms;
    r.interval_ms = opt.interval_ms;
    r.valid_samples = t.valid_samples;

    r.tx_avg_gbps = t.tx_avg_gbps;
    r.rx_avg_gbps = t.rx_avg_gbps;
    r.combined_gbps = r.tx_avg_gbps + r.rx_avg_gbps;

    // Tier-2 after snapshot
    if (opt.integrity) {
        aer_after = read_aer_snapshot(r.bus_id);
        if (aer_before.valid && aer_after.valid) {
            r.aer_available = true;
            r.aer_corr_delta = std::max(0LL, aer_after.correctable - aer_before.correctable);
            r.aer_nonfatal_delta = std::max(0LL, aer_after.nonfatal - aer_before.nonfatal);
            r.aer_fatal_delta = std::max(0LL, aer_after.fatal - aer_before.fatal);
            r.integrity_note = "AER counters read via sysfs";
        } else {
            r.aer_available = false;
            r.integrity_note = "AER counters unavailable; cannot access";
        }
    } else {
        r.integrity_note = ""; // not requested; do not print
    }

    // Efficiency: prefer NVML combined when valid; fall back to peak direction max if NVML invalid.
    double measured_for_eff = 0.0;
    if (r.valid_samples > 0 && r.combined_gbps > 0.0) {
        measured_for_eff = r.combined_gbps;
    } else {
        measured_for_eff = std::max(r.h2d_peak_gbps, r.d2h_peak_gbps);
    }

    r.efficiency_pct = 0.0;
    if (r.theoretical_gbps > 0.0) {
        r.efficiency_pct = (measured_for_eff / r.theoretical_gbps) * 100.0;
    }

    r.verdict = compute_verdict(r.cur_gen, r.cur_width, r.max_gen, r.max_width, r.efficiency_pct);

    return r;
}

// -----------------------------
// Output formatting
// -----------------------------
static void print_human_report(const RunResult &r, bool show_advisory, bool show_integrity_section) {
    std::printf("════════════════════════════════════════════════════════════════\n");
    std::printf("GPU PCIe Diagnostic & Bandwidth Analysis v2.7.4\n");
    std::printf("════════════════════════════════════════════════════════════════\n");
    std::printf("GPU:   %s\n", r.gpu_name.c_str());
    std::printf("BDF:   %s\n", r.bus_id.c_str());
    std::printf("UUID:  %s\n\n", r.gpu_uuid.c_str());

    std::printf("▸ PCIe Link\n");
    std::printf("  Current: Gen%u x%u\n", r.cur_gen, r.cur_width);
    std::printf("  Max Cap: Gen%u x%u\n", r.max_gen, r.max_width);
    std::printf("  Theoretical (payload): %.2f GB/s\n", r.theoretical_gbps);
    std::printf("  Transfer Size: %zu MiB\n\n", r.transfer_mb);

    std::printf("▸ Peak Copy Bandwidth (explicit memcpy)\n");
    std::printf("  Host → Device: %.2f GB/s\n", r.h2d_peak_gbps);
    std::printf("  Device → Host: %.2f GB/s\n\n", r.d2h_peak_gbps);

    std::printf("▸ Telemetry (NVML, under load window)\n");
    std::printf("  Window: %.1f s (%d samples @ %d ms)\n",
                r.window_ms / 1000.0, std::max(1, r.window_ms / r.interval_ms), r.interval_ms);
    if (r.valid_samples > 0) {
        std::printf("  TX avg: %.2f GB/s\n", r.tx_avg_gbps);
        std::printf("  RX avg: %.2f GB/s\n", r.rx_avg_gbps);
        std::printf("  Combined: %.2f GB/s\n\n", r.combined_gbps);
    } else {
        std::printf("  TX/RX counters: unavailable during window\n\n");
    }

    std::printf("▸ Verdict\n");
    std::printf("  State: %s\n", r.verdict.state.c_str());
    std::printf("  Reason: %s\n", r.verdict.reason.c_str());
    std::printf("  Efficiency: %.2f%%\n\n", r.efficiency_pct);

    std::printf("▸ System Signals (informational)\n");
    if (r.max_read_req_bytes > 0) {
        std::printf("  MaxReadReq: %d bytes\n", r.max_read_req_bytes);
    } else {
        std::printf("  MaxReadReq: unavailable\n");
    }
    std::printf("  Persistence Mode: %s\n", r.persistence_enabled ? "Enabled" : "Disabled");
    std::printf("  ASPM Policy (sysfs string): %s\n", r.aspm_policy.c_str());
    std::printf("  IOMMU: %s\n\n", r.iommu_status.c_str());

    if (show_integrity_section) {
        std::printf("▸ Integrity Counters (read-only)\n");
        if (r.aer_available) {
            std::printf("  AER deltas: correctable=%lld, nonfatal=%lld, fatal=%lld\n\n",
                        r.aer_corr_delta, r.aer_nonfatal_delta, r.aer_fatal_delta);
        } else {
            std::printf("  %s\n\n", r.integrity_note.c_str());
        }
    }

    if (show_advisory && !r.verdict.advisory.empty()) {
        std::printf("%s\n\n", r.verdict.advisory.c_str());
    }
}

static std::string csv_header() {
    return std::string(
        "timestamp_utc,gpu_model,gpu_uuid,bus_id,"
        "cur_gen,cur_width,max_gen,max_width,"
        "transfer_mb,theoretical_gbps,"
        "h2d_peak_gbps,d2h_peak_gbps,"
        "tx_avg_gbps,rx_avg_gbps,combined_gbps,efficiency_pct,"
        "verdict_state,verdict_reason,"
        "max_read_req_bytes,persistence_enabled,aspm_policy,iommu_status,"
        "integrity_enabled,aer_available,aer_correctable_delta,aer_nonfatal_delta,aer_fatal_delta,integrity_note"
    );
}

static std::string csv_row(const RunResult &r, bool integrity_enabled) {
    std::ostringstream o;
    o << r.timestamp_utc << ","
      << "\"" << r.gpu_name << "\"" << ","
      << "\"" << r.gpu_uuid << "\"" << ","
      << "\"" << r.bus_id << "\"" << ","
      << r.cur_gen << "," << r.cur_width << ","
      << r.max_gen << "," << r.max_width << ","
      << r.transfer_mb << ","
      << std::fixed << std::setprecision(3) << r.theoretical_gbps << ","
      << std::fixed << std::setprecision(3) << r.h2d_peak_gbps << ","
      << std::fixed << std::setprecision(3) << r.d2h_peak_gbps << ","
      << std::fixed << std::setprecision(3) << r.tx_avg_gbps << ","
      << std::fixed << std::setprecision(3) << r.rx_avg_gbps << ","
      << std::fixed << std::setprecision(3) << r.combined_gbps << ","
      << std::fixed << std::setprecision(2) << r.efficiency_pct << ","
      << "\"" << r.verdict.state << "\"" << ","
      << "\"" << r.verdict.reason << "\"" << ","
      << r.max_read_req_bytes << ","
      << (r.persistence_enabled ? "1" : "0") << ","
      << "\"" << r.aspm_policy << "\"" << ","
      << "\"" << r.iommu_status << "\"" << ","
      << (integrity_enabled ? "1" : "0") << ","
      << (r.aer_available ? "1" : "0") << ","
      << r.aer_corr_delta << ","
      << r.aer_nonfatal_delta << ","
      << r.aer_fatal_delta << ","
      << "\"" << r.integrity_note << "\"";
    return o.str();
}

static std::string json_escape(const std::string &s) {
    std::ostringstream o;
    for (char c : s) {
        switch (c) {
            case '\\': o << "\\\\"; break;
            case '"':  o << "\\\""; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c << std::dec;
                } else {
                    o << c;
                }
        }
    }
    return o.str();
}

static std::string json_object(const RunResult &r, bool integrity_enabled) {
    std::ostringstream o;
    o << "  {\n";
    o << "    \"timestamp_utc\": \"" << json_escape(r.timestamp_utc) << "\",\n";
    o << "    \"gpu\": \"" << json_escape(r.gpu_name) << "\",\n";
    o << "    \"gpu_uuid\": \"" << json_escape(r.gpu_uuid) << "\",\n";
    o << "    \"bus_id\": \"" << json_escape(r.bus_id) << "\",\n";
    o << "    \"pcie\": {\n";
    o << "      \"cur_gen\": " << r.cur_gen << ",\n";
    o << "      \"cur_width\": " << r.cur_width << ",\n";
    o << "      \"max_gen\": " << r.max_gen << ",\n";
    o << "      \"max_width\": " << r.max_width << ",\n";
    o << "      \"theoretical_gbps\": " << std::fixed << std::setprecision(3) << r.theoretical_gbps << "\n";
    o << "    },\n";
    o << "    \"bandwidth\": {\n";
    o << "      \"transfer_mb\": " << r.transfer_mb << ",\n";
    o << "      \"h2d_peak_gbps\": " << std::fixed << std::setprecision(3) << r.h2d_peak_gbps << ",\n";
    o << "      \"d2h_peak_gbps\": " << std::fixed << std::setprecision(3) << r.d2h_peak_gbps << ",\n";
    o << "      \"tx_avg_gbps\": " << std::fixed << std::setprecision(3) << r.tx_avg_gbps << ",\n";
    o << "      \"rx_avg_gbps\": " << std::fixed << std::setprecision(3) << r.rx_avg_gbps << ",\n";
    o << "      \"combined_gbps\": " << std::fixed << std::setprecision(3) << r.combined_gbps << ",\n";
    o << "      \"efficiency_pct\": " << std::fixed << std::setprecision(2) << r.efficiency_pct << ",\n";
    o << "      \"telemetry_window_ms\": " << r.window_ms << ",\n";
    o << "      \"telemetry_interval_ms\": " << r.interval_ms << ",\n";
    o << "      \"telemetry_valid_samples\": " << r.valid_samples << "\n";
    o << "    },\n";
    o << "    \"verdict\": {\n";
    o << "      \"state\": \"" << json_escape(r.verdict.state) << "\",\n";
    o << "      \"reason\": \"" << json_escape(r.verdict.reason) << "\"\n";
    o << "    },\n";
    o << "    \"system\": {\n";
    o << "      \"max_read_req_bytes\": " << r.max_read_req_bytes << ",\n";
    o << "      \"persistence_enabled\": " << (r.persistence_enabled ? "true" : "false") << ",\n";
    o << "      \"aspm_policy\": \"" << json_escape(r.aspm_policy) << "\",\n";
    o << "      \"iommu_status\": \"" << json_escape(r.iommu_status) << "\"\n";
    o << "    },\n";
    o << "    \"integrity\": {\n";
    o << "      \"enabled\": " << (integrity_enabled ? "true" : "false") << ",\n";
    o << "      \"aer_available\": " << (r.aer_available ? "true" : "false") << ",\n";
    o << "      \"aer_correctable_delta\": " << r.aer_corr_delta << ",\n";
    o << "      \"aer_nonfatal_delta\": " << r.aer_nonfatal_delta << ",\n";
    o << "      \"aer_fatal_delta\": " << r.aer_fatal_delta << ",\n";
    o << "      \"note\": \"" << json_escape(r.integrity_note) << "\"\n";
    o << "    }\n";
    o << "  }";
    return o.str();
}

// -----------------------------
// main
// -----------------------------
int main(int argc, char **argv) {
    Options opt{};
    if (!parse_args(argc, argv, opt)) {
        print_usage(argv[0]);
        return 1;
    }

    // Init NVML once
    NVML_CHECK(nvmlInit_v2());

    int cuda_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&cuda_count));
    unsigned int nvml_count = 0;
    NVML_CHECK(nvmlDeviceGetCount(&nvml_count));

    int count = std::min<int>(cuda_count, (int)nvml_count);
    if (count <= 0) {
        std::fprintf(stderr, "No CUDA/NVML devices found.\n");
        nvmlShutdown();
        return 1;
    }

    std::vector<int> gpu_indices;
    if (opt.all_gpus) {
        for (int i = 0; i < count; ++i) gpu_indices.push_back(i);
    } else {
        if (opt.gpu_index < 0 || opt.gpu_index >= count) {
            std::fprintf(stderr, "Invalid --gpu-index %d (device count=%d)\n", opt.gpu_index, count);
            nvmlShutdown();
            return 1;
        }
        gpu_indices.push_back(opt.gpu_index);
    }

    // Prepare output directories (fixed paths)
    std::string run_dir;
    std::string csv_path, json_path;

    const bool want_logs = opt.log && (opt.csv || opt.json);
    if (want_logs) {
        // Fixed output locations (GitHub-friendly):
        //   CSV  -> <out_root>/csv/pcie_log.csv
        //   JSON -> <out_root>/json/pcie_sessions.json   (valid JSON array)
        ensure_dir(opt.out_root);
        ensure_dir(join_path(opt.out_root, "csv"));
        ensure_dir(join_path(opt.out_root, "json"));

        if (opt.csv)  csv_path  = join_path(join_path(opt.out_root, "csv"),  "pcie_log.csv");
        if (opt.json) json_path = join_path(join_path(opt.out_root, "json"), "pcie_sessions.json");
    }

    for (size_t idx = 0; idx < gpu_indices.size(); ++idx) {
        const int g = gpu_indices[idx];

        RunResult r = run_one_gpu(g, opt);

        // Human output: no “integrity” section unless user requested it.
        print_human_report(r, /*show_advisory=*/true, /*show_integrity_section=*/opt.integrity);

        // Logs
        if (want_logs) {
            if (opt.csv) {
                append_csv_row(csv_path, csv_header(), csv_row(r, opt.integrity));
            }
            if (opt.json) {
                append_json_array_object(json_path, json_object(r, opt.integrity));
            }
        }
    }

    if (want_logs) {
        std::printf("Logs:\n");
        if (opt.csv)  std::printf("  CSV:  %s\n", csv_path.c_str());
        if (opt.json) std::printf("  JSON: %s\n", json_path.c_str());
    }

    nvmlShutdown();
    return 0;
}
