# harakiri-dash

A single-binary web dashboard for the Harakiri CUDA sweep. It replaces the
`nvidia-smi --format=csv -l 5 -f gpu_monitor.csv` workflow with a live page:
start/stop the sweep, watch every GPU on realtime charts, and set power limits
per card.

One `go build` produces one `.exe` with the UI embedded. No Node, no CDN, no
runtime files, no external Go modules.

## What it does

- **Start / stop the sweep** with a form: mode (`global`, `chunks`, `sections`,
  `backward`), range, target, batch, jobs, sections, gpui, maxbatch, out, and
  free-form extra args. The process is started directly (no shell) and killed
  as a process tree.
- **Realtime telemetry** by polling
  `nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used,memory.total,pcie.link.gen.current,pcie.link.width.current,clocks_throttle_reasons.active --format=csv,noheader,nounits`
  once per second. It does not write or tail a CSV file.
- **Charts**: one combined chart with a line per GPU (power / temp / util /
  clocks / mem, 1-30 min window) plus a card per GPU with live values,
  four sparklines, PCIe link and decoded throttle reasons.
- **Power limits**: an input per detected GPU, `fill all` and `Apply limits`
  (`nvidia-smi -i N -pl W`). Requires an elevated shell on Windows.
- **Live log** of the sweep with found-key highlighting (`FOUND`, `MATCH : YES`),
  last throughput, elapsed time and exit code.
- GPU count is detected automatically; `gpui` / `sections` are pre-filled from
  it on first load.

## Build

Requires Go 1.22+ (tested with 1.26).

```powershell
# ******* RUN As Admin *******
cd dashboard
go build -o harakiri-dash.exe .
```

## Run

From the directory that contains `harakiri_sweep.exe`:

```powershell
# ******* RUN As Admin *******
.\harakiri-dash.exe
```

Open <http://127.0.0.1:8099>.

If the dashboard lives elsewhere:

```powershell
# ******* RUN As Admin *******
.\harakiri-dash.exe -dir C:\path\to\sweep -exe harakiri_sweep.exe
```

Flags:

| flag | default | meaning |
|---|---|---|
| `-addr` | `127.0.0.1:8099` | listen address |
| `-exe` | `harakiri_sweep.exe` | sweep binary (relative to `-dir`) |
| `-dir` | `.` | working directory for the sweep |
| `-poll` | `1s` | nvidia-smi polling interval (min 250 ms) |
| `-history` | `900` | seconds of telemetry kept in memory |
| `-user` / `-pass` | empty | HTTP basic auth (both required together) |
| `-tls-cert` / `-tls-key` | empty | serve HTTPS |

## Remote access

The default bind is `127.0.0.1`, which is the safe choice. Anyone who can reach
the port can start/stop the sweep and change GPU power limits, so do not expose
it naked.

Recommended options:

1. **SSH tunnel** (simplest, no auth flags needed):

   ```powershell
   # ******* RUN As Admin *******
   ssh -L 8099:127.0.0.1:8099 user@gpu-box
   ```

   then open <http://127.0.0.1:8099> locally.

2. **Reverse proxy** (nginx/Caddy) with TLS and auth in front of
   `127.0.0.1:8099`.

3. **Built-in auth/TLS**:

   ```powershell
   # ******* RUN As Admin *******
   .\harakiri-dash.exe -addr 0.0.0.0:8099 -user admin -pass 'long-random-pass' `
     -tls-cert cert.pem -tls-key key.pem
   ```

   Without `-tls-cert` the password crosses the network in cleartext (basic
   auth); only use that on a trusted LAN or behind a TLS proxy.

## Notes

- Power-limit changes need Administrator on Windows. If the dashboard runs
  normally, `Apply limits` returns `nvidia-smi`'s access error for each GPU.
- The telemetry query omits `timestamp` (the server stamps samples itself) and
  is polled instead of streamed to a CSV, so there is no growing
  `gpu_monitor.csv` to clean up.
- History is in memory only; restarting the dashboard starts the charts empty.
- The UI reconnects automatically if the server restarts.
