// harakiri-dash: single-binary web dashboard for the Harakiri CUDA sweep.
//
//	harakiri-dash.exe -dir . -exe harakiri_sweep.exe -addr 127.0.0.1:8099
//
// It polls nvidia-smi for realtime GPU telemetry, streams it to the embedded
// web UI over SSE, starts/stops the sweep process and applies per-GPU power
// limits. No external dependencies, no CDN, no files besides the exe.
package main

import (
	"bufio"
	"context"
	_ "embed"
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

//go:embed ui.html
var uiHTML []byte

//go:embed ui.css
var uiCSS []byte

//go:embed ui.js
var uiJS []byte

// ---------------------------------------------------------------- telemetry

type GPU struct {
	Index      int      `json:"index"`
	Name       string   `json:"name"`
	Temp       *float64 `json:"temp"`
	PowerDraw  *float64 `json:"power_draw"`
	PowerLimit *float64 `json:"power_limit"`
	ClockSM    *float64 `json:"clock_sm"`
	ClockMem   *float64 `json:"clock_mem"`
	UtilGPU    *float64 `json:"util_gpu"`
	UtilMem    *float64 `json:"util_mem"`
	MemUsed    *float64 `json:"mem_used"`
	MemTotal   *float64 `json:"mem_total"`
	PCIeGen    string   `json:"pcie_gen"`
	PCIeWidth  string   `json:"pcie_width"`
	Throttle   string   `json:"throttle"`
}

type Sample struct {
	T    int64 `json:"t"`
	GPUs []GPU `json:"gpus"`
}

var gpuQuery = []string{
	"--query-gpu=index,name,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used,memory.total,pcie.link.gen.current,pcie.link.width.current,clocks_throttle_reasons.active",
	"--format=csv,noheader,nounits",
}

func runTimeout(d time.Duration, name string, args ...string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), d)
	defer cancel()
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.SysProcAttr = sysProcAttr()
	out, err := cmd.CombinedOutput()
	if ctx.Err() == context.DeadlineExceeded {
		return out, fmt.Errorf("%s timed out after %s", name, d)
	}
	return out, err
}

func num(s string) *float64 {
	s = strings.TrimSpace(s)
	if s == "" || strings.Contains(s, "N/A") || strings.Contains(s, "Not Supported") {
		return nil
	}
	f := strings.Fields(s)
	if len(f) == 0 {
		return nil
	}
	v, err := strconv.ParseFloat(f[0], 64)
	if err != nil {
		return nil
	}
	return &v
}

func atoi(s string) int {
	v, _ := strconv.Atoi(strings.TrimSpace(s))
	return v
}

func parseGPUs(data []byte) ([]GPU, error) {
	r := csv.NewReader(strings.NewReader(string(data)))
	r.TrimLeadingSpace = true
	r.FieldsPerRecord = -1
	out := []GPU{}
	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if len(rec) < 14 {
			continue
		}
		out = append(out, GPU{
			Index:      atoi(rec[0]),
			Name:       strings.TrimSpace(rec[1]),
			Temp:       num(rec[2]),
			PowerDraw:  num(rec[3]),
			PowerLimit: num(rec[4]),
			ClockSM:    num(rec[5]),
			ClockMem:   num(rec[6]),
			UtilGPU:    num(rec[7]),
			UtilMem:    num(rec[8]),
			MemUsed:    num(rec[9]),
			MemTotal:   num(rec[10]),
			PCIeGen:    strings.TrimSpace(rec[11]),
			PCIeWidth:  strings.TrimSpace(rec[12]),
			Throttle:   strings.TrimSpace(rec[13]),
		})
	}
	if len(out) == 0 {
		return nil, errors.New("nvidia-smi returned no GPUs")
	}
	return out, nil
}

// ------------------------------------------------------------------ process

const logCap = 4000

type LogLine struct {
	Seq  uint64 `json:"seq"`
	T    int64  `json:"t"`
	Line string `json:"line"`
}

type ProcInfo struct {
	Running  bool     `json:"running"`
	PID      int      `json:"pid"`
	Started  int64    `json:"started"`
	Elapsed  float64  `json:"elapsed"`
	ExitCode *int     `json:"exit_code"`
	Stopped  bool     `json:"stopped"`
	Found    string   `json:"found,omitempty"`
	Match    bool     `json:"match"`
	Rate     string   `json:"rate,omitempty"`
	Args     []string `json:"args,omitempty"`
}

type Proc struct {
	mu      sync.Mutex
	cmd     *exec.Cmd
	running bool
	pid     int
	started time.Time
	stopped time.Time
	exit    *int
	found   string
	match   bool
	killed  bool
	rate    string
	args    []string
	lines   []LogLine
	seq     uint64
	done    chan struct{}
}

func (p *Proc) addLine(s string) {
	s = strings.TrimRight(s, "\r")
	p.mu.Lock()
	defer p.mu.Unlock()
	p.seq++
	p.lines = append(p.lines, LogLine{Seq: p.seq, T: time.Now().UnixMilli(), Line: s})
	if len(p.lines) > logCap {
		p.lines = append(p.lines[:0], p.lines[len(p.lines)-logCap:]...)
	}
	if i := strings.Index(s, "FOUND scalar:"); i >= 0 {
		p.found = strings.TrimSpace(s[i+len("FOUND scalar:"):])
	}
	if strings.Contains(s, "MATCH") && strings.Contains(s, "YES") {
		p.match = true
	}
	if i := strings.Index(s, "throughput :"); i >= 0 {
		p.rate = strings.TrimSpace(s[i+len("throughput :"):])
	}
}

func (p *Proc) LogsSince(seq uint64) ([]string, uint64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := []string{}
	for _, l := range p.lines {
		if l.Seq > seq {
			out = append(out, l.Line)
		}
	}
	return out, p.seq
}

func (p *Proc) Start(exe, dir string, args []string) error {
	p.mu.Lock()
	if p.running {
		p.mu.Unlock()
		return errors.New("a sweep is already running")
	}
	p.mu.Unlock()

	cmd := exec.Command(exe, args...)
	cmd.Dir = dir
	cmd.SysProcAttr = sysProcAttr()

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return err
	}

	p.mu.Lock()
	p.cmd = cmd
	p.running = true
	p.pid = cmd.Process.Pid
	p.started = time.Now()
	p.stopped = time.Time{}
	p.exit = nil
	p.found = ""
	p.match = false
	p.killed = false
	p.rate = ""
	p.args = append([]string(nil), args...)
	p.done = make(chan struct{})
	done := p.done
	p.mu.Unlock()

	p.addLine("$ " + filepath.Base(exe) + " " + strings.Join(args, " "))

	var wg sync.WaitGroup
	ch := make(chan string, 256)
	scan := func(r io.Reader) {
		defer wg.Done()
		sc := bufio.NewScanner(r)
		sc.Buffer(make([]byte, 64*1024), 1<<20)
		for sc.Scan() {
			ch <- sc.Text()
		}
	}
	wg.Add(2)
	go scan(stdout)
	go scan(stderr)
	go func() {
		wg.Wait()
		close(ch)
	}()
	go func() {
		for line := range ch {
			p.addLine(line)
		}
	}()

	go func() {
		err := cmd.Wait()
		p.mu.Lock()
		p.running = false
		p.stopped = time.Now()
		code := 0
		if err != nil {
			var ee *exec.ExitError
			if errors.As(err, &ee) {
				code = ee.ExitCode()
			} else {
				code = -1
			}
		}
		p.exit = &code
		close(done)
		p.mu.Unlock()
	}()
	return nil
}

func (p *Proc) Stop(timeout time.Duration) error {
	p.mu.Lock()
	if !p.running {
		p.mu.Unlock()
		return errors.New("no sweep is running")
	}
	pid := p.pid
	done := p.done
	p.killed = true
	p.mu.Unlock()

	if err := killTree(pid); err != nil {
		return err
	}
	select {
	case <-done:
		return nil
	case <-time.After(timeout):
		return fmt.Errorf("pid %d did not exit within %s", pid, timeout)
	}
}

func (p *Proc) Info() ProcInfo {
	p.mu.Lock()
	defer p.mu.Unlock()
	el := 0.0
	if p.running {
		el = time.Since(p.started).Seconds()
	} else if !p.stopped.IsZero() {
		el = p.stopped.Sub(p.started).Seconds()
	}
	return ProcInfo{
		Running:  p.running,
		PID:      p.pid,
		Started:  p.started.UnixMilli(),
		Elapsed:  el,
		ExitCode: p.exit,
		Stopped:  p.killed,
		Found:    p.found,
		Match:    p.match,
		Rate:     p.rate,
		Args:     append([]string(nil), p.args...),
	}
}

// ---------------------------------------------------------------------- app

type Snapshot struct {
	Type     string   `json:"type"`
	T        int64    `json:"t"`
	Epoch    string   `json:"epoch"`
	Exe      string   `json:"exe"`
	GPUs     []GPU    `json:"gpus"`
	Samples  []Sample `json:"samples,omitempty"`
	Proc     ProcInfo `json:"proc"`
	Log      LogDelta `json:"log"`
	Error    string   `json:"error,omitempty"`
	GPUCount int      `json:"gpu_count"`
	PollMS   int64    `json:"poll_ms"`
}

type LogDelta struct {
	Seq   uint64   `json:"seq"`
	Lines []string `json:"lines"`
}

type App struct {
	mu      sync.RWMutex
	gpus    []GPU
	hist    []Sample
	histMax int
	poll    time.Duration
	lastErr string
	epoch   string
	pm      *Proc
	dir     string
	exe     string
}

func (a *App) pollOnce() {
	out, err := runTimeout(12*time.Second, "nvidia-smi", gpuQuery...)
	if err != nil {
		msg := strings.TrimSpace(string(out))
		if msg == "" {
			msg = err.Error()
		} else {
			msg = err.Error() + ": " + msg
		}
		a.mu.Lock()
		a.lastErr = msg
		a.mu.Unlock()
		return
	}
	gpus, err := parseGPUs(out)
	if err != nil {
		a.mu.Lock()
		a.lastErr = err.Error()
		a.mu.Unlock()
		return
	}
	s := Sample{T: time.Now().UnixMilli(), GPUs: gpus}
	a.mu.Lock()
	a.gpus = gpus
	a.lastErr = ""
	a.hist = append(a.hist, s)
	if len(a.hist) > a.histMax {
		a.hist = append(a.hist[:0], a.hist[len(a.hist)-a.histMax:]...)
	}
	a.mu.Unlock()
}

func (a *App) snapshot(hello bool, since uint64) Snapshot {
	a.mu.RLock()
	gpus := append([]GPU{}, a.gpus...)
	var samples []Sample
	if hello {
		samples = append([]Sample{}, a.hist...)
	}
	errMsg := a.lastErr
	a.mu.RUnlock()

	lines, seq := a.pm.LogsSince(since)
	s := Snapshot{
		Type:     "tick",
		T:        time.Now().UnixMilli(),
		Epoch:    a.epoch,
		Exe:      filepath.Base(a.exe),
		GPUs:     gpus,
		Proc:     a.pm.Info(),
		Log:      LogDelta{Seq: seq, Lines: lines},
		Error:    errMsg,
		GPUCount: len(gpus),
		PollMS:   a.poll.Milliseconds(),
	}
	if hello {
		s.Type = "hello"
		s.Samples = samples
	}
	return s
}

// ------------------------------------------------------------------ handlers

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	enc := json.NewEncoder(w)
	_ = enc.Encode(v)
}

func (a *App) handleState(w http.ResponseWriter, r *http.Request) {
	since, _ := strconv.ParseUint(r.URL.Query().Get("logsince"), 10, 64)
	writeJSON(w, 200, a.snapshot(true, since))
}

func (a *App) handleEvents(w http.ResponseWriter, r *http.Request) {
	fl, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", 500)
		return
	}
	h := w.Header()
	h.Set("Content-Type", "text/event-stream; charset=utf-8")
	h.Set("Cache-Control", "no-cache, no-transform")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")

	since, _ := strconv.ParseUint(r.URL.Query().Get("logsince"), 10, 64)
	enc := json.NewEncoder(w)

	send := func(v any) bool {
		if _, err := io.WriteString(w, "data: "); err != nil {
			return false
		}
		if err := enc.Encode(v); err != nil {
			return false
		}
		if _, err := io.WriteString(w, "\n"); err != nil {
			return false
		}
		fl.Flush()
		return true
	}

	hello := a.snapshot(true, since)
	if !send(hello) {
		return
	}
	since = hello.Log.Seq

	tick := a.poll
	if tick < 500*time.Millisecond {
		tick = 500 * time.Millisecond
	}
	tk := time.NewTicker(tick)
	defer tk.Stop()
	keep := time.NewTicker(15 * time.Second)
	defer keep.Stop()

	for {
		select {
		case <-r.Context().Done():
			return
		case <-tk.C:
			s := a.snapshot(false, since)
			if !send(s) {
				return
			}
			since = s.Log.Seq
		case <-keep.C:
			if _, err := io.WriteString(w, ": ping\n\n"); err != nil {
				return
			}
			fl.Flush()
		}
	}
}

type startReq struct {
	Mode     string `json:"mode"`
	Range    string `json:"range"`
	Target   string `json:"target"`
	Batch    int    `json:"batch"`
	Jobs     int    `json:"jobs"`
	Sections int    `json:"sections"`
	GPUI     string `json:"gpui"`
	MaxBatch int    `json:"maxbatch"`
	Out      string `json:"out"`
	Extra    string `json:"extra"`
}

func splitArgs(s string) ([]string, error) {
	rs := []rune(s)
	var out []string
	var cur strings.Builder
	var quote rune
	flush := func() {
		if cur.Len() > 0 {
			out = append(out, cur.String())
			cur.Reset()
		}
	}
	for i := 0; i < len(rs); i++ {
		r := rs[i]
		switch {
		case quote != 0:
			if r == quote {
				quote = 0
				continue
			}
			if r == '\\' && quote == '"' && i+1 < len(rs) && (rs[i+1] == '"' || rs[i+1] == '\\') {
				i++
				cur.WriteRune(rs[i])
				continue
			}
			cur.WriteRune(r)
		case r == '\'' || r == '"':
			quote = r
		case r == '\\' && i+1 < len(rs) && (rs[i+1] == '"' || rs[i+1] == '\'' || rs[i+1] == ' '):
			i++
			cur.WriteRune(rs[i])
		case r == ' ' || r == '\t' || r == '\n' || r == '\r':
			flush()
		default:
			cur.WriteRune(r)
		}
	}
	if quote != 0 {
		return nil, errors.New("unterminated quote in extra args")
	}
	flush()
	return out, nil
}

func buildArgs(req startReq) ([]string, error) {
	if strings.TrimSpace(req.Range) == "" {
		return nil, errors.New("range is required")
	}
	if strings.TrimSpace(req.Target) == "" {
		return nil, errors.New("target is required")
	}
	args := []string{}
	switch req.Mode {
	case "", "sections":
	case "global":
		args = append(args, "--global")
	case "chunks":
		args = append(args, "--chunks")
	case "backward":
		args = append(args, "--backward")
	default:
		return nil, fmt.Errorf("unknown mode %q", req.Mode)
	}
	args = append(args, "--range", strings.TrimSpace(req.Range), "--target", strings.TrimSpace(req.Target))
	if req.Batch > 0 {
		args = append(args, "--batch", strconv.Itoa(req.Batch))
	}
	if req.Jobs > 0 {
		args = append(args, "--jobs", strconv.Itoa(req.Jobs))
	}
	if req.Sections > 0 {
		args = append(args, "--sections", strconv.Itoa(req.Sections))
	}
	if g := strings.TrimSpace(req.GPUI); g != "" {
		args = append(args, "--gpui", g)
	}
	if req.MaxBatch > 0 {
		args = append(args, "--maxbatch", strconv.Itoa(req.MaxBatch))
	}
	if o := strings.TrimSpace(req.Out); o != "" {
		args = append(args, "--out", o)
	}
	if strings.TrimSpace(req.Extra) != "" {
		extra, err := splitArgs(req.Extra)
		if err != nil {
			return nil, err
		}
		args = append(args, extra...)
	}
	return args, nil
}

func (a *App) handleStart(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, 405, map[string]string{"error": "POST only"})
		return
	}
	var req startReq
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		writeJSON(w, 400, map[string]string{"error": "bad json: " + err.Error()})
		return
	}
	args, err := buildArgs(req)
	if err != nil {
		writeJSON(w, 400, map[string]string{"error": err.Error()})
		return
	}
	if err := a.pm.Start(a.exe, a.dir, args); err != nil {
		writeJSON(w, 409, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, 200, a.pm.Info())
}

func (a *App) handleStop(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, 405, map[string]string{"error": "POST only"})
		return
	}
	if err := a.pm.Stop(10 * time.Second); err != nil {
		writeJSON(w, 409, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, 200, a.pm.Info())
}

type powerLimit struct {
	Index int `json:"index"`
	Watts int `json:"watts"`
}

type powerReq struct {
	Limits []powerLimit `json:"limits"`
}

type powerResult struct {
	Index  int    `json:"index"`
	OK     bool   `json:"ok"`
	Output string `json:"output"`
}

func (a *App) handlePower(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, 405, map[string]string{"error": "POST only"})
		return
	}
	var req powerReq
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		writeJSON(w, 400, map[string]string{"error": "bad json: " + err.Error()})
		return
	}
	if len(req.Limits) == 0 {
		writeJSON(w, 400, map[string]string{"error": "no limits given"})
		return
	}
	results := []powerResult{}
	for _, l := range req.Limits {
		if l.Index < 0 || l.Watts < 50 || l.Watts > 1500 {
			results = append(results, powerResult{Index: l.Index, OK: false, Output: "invalid index/watt value"})
			continue
		}
		out, err := runTimeout(30*time.Second, "nvidia-smi", "-i", strconv.Itoa(l.Index), "-pl", strconv.Itoa(l.Watts))
		text := strings.TrimSpace(string(out))
		if text == "" && err != nil {
			text = err.Error()
		}
		results = append(results, powerResult{Index: l.Index, OK: err == nil, Output: text})
	}
	writeJSON(w, 200, map[string]any{"results": results})
}

func (a *App) handleLogs(w http.ResponseWriter, r *http.Request) {
	since, _ := strconv.ParseUint(r.URL.Query().Get("since"), 10, 64)
	lines, seq := a.pm.LogsSince(since)
	writeJSON(w, 200, LogDelta{Seq: seq, Lines: lines})
}

// -------------------------------------------------------------------- server

func securityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		h.Set("X-Content-Type-Options", "nosniff")
		h.Set("Referrer-Policy", "no-referrer")
		h.Set("X-Frame-Options", "DENY")
		h.Set("Content-Security-Policy", "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
		next.ServeHTTP(w, r)
	})
}

func basicAuth(next http.Handler, user, pass string) http.Handler {
	if user == "" {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		u, p, ok := r.BasicAuth()
		if !ok || !constEq(u, user) || !constEq(p, pass) {
			w.Header().Set("WWW-Authenticate", `Basic realm="harakiri", charset="UTF-8"`)
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func constEq(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	var v byte
	for i := 0; i < len(a); i++ {
		v |= a[i] ^ b[i]
	}
	return v == 0
}

func main() {
	addr := flag.String("addr", "127.0.0.1:8099", "listen address (use 0.0.0.0:8099 to expose on the network)")
	exe := flag.String("exe", "harakiri_sweep.exe", "sweep binary (absolute, or relative to -dir)")
	dir := flag.String("dir", ".", "working directory for the sweep process")
	poll := flag.Duration("poll", time.Second, "GPU polling interval")
	histSec := flag.Int("history", 900, "seconds of telemetry kept in memory")
	user := flag.String("user", "", "basic auth username (empty disables auth)")
	pass := flag.String("pass", "", "basic auth password")
	cert := flag.String("tls-cert", "", "TLS certificate file (enables https)")
	key := flag.String("tls-key", "", "TLS private key file")
	flag.Parse()

	if *poll < 250*time.Millisecond {
		*poll = 250 * time.Millisecond
	}
	if (*cert == "") != (*key == "") {
		log.Fatal("-tls-cert and -tls-key must be given together")
	}
	if *user != "" && *pass == "" {
		log.Fatal("-user requires -pass")
	}

	absDir, err := filepath.Abs(*dir)
	if err != nil {
		log.Fatalf("bad -dir: %v", err)
	}
	exePath := *exe
	if !filepath.IsAbs(exePath) {
		exePath = filepath.Join(absDir, exePath)
	}
	exePath, _ = filepath.Abs(exePath)
	if st, err := os.Stat(exePath); err != nil {
		log.Fatalf("sweep binary not found: %s (%v)", exePath, err)
	} else if st.IsDir() {
		log.Fatalf("sweep binary is a directory: %s", exePath)
	}

	histMax := int(time.Duration(*histSec) * time.Second / *poll)
	if histMax < 10 {
		histMax = 10
	}

	app := &App{
		histMax: histMax,
		poll:    *poll,
		epoch:   strconv.FormatInt(time.Now().UnixNano(), 36),
		pm:      &Proc{},
		dir:     absDir,
		exe:     exePath,
	}

	go func() {
		t := time.NewTicker(*poll)
		defer t.Stop()
		app.pollOnce()
		for range t.C {
			app.pollOnce()
		}
	}()

	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(uiHTML)
	})
	mux.HandleFunc("/ui.css", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/css; charset=utf-8")
		_, _ = w.Write(uiCSS)
	})
	mux.HandleFunc("/ui.js", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/javascript; charset=utf-8")
		_, _ = w.Write(uiJS)
	})
	mux.HandleFunc("/favicon.ico", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})
	mux.HandleFunc("/api/state", app.handleState)
	mux.HandleFunc("/api/events", app.handleEvents)
	mux.HandleFunc("/api/start", app.handleStart)
	mux.HandleFunc("/api/stop", app.handleStop)
	mux.HandleFunc("/api/power", app.handlePower)
	mux.HandleFunc("/api/log", app.handleLogs)

	handler := securityHeaders(basicAuth(mux, *user, *pass))

	host, _, err := net.SplitHostPort(*addr)
	if err == nil && host != "" && host != "127.0.0.1" && host != "::1" && host != "localhost" && *user == "" {
		log.Printf("WARNING: listening on %s without authentication. Traffic is unencrypted and anyone who can reach this port can start/stop the sweep and change GPU power limits.", *addr)
	}

	srv := &http.Server{
		Addr:              *addr,
		Handler:           handler,
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
	}

	go func() {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, os.Interrupt)
		<-sig
		log.Println("shutting down")
		_ = app.pm.Stop(5 * time.Second)
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_ = srv.Shutdown(ctx)
		os.Exit(0)
	}()

	scheme := "http"
	if *cert != "" {
		scheme = "https"
	}
	log.Printf("harakiri dashboard on %s://%s", scheme, *addr)
	log.Printf("sweep binary: %s", exePath)
	log.Printf("working dir : %s", absDir)
	log.Printf("poll %s, history %d samples", *poll, histMax)

	if *cert != "" {
		err = srv.ListenAndServeTLS(*cert, *key)
	} else {
		err = srv.ListenAndServe()
	}
	if err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}
