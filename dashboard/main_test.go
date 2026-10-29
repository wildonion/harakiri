package main

import (
	"reflect"
	"strings"
	"testing"
)

func TestSplitArgs(t *testing.T) {
	cases := []struct {
		in   string
		want []string
		bad  bool
	}{
		{`--verbose --multibuffer`, []string{"--verbose", "--multibuffer"}, false},
		{`--out "C:\Program Files\x.txt"`, []string{"--out", `C:\Program Files\x.txt`}, false},
		{`C:\temp\file`, []string{`C:\temp\file`}, false},
		{`'a b'   c`, []string{"a b", "c"}, false},
		{`--foo "a\"b"`, []string{"--foo", `a"b`}, false},
		{`--flag`, []string{"--flag"}, false},
		{``, nil, false},
		{`"unterminated`, nil, true},
	}
	for _, c := range cases {
		got, err := splitArgs(c.in)
		if c.bad {
			if err == nil {
				t.Errorf("splitArgs(%q): expected error", c.in)
			}
			continue
		}
		if err != nil {
			t.Errorf("splitArgs(%q): %v", c.in, err)
			continue
		}
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("splitArgs(%q) = %#v, want %#v", c.in, got, c.want)
		}
	}
}

func TestBuildArgs(t *testing.T) {
	args, err := buildArgs(startReq{
		Mode:     "chunks",
		Range:    "200000000000:3fffffffffff",
		Target:   "1F3JRMWudBaj48EhwcHDdpeuy2jwACNxjP",
		Batch:    1024,
		Jobs:     87040,
		Sections: 3,
		GPUI:     "0,1,2",
		MaxBatch: 200,
		Out:      "found.txt",
		Extra:    "--verbose",
	})
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(args, " ")
	for _, want := range []string{
		"--chunks", "--range 200000000000:3fffffffffff",
		"--target 1F3JRMWudBaj48EhwcHDdpeuy2jwACNxjP",
		"--batch 1024", "--jobs 87040", "--sections 3",
		"--gpui 0,1,2", "--maxbatch 200", "--out found.txt", "--verbose",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("args %q missing %q", joined, want)
		}
	}
	if strings.Contains(joined, "--global") {
		t.Errorf("chunks request must not contain --global: %q", joined)
	}

	if _, err := buildArgs(startReq{Mode: "global"}); err == nil {
		t.Error("expected error for empty range/target")
	}
	if _, err := buildArgs(startReq{Range: "1:2", Target: "x", Mode: "nope"}); err == nil {
		t.Error("expected error for unknown mode")
	}
}

func TestParseGPUs(t *testing.T) {
	csv := `0, NVIDIA GeForce RTX 5090, 67, 575.61, 575.00, 2400, 7000, 100, 18, 1234, 32607, 5, 8, 0x0000000000000004
1, NVIDIA GeForce RTX 5090, 70, 560.10, 575.00, 2350, 7000, 99, 17, 1200, 32607, 5, 8, 0x0000000000000000
`
	gpus, err := parseGPUs([]byte(csv))
	if err != nil {
		t.Fatal(err)
	}
	if len(gpus) != 2 {
		t.Fatalf("got %d gpus, want 2", len(gpus))
	}
	g := gpus[0]
	if g.Name != "NVIDIA GeForce RTX 5090" || g.Index != 0 {
		t.Errorf("bad identity: %+v", g)
	}
	if g.Temp == nil || *g.Temp != 67 || g.PowerDraw == nil || *g.PowerDraw != 575.61 {
		t.Errorf("bad numerics: %+v", g)
	}
	if g.PCIeGen != "5" || g.PCIeWidth != "8" || g.Throttle != "0x0000000000000004" {
		t.Errorf("bad pcie/throttle: %+v", g)
	}

	na := `0, GPU, 50, [N/A], 575.00, 2000, 7000, 0, 0, 100, 8000, 4, 8, 0x0000000000000001`
	gpus, err = parseGPUs([]byte(na))
	if err != nil {
		t.Fatal(err)
	}
	if gpus[0].PowerDraw != nil {
		t.Errorf("N/A should parse to nil, got %v", *gpus[0].PowerDraw)
	}
	if gpus[0].PowerLimit == nil || *gpus[0].PowerLimit != 575 {
		t.Errorf("power limit not parsed: %+v", gpus[0])
	}

	if _, err := parseGPUs([]byte("")); err == nil {
		t.Error("expected error for empty output")
	}
}
