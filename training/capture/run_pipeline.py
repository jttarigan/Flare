"""Run capture + compress + manifest for the 5 synthetic FlareLab sessions.

For each session bin under FlareLab/sessions/:
    1. flarelab.exe --replay <bin> --capture <tmp samples.bin>
    2. zstd-compress with the `zstandard` Python module → samples.zst
    3. Write meta.json (capture date, repo SHAs, GT settings, sample count,
       raw/compressed sizes).

Output layout (per DATA_SPEC v0.2 §5):
    Flare/training/data/<session_dir>/
        samples.zst
        meta.json

Run sequentially. Each session is ~2 min wallclock (FlareLab runs vsynced at
60 FPS, forced dt=1/60 in replay mode), so the full pipeline is ~10 min.
Don't alt-tab during capture.
"""

import datetime
import json
import os
import shutil
import struct
import subprocess
import sys
import time

import zstandard

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
FLARELAB_DIR     = os.path.join(REPO_ROOT, "FlareLab")
FLARELAB_EXE     = os.path.join(FLARELAB_DIR, "build", "Release", "flarelab.exe")
SESSIONS_DIR     = os.path.join(FLARELAB_DIR, "cursor_scripts")
ITHAPPYGAME_DIR  = os.path.join(REPO_ROOT, "ITHappyGame")
OUT_BASE         = os.path.join(REPO_ROOT, "Flare", "training", "data")
TMP_DIR          = os.path.join(REPO_ROOT, "FlareLab", "build", "capture_tmp")

SAMPLE_BYTES = 263336  # DATA_SPEC v0.2 wire format

SESSIONS = [
    ("session_01_idle.bin",  "session_01_idle",  "idle"),
    ("session_02_slow.bin",  "session_02_slow",  "slow"),
    ("session_03_chase.bin", "session_03_chase", "chase"),
    ("session_04_scrum.bin", "session_04_scrum", "scrum"),
    ("session_05_empty.bin", "session_05_empty", "empty"),
]

GT_SETTINGS = {
    "shadow_resolution":   "4096^2 cube",
    "pcf_taps":            16,
    "pcf_kernel":          "Poisson disk",
    "output_resolution":   "256x256",
    "output_format":       "f16",
    "cadence_frames":      4,
}


def git_sha(repo_dir):
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return out.stdout.decode().strip()
    except Exception as e:
        return f"<unavailable: {e}>"


def assert_paths():
    if not os.path.isfile(FLARELAB_EXE):
        sys.exit(f"flarelab.exe not found: {FLARELAB_EXE}\n"
                 f"build it with: cmake --build FlareLab/build --config Release --target flarelab")
    for binname, _, _ in SESSIONS:
        p = os.path.join(SESSIONS_DIR, binname)
        if not os.path.isfile(p):
            sys.exit(f"missing input log: {p}\nrun generate_sessions.py first")


def run_capture(input_bin, output_samples_bin):
    if os.path.exists(output_samples_bin):
        os.remove(output_samples_bin)
    cmd = [FLARELAB_EXE,
           "--cursor-script", input_bin,
           "--capture",       output_samples_bin]
    print(f"    $ flarelab --cursor-script {os.path.basename(input_bin)} --capture {os.path.basename(output_samples_bin)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=FLARELAB_DIR)
    dt = time.time() - t0
    if result.returncode != 0:
        sys.exit(f"flarelab.exe exited with code {result.returncode}")
    if not os.path.isfile(output_samples_bin):
        sys.exit(f"flarelab did not produce {output_samples_bin}")
    sz = os.path.getsize(output_samples_bin)
    if sz % SAMPLE_BYTES != 0:
        sys.exit(f"raw size {sz} is not a multiple of {SAMPLE_BYTES}")
    n_samples = sz // SAMPLE_BYTES
    print(f"      {n_samples} samples, {sz} B raw, {dt:.1f} s wall")
    return n_samples, sz, dt


def zstd_compress(src_path, dst_path):
    cctx = zstandard.ZstdCompressor(level=1)
    t0 = time.time()
    with open(src_path, "rb") as fin, open(dst_path, "wb") as fout:
        cctx.copy_stream(fin, fout)
    dt = time.time() - t0
    sz = os.path.getsize(dst_path)
    raw = os.path.getsize(src_path)
    print(f"      zstd-1: {raw} -> {sz} B ({sz/raw*100:.1f}%), {dt:.2f} s")
    return sz


def first_sample_summary(samples_bin_path):
    """Parse just the header of the first Sample for the meta file."""
    with open(samples_bin_path, "rb") as f:
        header = f.read(168)  # 8+8+8 + 64+64 + 12 + 4 = 168
    fi, ulo, uhi = struct.unpack_from("<QQQ", header, 0)
    # view[16] at 24, proj[16] at 88
    proj_diag = struct.unpack_from("<4f", header, 88 + 0*16)[0], \
                struct.unpack_from("<4f", header, 88 + 1*16)[1], \
                struct.unpack_from("<4f", header, 88 + 2*16)[2], \
                struct.unpack_from("<4f", header, 88 + 3*16)[3]
    hero = struct.unpack_from("<3f", header, 152)
    n_active = struct.unpack_from("<I", header, 164)[0]
    return {
        "first_frame_index": int(fi),
        "session_uuid_low":  hex(ulo),
        "session_uuid_high": hex(uhi),
        "proj_diag":         [float(v) for v in proj_diag],
        "hero_light_pos":    [float(v) for v in hero],
        "active_light_count": int(n_active),
    }


def write_meta(meta_path, *, tag, input_bin, n_samples, raw_size, zst_size,
               wall_seconds, flarelab_sha, ithappygame_sha, first_sample):
    meta = {
        "tag":              tag,
        "input_bin":        os.path.relpath(input_bin, REPO_ROOT).replace("\\", "/"),
        "captured_at":      datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "wall_seconds":     round(wall_seconds, 1),
        "n_samples":        n_samples,
        "raw_bytes":        raw_size,
        "compressed_bytes": zst_size,
        "compression":      "zstd level 1 (zstandard python module)",
        "sample_bytes":     SAMPLE_BYTES,
        "wire_format":      "DATA_SPEC v0.2",
        "gt_settings":      GT_SETTINGS,
        "flarelab_sha":     flarelab_sha,
        "ithappygame_sha":  ithappygame_sha,
        "first_sample":     first_sample,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"      meta.json <- {meta_path}")


def main():
    assert_paths()
    os.makedirs(OUT_BASE, exist_ok=True)
    os.makedirs(TMP_DIR,  exist_ok=True)

    flarelab_sha    = git_sha(FLARELAB_DIR)
    ithappygame_sha = git_sha(ITHAPPYGAME_DIR)
    print(f"FlareLab    SHA: {flarelab_sha}")
    print(f"ITHappyGame SHA: {ithappygame_sha}")
    print()

    grand_total_t = time.time()
    for binname, dirname, tag in SESSIONS:
        print(f"=== {dirname} ===")
        input_bin = os.path.join(SESSIONS_DIR, binname)
        out_dir   = os.path.join(OUT_BASE, dirname)
        os.makedirs(out_dir, exist_ok=True)

        tmp_raw = os.path.join(TMP_DIR, f"{dirname}.bin")
        n_samples, raw_size, wall = run_capture(input_bin, tmp_raw)

        zst_path = os.path.join(out_dir, "samples.zst")
        zst_size = zstd_compress(tmp_raw, zst_path)

        first = first_sample_summary(tmp_raw)
        write_meta(os.path.join(out_dir, "meta.json"),
                   tag=tag, input_bin=input_bin,
                   n_samples=n_samples, raw_size=raw_size, zst_size=zst_size,
                   wall_seconds=wall,
                   flarelab_sha=flarelab_sha, ithappygame_sha=ithappygame_sha,
                   first_sample=first)
        os.remove(tmp_raw)
        print()

    if os.path.isdir(TMP_DIR) and not os.listdir(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    print(f"all 5 sessions complete in {time.time()-grand_total_t:.1f} s wall")


if __name__ == "__main__":
    main()
