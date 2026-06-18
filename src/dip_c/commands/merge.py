"""Hierarchical sort-merge of .pairs.gz files.

Merges multiple .pairs.gz files into a single sorted, compressed
.pairs.gz file using GNU sort (merge mode) and pigz.

Inputs may be supplied as:
  * Positional arguments: individual .pairs.gz files and/or
    directories (each directory is expanded to its *.pairs.gz files).
  * A manifest file via -T/--files-from (one path per line; use '-'
    to read from stdin).
The two sources may be combined; duplicates are removed.

Requires:  sort, gunzip, pigz  (standard on HPC systems)

Usage:
    dip-c merge [INPUT ...] [-T FILE] -o <output> [-g <genome>]
                [-h <header>] [-j <jobs>] [-m <memory>]
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from dip_c.merge_utils import (
    SORT_KEYS,
    check_required_tools,
    chk_chrom_and_fields,
    chrom_index_map,
    choose_grouping_factors,
    compute_batch_size,
    count_lines,
    decompress_strip_header,
    detect_cpus,
    estimate_file_sizes,
    extract_header,
)


# ══════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════

def _log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    sys.stderr.write("[M::merge] [%s] %s\n" % (ts, msg))


def _elapsed(start):
    secs = int(time.time() - start)
    h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
    return "%02d:%02d:%02d" % (h, m, s)


# ══════════════════════════════════════════════════════════════════════════
# Sort-merge helper (called in worker processes)
# ══════════════════════════════════════════════════════════════════════════

def _sort_merge_batch(file_list, output_path, sort_mem_gb, tmpdir):
    """Run ``sort -m`` on a list of pre-sorted files."""
    cmd = (
        ["sort", "-m", "-S", "%dG" % sort_mem_gb, "-T", tmpdir]
        + SORT_KEYS
        + file_list
    )
    with open(output_path, "w") as out:
        subprocess.run(cmd, stdout=out, stderr=subprocess.DEVNULL, check=True)
    return output_path


# ══════════════════════════════════════════════════════════════════════════
# Phase implementations
# ══════════════════════════════════════════════════════════════════════════

def _phase1_decompress(pairs_files, work_dir, max_jobs):
    """Decompress .pairs.gz files, stripping headers."""
    _log("Phase 1: Decompressing %d .pairs.gz files..." % len(pairs_files))

    # Index-prefix the output filename so that two inputs with the same
    # basename (from different directories) cannot collide in work_dir.
    # The original basename is kept for debuggability.
    width = max(4, len(str(len(pairs_files) - 1)))
    noheader_files = []
    with ProcessPoolExecutor(max_workers=max_jobs) as pool:
        futures = {}
        for idx, pf in enumerate(pairs_files):
            stem = os.path.basename(pf)
            if stem.endswith(".pairs.gz"):
                stem = stem[: -len(".pairs.gz")]
            out_name = "%0*d_%s.noheader.pairs" % (width, idx, stem)
            out = os.path.join(work_dir, out_name)
            noheader_files.append(out)
            futures[pool.submit(decompress_strip_header, pf, out)] = pf

        for fut in as_completed(futures):
            fut.result()  # raise on error

    _log("Phase 1: Decompressed %d files" % len(noheader_files))
    return noheader_files


def _phase2_estimate(noheader_files):
    """Estimate file sizes by sampling."""
    _log("Phase 2: Estimating file sizes...")
    median, mean, stdev, minv, maxv = estimate_file_sizes(noheader_files)

    if len(noheader_files) < 10:
        _log("All %d files counted. Median: %d contacts"
             % (len(noheader_files), median))
    else:
        _log("Sample statistics (10 random files):")
        sys.stderr.write("  Mean: %d contacts\n" % mean)
        sys.stderr.write("  StdDev: %d contacts\n" % stdev)
        sys.stderr.write("  Median: %d contacts\n" % median)
        sys.stderr.write("  Range: %d - %d contacts\n" % (minv, maxv))

    return median


def _phase3_batch_size(num_files, median, sort_mem_per_job, jobs):
    """Compute optimal batch size."""
    batch_size, mem_limit, cpu_opt = compute_batch_size(
        num_files, median, sort_mem_per_job, jobs,
    )
    _log("Phase 3: Batch size calculation:")
    sys.stderr.write("  Memory limit batch size: %d\n" % mem_limit)
    sys.stderr.write("  CPU optimal batch size: %d\n" % cpu_opt)
    sys.stderr.write("  Selected batch size: %d\n" % batch_size)
    sys.stderr.write("  Expected batches: %d\n"
                     % ((num_files + batch_size - 1) // batch_size))
    return batch_size


def _phase4_initial_sort(noheader_files, batch_size, sort_mem_gb,
                         max_jobs, work_dir):
    """Split into batches and parallel sort-merge each batch."""
    # Chunk the file list
    batches = []
    for i in range(0, len(noheader_files), batch_size):
        batches.append(noheader_files[i:i + batch_size])

    _log("Phase 4: Sorting %d files in %d batches..."
         % (len(noheader_files), len(batches)))

    tmpdir = os.path.join(work_dir, "sort_tmp")
    os.makedirs(tmpdir, exist_ok=True)

    sorted_paths = []
    with ProcessPoolExecutor(max_workers=max_jobs) as pool:
        futures = {}
        for idx, batch in enumerate(batches):
            out = os.path.join(work_dir, "batch_%04d_sorted" % idx)
            sorted_paths.append(out)
            futures[pool.submit(
                _sort_merge_batch, batch, out, sort_mem_gb, tmpdir,
            )] = idx

        for fut in as_completed(futures):
            fut.result()  # raise on error

    _log("Phase 4: Batch sort complete (%d sorted batches)" % len(sorted_paths))
    return sorted_paths


def _phase5_validate(sorted_batches, genome, max_jobs):
    """Chromosome validation on each sorted batch."""
    chrom_idx = chrom_index_map(genome)
    if chrom_idx is None:
        _log("Phase 5: genome=any, checking field count only")
    else:
        _log("Phase 5: Validating chromosome ordering (%s)..." % genome)

    total_removed = 0
    with ProcessPoolExecutor(max_workers=max_jobs) as pool:
        futures = {
            pool.submit(chk_chrom_and_fields, path, chrom_idx): path
            for path in sorted_batches
        }
        for fut in as_completed(futures):
            removed = fut.result()
            if removed > 0:
                path = futures[fut]
                sys.stderr.write("  %s: %d lines removed\n"
                                 % (os.path.basename(path), removed))
            total_removed += removed

    _log("Phase 5: Validation complete (%d total lines removed)"
         % total_removed)


def _phase6_hierarchical_merge(sorted_batches, median, sort_mem_gb,
                                max_jobs, work_dir):
    """Multi-level hierarchical merge using sort -m."""
    factors = choose_grouping_factors(len(sorted_batches), median)
    _log("Phase 6: Hierarchical merge (%d batches, factors: %s)"
         % (len(sorted_batches), factors))

    tmpdir = os.path.join(work_dir, "sort_tmp")
    os.makedirs(tmpdir, exist_ok=True)

    current_files = list(sorted_batches)
    level = 1

    while len(current_files) > 1:
        factor_idx = min(level - 1, len(factors) - 1)
        group_size = factors[factor_idx]
        group_size = min(group_size, len(current_files))

        _log("Level %d: %d files, group size %d"
             % (level, len(current_files), group_size))

        level_dir = os.path.join(work_dir, "merge_level%d" % level)
        os.makedirs(level_dir, exist_ok=True)

        # Build groups
        groups = []
        for i in range(0, len(current_files), group_size):
            groups.append(current_files[i:i + group_size])

        # Parallel merge
        next_files = []
        with ProcessPoolExecutor(max_workers=max_jobs) as pool:
            futures = {}
            for idx, group in enumerate(groups):
                out = os.path.join(level_dir, "merged_%04d" % idx)
                next_files.append(out)
                futures[pool.submit(
                    _sort_merge_batch, group, out, sort_mem_gb, tmpdir,
                )] = idx

            for fut in as_completed(futures):
                fut.result()

        # Cleanup previous level
        if level == 1:
            for f in sorted_batches:
                if os.path.exists(f):
                    os.remove(f)
        else:
            prev_dir = os.path.join(work_dir, "merge_level%d" % (level - 1))
            if os.path.isdir(prev_dir):
                shutil.rmtree(prev_dir)

        current_files = next_files
        level += 1

    return current_files[0]


def _phase7_compress(final_sorted, header_text, output_path, pigz_threads):
    """Prepend header and compress with pigz, atomically.

    Writes to ``<output_path>.tmp.<pid>`` and ``os.replace``s it onto
    the final path only on success. On any failure (pigz error,
    SIGINT/Ctrl-C, broken pipe), the partial temp file is removed and
    the running pigz process is killed, so callers never observe a
    half-written ``output_path``.
    """
    _log("Phase 7: Compressing with pigz (%d threads)..." % pigz_threads)
    tmp_out = "%s.tmp.%d" % (output_path, os.getpid())
    pigz = None
    pigz_err = b""
    try:
        with open(tmp_out, "wb") as out:
            pigz = subprocess.Popen(
                ["pigz", "-p", str(pigz_threads)],
                stdin=subprocess.PIPE, stdout=out, stderr=subprocess.PIPE,
            )
            try:
                pigz.stdin.write(header_text.encode("utf-8"))
                with open(final_sorted, "rb") as f:
                    shutil.copyfileobj(f, pigz.stdin, length=1024 * 1024)
            except BrokenPipeError:
                # pigz died early; the real error is on its stderr,
                # which we drain below before raising.
                pass
            finally:
                try:
                    pigz.stdin.close()
                except (OSError, BrokenPipeError):
                    pass
            # pigz stderr is small (a single error line, if any), so a
            # blocking read here is safe and avoids communicate()'s
            # double-flush of an already-closed stdin.
            pigz_err = pigz.stderr.read()
            pigz.stderr.close()
            pigz.wait()
        if pigz.returncode != 0:
            raise RuntimeError(
                "pigz failed (exit %d): %s"
                % (pigz.returncode,
                   (pigz_err or b"").decode("utf-8", "replace").strip())
            )
        os.replace(tmp_out, output_path)
    except BaseException:
        # Kill pigz if still running, then remove the partial output.
        if pigz is not None and pigz.poll() is None:
            try:
                pigz.kill()
            except OSError:
                pass
            try:
                pigz.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if os.path.exists(tmp_out):
            try:
                os.remove(tmp_out)
            except OSError:
                pass
        raise


# ══════════════════════════════════════════════════════════════════════════
# Argument parser
# ══════════════════════════════════════════════════════════════════════════

VALID_GENOMES = list(sorted(
    ["mm10", "hg19", "hg38", "any"]
))

def _build_parser():
    p = argparse.ArgumentParser(
        prog="dip-c merge",
        description="Hierarchical sort-merge of .pairs.gz files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        add_help=False,
        epilog="""\
Usage:
  dip-c merge [INPUT ...] [-T FILE] -o <output> [-g <genome>]
              [-h <header.pairs>] [-j <max_jobs>] [-m <max_mem_gb>]

Inputs (one or both required):
  INPUT ...           Positional: any number of .pairs.gz files and/or
                      directories. Each directory is expanded to its
                      *.pairs.gz files.
  -T, --files-from    Manifest file with one path per line. Use '-' to
                      read paths from stdin. Lines that are blank or
                      start with '#' are ignored. Relative paths are
                      resolved against the manifest's directory (or
                      the current directory, for stdin).

Required:
  -o <output>         Output file name (.pairs.gz appended if absent)

Optional:
  -g <genome>         Genome ID: mm10, hg19, hg38
                      If omitted, chrom-order validation is skipped
  -h <header>         Header file (default: auto-extracted from first file)
  -j <max_jobs>       Max parallel jobs (default: auto-detect CPUs, cap 16)
  -m <max_mem_gb>     Max memory in GB (default: 32)

Resource limits (-j, -m) are upper bounds for the adaptive algorithm.
Sort memory per job = min(total_mem / jobs, 30 GB).
Batch sizes and grouping factors are computed dynamically from file sizes.

Examples:
  # Glob expansion via the shell
  dip-c merge data/*.pairs.gz -g mm10 -o merged

  # A whole directory
  dip-c merge /path/to/pairsgz_dir -g mm10 -o merged.pairs.gz

  # Mix files and directories freely
  dip-c merge a.pairs.gz extra_dir/ b.pairs.gz -g hg38 -o merged

  # Manifest file (one path per line)
  dip-c merge -T samples.txt -g mm10 -o merged

  # Manifest from stdin (composes with find/xargs)
  find data -name '*.pairs.gz' | dip-c merge -T - -g mm10 -o merged

  # Manifest plus extra files
  dip-c merge -T core.txt extra1.pairs.gz extra2.pairs.gz -o merged

  # Skip chrom validation (omit -g)
  dip-c merge data/*.pairs.gz -o merged
""",
    )

    p.add_argument(
        "inputs", nargs="*", metavar="INPUT",
        help="One or more .pairs.gz files and/or directories containing "
             "them. Directories are expanded to their *.pairs.gz files. "
             "May be combined with -T.",
    )
    p.add_argument(
        "-T", "--files-from", dest="files_from", default=None,
        metavar="FILE",
        help="Read input paths from FILE (one per line). Use '-' to read "
             "from stdin. Blank lines and '#' comments are ignored.",
    )
    p.add_argument(
        "-g", "--genome", required=False, default=None,
        metavar="GENOME",
        help="Genome ID: mm10, hg19, hg38. "
             "If omitted, chromosome-order validation is skipped "
             "(only field count is checked).",
    )
    p.add_argument(
        "-o", "--output", required=True,
        metavar="FILE",
        help="Output file name. If it does not end with .pairs.gz, "
             "the suffix is appended automatically.",
    )
    p.add_argument(
        "-h", "--header", default=None,
        metavar="FILE",
        help="Header file to prepend. "
             "Default: auto-extracted from the first .pairs.gz file.",
    )
    p.add_argument(
        "-j", "--jobs", type=int, default=None,
        metavar="N",
        help="Max parallel jobs (default: auto-detect, capped at 16).",
    )
    p.add_argument(
        "-m", "--memory", type=int, default=32,
        metavar="GB",
        help="Max memory in GB (default: 32).",
    )
    p.add_argument(
        "--help", action="help",
        help="Show this help message and exit.",
    )

    return p


# ══════════════════════════════════════════════════════════════════════════
# Input resolution
# ══════════════════════════════════════════════════════════════════════════

def _resolve_inputs(args, parser):
    """Collect .pairs.gz paths from positionals and/or a manifest file.

    Resolution rules:
      * Each positional that is a directory is expanded to its
        ``*.pairs.gz`` files; each positional that is a file must end
        with ``.pairs.gz``.
      * ``-T FILE`` reads one path per line. Blank lines and lines
        starting with ``#`` are ignored. Relative paths are resolved
        against the manifest's parent directory (or CWD when reading
        from stdin via ``-T -``).
      * Duplicates are removed (first occurrence wins) and the final
        list is sorted for reproducible batch numbering.

    Exits via ``parser.error`` if no inputs are supplied or if any
    referenced path is missing or has the wrong extension.
    """
    paths = []

    # 1) Positional inputs: files and directories.
    for entry in args.inputs:
        p = os.path.abspath(entry)
        if os.path.isdir(p):
            found = sorted(glob.glob(os.path.join(p, "*.pairs.gz")))
            if not found:
                parser.error("No .pairs.gz files in directory: %s" % p)
            paths.extend(found)
        elif os.path.isfile(p):
            if not p.endswith(".pairs.gz"):
                parser.error("Not a .pairs.gz file: %s" % entry)
            paths.append(p)
        else:
            parser.error("Not a file or directory: %s" % entry)

    # 2) Manifest file (or stdin) via -T.
    if args.files_from is not None:
        if args.files_from == "-":
            raw_lines = sys.stdin.read().splitlines()
            base_dir = os.getcwd()
            src_label = "<stdin>"
        else:
            manifest = os.path.abspath(args.files_from)
            if not os.path.isfile(manifest):
                parser.error("Manifest file not found: %s" % args.files_from)
            with open(manifest, "r") as f:
                raw_lines = f.read().splitlines()
            base_dir = os.path.dirname(manifest)
            src_label = manifest

        for lineno, raw in enumerate(raw_lines, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            p = line if os.path.isabs(line) else os.path.join(base_dir, line)
            p = os.path.abspath(p)
            if os.path.isdir(p):
                parser.error(
                    "%s:%d: manifest entries must be .pairs.gz files, "
                    "not directories: %s "
                    "(pass directories as positional arguments instead)"
                    % (src_label, lineno, line)
                )
            if not os.path.isfile(p):
                parser.error(
                    "%s:%d: path does not exist: %s"
                    % (src_label, lineno, line)
                )
            if not p.endswith(".pairs.gz"):
                parser.error(
                    "%s:%d: not a .pairs.gz file: %s"
                    % (src_label, lineno, line)
                )
            paths.append(p)

    if not paths:
        parser.error(
            "No input files. Provide one or more .pairs.gz files or "
            "directories as positional arguments, and/or use "
            "-T <manifest> (use '-' for stdin)."
        )

    # Deduplicate (preserve first occurrence) and sort for reproducibility.
    return sorted(dict.fromkeys(paths))


# ══════════════════════════════════════════════════════════════════════════
# CLI entry point  –  called from dip_c.cli as  ``dip-c merge …``
# ══════════════════════════════════════════════════════════════════════════

def merge(argv):
    """Parse flags and run the hierarchical sort-merge pipeline."""
    parser = _build_parser()
    args = parser.parse_args(argv[1:])

    # -- Resolve and validate inputs ---------------------------------------
    pairs_files = _resolve_inputs(args, parser)

    genome = args.genome
    if genome is None:
        genome = "any"
    if genome not in VALID_GENOMES:
        parser.error(
            "Unknown genome '%s'. Supported: %s (or omit -g to skip validation)"
            % (genome, ", ".join(g for g in VALID_GENOMES if g != "any"))
        )

    output = args.output
    if not output.endswith(".pairs.gz"):
        output = output + ".pairs.gz"
    output = os.path.abspath(output)
    output_parent = os.path.dirname(output) or os.getcwd()
    if not os.path.isdir(output_parent):
        parser.error("Output directory does not exist: %s" % output_parent)

    max_jobs = args.jobs or detect_cpus()
    max_mem = args.memory

    if max_mem < 1:
        parser.error("--memory must be at least 1 GB (got %d)" % max_mem)
    if max_jobs < 1:
        parser.error("--jobs must be at least 1 (got %d)" % max_jobs)

    # -m is a hard constraint (HPC schedulers OOM-kill jobs that exceed
    # it); -j is a soft constraint (just performance). If their ratio
    # gives <1 GB/job, reduce -j rather than violate -m.
    sort_mem_per_job = max_mem // max_jobs
    if sort_mem_per_job < 1:
        new_jobs = max_mem  # gives exactly 1 GB/job
        _log("Warning: -m %d GB / -j %d gives <1 GB/job; "
             "reducing jobs to %d to honour the memory budget."
             % (max_mem, max_jobs, new_jobs))
        max_jobs = new_jobs
        sort_mem_per_job = max_mem // max_jobs
    sort_mem_per_job = min(sort_mem_per_job, 30)

    # -- Check tools -------------------------------------------------------
    check_required_tools()

    # -- Log setup ---------------------------------------------------------
    start_time = time.time()
    _log("Merge started")
    _log("Output: %s" % output)
    _log("Genome: %s" % genome)
    _log("Found %d .pairs.gz files" % len(pairs_files))
    _log("Resource limits: %d jobs, %d GB memory (%d GB/job for sort)"
         % (max_jobs, max_mem, sort_mem_per_job))

    # -- Single-file shortcut ----------------------------------------------
    # Only valid when the result would be byte-identical to a copy:
    # no custom header, and no chrom-order validation (which may
    # remove rows).
    if len(pairs_files) == 1 and not args.header and genome == "any":
        _log("Only 1 file and no -h/-g requested, copying to output")
        tmp_out = "%s.tmp.%d" % (output, os.getpid())
        try:
            shutil.copy2(pairs_files[0], tmp_out)
            os.replace(tmp_out, output)
        except BaseException:
            if os.path.exists(tmp_out):
                try:
                    os.remove(tmp_out)
                except OSError:
                    pass
            raise
        _log("Done in %s" % _elapsed(start_time))
        return 0
    if len(pairs_files) == 1:
        _log("Only 1 file found; running pipeline to honour -h/-g")

    # -- Header extraction -------------------------------------------------
    if args.header:
        with open(args.header, "r") as f:
            header_text = f.read()
        _log("Header loaded from %s" % args.header)
    else:
        header_text = extract_header(pairs_files[0])
        _log("Header auto-extracted from %s (%d lines)"
             % (os.path.basename(pairs_files[0]),
                header_text.count("\n")))

    # -- Create work directory ---------------------------------------------
    # Placed next to the output so spill/staging stays on the same
    # filesystem as the final file.
    work_dir = tempfile.mkdtemp(prefix=".merge_work_", dir=output_parent)
    _log("Work directory: %s" % work_dir)

    try:
        # Phase 1: Decompress
        noheader_files = _phase1_decompress(pairs_files, work_dir, max_jobs)

        # Phase 2: Estimate sizes
        median = _phase2_estimate(noheader_files)

        # Phase 3: Batch size
        batch_size = _phase3_batch_size(
            len(noheader_files), median, sort_mem_per_job, max_jobs,
        )

        # Phase 4: Initial batch sort
        sorted_batches = _phase4_initial_sort(
            noheader_files, batch_size, sort_mem_per_job, max_jobs, work_dir,
        )

        # Clean up decompressed files (no longer needed)
        for f in noheader_files:
            if os.path.exists(f):
                os.remove(f)

        # Phase 5: Chromosome validation
        _phase5_validate(sorted_batches, genome, max_jobs)

        # Phase 6: Hierarchical merge
        final_sorted = _phase6_hierarchical_merge(
            sorted_batches, median, sort_mem_per_job, max_jobs, work_dir,
        )

        # Count contacts
        total_contacts = count_lines(final_sorted)
        _log("Total contacts: %d" % total_contacts)

        # Phase 7: Compress
        pigz_threads = min(max_jobs, 16)
        _phase7_compress(final_sorted, header_text, output, pigz_threads)

    finally:
        # Cleanup work directory
        if os.path.isdir(work_dir):
            shutil.rmtree(work_dir)
            _log("Work directory cleaned up")

    # -- Final stats -------------------------------------------------------
    if os.path.exists(output):
        size_mb = os.path.getsize(output) / (1024 * 1024)
        _log("Output: %s (%.1f MB, %d contacts)" % (output, size_mb, total_contacts))
    else:
        sys.stderr.write("[E::merge] Output file not created: %s\n" % output)
        return 1

    _log("Merge completed in %s" % _elapsed(start_time))
    return 0
