#!/usr/bin/env python3
"""Independent, CPU-only final SciFact submission gate and optional ZIP writer.

No project modules, model weights, or GPU libraries are imported. The staging
directory is never modified. Category semantics and claimed runtime measurements
require semantic review; this tool checks coverage and internal consistency.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


EVAL_RUNS = (
    "t1_bm25.trec", "t2_general.trec", "t2_scientific.trec",
    "t3_hybrid.trec", "t3_rerank.trec", "t4_random.trec", "t4_hard.trec",
)
VAL_RUNS = ("val_bm25.trec", "val_general.trec")
CSV_FILES = ("t5_verification.csv", "t5_val_verification.csv", "t5_val_errors.csv")
REQUIRED_PREDICTIONS = set(EVAL_RUNS + VAL_RUNS + CSV_FILES)
ERROR_CATEGORIES = {
    "retrieval_miss", "negation", "numerical", "entity_mismatch",
    "needs_multiple_docs", "other",
}
OPTIONAL_METRICS = {
    "scientific": "task2.val_ndcg10_scientific",
    "hybrid": "task3.val_ndcg10_hybrid",
    "rerank": "task3.val_ndcg10_rerank",
    "random": "task4.val_ndcg10_random_negatives",
    "hard": "task4.val_ndcg10_hard_negatives",
}
PLACEHOLDER = re.compile(
    r"\b(?:TODO|TBD|FIXME|PLACEHOLDER|PENDING|FILL(?:\s+THIS)?\s+IN|"
    r"TO\s+BE\s+(?:DONE|COMPLETED|WRITTEN))\b|\[\s*(?:Q[1-5]|INSERT|FILL)",
    re.I,
)


class Gate:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.notes: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def load_json(self, path: Path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.errors.append(f"{path}: cannot read JSON: {exc}")
            return None


def load_jsonl(path: Path, gate: Gate) -> list[dict]:
    rows = []
    try:
        with path.open(encoding="utf-8") as stream:
            for lineno, line in enumerate(stream, 1):
                if line.strip():
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        raise ValueError(f"line {lineno} is not an object")
                    rows.append(obj)
    except (OSError, ValueError) as exc:
        gate.errors.append(f"{path}: cannot read JSONL: {exc}")
    return rows


def integer(value: str, where: str) -> int:
    if not re.fullmatch(r"[0-9]+", value):
        raise ValueError(f"{where}: expected nonnegative integer, got {value!r}")
    return int(value)


def read_csv(path: Path, header: tuple[str, ...], gate: Gate) -> list[list[str]] | None:
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.reader(stream))
        if not rows or tuple(rows[0]) != header:
            raise ValueError(f"header must be {','.join(header)}")
        for lineno, row in enumerate(rows[1:], 2):
            if len(row) != len(header) or any(not cell or cell != cell.strip() for cell in row):
                raise ValueError(f"row {lineno}: expected {len(header)} nonblank, trimmed fields")
        return rows[1:]
    except (OSError, ValueError) as exc:
        gate.errors.append(f"{path}: {exc}")
        return None


def read_trec(path: Path, claim_ids: set[int], doc_ids: set[int], gate: Gate) -> dict[int, list[int]] | None:
    groups: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    previous_key = (-1, 0)
    try:
        with path.open(encoding="utf-8") as stream:
            for lineno, line in enumerate(stream, 1):
                fields = line.split()
                if len(fields) != 6 or fields[1] != "Q0" or not fields[5]:
                    raise ValueError(f"line {lineno}: require six TREC fields with literal Q0")
                cid = integer(fields[0], f"line {lineno} claim ID")
                did = integer(fields[2], f"line {lineno} document ID")
                rank = integer(fields[3], f"line {lineno} rank")
                score = float(fields[4])
                if cid not in claim_ids or did not in doc_ids:
                    raise ValueError(f"line {lineno}: unknown claim/document ID {cid}/{did}")
                if not math.isfinite(score):
                    raise ValueError(f"line {lineno}: score is not finite")
                if (cid, rank) <= previous_key:
                    raise ValueError(f"line {lineno}: rows must sort by claim ID then rank")
                previous_key = (cid, rank)
                groups[cid].append((rank, did, score))
        if set(groups) != claim_ids:
            missing, extra = claim_ids - set(groups), set(groups) - claim_ids
            raise ValueError(f"claim coverage differs: missing={len(missing)}, extra={len(extra)}")
        for cid, rows in groups.items():
            if [r for r, _, _ in rows] != list(range(1, 101)):
                raise ValueError(f"claim {cid}: ranks must be exactly 1..100")
            if len({did for _, did, _ in rows}) != 100:
                raise ValueError(f"claim {cid}: document IDs are duplicated")
            if any(a[2] < b[2] for a, b in zip(rows, rows[1:])):
                raise ValueError(f"claim {cid}: scores increase down the ranking")
        return {cid: [did for _, did, _ in rows] for cid, rows in groups.items()}
    except (OSError, ValueError) as exc:
        gate.errors.append(f"{path}: {exc}")
        return None


def read_verification(path: Path, claim_ids: set[int], doc_ids: set[int], gate: Gate) -> set[tuple[int, int, str]] | None:
    rows = read_csv(path, ("claim_id", "doc_id", "label"), gate)
    if rows is None:
        return None
    triples = set()
    pairs = set()
    counts: Counter[int] = Counter()
    try:
        for lineno, (raw_cid, raw_did, label) in enumerate(rows, 2):
            cid = integer(raw_cid, f"row {lineno} claim ID")
            did = integer(raw_did, f"row {lineno} document ID")
            if cid not in claim_ids or did not in doc_ids:
                raise ValueError(f"row {lineno}: unknown claim/document ID {cid}/{did}")
            if label not in {"SUPPORT", "CONTRADICT"}:
                raise ValueError(f"row {lineno}: invalid label {label!r}")
            if (cid, did) in pairs:
                raise ValueError(f"row {lineno}: duplicate claim/document pair {cid}/{did}")
            pairs.add((cid, did))
            triples.add((cid, did, label))
            counts[cid] += 1
            if counts[cid] > 3:
                raise ValueError(f"claim {cid}: more than three verification rows")
        return triples
    except ValueError as exc:
        gate.errors.append(f"{path}: {exc}")
        return None


def read_error_categories(path: Path, claim_ids: set[int], gate: Gate) -> dict[int, str] | None:
    rows = read_csv(path, ("claim_id", "category"), gate)
    if rows is None:
        return None
    result = {}
    try:
        for lineno, (raw_cid, category) in enumerate(rows, 2):
            cid = integer(raw_cid, f"row {lineno} claim ID")
            if cid not in claim_ids or cid in result or category not in ERROR_CATEGORIES:
                raise ValueError(f"row {lineno}: unknown/duplicate claim or invalid category")
            result[cid] = category
        return result
    except ValueError as exc:
        gate.errors.append(f"{path}: {exc}")
        return None


def ndcg10(run: dict[int, list[int]], claims: list[dict]) -> float:
    per = []
    for claim in claims:
        gold = {e["doc_id"] for e in claim["evidence"]}
        if not gold:
            continue
        dcg = sum((did in gold) / math.log2(rank + 1) for rank, did in enumerate(run[claim["id"]][:10], 1))
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(10, len(gold)) + 1))
        per.append(dcg / ideal)
    if not per:
        raise ValueError("no evidence-bearing validation claims")
    return sum(per) / len(per)


def nested(obj: dict, dotted: str):
    value = obj
    for key in dotted.split("."):
        value = value[key]
    return value


def compare_number(gate: Gate, name: str, claimed, computed: float, tolerance: float = 1e-8) -> None:
    if isinstance(claimed, bool) or not isinstance(claimed, (int, float)) or not math.isfinite(claimed):
        gate.errors.append(f"results.json {name}: expected finite number")
    elif abs(claimed - computed) > tolerance:
        gate.errors.append(f"results.json {name}: {claimed} differs from computed {computed:.12g}")


def verify_result_tree(gate: Gate, result: dict, template: dict, prefix: str = "") -> None:
    for key, shape in template.items():
        name = prefix + key
        if key not in result:
            gate.errors.append(f"results.json missing {name}")
            continue
        value = result[key]
        if isinstance(shape, dict):
            if not isinstance(value, dict):
                gate.errors.append(f"results.json {name} must be an object")
            else:
                verify_result_tree(gate, value, shape, name + ".")
        elif value is None or value == "":
            gate.errors.append(f"results.json {name} is blank")
        elif isinstance(value, float) and not math.isfinite(value):
            gate.errors.append(f"results.json {name} is not finite")
        elif isinstance(value, str) and PLACEHOLDER.search(value):
            gate.errors.append(f"results.json {name} contains placeholder language")


def validate(staging: Path, data: Path, optional_runs: dict[str, Path], gate: Gate, pending_review: bool) -> dict:
    info: dict = {}
    for name in ("corpus", "val_claims", "eval_claims"):
        if not (data / f"{name}.jsonl").is_file():
            gate.errors.append(f"missing assignment input {data / (name + '.jsonl')}")
    if gate.errors:
        return info
    docs = load_jsonl(data / "corpus.jsonl", gate)
    val_claims = load_jsonl(data / "val_claims.jsonl", gate)
    eval_claims = load_jsonl(data / "eval_claims.jsonl", gate)
    if gate.errors:
        return info
    try:
        doc_ids = {int(d["doc_id"]) for d in docs}
        val_ids = {int(c["id"]) for c in val_claims}
        eval_ids = {int(c["id"]) for c in eval_claims}
        assert len(doc_ids) == len(docs) and len(val_ids) == len(val_claims) and len(eval_ids) == len(eval_claims)
        assert len(val_ids) == 162 and len(eval_ids) == 300
    except (KeyError, TypeError, ValueError, AssertionError) as exc:
        gate.errors.append(f"assignment data IDs invalid or unexpected counts: {exc}")
        return info

    pred = staging / "predictions"
    if not pred.is_dir():
        gate.errors.append(f"missing {pred}")
        return info
    present = {p.name for p in pred.iterdir() if p.is_file()}
    for name in sorted(REQUIRED_PREDICTIONS - present):
        message = f"missing predictions/{name}"
        (gate.notes if pending_review and name == "t5_val_errors.csv" else gate.errors).append(message)
    for name in sorted(present - REQUIRED_PREDICTIONS):
        gate.errors.append(f"unexpected predictions/{name}")
    if any(p.is_dir() for p in pred.iterdir()):
        gate.errors.append("predictions/ must contain files only")
    runs = {}
    for name in EVAL_RUNS + VAL_RUNS:
        if name in present:
            run = read_trec(pred / name, val_ids if name.startswith("val_") else eval_ids, doc_ids, gate)
            if run is not None:
                runs[name] = run
    info["trec_validated"] = sorted(runs)
    eval_pred = read_verification(pred / "t5_verification.csv", eval_ids, doc_ids, gate) if "t5_verification.csv" in present else None
    val_pred = read_verification(pred / "t5_val_verification.csv", val_ids, doc_ids, gate) if "t5_val_verification.csv" in present else None
    categories = read_error_categories(pred / "t5_val_errors.csv", val_ids, gate) if "t5_val_errors.csv" in present else None
    if eval_pred is not None:
        info["eval_verification_rows"] = len(eval_pred)
    result = gate.load_json(staging / "results.json")
    template_path = data.parent / "results_template.json"
    if not template_path.is_file():
        template_path = staging / "code/results_template.json"
    template = gate.load_json(template_path) if template_path.is_file() else None
    if template is None:
        gate.errors.append("results_template.json required beside data/ for exact result-field coverage")
    if not isinstance(result, dict):
        result = None
    elif isinstance(template, dict):
        verify_result_tree(gate, result, template)

    if val_pred is not None:
        gold = {(c["id"], e["doc_id"], e["label"]) for c in val_claims for e in c["evidence"]}
        tp, fp, fn = len(gold & val_pred), len(val_pred - gold), len(gold - val_pred)
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        by_gold = defaultdict(set)
        by_pred = defaultdict(set)
        for cid, did, lab in gold:
            by_gold[cid].add((did, lab))
        for cid, did, lab in val_pred:
            by_pred[cid].add((did, lab))
        errors = {cid for cid in val_ids if by_gold[cid] != by_pred[cid]}
        info.update(val_f1=f1, validation_errors=len(errors), val_tp=tp, val_fp=fp, val_fn=fn)
        if categories is not None and set(categories) != errors:
            gate.errors.append(f"t5_val_errors.csv exact-set coverage mismatch: missing={sorted(errors - set(categories))[:8]}, extra={sorted(set(categories) - errors)[:8]}")
        if result is not None:
            try:
                compare_number(gate, "task5.val_f1", nested(result, "task5.val_f1"), f1)
                diag = nested(result, "diagnostics.verification")
                for key, actual in (("true_positives", tp), ("false_positives", fp), ("false_negatives", fn), ("error_count", len(errors)), ("denominator_claims", len(val_ids)), ("f1", f1), ("precision", tp / (tp + fp) if tp + fp else 0), ("recall", tp / (tp + fn) if tp + fn else 0), ("exact_set_accuracy", 1 - len(errors) / len(val_ids))):
                    if key in diag:
                        compare_number(gate, "diagnostics.verification." + key, diag[key], actual)
                if "error_ids" in diag and (not isinstance(diag["error_ids"], list) or set(diag["error_ids"]) != errors or len(diag["error_ids"]) != len(errors)):
                    gate.errors.append("results.json diagnostics.verification.error_ids differs from exact-set errors")
            except (KeyError, TypeError) as exc:
                gate.errors.append(f"results.json verification diagnostics missing/malformed: {exc}")

    for name, field in (("val_bm25.trec", "task1.val_ndcg10"), ("val_general.trec", "task2.val_ndcg10_general")):
        if name in runs and result is not None:
            try:
                compare_number(gate, field, nested(result, field), ndcg10(runs[name], val_claims))
            except (KeyError, TypeError, ValueError) as exc:
                gate.errors.append(f"cannot compare {field}: {exc}")
    if result is not None and "val_bm25.trec" in runs and "val_general.trec" in runs:
        for key, winner, loser in (("bm25_wins", "val_bm25.trec", "val_general.trec"), ("general_wins", "val_general.trec", "val_bm25.trec")):
            try:
                ids = result["report_q1"][key]
                if not isinstance(ids, list) or len(ids) != 2 or len(set(ids)) != 2 or not set(ids) <= val_ids:
                    raise ValueError("must contain two distinct validation claim IDs")
                by_id = {c["id"]: c for c in val_claims}
                if any(ndcg10({cid: runs[winner][cid]}, [by_id[cid]]) <= ndcg10({cid: runs[loser][cid]}, [by_id[cid]]) for cid in ids):
                    raise ValueError("listed winner does not win on every claim")
            except (KeyError, TypeError, ValueError) as exc:
                gate.errors.append(f"results.json report_q1.{key}: {exc}")

    if "t3_hybrid.trec" in runs and "t3_rerank.trec" in runs and result is not None:
        try:
            k = nested(result, "task3.rerank_top_k")
            if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 100:
                raise ValueError("rerank_top_k must be an integer 1..100")
            for cid, hybrid in runs["t3_hybrid.trec"].items():
                rerank = runs["t3_rerank.trec"][cid]
                if set(hybrid[:k]) != set(rerank[:k]) or hybrid[k:] != rerank[k:]:
                    raise ValueError(f"claim {cid}: rerank prefix/tail differs from hybrid k={k}")
            info["rerank_tail_invariance"] = f"PASS k={k}"
        except (KeyError, TypeError, ValueError) as exc:
            gate.errors.append(f"rerank prefix/tail: {exc}")

    for alias, path in optional_runs.items():
        run = read_trec(path, val_ids, doc_ids, gate)
        if run is not None and result is not None:
            field = OPTIONAL_METRICS[alias]
            try:
                compare_number(gate, field, nested(result, field), ndcg10(run, val_claims))
            except (KeyError, TypeError, ValueError) as exc:
                gate.errors.append(f"cannot compare optional {field}: {exc}")
    if len(optional_runs) < len(OPTIONAL_METRICS):
        gate.notes.append("Scientific/hybrid/rerank/random/hard validation nDCG cannot be recomputed without corresponding --val-run files.")

    if result is not None:
        try:
            completion = result["completion"]
            review_complete = completion.get("human_review_complete") is True
            if completion.get("delegated_review_complete") is True:
                provenance = gate.load_json(staging / "code/run_records/final-review-provenance.json")
                decisions = gate.load_json(staging / "code/run_records/final-category-decisions.json")
                delegated_valid = (
                    isinstance(provenance, dict)
                    and provenance.get("status") == "final"
                    and bool(provenance.get("authorization"))
                    and provenance.get("human_individual_case_review") is False
                    and provenance.get("independent_review_reconciled") is True
                    and bool(completion.get("review_authorization"))
                    and isinstance(decisions, dict)
                    and categories is not None
                    and {str(k): v for k, v in categories.items()}
                    == {k: v.get("category") for k, v in decisions.items()}
                )
                if not delegated_valid:
                    gate.errors.append("Delegated review provenance or final category decisions missing/inconsistent")
                review_complete = review_complete or delegated_valid
            if not review_complete:
                message = "results.json: completed human or explicitly delegated semantic review required"
                (gate.notes if pending_review else gate.errors).append(message)
            if completion.get("pending_gates"):
                message = "results.json completion.pending_gates is nonempty"
                (gate.notes if pending_review else gate.errors).append(message)
            roll = result["roll_no"]
            if not isinstance(roll, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", roll):
                gate.errors.append("results.json roll_no missing or invalid for ZIP name")
            if not isinstance(result["name"], str) or not result["name"].strip():
                gate.errors.append("results.json name missing")
            for section in ("task1", "task2", "task3", "task4", "task5"):
                if not isinstance(result[section], dict):
                    raise ValueError(f"{section} must be an object")
            metric_fields = (
                "task1.val_ndcg10", "task2.val_ndcg10_general", "task2.val_ndcg10_scientific",
                "task3.val_ndcg10_hybrid", "task3.val_ndcg10_rerank",
                "task4.val_ndcg10_random_negatives", "task4.val_ndcg10_hard_negatives", "task5.val_f1",
            )
            for field in metric_fields:
                value = nested(result, field)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    gate.errors.append(f"results.json {field} must be finite in [0,1]")
            if "diagnostics" in result and isinstance(result["diagnostics"], dict):
                table = result["diagnostics"].get("retrieval_table", {})
                field_by_name = {
                    "bm25": "task1.val_ndcg10", "general": "task2.val_ndcg10_general",
                    "scientific": "task2.val_ndcg10_scientific", "hybrid": "task3.val_ndcg10_hybrid",
                    "rerank": "task3.val_ndcg10_rerank", "random": "task4.val_ndcg10_random_negatives",
                    "hard": "task4.val_ndcg10_hard_negatives",
                }
                for name, field in field_by_name.items():
                    if name in table and "ndcg10" in table[name]:
                        compare_number(gate, f"diagnostics.retrieval_table.{name}.ndcg10", table[name]["ndcg10"], nested(result, field))
        except (KeyError, TypeError, ValueError) as exc:
            gate.errors.append(f"results.json missing/malformed final fields: {exc}")

    report_path = staging / "report.md"
    try:
        report = report_path.read_text(encoding="utf-8")
        headings = list(re.finditer(r"(?m)^## Q([1-5])\s*$", report))
        if [m.group(1) for m in headings] != list("12345"):
            raise ValueError("require exactly ## Q1 through ## Q5 in order")
        counts = {}
        for index, heading in enumerate(headings):
            answer = report[heading.end():headings[index + 1].start() if index + 1 < len(headings) else len(report)].strip()
            count = len(answer.split())
            counts[f"Q{index + 1}"] = count
            if not answer or count > 100:
                gate.errors.append(f"report.md Q{index + 1}: blank or over 100 words ({count} words)")
            elif PLACEHOLDER.search(answer):
                message = f"report.md Q{index + 1}: contains placeholder language ({count} words)"
                (gate.notes if pending_review and index == 3 else gate.errors).append(message)
        info["report_words"] = counts
    except (OSError, ValueError) as exc:
        gate.errors.append(f"report.md: {exc}")

    if not (staging / "code" / "README.md").is_file():
        gate.errors.append("missing code/README.md")
    return info


def staging_files(staging: Path, gate: Gate) -> dict[str, Path]:
    files = {}
    # An extracted copy may also contain local inputs and a reproduction environment.
    # Only the four assignment deliverables belong in the archive.
    allowed = {"code", "predictions", "results.json", "report.md"}
    local_only = {"data", "artifacts", ".venv", ".git", ".pytest_cache", "__pycache__"}
    for entry in staging.iterdir():
        if entry.name not in allowed and entry.name not in local_only:
            gate.errors.append(f"unexpected root entry: {entry.name}; keep local output outside submission root")
    paths = []
    for name in sorted(allowed):
        root = staging / name
        if root.is_symlink():
            gate.errors.append(f"symlink in staging: {root}")
        elif root.is_dir():
            paths.extend(root.rglob("*"))
        elif root.is_file():
            paths.append(root)
    for path in paths:
        if "__pycache__" in path.parts or ".pytest_cache" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            gate.errors.append(f"symlink in staging: {path}")
        elif path.is_file():
            relative = path.relative_to(staging).as_posix()
            if relative.split("/")[0] not in {"code", "predictions", "results.json", "report.md"}:
                gate.errors.append(f"unexpected ZIP root entry: {relative}")
            if any(part in {"__pycache__", ".venv", ".git"} for part in path.parts) or path.suffix in {".pt", ".pth", ".safetensors", ".bin"}:
                gate.errors.append(f"runtime cache/weight file in staging: {relative}")
            files[relative] = path
    for name in ("results.json", "report.md", "code/README.md"):
        if name not in files:
            gate.errors.append(f"missing ZIP member {name}")
    return files


def check_zip(path: Path, files: dict[str, Path], gate: Gate) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [m.filename for m in members if not m.is_dir()]
            if len(names) != len(set(names)):
                gate.errors.append("ZIP contains duplicate member names")
            for m in members:
                name = m.filename
                pure = PurePosixPath(name)
                if name.startswith("/") or ".." in pure.parts or "\\" in name or pure.parts[0] not in {"code", "predictions", "results.json", "report.md"}:
                    gate.errors.append(f"ZIP unsafe/unexpected member: {name}")
                if (m.external_attr >> 16) & 0o170000 == 0o120000:
                    gate.errors.append(f"ZIP symlink member: {name}")
            if set(names) != set(files):
                gate.errors.append(f"ZIP layout differs from staging: missing={sorted(set(files)-set(names))[:8]}, extra={sorted(set(names)-set(files))[:8]}")
            if archive.testzip() is not None:
                gate.errors.append("ZIP CRC/integrity check failed")
            if not gate.errors:
                for name, source in files.items():
                    if archive.read(name) != source.read_bytes():
                        gate.errors.append(f"ZIP content differs from staging: {name}")
                        break
    except (OSError, zipfile.BadZipFile, IndexError) as exc:
        gate.errors.append(f"ZIP cannot be read: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True, help="explicit submission root with code/, predictions/, report.md, results.json")
    parser.add_argument("--data", type=Path, required=True, help="assignment data/; template is read beside data or from staging/code/")
    parser.add_argument("--val-run", action="append", default=[], metavar="NAME=PATH", help="optional validation TREC: scientific, hybrid, rerank, random, hard")
    parser.add_argument("--check-zip", type=Path, help="compare an existing ZIP byte-for-byte to staging")
    parser.add_argument("--write-zip", type=Path, help="write final ZIP only after all gates pass; refuses overwrite")
    parser.add_argument("--pending-review", action="store_true", help="preview staging while human review is pending; never writes a ZIP")
    args = parser.parse_args()
    gate = Gate()
    optional = {}
    for item in args.val_run:
        name, sep, raw_path = item.partition("=")
        if not sep or name not in OPTIONAL_METRICS or name in optional or not raw_path:
            parser.error(f"invalid --val-run {item!r}; expected unique NAME=PATH, NAME in {', '.join(OPTIONAL_METRICS)}")
        optional[name] = Path(raw_path).expanduser().resolve()
    staging = args.staging.expanduser().resolve()
    data = args.data.expanduser().resolve()
    if not staging.is_dir():
        parser.error("--staging must be a directory")
    if args.check_zip and args.write_zip:
        parser.error("use either --check-zip or --write-zip")
    if args.pending_review and args.write_zip:
        parser.error("--pending-review cannot be combined with --write-zip")
    info = validate(staging, data, optional, gate, args.pending_review)
    files = staging_files(staging, gate)
    if args.check_zip:
        check_zip(args.check_zip.expanduser().resolve(), files, gate)
    if not gate.errors and args.write_zip:
        output = args.write_zip.expanduser().resolve()
        roll = json.loads((staging / "results.json").read_text(encoding="utf-8"))["roll_no"]
        expected = f"{roll}_nlp_bootcamp.zip"
        if output.name != expected:
            gate.errors.append(f"ZIP filename must be {expected}")
        elif output.exists():
            gate.errors.append(f"refusing to overwrite {output}")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            pending = output.with_name(output.name + ".pending")
            if pending.exists():
                gate.errors.append(f"refusing to overwrite {pending}")
            else:
                try:
                    with zipfile.ZipFile(pending, "w", zipfile.ZIP_DEFLATED) as archive:
                        for name, source in sorted(files.items()):
                            archive.write(source, arcname=name)
                    check_zip(pending, files, gate)
                    if not gate.errors:
                        pending.replace(output)
                        info["zip_written"] = str(output)
                    else:
                        pending.unlink(missing_ok=True)
                except (OSError, zipfile.BadZipFile) as exc:
                    gate.errors.append(f"cannot write ZIP: {exc}")
                    pending.unlink(missing_ok=True)
    status = "FAIL" if gate.errors else "PREVIEW_PENDING_REVIEW" if args.pending_review else "PASS"
    print(json.dumps({"status": status, "errors": gate.errors, "notes": gate.notes, "checks": info}, indent=2))
    return 0 if not gate.errors else 1


if __name__ == "__main__":
    sys.exit(main())
