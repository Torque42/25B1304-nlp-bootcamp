"""Validation-selected NLI decisions and a human-only error categorization gate."""

import argparse, csv, html, json, time
import numpy as np
from common import *
from neural import Encoder, sync

LABELS = ["CONTRADICT", "SUPPORT", "NEUTRAL"]
THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def classify(rows, k, threshold):
    byclaim = {}
    decisions = []
    for row in rows:
        if row["retrieval_rank"] > k:
            continue
        label = int(np.argmax(row["probabilities"]))
        probability = row["probabilities"][label]
        reason = (
            "neutral"
            if label == 2
            else "below_threshold"
            if probability < threshold
            else "accepted_before_cap"
        )
        item = {
            **row,
            "chosen_label": LABELS[label],
            "chosen_probability": probability,
            "threshold": threshold,
            "decision": reason,
        }
        if reason == "accepted_before_cap":
            byclaim.setdefault(row["claim_id"], []).append(item)
        decisions.append(item)
    selected = set()
    for cid, items in byclaim.items():
        items.sort(
            key=lambda x: (-x["chosen_probability"], x["retrieval_rank"], x["doc_id"])
        )
        for item in items[:3]:
            selected.add((cid, item["doc_id"]))
    triples = []
    for item in decisions:
        key = (item["claim_id"], item["doc_id"])
        if item["decision"] == "accepted_before_cap":
            item["decision"] = "accepted" if key in selected else "three_row_cap"
        if item["decision"] == "accepted":
            triples.append((item["claim_id"], item["doc_id"], item["chosen_label"]))
    triples = [
        (cid, item["doc_id"], item["chosen_label"])
        for cid in sorted(byclaim)
        for item in byclaim[cid][:3]
    ]
    return triples, decisions


def score_pairs(model, public, rankings, bydoc, k, split, system):
    identity = {
        "model": {a: b for a, b in model.metadata.items() if a != "load_time_sec"},
        "inputs_hash": digest([public, bydoc]),
        "retrieval_hash": digest(rankings),
        "system": system,
        "truncation": "only_first",
        "max_length": 512,
    }
    path = ART / "cache" / f"nli-{split}-{digest(identity)}.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    inputs = []
    missing = []
    for c in public:
        for rank, d in enumerate(rankings[str(c["id"])][:k], 1):
            key = f"{c['id']}:{d}"
            if key not in cache:
                inputs.append((document_text(bydoc[d]), c["claim"]))
                missing.append((key, c["id"], d, rank))
    if inputs:
        metadata = model.token_metadata(inputs)
        t = time.perf_counter()
        logits = model.encode(inputs, 8, True)
        sync(model.device)
        elapsed = time.perf_counter() - t
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        for (key, cid, d, rank), pair, logit, probability, tok in zip(
            missing, inputs, logits, probabilities, metadata
        ):
            encoded = model.tokenizer(
                pair[0],
                pair[1],
                truncation="only_first",
                max_length=512,
                return_offsets_mapping=True,
            )
            seq = encoded.sequence_ids()
            offsets = encoded["offset_mapping"]
            end = max((b for (a, b), s in zip(offsets, seq) if s == 0), default=0)
            cache[key] = {
                "claim_id": cid,
                "doc_id": d,
                "retrieval_rank": rank,
                "logits": logit.tolist(),
                "probabilities": probability.tolist(),
                "probability_order": LABELS,
                "tokens": {
                    **tok,
                    "input_ids": encoded["input_ids"],
                    "premise_visible_char_end": end,
                    "premise_sha256": digest(pair[0]),
                    "hypothesis_sha256": digest(pair[1]),
                    "truncation": "only_first",
                    "tokenizer_revision": model.revision,
                },
                "premise": pair[0],
                "hypothesis": pair[1],
            }
        save_json(path, cache)
        print(f"{split}: {len(inputs)} new NLI pairs in {elapsed:.1f}s", flush=True)
    return [cache[f"{c['id']}:{d}"] for c in public for d in rankings[str(c["id"])][:k]]


def write_verification(split, triples):
    path = PRED / (
        "t5_val_verification.csv" if split == "val" else "t5_verification.csv"
    )
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["claim_id", "doc_id", "label"])
        writer.writerows(triples)


def sentence_spans(doc):
    pos = len(doc["title"]) + 1
    out = []
    for sentence in doc["abstract"]:
        out.append((pos, pos + len(sentence)))
        pos += len(sentence) + 1
    return out


def make_review(claims, bydoc, rankings, decisions, triples, info):
    pred = {
        c["id"]: {(d, l) for cid, d, l in triples if cid == c["id"]} for c in claims
    }
    trace = {(r["claim_id"], r["doc_id"]): r for r in decisions}
    system = info["retriever"]
    first = info["first_stage"]
    first_rankings = read_run(first, "val")
    rows = []
    for c in claims:
        gold = gold_pairs(c)
        predicted = pred[c["id"]]
        if gold == predicted:
            continue
        rank = rankings[str(c["id"])]
        fst = first_rankings[str(c["id"])]
        docs = []
        flags = []
        for e in c["evidence"]:
            d = e["doc_id"]
            doc = bydoc[d]
            nli = trace.get((c["id"], d))
            spans = sentence_spans(doc)
            sent = rank.index(d) < info["candidate_k"]
            coverage = [
                bool(
                    nli
                    and (
                        s in nli["tokens"]["visible_document_sentence_indices"]
                        if "visible_document_sentence_indices" in nli["tokens"]
                        else spans[s][1] <= nli["tokens"]["premise_visible_char_end"]
                    )
                )
                for s in e["sentences"]
            ]
            docs.append(
                {
                    "doc_id": d,
                    "gold_label": e["label"],
                    "title": doc["title"],
                    "gold_sentence_indices": e["sentences"],
                    "gold_sentence_texts": [doc["abstract"][s] for s in e["sentences"]],
                    "gold_sentence_fully_visible": coverage,
                    "first_stage_rank": fst.index(d) + 1,
                    "final_rank": rank.index(d) + 1,
                    "in_first_stage_top100": d in fst[:100],
                    "in_final_top100": d in rank[:100],
                    "sent_to_nli": sent,
                    "nli": nli,
                }
            )
            if not sent:
                flags.append(
                    "gold_absent_top100"
                    if d not in rank[:100]
                    else "gold_below_nli_cutoff"
                )
            elif not all(coverage):
                flags.append("gold_sentence_truncated")
            if nli and (d, e["label"]) not in predicted:
                flags.append("gold_examined_but_wrong_or_rejected")
        if not gold and predicted:
            flags.append("gold_nei_false_positive")
        if predicted - gold:
            flags.append("extra_pairs")
        claim_trace = [r for r in decisions if r["claim_id"] == c["id"]]
        if any(r["decision"] == "three_row_cap" for r in claim_trace):
            flags.append("three_row_cap")
        rows.append(
            {
                "claim_id": c["id"],
                "claim": c["claim"],
                "gold_pairs": [list(x) for x in sorted(gold)],
                "predicted_pairs": [list(x) for x in sorted(predicted)],
                "correct_pairs": [list(x) for x in sorted(gold & predicted)],
                "missing_pairs": [list(x) for x in sorted(gold - predicted)],
                "extra_pairs": [list(x) for x in sorted(predicted - gold)],
                "gold_documents": docs,
                "false_positive_documents": [
                    {
                        "doc_id": d,
                        "predicted_label": l,
                        "title": bydoc[d]["title"],
                        "abstract": bydoc[d]["abstract"],
                        "nli": trace.get((c["id"], d)),
                    }
                    for d, l in sorted(predicted - gold)
                ],
                "nli_trace": claim_trace,
                "configuration": info,
                "suggested_flags": sorted(set(flags)),
                "suggested_category": "retrieval_miss"
                if any(
                    f in flags for f in ["gold_absent_top100", "gold_below_nli_cutoff"]
                )
                else "",
                "final_category": "",
                "reviewer_note": "",
            }
        )
    save_jsonl(ART / "validation_error_review.jsonl", rows)
    with (ART / "validation_error_review.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "claim_id",
                "claim",
                "gold_pairs",
                "predicted_pairs",
                "suggested_flags",
                "suggested_category",
                "final_category",
                "reviewer_note",
            ],
        )
        w.writeheader()
        for row in rows:
            w.writerow(
                {
                    k: json.dumps(row[k], ensure_ascii=False)
                    if isinstance(row[k], list)
                    else row[k]
                    for k in w.fieldnames
                }
            )
    render_review(rows)
    return rows


def render_review(rows):
    # Entire review is portable; no server, model, external JS, or network calls.
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    text = """<!doctype html><meta charset="utf-8"><title>SciFact validation error review</title>
<style>body{font:16px system-ui;max-width:1150px;margin:30px auto;padding:0 20px;background:#f7f8fa;color:#18202c}header{position:sticky;top:0;background:#f7f8fa;padding:16px 0;z-index:1}button,select,input,textarea{font:inherit;padding:8px}button{cursor:pointer}article{padding:22px;background:white;border:1px solid #d5dce5;border-radius:8px;margin:20px 0}p{line-height:1.5}.flag{color:#8c4d00}pre{white-space:pre-wrap;font-size:13px}textarea{width:95%;height:60px}details{margin:12px 0}summary{cursor:pointer;font-weight:600}.bad{color:#b82430}.good{color:#18653b}label{display:block;margin:12px 0}</style>
<header><h1>Validation error review</h1><p id="progress"></p><button onclick="download()">Download reviewed CSV</button> <button onclick="save()">Save in this browser</button> <label>Filter <select id="filter" onchange="render()"><option value="all">All errors</option><option value="pending">Pending review</option><option value="retrieval_miss">Suggested retrieval misses</option></select></label><p>Confirm one primary category per incorrect claim. Suggestions are diagnostic flags only. Semantic categories require your judgement. CSV import validates every and only erroneous claim.</p></header><main id="cases"></main>
<script>const cases=__DATA__;const cats=['retrieval_miss','negation','numerical','entity_mismatch','needs_multiple_docs','other'];const identity='scifact-review-__REVIEW_FINGERPRINT__';const saved=JSON.parse(localStorage.getItem(identity)||'{}');for(const c of cases){if(saved[c.claim_id])Object.assign(c,saved[c.claim_id])}const esc=x=>String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');function update(id,key,value){cases.find(x=>x.claim_id===id)[key]=value;save();progress()}function save(){localStorage.setItem(identity,JSON.stringify(Object.fromEntries(cases.map(c=>[c.claim_id,{final_category:c.final_category,reviewer_note:c.reviewer_note}]))))}function progress(){document.getElementById('progress').textContent=cases.filter(x=>x.final_category).length+' / '+cases.length+' categorized'}function render(){const filter=document.getElementById('filter').value;document.getElementById('cases').innerHTML=cases.filter(c=>filter==='all'||filter==='pending'&&!c.final_category||filter==='retrieval_miss'&&c.suggested_category==='retrieval_miss').map(c=>`<article><h2>Claim ${c.claim_id}</h2><p>${esc(c.claim)}</p><p><b>Gold:</b> ${esc(JSON.stringify(c.gold_pairs))}<br><b>Predicted:</b> ${esc(JSON.stringify(c.predicted_pairs))}</p><p class="flag">Flags: ${esc(c.suggested_flags.join(', '))}</p>${c.gold_documents.map(d=>`<details open><summary>Gold ${d.doc_id}: ${esc(d.gold_label)}, final rank ${d.final_rank}, ${d.sent_to_nli?'examined by NLI':'outside NLI candidates'}</summary><p>${esc(d.title)}</p>${d.gold_sentence_texts.map((s,i)=>`<p><b>Sentence ${d.gold_sentence_indices[i]} (${d.gold_sentence_fully_visible[i]?'fully visible':'not fully visible to NLI'}):</b> ${esc(s)}</p>`).join('')}${d.nli?`<pre>${esc(JSON.stringify({probabilities:d.nli.probabilities,order:d.nli.probability_order,decision:d.nli.decision,threshold:d.nli.threshold},null,2))}</pre>`:''}</details>`).join('')}${c.false_positive_documents.map(d=>`<details><summary>Extra ${d.doc_id}: ${esc(d.predicted_label)} — ${esc(d.title)}</summary><p>${esc(d.abstract.join(' '))}</p></details>`).join('')}<details><summary>Complete NLI and input trace</summary><pre>${esc(JSON.stringify(c.nli_trace,null,2))}</pre></details><label>Final category <select onchange="update(${c.claim_id},'final_category',this.value)"><option value="">Choose after review</option>${cats.map(x=>`<option ${x===c.final_category?'selected':''}>${x}</option>`).join('')}</select></label><label>Reviewer note<textarea onchange="update(${c.claim_id},'reviewer_note',this.value)">${esc(c.reviewer_note)}</textarea></label></article>`).join('');progress()}function download(){save();const quote=x=>'"'+String(x).replaceAll('"','""')+'"';const lines=['claim_id,final_category,reviewer_note',...cases.map(c=>[c.claim_id,c.final_category,c.reviewer_note].map(quote).join(','))];const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\\n')+'\\n'],{type:'text/csv'}));a.download='reviewed_categories.csv';a.click();URL.revokeObjectURL(a.href)}render();</script>"""
    (ART / "validation_error_review.html").write_text(
        text.replace("__DATA__", data).replace("__REVIEW_FINGERPRINT__", digest(rows))
    )


def main(device, k):
    seed_all()
    docs, claims = load_data()
    bydoc = {d["doc_id"]: d for d in docs}
    system = choose_best(
        ["bm25", "general", "scientific", "hybrid", "rerank", "random", "hard"]
    )
    first = (
        json.loads((ART / "rerank_metadata.json").read_text())["config"]["first_stage"]
        if system == "rerank"
        else system
    )
    save_json(
        ART / "verification_retriever_selection.json",
        {
            "winner": system,
            "selected_on": "validation nDCG@10",
            "frozen_before_eval": True,
        },
    )
    model = Encoder("nli", device)
    val_rankings = read_run(system, "val")
    val_rows = score_pairs(
        model, inference_claims(claims["val"]), val_rankings, bydoc, k, "val", system
    )
    sweep = []
    for candidate_k in [5, 10] if k == 10 else [5]:
        for threshold in THRESHOLDS:
            triples, _ = classify(val_rows, candidate_k, threshold)
            metrics = triple_metrics(triples, claims["val"])
            sweep.append(
                {"candidate_k": candidate_k, "threshold": threshold, "metrics": metrics}
            )
    best = max(
        sweep,
        key=lambda x: (
            x["metrics"]["f1"],
            x["metrics"]["precision"],
            x["threshold"],
            -x["candidate_k"],
        ),
    )
    chosen_k = best["candidate_k"]
    threshold = best["threshold"]
    save_json(ART / "nli_threshold_sweep.json", sweep)
    save_json(ART / f"nli_k{k}_threshold_sweep.json", sweep)
    info = {
        "retriever": system,
        "first_stage": first,
        "candidate_k": chosen_k,
        "threshold": threshold,
        "max_rows": 3,
        "model": model.metadata,
        "metrics": best["metrics"],
        "zero_shot": True,
        "frozen_before_evaluation": True,
    }
    save_json(ART / "nli_selection.json", info)
    save_json(ART / f"nli_k{k}_selection.json", info)
    for split in ["val", "eval"]:
        rankings = read_run(system, split)
        public = inference_claims(claims[split])
        rows = (
            val_rows
            if split == "val"
            else score_pairs(model, public, rankings, bydoc, chosen_k, split, system)
        )
        triples, decisions = classify(rows, chosen_k, threshold)
        write_verification(split, triples)
        save_jsonl(ART / f"nli_{split}_trace.jsonl", decisions)
        first_rankings = read_run(first, split)
        first_raw_path = ART / f"{first}_{split}_raw_scores.json"
        first_raw = (
            json.loads(first_raw_path.read_text()) if first_raw_path.exists() else {}
        )
        final_raw_path = ART / f"{system}_{split}_raw_scores.json"
        final_raw = (
            json.loads(final_raw_path.read_text()) if final_raw_path.exists() else {}
        )
        save_jsonl(
            ART / f"verification_{split}_retrieval_trace.jsonl",
            [
                {
                    "claim_id": c["id"],
                    "doc_id": d,
                    "first_stage_rank": first_rankings[str(c["id"])].index(d) + 1,
                    "final_rank": r,
                    "first_stage_raw_score": first_raw.get(str(c["id"]), [])[
                        first_rankings[str(c["id"])].index(d)
                    ]
                    if d in first_rankings[str(c["id"])][:100]
                    and str(c["id"]) in first_raw
                    else None,
                    "final_raw_score": final_raw[str(c["id"])][r - 1]
                    if str(c["id"]) in final_raw
                    else None,
                    "final_rank_score": 101 - r,
                    "sent_to_nli": r <= chosen_k,
                }
                for c in public
                for r, d in enumerate(rankings[str(c["id"])][:100], 1)
            ],
        )
        if split == "val":
            review = make_review(
                claims["val"], bydoc, rankings, decisions, triples, info
            )
            assert len(review) == best["metrics"]["error_count"]
    record_stage("nli", {"config": info, "metrics": best["metrics"]})
    print(
        f"Human review required for {len(review)} validation errors. F1={best['metrics']['f1']:.6f}",
        flush=True,
    )
    model.close()


def export_review(path):
    expected = {r["claim_id"] for r in jsonl(ART / "validation_error_review.jsonl")}
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    categories = {}
    notes = {}
    for row in rows:
        cid = int(row["claim_id"])
        cat = row.get("final_category", row.get("category", "")).strip()
        if cid in categories:
            raise ValueError(f"Duplicate claim {cid}")
        if cat not in CATEGORIES:
            raise ValueError(f"Claim {cid}: missing or invalid final category {cat!r}")
        categories[cid] = cat
        notes[cid] = row.get("reviewer_note", "")
    if set(categories) != expected:
        raise ValueError(
            f"Coverage mismatch: missing={sorted(expected - set(categories))}, extra={sorted(set(categories) - expected)}"
        )
    with (PRED / "t5_val_errors.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["claim_id", "category"])
        w.writerows(sorted(categories.items()))
    save_json(
        ART / "human_review.json",
        {
            "reviewed_input_sha256": file_hash(path),
            "categories": categories,
            "reviewer_notes": notes,
            "error_count": len(expected),
            "retrieval_miss_count": sum(
                c == "retrieval_miss" for c in categories.values()
            ),
            "retrieval_miss_fraction": sum(
                c == "retrieval_miss" for c in categories.values()
            )
            / len(expected)
            if expected
            else None,
        },
    )
    print("Reviewed categories exported.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--k", type=int, default=5, choices=[5, 10])
    p.add_argument("--import-review")
    a = p.parse_args()
    if a.import_review:
        export_review(a.import_review)
    else:
        main(a.device, a.k)
