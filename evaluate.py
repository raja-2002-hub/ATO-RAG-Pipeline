"""
evaluate.py — Production-grade evaluation of the ATO RAG system.

100 questions across 4 tiers. Measures retrieval quality, answer accuracy,
hybrid search value, and safety. Uses GPT to auto-grade answers.

Usage:
    python evaluate.py --no-llm                  # retrieval only (fast, free)
    python evaluate.py                            # full eval with GPT grading
    python evaluate.py --compare-search           # compare dense vs hybrid
"""

import argparse, json, time, sys, os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from retrieval.retriever import retrieve, dense_search, sparse_search, hybrid_search, group_by_url
from retrieval.reranker import Reranker


def load_test_set():
    p = Path(__file__).parent / "data" / "test_questions.json"
    with p.open("r", encoding="utf-8") as f:
        return json.loads(f.read())


def check_url_match(results, frags):
    if not frags: return True
    for r in results:
        for f in frags:
            if f.lower() in r.get("url","").lower(): return True
    return False


def find_rank(results, frags):
    if not frags: return 1
    for i, r in enumerate(results):
        for f in frags:
            if f.lower() in r.get("url","").lower(): return i+1
    return None


def grade_with_gpt(question, answer, ground_truth):
    import openai
    client = openai.OpenAI()
    prompt = f"""Grade this tax system answer. Score each 0-2 (0=bad, 1=partial, 2=good).

Question: {question}
Expected: {ground_truth}
Answer: {answer}

Respond ONLY JSON: {{"accuracy":0,"completeness":0,"groundedness":0,"safety":0,"relevance":0,"comment":"brief"}}"""
    try:
        r = client.chat.completions.create(model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}], temperature=0, max_tokens=150)
        t = r.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        return json.loads(t)
    except:
        return {"accuracy":-1,"completeness":-1,"groundedness":-1,"safety":-1,"relevance":-1}


def run(args):
    tests = load_test_set()
    print(f"{'='*60}\n  ATO RAG — Evaluation ({len(tests)} questions)\n{'='*60}")

    reranker = None
    try: reranker = Reranker(); print("  Reranker: loaded")
    except: print("  Reranker: unavailable")

    call_llm, fmt_ev = None, None
    if not args.no_llm:
        try:
            from api.app import call_llm as _c, format_evidence as _f
            call_llm, fmt_ev = _c, _f; print("  LLM: ready")
        except Exception as e: print(f"  LLM: unavailable ({e})")

    print()
    all_res, ret_evals, grades, latencies = [], [], [], []
    tiers = {"T1":[],"T2":[],"T3":[],"T4":[]}
    cats = {}

    for tc in tests:
        print(f"  [{tc['id']:3d}/100] {tc['question'][:52]}...", end=" ", flush=True)
        t0 = time.time()
        results = retrieve(tc["question"], reranker=reranker)
        lat = time.time()-t0; latencies.append(lat)

        url_match = check_url_match(results, tc["expected_url_contains"])
        rank = find_rank(results, tc["expected_url_contains"])
        mrr = 1.0/rank if rank else 0.0
        all_text = " ".join(r.get("text","") for r in results).lower()
        kw_found = [k for k in tc["expected_answer_contains"] if k.lower() in all_text]
        kw_recall = len(kw_found)/max(len(tc["expected_answer_contains"]),1)

        ret = {"url_match":url_match,"rank":rank,"mrr":mrr,"kw_recall":kw_recall}
        ret_evals.append(ret)
        tiers[tc["tier"]].append(ret)
        cats.setdefault(tc["category"],[]).append(ret)

        sym = "✓" if url_match else "✗"
        print(f"{sym} MRR={mrr:.2f} KW={kw_recall:.0%} {lat:.1f}s", end="")

        answer, grade = "", {}
        if call_llm and fmt_ev:
            try:
                ev = fmt_ev(results)
                answer, err = call_llm(f"Question: {tc['question']}\n\nEvidence:\n{ev}\n\nAnswer using evidence.")
                if answer and not err:
                    grade = grade_with_gpt(tc["question"], answer, tc["ground_truth"])
                    grades.append(grade)
                    print(f" A={grade.get('accuracy','?')}/2", end="")
            except: pass

        print()
        all_res.append({"id":tc["id"],"tier":tc["tier"],"category":tc["category"],
            "question":tc["question"],"retrieval":ret,"latency":round(lat,2),
            "answer":answer[:500] if answer else None,"grade":grade,
            "top3":[{"title":r.get("title",""),"url":r.get("url","")} for r in results[:3]]})

    # ── Metrics ──
    N = len(ret_evals)
    hits = sum(1 for r in ret_evals if r["url_match"])
    mrr = sum(r["mrr"] for r in ret_evals)/N
    r1 = sum(1 for r in ret_evals if r["rank"]==1)/N
    r3 = sum(1 for r in ret_evals if r["rank"] and r["rank"]<=3)/N
    r5 = sum(1 for r in ret_evals if r["rank"] and r["rank"]<=5)/N
    kw = sum(r["kw_recall"] for r in ret_evals)/N

    print(f"\n{'='*60}\n  RETRIEVAL METRICS\n{'='*60}")
    print(f"\n  Hit rate  : {hits}/{N} ({hits*100//N}%)")
    print(f"  MRR       : {mrr:.3f}")
    print(f"  Recall@1  : {r1:.1%}")
    print(f"  Recall@3  : {r3:.1%}")
    print(f"  Recall@5  : {r5:.1%}")
    print(f"  Keyword   : {kw:.1%}")
    print(f"  Latency   : avg={sum(latencies)/N:.2f}s  p95={sorted(latencies)[int(N*0.95)]:.2f}s")

    print(f"\n  Per tier:")
    for t in ["T1","T2","T3","T4"]:
        e = tiers[t]
        if not e: continue
        h = sum(1 for r in e if r["url_match"])
        m = sum(r["mrr"] for r in e)/len(e)
        print(f"    {t}: {h}/{len(e)} ({h*100//len(e)}%) MRR={m:.3f}")

    print(f"\n  Per category:")
    for c in sorted(cats):
        e = cats[c]
        h = sum(1 for r in e if r["url_match"])
        m = sum(r["mrr"] for r in e)/len(e)
        print(f"    {c:20s} {h}/{len(e)} MRR={m:.3f}")

    failed = [tc for tc,r in zip(tests,ret_evals) if not r["url_match"]]
    if failed:
        print(f"\n  Failed ({len(failed)}):")
        for tc in failed[:10]:
            print(f"    [{tc['id']}] {tc['question'][:55]}")

    if grades:
        valid = [g for g in grades if g.get("accuracy",-1)>=0]
        if valid:
            print(f"\n{'='*60}\n  ANSWER QUALITY (GPT-graded, {len(valid)} answers)\n{'='*60}")
            for m in ["accuracy","completeness","groundedness","safety","relevance"]:
                s = [g[m] for g in valid if m in g]
                avg = sum(s)/len(s) if s else 0
                perfect = sum(1 for x in s if x==2)
                print(f"  {m:15s}: {avg:.2f}/2  ({perfect}/{len(s)} perfect)")
            hall = sum(1 for g in valid if g.get("accuracy")==0)
            print(f"\n  Hallucination : {hall}/{len(valid)} ({hall*100//len(valid)}%)")

    # Compare search
    search_comp = {}
    if args.compare_search:
        print(f"\n{'='*60}\n  SEARCH COMPARISON\n{'='*60}")
        for name, fn in [("dense",dense_search),("bm25",sparse_search),("hybrid",hybrid_search)]:
            h = 0; ms = 0; tot = 0
            for tc in tests:
                if not tc["expected_url_contains"]: continue
                tot += 1; raw = fn(tc["question"], k=40); top = group_by_url(raw)[:5]
                if check_url_match(top, tc["expected_url_contains"]): h+=1
                rk = find_rank(top, tc["expected_url_contains"])
                ms += (1.0/rk) if rk else 0
            search_comp[name] = {"hits":h,"total":tot,"mrr":round(ms/tot,3) if tot else 0}
            print(f"  {name:8s}: {h}/{tot} ({h*100//tot}%) MRR={search_comp[name]['mrr']:.3f}")

    # Save
    output = {"timestamp":datetime.now().isoformat(),"n":N,
        "retrieval":{"hit_rate":round(hits/N,3),"mrr":round(mrr,3),"r1":round(r1,3),
            "r3":round(r3,3),"r5":round(r5,3),"kw":round(kw,3)},
        "tiers":{t:{"hit":round(sum(1 for r in e if r["url_match"])/len(e),3),
            "mrr":round(sum(r["mrr"] for r in e)/len(e),3)} for t,e in tiers.items() if e},
        "search_comparison":search_comp, "results":all_res}
    if grades:
        valid = [g for g in grades if g.get("accuracy",-1)>=0]
        if valid:
            output["answer_quality"] = {m:round(sum(g[m] for g in valid)/len(valid),2)
                for m in ["accuracy","completeness","groundedness","safety","relevance"]}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output,indent=2,ensure_ascii=False), encoding="utf-8")
    print(f"\n{'='*60}\n  Saved: {out_path}\n{'='*60}")


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--compare-search", action="store_true")
    ap.add_argument("--output", default="data/eval_results.json")
    run(ap.parse_args())
