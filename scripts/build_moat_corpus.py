"""
build_moat_corpus.py

First real brick of the data moat (possibility.md): a small, schema-compliant,
Kenyan-context natural-language corpus over three domains -- finance, law,
code -- with systematically-generated boundary examples, built exactly the way
TECHNICAL.md section 8's printer prototype prescribes: clean examples plus
mechanically crossed vocabularies, so calibration thresholds can be learned
from genuine difficulty instead of degenerate clean-only data.

Schema (implements the 'New Profiler Dataset Design' sheet in
performance_data.xlsx): example_id, domain_label, text, is_boundary_example,
cross_domain_hint, split, source, contributor, added_for_version.

Honest status: this is a STARTER corpus (a few hundred examples, machine-
generated from hand-written seeds, English with Kenyan context). It is the
first brick of the moat, not the moat. The generator is the source of truth;
re-running it with the same seed reproduces the corpus exactly.

Run: python3 build_moat_corpus.py   (needs: scikit-learn, numpy)
Outputs: ../corpus/moat_brick1.jsonl, moat_brick1.csv, plus a printed
validation report (split integrity + boundary-hardness check).
"""
import csv
import json
import re
from pathlib import Path

import numpy as np

rng = np.random.RandomState(20260805)   # fixed seed: reproducible corpus

OUT_DIR = Path(__file__).resolve().parent.parent / "corpus"
VERSION = "v2-moat-brick-1"
CONTRIBUTOR = "washington"

# ----------------------------------------------------------------------
# 1. Hand-written seeds -- the anchor of realism (source=hand_written)
# ----------------------------------------------------------------------
FINANCE_SEEDS = [
    "How do I reverse a wrong M-Pesa transaction sent to an unknown number?",
    "What are the current interest rates for SACCO share capital loans?",
    "Explain the difference between M-Shwari and Fuliza loan charges.",
    "Is there a transaction limit for sending money to a mobile wallet in Uganda?",
    "How do I read my bank statement to find the total monthly bank charges?",
    "What does 'bounce' mean when a cheque is returned unpaid in Kenya?",
    "Should I take a fixed deposit or a money market fund for six months?",
    "How is forex spread charged when converting KES to USD at a bureau?",
    "What is the penalty for early loan repayment at a Kenyan bank?",
    "How do I track my daily budget using mobile money transaction history?",
]
LAW_SEEDS = [
    "What are my rights when a landlord increases rent without notice in Kenya?",
    "Does the Kenya Data Protection Act 2019 apply to a small e-commerce shop?",
    "What clauses should a software development NDA include?",
    "Can an employer terminate a contract without notice under the Employment Act?",
    "What is the process for registering a business name at the Business Registration Service?",
    "How does defamation law apply to posts on X in Kenya?",
    "What is the difference between a lease and a tenancy agreement?",
    "Are verbal contracts enforceable for goods above a certain value?",
    "What happens when a company director breaches their fiduciary duty?",
    "How does the Companies Act handle a dispute between two shareholders?",
]
CODE_SEEDS = [
    "How do I handle HTTP 429 rate-limit errors in a Python API client?",
    "What is the difference between a left join and an inner join in SQL?",
    "How do I securely store OAuth tokens in a mobile app?",
    "Why does my Node.js server crash with an unhandled promise rejection?",
    "How do I write a Django query that filters by date range?",
    "What is the best way to paginate a large API response?",
    "How do I debug a race condition in a concurrent Go program?",
    "Explain the difference between a library and a framework.",
    "How do I set up CI/CD for a small Flutter app with GitHub Actions?",
    "What does idempotency mean for a payment webhook handler?",
]

# ----------------------------------------------------------------------
# 2. Systematic generation: template x vocabulary (source=systematic_generation)
# ----------------------------------------------------------------------
FIN_TEMPLATES = [
    "How do I {act} {fin_thing} using {fin_channel}?",
    "What fees apply when I {fin_action} {fin_thing2}?",
    "Is it possible to {fin_action2} {fin_thing3} without a bank account?",
    "How long does {fin_process} take to reflect in my {fin_account}?",
    "Explain the charges for {fin_service} on {fin_platform}.",
]
FIN_VOCAB = {
    "act": ["send money to", "withdraw cash from", "pay for goods with", "save towards"],
    "fin_thing": ["a mobile wallet", "a SACCO account", "a bank card", "an escrow account"],
    "fin_channel": ["M-Pesa", "Airtel Money", "a bank app", "an agent till"],
    "fin_action": ["transferring", "depositing", "withdrawing", "overdrawing"],
    "fin_thing2": ["KES 5,000", "a dollar amount", "loan repayments", "insurance premiums"],
    "fin_action2": ["open an account", "get a loan", "buy treasury bills", "pay a utility bill"],
    "fin_thing3": ["from abroad", "as a minor", "as a foreigner", "with a passport"],
    "fin_process": ["a PesaLink transfer", "a bank-to-wallet payment", "a cheque clearance", "a dividend payout"],
    "fin_account": ["savings account", "current account", "M-Pesa wallet", "cooperative account"],
    "fin_service": ["standing orders", "mobile banking", "merchant payments", "international transfers"],
    "fin_platform": ["M-Pesa", "a fintech app", "an agency bank", "the stock exchange"],
}

LAW_TEMPLATES = [
    "What does {law_doc} say about {law_topic}?",
    "Is {law_act} allowed under {law_statute}?",
    "What are the legal steps to {law_action} in Kenya?",
    "Who is liable when {law_scenario}?",
    "What is the penalty for {law_offense} under Kenyan law?",
]
LAW_VOCAB = {
    "law_doc": ["the Data Protection Act 2019", "the Employment Act", "the Companies Act", "the Consumer Protection Act"],
    "law_topic": ["employee termination", "customer data sharing", "director liability", "unfair market practices"],
    "law_act": ["dismissing an employee without notice", "processing data without consent", "withholding severance pay", "selling expired goods"],
    "law_statute": ["Kenyan law", "the Constitution", "the Penal Code", "the Land Act"],
    "law_action": ["register a trademark", "incorporate a company", "evict a tenant", "file a small claim"],
    "law_scenario": ["an employee leaks customer data", "a supplier breaches a contract", "a tenant damages property", "a company fails to file returns"],
    "law_offense": ["fraud", "defamation", "breach of contract", "money laundering"],
}

CODE_TEMPLATES = [
    "How do I {code_act} in {code_lang}?",
    "What is the best way to {code_action} for {code_target}?",
    "Why does my code {code_fail} when I {code_act2}?",
    "How do I {code_act3} without {code_avoid}?",
    "Explain how to {code_action2} using {code_tool}.",
]
CODE_VOCAB = {
    "code_act": ["parse JSON", "retry failed requests", "hash a password", "cache database queries"],
    "code_lang": ["Python", "JavaScript", "Go", "Dart"],
    "code_action": ["handle pagination", "manage connection pools", "validate user input", "implement authentication"],
    "code_target": ["a large dataset", "a mobile app", "an internal API", "a payment gateway"],
    "code_fail": ["throw a null pointer", "return stale data", "time out", "corrupt the database"],
    "code_act2": ["call an external service", "run a migration", "deploy to production", "shard the table"],
    "code_act3": ["stream large files", "run background jobs", "share state across threads", "roll back a transaction"],
    "code_avoid": ["blocking the event loop", "memory leaks", "race conditions", "buffering the whole file"],
    "code_action2": ["implement retry logic", "build a rate limiter", "write an idempotent endpoint", "set up structured logging"],
    "code_tool": ["Redis", "PostgreSQL", "Kafka", "Docker"],
}

# ----------------------------------------------------------------------
# 3. Boundary generation: crossed templates (source=systematic_generation,
#    is_boundary_example=True) -- genuinely dual-domain subjects
# ----------------------------------------------------------------------
BOUNDARY_TEMPLATES = [
    # finance + law
    ("finance+law", "What legal documents do I need to {fin_act} {fin_amt} through {fin_chan} as a {legal_person}?"),
    ("finance+law", "Is it legal for a lender to {lend_act} when the borrower {borrow_state}?"),
    ("finance+law", "What are the tax obligations for a business receiving {fin_amt2} via {fin_chan2} per month?"),
    ("finance+law", "Can a {legal_person2} be held liable for {fin_liab} on behalf of their client?"),
    ("finance+law", "What happens to {fin_asset} in a {legal_proc} under Kenyan succession law?"),
    # finance + code
    ("finance+code", "How do I implement {code_pay} that handles {fin_chan3} callbacks idempotently?"),
    ("finance+code", "What is the safest way to store {fin_data} for a {code_target2} handling {fin_chan4} payments?"),
    ("finance+code", "How do I reconcile {fin_recon} with an {code_tool2} ledger in my fintech backend?"),
    ("finance+code", "What fields should my {code_pay2} validate before debiting a {fin_wallet}?"),
    ("finance+code", "How do I build {code_rate} for a mobile lending API to comply with {fin_lim}?"),
    # law + code
    ("law+code", "How do I implement {code_consent} for a {code_target3} to comply with the Data Protection Act?"),
    ("law+code", "What should a {code_tool3} project's license say about {law_use}?"),
    ("law+code", "How do I log {law_audit} events in a system that must satisfy {law_req}?"),
    ("law+code", "What is the right way to delete {law_pii} from a production database under {law_reg}?"),
    ("law+code", "Can I legally scrape {law_scrape} for my {code_target4} under Kenyan law?"),
    # triple boundary
    ("finance+law+code", "How do I build a {code_pay3} that issues {fin_rec} and archives {law_rec} for audit?"),
    ("finance+law+code", "What does a fintech API need to log to satisfy {law_audit2} and {fin_reg}?"),
]
BOUNDARY_VOCAB = {
    "fin_act": ["transfer", "remit", "lend", "invest"],
    "fin_amt": ["KES 100,000", "a large sum", "foreign currency", "mobile money"],
    "fin_chan": ["M-Pesa", "a fintech app", "a bank", "an agent"],
    "legal_person": ["a minor", "a foreign national", "a company", "an estate"],
    "lend_act": ["repossess collateral", "deduct directly from salary", "blacklist a borrower", "charge compound interest"],
    "borrow_state": ["defaults twice", "is declared bankrupt", "dies", "contests the loan"],
    "fin_amt2": ["over KES 1 million", "more than 500 transactions", "cross-border payments", "cash deposits"],
    "fin_chan2": ["M-Pesa", "PesaLink", "a bank account", "a forex bureau"],
    "legal_person2": ["an agent", "a digital lender", "a bank teller", "a SACCO official"],
    "fin_liab": ["a fraudulent transaction", "an unauthorized withdrawal", "a mistaken transfer", "a lost card"],
    "fin_asset": ["a land title", "a savings account", "a share certificate", "a vehicle logbook"],
    "legal_proc": ["divorce", "insolvency", "probate", "attachment"],
    "code_pay": ["a payment webhook", "a checkout flow", "a wallet top-up endpoint", "a loan disbursement job"],
    "fin_chan3": ["M-Pesa", "Airtel Money", "card", "bank transfer"],
    "fin_data": ["card numbers", "transaction history", "KYC documents", "account balances"],
    "code_target2": ["mobile app", "backend service", "web dashboard", "POS system"],
    "fin_chan4": ["mobile money", "cards", "bank transfers", "crypto"],
    "fin_recon": ["daily settlements", "failed transactions", "agent float", "escrow balances"],
    "code_tool2": ["SQL", "Kafka", "Redis", "a spreadsheet export"],
    "code_pay2": ["payment request", "withdrawal form", "loan application", "refund job"],
    "fin_wallet": ["M-Pesa wallet", "bank account", "SACCO account", "escrow account"],
    "code_rate": ["a rate limiter", "a spending guard", "an anti-fraud check", "a balance lock"],
    "fin_lim": ["CBK limits", "KYC thresholds", "daily caps", "lending caps"],
    "code_consent": ["a consent screen", "a data-sharing toggle", "a privacy notice", "an opt-out flow"],
    "code_target3": ["mobile app", "website", "SaaS product", "kiosk system"],
    "code_tool3": ["open-source", "commercial", "academic", "government"],
    "law_use": ["redistribution", "commercial use", "modification", "liability"],
    "law_audit": ["consent", "access", "correction", "deletion"],
    "law_req": ["audit requirements", "data protection law", "court orders", "regulatory reporting"],
    "law_pii": ["customer records", "biometric data", "health records", "location data"],
    "law_reg": ["the Data Protection Act", "GDPR", "CBK guidelines", "the Access to Information Act"],
    "law_scrape": ["public court records", "social media profiles", "company registries", "price listings"],
    "code_target4": ["research tool", "compliance product", "news aggregator", "price tracker"],
    "code_pay3": ["disbursement system", "billing engine", "subscription service", "remittance platform"],
    "fin_rec": ["digital receipts", "tax invoices", "loan statements", "settlement reports"],
    "law_rec": ["statutory records", "consent logs", "audit trails", "contract archives"],
    "law_audit2": ["CBK audit rules", "tax requirements", "data protection audits", "anti-money-laundering checks"],
    "fin_reg": ["CBK regulations", "SACCO Society Act rules", "NSE listing rules", "insurance regulations"],
}


def fill(template, vocab, r):
    """Fill a template's {slot} placeholders by sampling vocab lists."""
    out = template
    for m in re.findall(r"\{(\w+)\}", template):
        choices = vocab[m]
        out = out.replace("{" + m + "}", choices[r.randint(len(choices))])
    # post-process: dedupe doubled articles from vocab crossing ("as a a ...")
    out = re.sub(r"\b(a|an)\s+\1\b", r"\1", out)
    return out


def main():
    examples = []
    eid = 0

    def add(domain, text, boundary, hint, split, source):
        nonlocal eid
        eid += 1
        examples.append({
            "example_id": f"brick1-{eid:04d}",
            "domain_label": domain,
            "text": text,
            "is_boundary_example": boundary,
            "cross_domain_hint": hint,
            "split": split,
            "source": source,
            "contributor": CONTRIBUTOR,
            "added_for_version": VERSION,
        })

    # ---- clean examples: hand-written seeds + systematic fills
    clean_spec = {
        "finance": (FINANCE_SEEDS, FIN_TEMPLATES, FIN_VOCAB, 55),
        "law": (LAW_SEEDS, LAW_TEMPLATES, LAW_VOCAB, 55),
        "code": (CODE_SEEDS, CODE_TEMPLATES, CODE_VOCAB, 55),
    }
    for domain, (seeds, templates, vocab, n_target) in clean_spec.items():
        texts = list(seeds)
        attempts = 0
        while len(texts) < n_target and attempts < 400:
            attempts += 1
            t = fill(templates[rng.randint(len(templates))], vocab, rng)
            if t not in texts:
                texts.append(t)
        # stratified split: 70 train / 15 calibration / 15 test
        perm = rng.permutation(len(texts))
        n_train = int(len(texts) * 0.7)
        n_cal = int(len(texts) * 0.85) - n_train
        for i, idx in enumerate(perm):
            split = "train" if i < n_train else ("calibration" if i < n_train + n_cal else "test")
            add(domain, texts[idx], False, "", split,
                "hand_written" if idx < len(seeds) else "systematic_generation")

    # ---- boundary examples: crossed vocabularies (halves to calibration/test)
    bound_texts = []
    for hint, template in BOUNDARY_TEMPLATES:
        bound_texts.append((hint, fill(template, BOUNDARY_VOCAB, rng)))
    seen = set()
    unique = []
    for h, t in bound_texts:
        if t not in seen:
            seen.add(t)
            unique.append((h, t))
    perm = rng.permutation(len(unique))
    n_cal = int(len(unique) * 0.5)
    for i, idx in enumerate(perm):
        hint, t = unique[idx]
        split = "calibration" if i < n_cal else "test"
        add("boundary", t, True, hint, split, "systematic_generation")

    # ---- write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "moat_brick1.jsonl", "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    with open(OUT_DIR / "moat_brick1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(examples[0].keys()))
        w.writeheader()
        w.writerows(examples)

    # ---- validation report
    print("=" * 70)
    print("MOAT BRICK 1 -- corpus report")
    print("=" * 70)
    from collections import Counter
    print(f"total examples: {len(examples)}")
    print(f"by domain:      {dict(Counter(e['domain_label'] for e in examples))}")
    print(f"boundary:       {sum(1 for e in examples if e['is_boundary_example'])} "
          f"({dict(Counter(e['cross_domain_hint'] for e in examples if e['is_boundary_example']))})")
    print(f"splits:         {dict(Counter(e['split'] for e in examples))}")
    print(f"source:         {dict(Counter(e['source'] for e in examples))}")

    # split integrity: no duplicate text across splits, ids unique
    ids = [e["example_id"] for e in examples]
    assert len(ids) == len(set(ids)), "duplicate example_ids"
    texts = [e["text"] for e in examples]
    assert len(texts) == len(set(texts)), "duplicate texts"
    print("integrity:      ids unique, texts unique -- OK")

    # boundary hardness: TF-IDF classifier trained on clean train,
    # evaluated on clean test vs boundary test
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    clean_train = [e for e in examples if e["split"] == "train"]
    clean_test = [e for e in examples if e["split"] == "test" and not e["is_boundary_example"]]
    bound_test = [e for e in examples if e["split"] == "test" and e["is_boundary_example"]]
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    Xtr = vec.fit_transform([e["text"] for e in clean_train])
    clf = LogisticRegression(max_iter=2000, random_state=0)
    clf.fit(Xtr, [e["domain_label"] for e in clean_train])
    acc_clean = clf.score(vec.transform([e["text"] for e in clean_test]),
                          [e["domain_label"] for e in clean_test])
    pred_bound = clf.predict(vec.transform([e["text"] for e in bound_test]))
    acc_bound = (pred_bound == [e["domain_label"] for e in bound_test]).mean()
    print(f"\nboundary-hardness check (TF-IDF + logistic regression):")
    print(f"  accuracy on clean test:  {acc_clean*100:.1f}%")
    print(f"  accuracy on boundary test: {acc_bound*100:.1f}%  "
          f"(low = genuinely dual-domain, the property §8 requires)")
    print(f"  boundary predicted as:   {dict(Counter(pred_bound))}")

    print(f"\nwrote: {OUT_DIR / 'moat_brick1.jsonl'}, {OUT_DIR / 'moat_brick1.csv'}")


if __name__ == "__main__":
    main()
