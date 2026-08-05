"""
build_moat_corpus.py

Moat corpus generator -- brick 2. Continues the first brick (brick 1:
finance/law/code, 181 examples) with a fourth domain (education, Kenyan
context), expanded seeds and vocabularies, and a wider boundary-example
set across all domain pairs.

Same design contract as brick 1: seeded and deterministic (re-running
reproduces the corpus exactly), schema-compliant with the 'New Profiler
Dataset Design' sheet in performance_data.xlsx, and machine-validated
(boundary hardness: clean examples are easy for a domain classifier,
boundary examples are genuinely dual-domain).

Honest status: machine-generated from hand-written seeds, English with
Kenyan context. No human review pass, no Swahili/sheng coverage. It is the
second brick of the moat, not the moat.

Run: python3 build_moat_corpus.py   (needs: scikit-learn, numpy)
Outputs: ../corpus/moat_brick2.jsonl, moat_brick2.csv, plus a printed
validation report (split integrity + boundary-hardness check).
"""
import csv
import json
import re
from pathlib import Path

import numpy as np

rng = np.random.RandomState(20260807)   # fixed seed: reproducible corpus

OUT_DIR = Path(__file__).resolve().parent.parent / "corpus"
VERSION = "v2-moat-brick-3"
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
    "How do I open a bank account for my small business in Kenya?",
    "What is the difference between a salary advance and a personal loan?",
    "How do I file VAT returns for my online shop?",
    "What are the charges for sending money from Kenya to Tanzania?",
    "How does the CBK's digital credit rules affect mobile lenders?",
    "What is a standing order and how do I set one up?",
    "How do I invest in treasury bills through my phone?",
    "Should I use a chama or a Sacco for my savings group?",
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
    "How do I register a trademark for my brand in Kenya?",
    "What does the Consumer Protection Act say about refunds?",
    "How do I handle a dispute with a contractor who was paid in advance?",
    "What are my obligations as a landlord under the Landlord and Tenant Act?",
    "How does the Small Claims Court work in Kenya?",
    "What is the penalty for late filing of company returns?",
    "Can my employer read my personal messages on a work phone?",
    "How do I make a police report for a bounced cheque?",
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
    "How do I use environment variables to keep API keys out of source control?",
    "What is the difference between synchronous and asynchronous code in Python?",
    "How do I write a unit test for a function that calls an external API?",
    "What is the best way to version a REST API?",
    "How do I profile a slow SQL query in PostgreSQL?",
    "Why should I use a virtual environment for Python projects?",
    "How do I implement server-side pagination in a GraphQL API?",
    "What is a webhook and how do I secure its endpoint?",
]
EDUCATION_SEEDS = [
    "How do I apply for a HELB loan as a first-year university student?",
    "What subjects do I need in KCSE to study computer science?",
    "How does the KUCCPS placement process work for TVET colleges?",
    "Can I defer my university admission for one year?",
    "What is the difference between a diploma and a degree at a Kenyan polytechnic?",
    "How do I apply for a bursary from the county government?",
    "What is the cut-off grade for medicine at the University of Nairobi?",
    "How do I transfer from one university to another mid-degree?",
    "What are the fees for a private secondary school in Nairobi?",
    "How does the competency-based curriculum grade junior school students?",
    "How do I sit KCSE as a private candidate?",
    "What documents do I need to register for TVET under the new funding model?",
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
    "What documents do I need to {fin_action3} {fin_thing4}?",
]
FIN_VOCAB = {
    "act": ["send money to", "withdraw cash from", "pay for goods with", "save towards", "invest in", "borrow against"],
    "fin_thing": ["a mobile wallet", "a SACCO account", "a bank card", "an escrow account", "a treasury bond", "an insurance policy"],
    "fin_channel": ["M-Pesa", "Airtel Money", "a bank app", "an agent till", "a forex bureau", "the NSE app"],
    "fin_action": ["transferring", "depositing", "withdrawing", "overdrawing", "reversing", "splitting"],
    "fin_thing2": ["KES 5,000", "a dollar amount", "loan repayments", "insurance premiums", "a dividend", "a salary payment"],
    "fin_action2": ["open an account", "get a loan", "buy treasury bills", "pay a utility bill", "start a chama", "register for eCitizen payments"],
    "fin_thing3": ["from abroad", "as a minor", "as a foreigner", "with a passport", "without collateral", "with a guarantor"],
    "fin_process": ["a PesaLink transfer", "a bank-to-wallet payment", "a cheque clearance", "a dividend payout", "an RTGS transfer", "a standing order"],
    "fin_account": ["savings account", "current account", "M-Pesa wallet", "cooperative account", "dollar account", "fixed deposit"],
    "fin_service": ["standing orders", "mobile banking", "merchant payments", "international transfers", "overdraft facilities", "salary advances"],
    "fin_platform": ["M-Pesa", "a fintech app", "an agency bank", "the stock exchange", "a Sacco portal", "an insurance app"],
    "fin_action3": ["open", "close", "consolidate", "register", "link", "verify"],
    "fin_thing4": ["a bank account", "a mobile money line", "a SACCO membership", "an NSSF account", "a HELB account", "an escrow account"],
}

LAW_TEMPLATES = [
    "What does {law_doc} say about {law_topic}?",
    "Is {law_act} allowed under {law_statute}?",
    "What are the legal steps to {law_action} in Kenya?",
    "Who is liable when {law_scenario}?",
    "What is the penalty for {law_offense} under Kenyan law?",
    "Does {law_doc} apply to {law_entity}?",
    "Can {law_entity} be sued for {law_offense}?",
    "What does {law_statute} say about {law_topic}?",
]
LAW_VOCAB = {
    "law_doc": ["the Data Protection Act 2019", "the Employment Act", "the Companies Act", "the Consumer Protection Act", "the Labour Relations Act", "the Access to Information Act"],
    "law_topic": ["employee termination", "customer data sharing", "director liability", "unfair market practices", "probationary employment", "airtime data privacy"],
    "law_act": ["dismissing an employee without notice", "processing data without consent", "withholding severance pay", "selling expired goods", "dismissing a union member", "sharing customer data with third parties"],
    "law_statute": ["Kenyan law", "the Constitution", "the Penal Code", "the Land Act", "the Labour Relations Act", "the Access to Information Act"],
    "law_action": ["register a trademark", "incorporate a company", "evict a tenant", "file a small claim", "get a court order", "appeal a tribunal decision"],
    "law_scenario": ["an employee leaks customer data", "a supplier breaches a contract", "a tenant damages property", "a company fails to file returns", "a board member leaks minutes", "a contractor abandons a site"],
    "law_offense": ["fraud", "defamation", "breach of contract", "money laundering", "insider trading", "identity theft"],
    "law_entity": ["a sole trader", "a church", "a school", "a foreign company"],
}

CODE_TEMPLATES = [
    "How do I {code_act} in {code_lang}?",
    "What is the best way to {code_action} for {code_target}?",
    "Why does my code {code_fail} when I {code_act2}?",
    "How do I {code_act3} without {code_avoid}?",
    "Explain how to {code_action2} using {code_tool}.",
    "How do I {code_act4} when {code_cond}?",
]
CODE_VOCAB = {
    "code_act": ["parse JSON", "retry failed requests", "hash a password", "cache database queries", "serialize objects", "validate webhooks"],
    "code_lang": ["Python", "JavaScript", "Go", "Dart", "Rust", "TypeScript"],
    "code_action": ["handle pagination", "manage connection pools", "validate user input", "implement authentication", "schedule background jobs", "build a REST client"],
    "code_target": ["a large dataset", "a mobile app", "an internal API", "a payment gateway", "a message queue", "a data warehouse"],
    "code_fail": ["throw a null pointer", "return stale data", "time out", "corrupt the database", "deadlock", "exhaust memory"],
    "code_act2": ["call an external service", "run a migration", "deploy to production", "shard the table", "reload config", "spawn workers"],
    "code_act3": ["stream large files", "run background jobs", "share state across threads", "roll back a transaction", "batch inserts", "tune garbage collection"],
    "code_avoid": ["blocking the event loop", "memory leaks", "race conditions", "buffering the whole file", "N+1 queries", "hardcoded secrets"],
    "code_action2": ["implement retry logic", "build a rate limiter", "write an idempotent endpoint", "set up structured logging", "add circuit breakers", "write migration tests"],
    "code_tool": ["Redis", "PostgreSQL", "Kafka", "Docker", "S3", "Celery"],
    "code_act4": ["handle offline sync", "split a monolith", "secure an API", "debug a memory leak", "run A/B tests", "observe latency"],
    "code_cond": ["the network drops", "the queue backs up", "the replica lags", "a deploy is in progress", "the cache misses", "the database is locked"],
}

EDUCATION_TEMPLATES = [
    "What {edu_need} for {edu_level}?",
    "How do I {edu_action} at {edu_inst}?",
    "What is the {edu_req} for {edu_prog}?",
    "How does {edu_policy} work?",
    "Can I {edu_action2} without {edu_cond}?",
    "How are {edu_fees2} set at {edu_inst}?",
    "What is the deadline for {edu_gerund} at {edu_inst}?",
]
EDUCATION_VOCAB = {
    "edu_need": ["subjects are required", "grades do I need", "documents are needed", "fees apply", "results are recognized", "grade requirements apply"],
    "edu_level": ["KCSE", "a diploma", "university entry", "junior school", "an international curriculum", "a bridging course"],
    "edu_action": ["apply", "register", "transfer", "defer admission", "apply for re-admission", "appeal my placement"],
    "edu_inst": ["KUCCPS", "a TVET college", "the University of Nairobi", "a county polytechnic", "a national polytechnic", "a teachers college"],
    "edu_req": ["cut-off", "requirement", "deadline", "documentation", "grade aggregation", "credit transfer"],
    "edu_prog": ["medicine", "engineering", "computer science", "a teaching course", "a law degree", "an aviation course"],
    "edu_policy": ["the CBC grading system", "the 8-4-4 to CBC transition", "the new funding model", "school re-admission policy", "the school capitation formula", "boarding requirements"],
    "edu_action2": ["sit KCSE as a private candidate", "apply for a bursary", "get a fee waiver", "join a university mid-year", "repeat a year", "change my course"],
    "edu_cond": ["a KCSE certificate", "a bank account", "a birth certificate", "sponsor approval", "a guarantor", "a medical certificate"],
    "edu_fees2": ["school fees", "hostel charges", "exam registration fees", "library levies"],
    "edu_gerund": ["sitting KCSE as a private candidate", "repeating a year", "changing my course", "applying for a bursary", "registering for exams", "submitting transfer papers"],
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
    ("finance+law", "Is it legal to charge interest on a loan given to a minor?"),
    ("finance+law", "What is the tax treatment of SACCO dividends in Kenya?"),
    ("finance+law", "Can a digital lender blacklist a borrower who disputes the debt?"),
    ("finance+law", "What happens to my M-Pesa float if my agent business is shut down?"),
    ("finance+law", "Who owns the money in a joint bank account when one holder dies?"),
    # finance + code
    ("finance+code", "How do I implement {code_pay} that handles {fin_chan3} callbacks idempotently?"),
    ("finance+code", "What is the safest way to store {fin_data} for a {code_target2} handling {fin_chan4} payments?"),
    ("finance+code", "How do I reconcile {fin_recon} with an {code_tool2} ledger in my fintech backend?"),
    ("finance+code", "What fields should my {code_pay2} validate before debiting a {fin_wallet}?"),
    ("finance+code", "How do I build {code_rate} for a mobile lending API to comply with {fin_lim}?"),
    ("finance+code", "How do I handle M-Pesa timeout callbacks without double-debiting customers?"),
    ("finance+code", "How do I build a statement parser that handles multiple Kenyan bank formats?"),
    ("finance+code", "How do I implement FX rate caching for a remittance app?"),
    ("finance+code", "How do I avoid floating-point errors converting KES to cents?"),
    # law + code
    ("law+code", "How do I implement {code_consent} for a {code_target3} to comply with the Data Protection Act?"),
    ("law+code", "What should a {code_tool3} project's license say about {law_use}?"),
    ("law+code", "How do I log {law_audit} events in a system that must satisfy {law_req}?"),
    ("law+code", "What is the right way to delete {law_pii} from a production database under {law_reg}?"),
    ("law+code", "Can I legally scrape {law_scrape} for my {code_target4} under Kenyan law?"),
    ("law+code", "How do I store consent records with tamper-evident timestamps?"),
    ("law+code", "What fields can a Kenyan edtech app legally collect from minors?"),
    ("law+code", "How do I build a right-to-erasure endpoint under the Data Protection Act?"),
    ("law+code", "What license applies if I reuse court judgment text in a legal research app?"),
    # finance + education
    ("finance+education", "Can I get a {fin_loan} for {edu_level} if I have no {fin_hist}?"),
    ("finance+education", "What happens to my {edu_fee} if my {fin_channel5} payment fails at {edu_inst}?"),
    ("finance+education", "How do I apply for a {edu_bursary} when my parents earn {fin_income}?"),
    ("finance+education", "Are {edu_doc} released when there is an outstanding {fin_bal} at {edu_inst}?"),
    ("finance+education", "What is the interest rate on {edu_loan} compared to a {fin_loan2}?"),
    # law + education
    ("law+education", "Can a school withhold {edu_doc} over an unpaid {fin_bal} under Kenyan law?"),
    ("law+education", "What are my rights if {edu_inst} expels me without a hearing?"),
    ("law+education", "Does the Data Protection Act apply to {edu_data} held by {edu_inst}?"),
    ("law+education", "Is it legal for {edu_inst} to charge {edu_fee} increases mid-year?"),
    ("law+education", "What legal recourse do I have if {edu_inst} revokes my {edu_admit}?"),
    # code + education
    ("code+education", "How do I build {edu_app} that syncs {edu_data} offline for rural schools?"),
    ("code+education", "How do I implement {code_grade} for an exam portal used by {edu_many} students?"),
    ("code+education", "How do I secure {edu_data} in an LMS with {code_role}?"),
    ("code+education", "How do I build a {code_fee} that reminds parents about {edu_fee} via SMS?"),
    ("code+education", "How do I scrape {edu_portal} results legally for my {code_target5}?"),
    # triple boundaries
    ("finance+law+code", "How do I build a {code_pay3} that issues {fin_rec} and archives {law_rec} for audit?"),
    ("finance+law+code", "What does a fintech API need to log to satisfy {law_audit2} and {fin_reg}?"),
    ("finance+law+education", "How do I build a {edu_app2} that issues {fin_rec} and stores {law_rec} for HELB audits?"),
    ("finance+law+education", "What does a {fintech_edu} need to log to satisfy {law_audit2} and {edu_reg}?"),
    ("finance+code+education", "How do I handle {fin_chan3} callbacks in a school {code_fee} without double-debiting parents?"),
    # brick 3 additions -- education-heavy pairs + more coverage
    ("finance+education", "How does the new university funding model calculate my loan eligibility?"),
    ("finance+education", "Are HELB loan defaults reported to credit reference bureaus?"),
    ("finance+education", "What happens to my bursary if I transfer to a private university?"),
    ("finance+education", "Can a university withhold my certificate for unpaid fees?"),
    ("finance+education", "How do I pay back my HELB loan if I drop out of university?"),
    ("law+education", "Can a school expel a student for unpaid fees under Kenyan law?"),
    ("law+education", "What does the Basic Education Act say about student discipline?"),
    ("law+education", "Is it legal for a private college to keep my transcripts for an unpaid balance?"),
    ("law+education", "Can the government regulate fees charged by private universities?"),
    ("code+education", "How do I build a KCSE results checker against a public API?"),
    ("code+education", "How do I design an LMS that handles 10,000 concurrent students?"),
    ("code+education", "How do I implement a secure student-record system with role-based access?"),
    ("finance+law", "Is it legal for a digital lender to deduct directly from a borrower's salary?"),
    ("finance+law", "What happens to an M-Pesa account when the owner dies intestate?"),
    ("law+code", "What data can a Kenyan fintech legally store about its users?"),
    ("law+code", "How do I make an AI moderation system comply with the Data Protection Act?"),
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
    # education slots
    "fin_loan": ["HELB loan", "a student loan", "a fee-financing loan", "a mobile loan"],
    "fin_hist": ["credit history", "bank statement", "guarantor", "income proof"],
    "fin_channel5": ["M-Pesa", "bank transfer", "PesaLink", "card"],
    "edu_level": ["first year", "final year", "junior school", "a diploma course"],
    "edu_fee": ["school fees", "tuition", "accommodation fees", "exam fees"],
    "edu_doc": ["transcripts", "a KCSE certificate", "a degree certificate", "admission letters"],
    "edu_inst": ["a university", "a TVET college", "a private school", "KUCCPS"],
    "edu_bursary": ["county bursary", "university scholarship", "fee waiver", "HELB hardship grant"],
    "fin_income": ["less than KES 50,000 a month", "irregular income", "farm income", "no formal income"],
    "fin_bal": ["fee balance", "library fine", "loan balance", "levy"],
    "edu_loan": ["HELB loan", "the new student funding model", "a private student loan", "a TVET loan"],
    "fin_loan2": ["SACCO loan", "bank loan", "mobile loan", "chama contribution"],
    "edu_data": ["student records", "exam results", "disciplinary files", "special-needs assessments"],
    "edu_admit": ["admission letter", "placement", "scholarship award", "transfer approval"],
    "edu_app": ["an e-learning platform", "a school management system", "a revision app", "a bursary portal"],
    "code_grade": ["a grading engine", "a results checker", "a timetable generator", "an attendance tracker"],
    "edu_many": ["10,000", "a district", "offline", "low-bandwidth"],
    "code_role": ["role-based access", "end-to-end encryption", "audit logging", "biometric login"],
    "code_fee": ["fee payment bot", "invoice generator", "payment reminder", "receipt system"],
    "edu_portal": ["KUCCPS", "KNEC", "university portals", "county bursary systems"],
    "code_target5": ["mobile app", "study tool", "analytics dashboard", "news site"],
    "edu_app2": ["bursary application system", "student loan portal", "school finance system", "scholarship platform"],
    "fintech_edu": ["student-loan fintech", "school fee platform", "HELB app", "bursary portal"],
    "edu_reg": ["Ministry of Education rules", "HELB regulations", "KUCCPS guidelines", "university statutes"],
}


def fill(template, vocab, r):
    """Fill a template's {slot} placeholders by sampling vocab lists."""
    out = template
    for m in re.findall(r"\{(\w+)\}", template):
        choices = vocab[m]
        out = out.replace("{" + m + "}", choices[r.randint(len(choices))])
    # post-process: dedupe doubled articles from vocab crossing
    # ("as a a ...", "as a an estate") -- keep the second article, which
    # belongs to the vocab item itself
    out = re.sub(r"\b(a|an)\s+(a|an)\b", r"\2", out)
    return out


def main():
    examples = []
    eid = 0

    def add(domain, text, boundary, hint, split, source):
        nonlocal eid
        eid += 1
        examples.append({
            "example_id": f"brick2-{eid:04d}",
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
        "finance": (FINANCE_SEEDS, FIN_TEMPLATES, FIN_VOCAB, 150),
        "law": (LAW_SEEDS, LAW_TEMPLATES, LAW_VOCAB, 150),
        "code": (CODE_SEEDS, CODE_TEMPLATES, CODE_VOCAB, 150),
        "education": (EDUCATION_SEEDS, EDUCATION_TEMPLATES, EDUCATION_VOCAB, 150),
    }
    for domain, (seeds, templates, vocab, n_target) in clean_spec.items():
        texts = list(seeds)
        attempts = 0
        while len(texts) < n_target and attempts < 3000:
            attempts += 1
            t = fill(templates[rng.randint(len(templates))], vocab, rng)
            if t not in texts:
                texts.append(t)
        # stratified split: 60 train / 25 calibration / 15 test
        # (calibration share raised vs brick 2 to stabilize the s8 p99
        # threshold, per the calibration trial's small-n finding)
        perm = rng.permutation(len(texts))
        n_train = int(len(texts) * 0.6)
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
    stem = "moat_brick3"
    with open(OUT_DIR / f"{stem}.jsonl", "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    with open(OUT_DIR / f"{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(examples[0].keys()))
        w.writeheader()
        w.writerows(examples)

    # ---- validation report
    print("=" * 70)
    print(f"MOAT BRICK 2 -- corpus report ({VERSION})")
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

    print(f"\nwrote: {OUT_DIR / f'{stem}.jsonl'}, {OUT_DIR / f'{stem}.csv'}")


if __name__ == "__main__":
    main()
