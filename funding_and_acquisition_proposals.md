# lorouter — Comprehensive Funding, Hackathon & Strategic Partner Proposals

> **Document Purpose:** 5 specialized, high-conviction grant applications, hackathon sponsorship proposals, and strategic partnership pitches tailored for `lorouter`.
> **Date:** August 2026

---

## Strategic Portfolio Overview

| # | Organization / Fund | Target Track / Mechanism | Direct Application / Portal Link | Funding / Value Type |
|---|---|---|---|---|
| **1** | **Emergent Ventures** (Mercatus Center / Tyler Cowen) | Fast-Grant for High-Agency Independent Researchers | [Emergent Ventures Portal](https://www.mercatus.org/emergent-ventures) | $10,000 – $50,000 unrestricted grant (Fast turnaround) |
| **2** | **Mozilla Technology Fund / Mozilla Builders** | Decentralized & Trustworthy AI Infrastructure | [Mozilla Technology Fund](https://foundation.mozilla.org/en/what-we-fund/awards/mozilla-technology-fund/) · [Mozilla Builders](https://builders.mozilla.org/) | $25,000 – $50,000 open-source grant |
| **3** | **Lacuna Fund / AI4D Africa** | African NLP & Low-Compute Public Goods | [Lacuna Fund](https://lacunafund.org/) · [AI4D Africa](https://ai4d.ai/) | $30,000 – $100,000 research & dataset grant |
| **4** | **Hugging Face Grants & Hackathon Sponsorship** | Open Source Ecosystem & PEFT Tooling | [Hugging Face Spaces & Grants](https://huggingface.co/grants) · [HF Community Contact](https://huggingface.co/contact) | $5,000 – $20,000 + GPU compute credits & co-branded event |
| **5** | **Epic Games MegaGrants** | Interactive Entertainment & Engine Tools | [Epic MegaGrants Portal](https://www.unrealengine.com/en-US/megagrants) | $25,000 – $75,000 non-dilutive grant |

---

## 1. Emergent Ventures (Mercatus Center)

* **Official Portal:** [https://www.mercatus.org/emergent-ventures](https://www.mercatus.org/emergent-ventures)
* **Application Direct Form:** [Mercatus Online Fast-Grant Submission Form](https://www.mercatus.org/emergent-ventures)
* **Decision Timeframe:** ~2–3 weeks (rolling submissions, zero bureaucracy).

### Why This is a Top-Tier Fit
Emergent Ventures is designed for high-agency, independent builders who do rigorous empirical work outside traditional institutional tracks. Your profile—a self-taught developer in Kenya producing 46 verified findings on CPU/small GPUs with mathematical swap guarantees—is the exact archetype Tyler Cowen backs with fast, zero-red-tape capital.

### Application Draft

**Project Title:** lorouter: Zero-Parameter Profile Routing for Decentralized Small-Model Multi-LoRA Serving

**Applicant:** Washingtone Imae (Independent Researcher, Kenya)

**1. What are you working on? (100 words)**
Serving systems (vLLM, S-LoRA) can load thousands of LoRA adapters on a single base model, but require the user to manually name the adapter. Existing automated routers (MoE-LoRA) use heavy learned gating networks that are black-box, risk catastrophic forgetting, and require global retraining whenever an adapter is added. 

I built `lorouter`: a zero-parameter routing layer that selects adapters via cosine similarity over calibrated domain competence profiles. It matches learned router accuracy (96.4% on 4 domains, 95.7% on 6 domains with 6/6 diagonal dominance), scales to 1,024 adapters, dispatches in 0.4 ms, and guarantees 0.00% swap collateral damage by structural construction.

**2. What have you already accomplished? (150 words)**
Over the past year, I built an empirical evidence chain of 46 numbered, reproducible findings backed by Python scripts:
- Proved 96.4% routing accuracy across SmolLM2-135M, 360M, and Qwen-0.8B models across multiple deterministic seeds (F5, F26, F27).
- Proven that adapter false-capture does not follow the classic independent-gate compounding failure mode in correlated pools (F34).
- Built a 3,010-example multi-domain calibration corpus (`moat_brick3.jsonl`) with 220 dual-domain boundary examples enforcing split discipline.
- Integrated hosted data optimization (Adaption Adaptive Data) to enhance training pairs and generated 420 held-out evaluation references (F43–F46).

**3. What will this grant fund? (100 words)**
A grant of $15,000 will fund:
1. **The 9-Domain Full-Scale Benchmark**: Finalizing training and evaluation across 9 real-world domains (including telecom, agriculture, and African fintech).
2. **vLLM/LoRAX Engine C++ Hook**: Embedding the policy directly into inference engines for production sub-millisecond dispatch.
3. **Compute & Open Release**: Publishing the dataset, 9 adapter checkpoints, and arXiv paper (`washi254/lorouter-moat-corpus-v4`).

**4. Why is this important? (100 words)**
It enables decentralized, permissionless AI customization on cheap hardware. Because the router has zero learned parameters and 0.00% swap collateral, anyone can inject a specialized adapter or Python script into a shared model without needing central retraining or expensive cluster compute.

---

## 2. Mozilla Builders / Mozilla Technology Fund

* **Official Portal:** [https://foundation.mozilla.org/en/what-we-fund/awards/mozilla-technology-fund/](https://foundation.mozilla.org/en/what-we-fund/awards/mozilla-technology-fund/)
* **Mozilla Builders Community:** [https://builders.mozilla.org/](https://builders.mozilla.org/)
* **Submission Cycle:** Annual / Bi-annual cohorts + rolling builder community calls.

### Why This is a Top-Tier Fit
Mozilla focuses on "Trustworthy AI," open-source commons, and anti-monopoly infrastructure. `lorouter` replaces centralized, closed, learned routing gates with open, auditable, permissionless capability dispatch where everyday users can contribute without gatekeeping.

### Application Draft

**Project Name:** lorouter — Open, Auditable, Zero-Parameter Capability Dispatch for Local and Edge LLMs

**Organization Track:** Open Source AI Infrastructure & Trustworthy AI

**Executive Summary:**
Current LLM serving forces a choice between monolithic models owned by centralized giants or complex MoE systems with opaque, learned routing weights. `lorouter` offers an open-source alternative: a zero-parameter routing plane that turns adapters, deterministic scripts, and community prompts into hot-swappable peers. It allows public institutions, community groups, and independent developers to inject localized intelligence into shared models with guaranteed zero collateral damage to other domains and full per-request decision auditability.

**Key Public-Interest Capabilities:**
1. **Traceable Decision Auditing**: Unlike learned gates that produce black-box forward passes, `lorouter` decisions reduce to verifiable scores ("Adapter X was chosen because query domain vector Y matched competence score Z"). This is essential for public sector, legal, and financial deployments.
2. **Permissionless Community Injection**: Grassroots developers can train lightweight 5MB LoRA adapters for underrepresented languages (e.g., Sheng, regional African dialects) and plug them into local models without requiring central infrastructure retraining.
3. **Neuro-Symbolic Efficiency**: Eliminates expensive JSON function-calling prompt bloat by routing directly to deterministic code or small local adapters in under 1 millisecond.

**Milestones & Deliverables:**
- **Month 1**: Release v1.0 of the `lorouter` Python package with plug-and-play support for custom user-injected Python scripts and LoRA weights.
- **Month 2**: Complete the 9-domain open benchmark including agriculture, public health, and fintech datasets.
- **Month 3**: Host a virtual "Community Capability Injection" developer challenge to crowdsource 50+ domain micro-adapters on Hugging Face.
- **Month 4**: Release an open-source vLLM / SGLang integration guide and reproducible evaluation suite.

**Funding Request:** $35,000 (Developer stipend, GPU training budget, community challenge prize pool).

---

## 3. Lacuna Fund / AI4D Africa

* **Official Portal (Lacuna Fund):** [https://lacunafund.org/apply/](https://lacunafund.org/apply/)
* **Official Portal (AI4D Africa):** [https://ai4d.ai/call/](https://ai4d.ai/call/)
* **Focus Area:** African-led AI datasets, regional language technologies, and public health/agriculture deployments.

### Why This is a Top-Tier Fit
Lacuna Fund and AI4D fund African-led, local-context AI datasets and systems that serve underserved populations. `lorouter`'s primary training foundation is Kenyan-grounded (SACCO finance, HELB education, KUCCPS admissions, Kenyan labor law, M-Pesa tariffs), and its small-model architecture directly targets low-connectivity African edge environments.

### Application Draft

**Project Title:** Democratizing Low-Resource African AI: Sub-1B Model Multi-Adapter Serving for Regional Public Services

**Focus Area:** African Language Technology & Localized Public Good AI

**The Problem in African Context:**
Deploying 70B+ frontier models for African public services (schools, agricultural extension, rural clinics, SACCO micro-finance) is economically unviable due to high API costs, bandwidth constraints, and lack of offline reliability. Furthermore, monolithic models lack nuanced training on regional realities like Kenyan statutes, agricultural pest management, or mobile money dispute flows.

**The lorouter Solution:**
`lorouter` enables a single, ultra-lightweight small language model (e.g., Qwen-0.8B or SmolLM-360M) operating offline on modest local hardware (Raspberry Pi, low-cost servers) to dynamically switch between dozens of domain-specific adapters:
- **Agriculture**: Crop disease triage and local planting guides.
- **Fintech**: SACCO bylaws, M-Pesa dispute resolutions, CBK digital lender regulations.
- **Education & Governance**: KUCCPS university placements, HELB student loan policies.
- **Linguistic Localization**: Multilingual and code-switched queries (Sheng, Swahili-English blend).

**Project Deliverables:**
1. **Curated African Moat Corpus (v4)**: Expanding the existing 3,010-example Kenyan corpus to 6,000+ human-reviewed pairs with native Swahili and Sheng coverage.
2. **9 Verified Open-Source Domain Adapters**: Released freely under permissive licenses for African developers.
3. **Offline Public Service Appliance Blueprint**: A turnkey software kit for deploying `lorouter` on low-power, offline edge hardware for rural schools and clinics.

**Funding Request:** $45,000 (Human annotation & Swahili/Sheng language review, local compute, hardware validation benches).

---

## 4. Hugging Face Community Grant & Hackathon Proposal

* **Official Portal (Grants & Compute):** [https://huggingface.co/grants](https://huggingface.co/grants)
* **Spaces / Partnership Inquiries:** [https://huggingface.co/contact](https://huggingface.co/contact)
* **Direct Outreach Channels:** Reach out to Developer Advocates (e.g., via Twitter/X or HF Discord @huggingface).

### Why This is a Top-Tier Fit
Hugging Face is the center of the open-source PEFT/LoRA ecosystem. `lorouter` represents the ideal substrate for a massive community event: **"The Micro-LoRA Ecosystem Hackathon."** Instead of building one giant model, thousands of developers build, calibrate, and submit tiny 5MB adapters to a shared global router matrix.

### Partnership & Sponsorship Pitch

**Event Concept:** *The Universal Capability Hackathon: Build the World's Largest Modular AI Pool*

**Target Platform:** Hugging Face Spaces + `peft` + `lorouter`

**The Pitch to Hugging Face:**
"We have built `lorouter`, a zero-parameter router that selects adapters and tools via calibrated competence profiles with 0% swap collateral damage. We want to partner with Hugging Face to host a 2-week global hackathon where community developers build specialized 5MB–10MB LoRA adapters and Python micro-tools across 100+ niche domains. Every participant's adapter is calibrated and automatically slotted into a single, unified Hugging Face Space running a shared base model."

**Why HF Wins:**
- Showcases the power of the `peft` and Hugging Face Hub ecosystem.
- Generates hundreds of high-quality, open-source community LoRAs and dataset cards on the Hub.
- Demonstrates how to serve thousands of community adapters on a single ZeroGPU / Spaces instance without VRAM explosion.

**What We Request:**
- **$10,000 Prize Pool Sponsorship** (shared among top domain contributions: Medical, Code, Creative, Regional Languages, Deterministic Tools).
- **ZeroGPU / Compute Credits** for hackathon participants.
- **Co-Marketing & Featured Space on the Hugging Face Daily Papers / Community Hub.**

---

## 5. Epic Games MegaGrants (Interactive Media & Game Development)

* **Official Portal:** [https://www.unrealengine.com/en-US/megagrants](https://www.unrealengine.com/en-US/megagrants)
* **Application Mechanism:** Rolling non-dilutive grant evaluations through the Unreal Engine developer dashboard.
* **Eligible Categories:** Open source developer tools, engine plugins, procedural generation, AI integration.

### Why This is a Top-Tier Fit
Epic Games MegaGrants provides non-dilutive, zero-strings funding ($5,000 to $250,000) for open-source tools and technology that advance 3D engines, virtual worlds, and game development workflows. `lorouter` directly solves the game industry's biggest AI bottleneck: running immersive, multi-character NPC intelligence locally on console/PC VRAM budgets without cloud API lag.

### Application Draft

**Project Title:** Sub-Millisecond Neuro-Symbolic Capability Dispatch for In-Engine NPCs and Procedural Asset Validation

**Category:** Open Source Tools for Unreal Engine & Real-Time Production

**Project Description:**
Real-time game engines face a severe constraint when integrating Generative AI:
1. **Cloud APIs introduce 2–3 second immersion-breaking latency** and unsustainable per-query recurring costs.
2. **Local 7B/14B models consume 8GB–16GB VRAM**, starving the engine of memory required for 4K textures, shaders, physics, and frame rendering.

`lorouter` introduces a lightweight, in-engine dispatch layer designed for game architectures:
- **Ultra-Low Memory Footprint**: A game runs a single quantized 0.8B/360M base model in ~400MB VRAM, while dynamically loading 5MB–10MB personality and faction adapters.
- **Sub-Millisecond Personality Swapping**: `lorouter`'s selection policy executes in **0.4 ms (17 µs – 380 µs)**, allowing the engine to swap NPC dialogue contexts between game ticks with zero frame drops.
- **Neuro-Symbolic Tool Execution**: Creative dialogue generation and deterministic engine tasks (e.g., HLSL shader generation, polygon budget verification, glTF schema checks) coexist in the same dispatch table.
- **Modder & Creator Ecosystem**: Modders can add new characters and questlines by distributing a tiny 8MB file without needing engine recompilation.

**MegaGrant Funding Goals:**
1. **Unreal Engine 5 Plugin (C++ / Blueprint)**: A native Unreal Engine plugin integrating `lorouter` for local on-device NPC dialogue and tool routing.
2. **Open Sample Project**: A playable UE5 fantasy tavern / open-world demo showcasing 20 distinct NPCs sharing a single 0.8B local model with dynamic adapter swapping.
3. **Asset Pipeline Tool Suite**: An automated Blender/glTF validator script running as a first-class programmatic expert.

**Funding Request:** $50,000 (Unreal Engine C++ development, open demo creation, performance optimization).

---

## Strategic Action Plan

| Priority | Opportunity | Direct Link | Action Required | Expected Timeline |
|---|---|---|---|---|
| **P1** | **Emergent Ventures** | [Apply Here](https://www.mercatus.org/emergent-ventures) | Submit 4-question fast grant online. | 2–3 weeks decision |
| **P2** | **Hugging Face Hackathon Pitch** | [Contact HF](https://huggingface.co/contact) | Reach out to HF Developer Advocates / Spaces team. | 3–4 weeks planning |
| **P3** | **Lacuna Fund / AI4D Africa** | [Lacuna Apply](https://lacunafund.org/apply/) | Submit when the 2026/2027 call opens (or adapt to direct RFP). | 1–2 months cycle |
| **P4** | **Mozilla Technology Fund** | [Mozilla MTF](https://foundation.mozilla.org/en/what-we-fund/awards/mozilla-technology-fund/) | Submit next open cohort application. | Bi-annual cycle |
| **P5** | **Epic Games MegaGrants** | [MegaGrants Portal](https://www.unrealengine.com/en-US/megagrants) | Submit proposal via the Unreal Engine MegaGrants portal. | Rolling evaluations |
