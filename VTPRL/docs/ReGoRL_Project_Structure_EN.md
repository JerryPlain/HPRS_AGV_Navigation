# ReGoRL Project Structure

## Overview

This project evolves an offline-to-online robotic reinforcement learning stack for safety-critical warehouse navigation into a broader research direction:

**ReGoRL: Reward Governance for Safety-Critical Robot Learning via Hierarchy-Guided Patches**

The repository currently contains:

- an AGV warehouse navigation pipeline,
- HPRS-based structured reward shaping,
- an LLM-assisted outer loop for constrained reward updates,
- and the technical foundation for extending the system into ReGoRL.

This document gives an English, diagram-first view of the full project structure.

## 1. Repository Map

```mermaid
flowchart TB
    root["HPRS_AGV_Navigation"]

    subgraph vt["VTPRL"]
        docs["docs<br/>methodology, repo guide,<br/>ReGoRL paper notes"]
        agent["agent<br/>training, evaluation,<br/>tools, logs"]
        shaping["auto-shaping<br/>HPRS specs, parser,<br/>reward wrappers"]
        sim["environment/simulator<br/>Unity binaries"]
        docker["Docker<br/>runtime and setup"]
        ext["external/stable-baselines3<br/>vendor dependency"]
        tests["testcases<br/>evaluation scenarios"]
        req["requirements_llm.txt"]
        readme["README.md"]
    end

    root --> vt
    vt --> readme
    vt --> docs
    vt --> agent
    vt --> shaping
    vt --> sim
    vt --> docker
    vt --> ext
    vt --> tests
    vt --> req

    classDef group fill:#f8fafc,stroke:#334155,stroke-width:1.5px,color:#0f172a;
    classDef core fill:#e0f2fe,stroke:#0369a1,stroke-width:1.5px,color:#082f49;
    classDef reward fill:#ecfccb,stroke:#4d7c0f,stroke-width:1.5px,color:#365314;
    classDef runtime fill:#ffedd5,stroke:#c2410c,stroke-width:1.5px,color:#7c2d12;
    classDef support fill:#f3e8ff,stroke:#7e22ce,stroke-width:1.5px,color:#581c87;

    class vt group;
    class agent core;
    class shaping reward;
    class sim,docker runtime;
    class docs,ext,tests,req,readme support;
```

## 2. Functional Architecture

```mermaid
flowchart LR
    subgraph data["Data and Training Pipeline"]
        expert["Expert Collection<br/>A* + DWA"]
        offline["Offline RL<br/>TD3-BC"]
        online["Online RL<br/>TD3 / warm start"]
        eval["Evaluation and Comparison"]
    end

    subgraph reward["Reward and Governance Layer"]
        hprs["HPRS Specification<br/>safety > target > comfort"]
        llm["LLM Patch Proposal"]
        verify["Validation / Acceptance"]
        logs["Logged Patch Trace<br/>accept / reject / rollback"]
    end

    subgraph infra["Infrastructure"]
        unity["Unity Simulator"]
        vec["VecEnv + Wrappers"]
        config["YAML Configs"]
    end

    expert --> offline --> online --> eval
    hprs --> vec
    llm --> verify --> logs
    online --> llm
    unity --> vec --> online
    config --> expert
    config --> offline
    config --> online
    config --> eval
    hprs --> verify

    classDef dataStyle fill:#dbeafe,stroke:#1d4ed8,stroke-width:1.5px,color:#1e3a8a;
    classDef rewardStyle fill:#dcfce7,stroke:#15803d,stroke-width:1.5px,color:#14532d;
    classDef infraStyle fill:#fff7ed,stroke:#ea580c,stroke-width:1.5px,color:#7c2d12;

    class expert,offline,online,eval dataStyle;
    class hprs,llm,verify,logs rewardStyle;
    class unity,vec,config infraStyle;
```

## 3. Internal Code Structure

```mermaid
flowchart TD
    agent["agent/"]

    subgraph a1["Training"]
        expert_mod["expert/<br/>expert trajectory generation"]
        td3bc["td3/td3bc/<br/>offline TD3-BC"]
        online_td3["td3/online_td3/<br/>online TD3"]
    end

    subgraph a2["Environment"]
        envs["envs/<br/>Gym/Unity interfaces"]
        vecenv["simulator_vec_env.py"]
        eval_agent["evaluate_agent.py"]
    end

    subgraph a3["Governance Tools"]
        summarize["llm_summarize_run.py"]
        propose["llm_propose_patch.py"]
        apply["llm_apply_patch.py"]
        loop["run_online_llm_loop.py"]
        compare["compare_models_with_hprs.py"]
    end

    subgraph a4["Artifacts"]
        logs["logs/<br/>models, runs,<br/>llm_runs, comparisons"]
        configs["config/<br/>experiment YAMLs"]
    end

    agent --> a1
    agent --> a2
    agent --> a3
    agent --> a4

    expert_mod --> td3bc --> online_td3
    envs --> vecenv --> online_td3
    online_td3 --> summarize --> propose --> apply --> loop
    loop --> compare
    configs --> expert_mod
    configs --> td3bc
    configs --> online_td3
    configs --> compare
    loop --> logs
    compare --> logs

    classDef train fill:#e0f2fe,stroke:#0284c7,stroke-width:1.5px,color:#0c4a6e;
    classDef env fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#78350f;
    classDef tool fill:#dcfce7,stroke:#16a34a,stroke-width:1.5px,color:#14532d;
    classDef artifact fill:#f3e8ff,stroke:#9333ea,stroke-width:1.5px,color:#581c87;

    class expert_mod,td3bc,online_td3 train;
    class envs,vecenv,eval_agent env;
    class summarize,propose,apply,loop,compare tool;
    class logs,configs artifact;
```

## 4. ReGoRL Method Positioning

```mermaid
flowchart TB
    train["Robotic RL Training Loop<br/>policy updates, replay buffer, diagnostics"]
    trigger{"Governance Trigger<br/>stagnation or safety degradation"}

    subgraph rego["ReGoRL Governance Module"]
        spec["Hierarchical Reward Specification"]
        space["Hierarchy-Guided Patch Space"]
        proposer["Patch Proposer"]
        verifier["Verifier"]
        critic["Safety Critic"]
        selector["Hierarchy-Aware Selector"]
        decision["Accept / Reject / Rollback"]
    end

    update["Updated Reward Specification"]

    train --> trigger
    trigger --> spec
    spec --> space --> proposer --> verifier --> critic --> selector --> decision --> update
    update --> train

    classDef loopStyle fill:#dbeafe,stroke:#1d4ed8,stroke-width:1.5px,color:#1e3a8a;
    classDef govStyle fill:#dcfce7,stroke:#15803d,stroke-width:1.5px,color:#14532d;
    classDef decisionStyle fill:#fee2e2,stroke:#dc2626,stroke-width:1.5px,color:#7f1d1d;

    class train,trigger,update loopStyle;
    class spec,space,proposer,verifier,critic,selector govStyle;
    class decision decisionStyle;
```

## 5. Directory Roles

- `agent/`: the operational core of the project, including expert collection, offline RL, online RL, evaluation, and LLM-driven orchestration.
- `auto-shaping/`: the structured reward layer, including HPRS specifications, parsing, and reward injection wrappers.
- `environment/simulator/`: Unity simulator binaries used for robot-environment interaction.
- `docs/`: methodology notes, repository guides, and ReGoRL paper-oriented drafts.
- `Docker/`: environment setup for reproducible execution.
- `testcases/`: benchmark and evaluation case definitions.

## 6. Why This Structure Matters

The repository is not just a training codebase. It is organized around a broader research claim:

- the robot policy is optimized in the inner loop,
- the reward specification is revised in the outer loop,
- and ReGoRL turns that outer loop into a governed, auditable, hierarchy-aware decision process.

That is the conceptual bridge from the current `VTPRL` implementation to the full `ReGoRL` paper direction.
