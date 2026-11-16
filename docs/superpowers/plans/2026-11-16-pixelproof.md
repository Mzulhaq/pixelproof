# PixelProof Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Publish one verified trained-model API project for the week of November16,2026.

**Architecture:** Fixed train/test experiment -> validated JSON coefficients -> immutable pure-Python inference -> bounded FastAPI endpoint. A frozen holdout checks exported-runtime agreement, baseline comparison and error analysis.

**Tech Stack:** Python3.11+, scikit-learn(training), FastAPI/uvicorn(runtime), pytest,Ruff,Docker,GitHubActions.

**Spec:** docs/design.md. User delegated sensible defaults and full execution; no additional design approval needed.

## Constraints and review focus

- 64integerpixels0..16; batch1..32; HTTP32KiB; model256KiB; no secrets/paidservices/private data.
- Fixed splitseed42 andtestfraction.25; no tuningagainsttest.
- Malformed/nonfinite/duplicateJSON values must fail before inference.
- Artifactdigest/schema/dimensions constrain safe loading; no pickle.
- Test data neverparticipates in fitting/baseline selection.
- Exported coefficients must match sklearn on every heldoutsample.
- Blank/noise probes expose limitation without falsegroundtruthlabels.

## Task1: training worker

Files training.py,scripts/train.py,test_training.py; interfaces in spec.
- [x] Write failing partition/artifact/baseline tests, implement fixedexperiment, run ownedchecks.
- [x] Generate candidate bundle once, record fixedparameters/actualmetrics and attribution.

## Task2: runtime primary

Files model.py,api.py,cli.py andtests.
- [x] Write failing artifact/schema/digest/pixelvalidation/math tests, implement immutablemodel.
- [x] Write failing HTTP/CLI tests, implement boundedjsonserving and localentrypoint.

## Task3: documentation worker

Files docs/interview-guide.md,docs/career-fit.md,data/README.md,LICENSE.
- [x] Explain sources/limits/provenance, architecture tradeoffs, exercise andresume drafts.

## Task4: integrate and publish

- [x] Verify actualtrainedbundle/runtimeagreement, realHTTPdemo,frozenevaluation,packages andlocks.
- [x] Independentreview,fixmaterialfindings; finaltests/lint andpublication scan.
- [x] Published the public repository once and verified all five source CI jobs. Final documentation commit verification and local ledger completion are recorded by the delivery run.
