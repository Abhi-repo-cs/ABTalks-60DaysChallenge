# Day 40 — A/B Testing Experiment Analysis Report

## 1. Executive Summary
This project evaluates a simulated A/B test comparing a control experience with an experimental variant. The primary KPI is conversion rate.

## 2. Experiment Design
- Control group: 5,000 users
- Experiment group: 5,000 users
- Primary metric: conversion rate
- Supporting metrics: revenue per user and session duration
- Statistical test: two-proportion z-test
- Significance level: α = 0.05

## 3. Statistical Method
For conversion rates p1 and p2, the pooled conversion rate is used under the null hypothesis:

z = (p2 - p1) / sqrt(p_pool(1-p_pool)(1/n1 + 1/n2))

A two-sided p-value below 0.05 is treated as statistically significant.

## 4. Results
Run `day40_ab_testing.ipynb` to calculate the exact conversion rates, absolute lift, relative lift, z-statistic, and p-value directly from the supplied dataset.

## 5. Business Decision Framework
If the experiment has a statistically significant positive conversion lift, recommend a controlled rollout while monitoring revenue per user, retention, and other guardrail metrics. If the result is not significant, continue testing or collect more observations rather than declaring a winner.

## 6. Important Caveats
Statistical significance does not automatically mean the effect is commercially meaningful. The experiment should also be evaluated for practical significance, sample-ratio mismatch, novelty effects, segment effects, and possible impact on downstream business metrics.

## 7. Conclusion
The repository demonstrates an end-to-end experimentation workflow: data preparation, KPI calculation, hypothesis testing, visualization, and business recommendation.
