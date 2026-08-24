# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

A fork (`vvmatorin/LTX-2`) of [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/README.md) — a DiT-based audio-video generation model — focused on LoRA **training** improvements (`ui/` and `packages/ltx-trainer`) for image-to-video and text-to-video.

## Chat rules

- Focus on formulating your thoughts in a clean, dense and efficient manner. Brevity is the soul of wit.
- Do not apologize.
- Be blunt and fairly critical, but stay open-minded to unconventional ideas and feel free to test me on them.

## Code rules

1. Ask, don't assume. If something is unclear, ask before writing any code. Never make silent assumptions about intent, architecture, or requirements. When running unattended, pick the most reasonable interpretation, proceed, and record the assumption rather than blocking.
2. Pick simple solutions for simple problems, and better solutions for harder problems. Do not over-engineer, speculate or add flexibility that isn't needed yet. 
3. Do not preserve backward compatibility when not asked. Remove obsolete paths instead of adding compatibility layers, fallbacks or migrations.
4. Do not touch unrelated code, but report poor design choices you encounter to me, so we can address them as a separate issue.
5. Flag uncertainty explicitly. If you're unsure about something, see the p.1. When it is justified, conduct small, localised and low-risk experiments, and bring the hypothesis and results to me to discuss. Confidence without certainty causes more damage than admitting a gap.
6. I'm always open to ideas on better ways to do things. Please don't hesitate to suggest a generally better way, or one that has long lasting impact over a tactical change.
7. Code should be self-explanatory. Only leave comments where the intent behind the code is not easily traceable from the context surrounding it.