You are a **reasoning annotator** for an embodied-vision SFT dataset.

You will be shown one trajectory of a multi-turn navigation conversation between a USER (who describes the task and shows images) and an ASSISTANT (who emits short `<action>...</action>` replies). The conversation is truncated so its **last** `[ASSISTANT N]` block is the **target turn** — the action whose reasoning you will explain. Earlier `[ASSISTANT i]` turns are shown only as trajectory history; do **not** re-annotate them.

Some intermediate user turns will appear with no `<image>` placeholders — those are turns whose images we have intentionally hidden so you focus on the views relevant to the target decision. Use the camera state described in their text plus the visible images of the kept turns.

Your job: produce **one** assistant reply for the target turn. The reply must contain, in order:
- a brief `<observation>...</observation>` describing what is visible in the relevant image(s) for this decision
- a short free-form thought (no tags) explaining *why* the target action is correct given the observation
- the target action wrapped in `<action>...</action>`, byte-identical to what the target turn already shows

Output strictly the body — no `<turn>` tags, no preamble, no commentary. Keep it short and on-topic.
