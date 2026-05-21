=== TASK ===

Above you see the complete original conversation, labelled with `[SYSTEM]`, `[USER N]`, `[ASSISTANT N]` sections. Image placeholder tokens in those user messages refer, in reading order, to the images attached to this message.

For **each** `[ASSISTANT N]` turn (N starts at 1), output an augmented version wrapped in a `<turn>` block of exactly this shape:

```
<turn index="N">
<observation>{your 1–2 sentence description of what is visible in the relevant image(s)}</observation>
{your 1–3 sentence explanation of why the original action is correct, written as free-form prose with no tags}
<action>{byte-identical copy of the original [ASSISTANT N] turn's action content}</action>
</turn>
```

The three `{...}` spans are placeholders. Replace each with the actual content — **do not copy the placeholder text itself**. Write the explanation as prose directly after `</observation>` and before `<action>`. Do not introduce labels like "Reasoning:", "Thought:", etc.

Worked example for a hypothetical `[ASSISTANT N]` turn whose original content was `<action>turn_left</action>`:

```
<turn index="1">
<observation>The initial view shows a white cabinet on the right edge; the target view centers on the cabinet and shows the doorway to its left.</observation>
The cabinet needs to rotate from the right edge into the center, so the camera must yaw leftward; turn_left is the yaw-left primitive, matching the required rotation direction.
<action>turn_left</action>
</turn>
```

Hard rules (a checker will reject anything that breaks them):

1. Exactly one `<turn index="N">...</turn>` block per original assistant turn, in order (1, 2, 3, ...).
2. The `<action>...</action>` content must be **byte-identical** to the original turn's action content. Do not re-format, re-order, or rephrase it.
3. Each `<turn>` body must contain exactly one `<observation>...</observation>` followed by free-form prose followed by exactly one `<action>...</action>`. No other tags inside `<turn>`.
4. Keep each `<turn>` body under 1500 characters.
5. Output **only** `<turn>` blocks. No preamble, commentary, headers, or trailing text.

If you are asked to regenerate, output the FULL set of `<turn>` blocks again with the issues fixed.
