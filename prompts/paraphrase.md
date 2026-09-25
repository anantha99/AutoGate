You are writing training data for AutoGate, a benchmark for in-car voice assistants in India. Rewrite one seed utterance into paraphrases that a driver or passenger might say to the car. The long-form rules are in docs/paraphrase-spec.md; this prompt is the short form.

## The seed

- intent: `{intent}` ({group}): {description}
- seed: `{seed_text}`
- style: `{style}`
- slots: {slots}

## What to write

Exactly {quota_plain} `plain`, {quota_hinglish} `hinglish`, {quota_kannada_english} `kannada_english` and {quota_disfluent} `disfluent` paraphrases, in that order.

- `plain`: natural Indian-English speech. Vary the vocabulary and the sentence shape (start with different words, move the slot around, change the verb). No two lines, and no line and the seed, should share more than about 60% of their words.
- `hinglish`: Hindi-English code-mixing in Latin script, spelled the common WhatsApp way ("AC chalu karo", "{contact} ko message bhejo ki main late hoon", but lowercase).
- `kannada_english`: Kannada-English code-mixing in Latin script, common Bengaluru usage ("ac haaku", "{contact} ge call maadu").
- `disfluent`: fillers, restarts and self-corrections ("uh can you um turn the ac on", "call {contact} no wait call {contact}"). Still one request, the same one.

## Hard rules (every line)

1. Same intent as the seed. Never add a second request.
2. Same style as the seed. A `command` stays an imperative, a `question` stays a question, an `outcome` states a need or situation without naming the action, an `indirect` line implies the request conversationally, and a `fragment` stays a terse few words.
3. Exactly the seed's slots: every `{slot}` in the seed appears, written exactly as in the seed, and no other `{slot}` appears. A slot may appear twice in a disfluent restart. If the seed has no slots, write none.
4. Never write a literal name, phone number, address, OTP, card number, time or place where the seed has a slot. Spell other numbers as words. No received-message content: the driver never quotes a message they received.
5. At most 24 words, all lowercase, ASCII only (code-mixed lines too), no punctuation except apostrophes and an optional trailing question mark on question-style lines.
6. No chatbot or product-manual register ("please could you kindly", "activate the climate control function"): this is speech in a moving car.
7. No duplicates, and no line that is only the seed with one synonym swapped.

## Output

Return only a JSON object matching this schema:

```json
{"type": "object", "properties": {"paraphrases": {"type": "array", "items": {"type": "object", "properties": {"text": {"type": "string"}, "variant": {"type": "string", "enum": ["plain", "hinglish", "kannada_english", "disfluent"]}}, "required": ["text", "variant"], "additionalProperties": false}}}, "required": ["paraphrases"], "additionalProperties": false}
```
