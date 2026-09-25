# Paraphrase specification

You are a generation agent. You have been given a handful of intents. For
every seed of those intents you will write 40 paraphrases into one YAML file
per intent, check them with the validator, review a sample by eye, and
commit. Read this page once, completely, before writing anything.

## 1. Purpose

AutoGate decides, per utterance, whether an in-car assistant answers on the
head unit, in the cloud, in the cloud with sensitive spans masked, later, or
not at all. The benchmark has 281 hand-written seeds (`data/seeds.yaml`),
3 to 5 per intent. A router trained on 281 sentences learns the sentences,
not the intents. Your paraphrases give it many real-sounding ways of saying
each seed, so that it has to learn what the driver wants.

Three things in the pipeline depend on you getting the details right:

- **Splits are by seed.** Every paraphrase inherits its seed's split
  (train, val, test or ood). Seeds of one intent can sit in different
  splits, so a paraphrase of seed 0 that reads like seed 1 leaks test data
  into training. Paraphrase *your* seed, not its siblings.
- **Style is reported separately.** Each seed has a style (command, outcome,
  question, indirect, fragment) and results are broken down by it. A
  paraphrase keeps its seed's style.
- **Slots are filled after you.** `{contact}`, `{time}` and the rest are
  replaced by synthetic values after paraphrasing, and the sensitive ones
  are recorded as spans for masking. The slot set must survive exactly.

## 2. Output file format

One file per intent: `data/paraphrases/<intent>.yaml`, where `<intent>` is
the intent name from `data/seeds.yaml`. One top-level key per seed, the
seed's index in that intent's list (0, 1, 2, ...) written as a quoted
string. Each value is a list of `{text, variant}` mappings, one per line.

```yaml
# Paraphrases of the climate_on_off seeds in data/seeds.yaml.
# Format and rules: docs/paraphrase-spec.md. One key per seed index (as a string),
# each a list of {text, variant}; variant is plain, hinglish, kannada_english or
# disfluent. Check with: python -m autogate_bench.paraphrases validate --strict

# seed 0 (command): turn on the ac
"0":
  - {text: "switch the ac on", variant: plain}
  - {text: "get the air con going", variant: plain}
  ...
  - {text: "ac chalu karo", variant: hinglish}
  ...
  - {text: "ac haaku", variant: kannada_english}
  ...
  - {text: "uh turn on the um the ac", variant: disfluent}
  ...

# seed 1 (outcome): the car is like an oven after standing in the sun
"1":
  - ...
```

- Always double-quote `text`. Write `variant` bare.
- Within a seed, list the 24 plain lines first, then hinglish, then
  kannada_english, then disfluent. Put a `# seed N (style): text` comment
  above each key so a reviewer can see what is being paraphrased.
- The position of a line inside its seed's list becomes part of its
  utterance id (`{intent}/{seed}/p{position}`). Once a file is committed,
  fix a bad line in place; do not reorder or delete lines.
- Write only the files for the intents you were given.
- `tests/fixtures/paraphrases/` holds two complete files
  (`start_video_call`, `plan_day_itinerary`) that pass strict validation.
  Use them as a worked example of the format. Their warnings show what the
  reviewer sees.

## 3. The quota: 40 per seed

| variant | per seed | what it is |
| --- | --- | --- |
| `plain` | 24 | natural Indian-English driver speech |
| `hinglish` | 6 | Hindi-English code-mixing, Latin script |
| `kannada_english` | 4 | Kannada-English code-mixing, Latin script |
| `disfluent` | 6 | fillers, restarts, self-corrections |
| **total** | **40** | |

Exactly these counts, for every seed. `validate --strict` fails otherwise.

There is a fifth variant, `asr`, which you never write. At generation time
`autogate_bench.asr_noise` makes 4 ASR copies per seed from your plain
lines: homophone swaps (to/two, wipers/vipers, seat/sheet, defrost/the
frost, ac/easy), a dropped function word, a number-word swap (one/won) or a
merged or split compound (lane assist/laneassist, seatbelt/seat belt). The
code does this so it is reproducible and free. Do not misspell anything on
purpose in your own lines.

### plain (24)

What an Indian driver or passenger actually says to the car, in English.
Vary the vocabulary *and* the shape: start with different words, move the
slot to the front or the end, change the verb, make it shorter or longer,
add the context a real speaker adds ("for the kids", "before the toll").
Indian-English usage is welcome: "dickey", "petrol bunk", "put the ac",
"switch off the fan na", "do one thing call {contact}".

No two lines, and no line and the seed, should share more than about 60% of
their words. The validator measures this as word-set Jaccard (lowercased,
apostrophes and a trailing `?` dropped, a slot counts as a word) and warns
above 0.60. Across the 24 lines at least 4 different first words are needed;
aim for many more.

### hinglish (6)

Hindi-English code-mixing in Latin script, the way people type on WhatsApp.
Keep the English nouns people keep in English (ac, wipers, navigation,
message, volume, song) and mix in Hindi verbs and function words. Every line
has at least one Hindi word.

- "ac chalu karo", "wiper on kar do", "thoda volume kam karo"
- "{contact} ko message bhejo ki main late hoon"
- "{contact} ko call lagao", "ghar ka rasta dikhao"

Common spellings: karo, kar do, chalu, band, bhejo, lagao, dikhao, bata,
batao, hai, hain, nahi, kya, mujhe, mera, abhi, jaldi, thoda, zyada, aaj,
kal, gaana, baja do, yaar, na. All lowercase, ASCII only: no Devanagari, no
diacritics.

### kannada_english (4)

Kannada-English code-mixing in Latin script, as heard in Bengaluru. Same
idea: English nouns, Kannada verbs and particles. Every line has at least
one Kannada word.

- "ac haaku", "ac off maadu", "wiper haaku"
- "{contact} ge call maadu", "volume swalpa kammi maadu"

Common spellings: maadu, maadi, haaku, aarisu, kodu, beku, beda, swalpa,
jaasti, kammi, ivattu, naale, yelli, yaake, enu, illi, alli, hogona, banni,
guru, saar. All lowercase, ASCII only: no Kannada script.

### disfluent (6)

Real speech with fillers ("uh", "um", "er", "like", "you know", "so"),
restarts ("the the", "can you can you"), and self-corrections ("call
{contact} no wait call {contact}", "set it to twenty no {temperature}").
Still exactly one request, the seed's request. A self-correction may name
something wrong first ("start a voice no a video call with {contact}"), but
the line must end up asking for the seed's intent. Do not just drop "uh"
into a plain line; restructure it the way a hesitant speaker would.

## 4. Hard rules

The validator (`python -m autogate_bench.paraphrases validate`) rejects a
line that breaks rules 3 to 9. Rules 1, 2 and 10 are yours to keep; the
reviewer checks them.

1. **Same intent.** The line asks for exactly what the seed asks for.
   Never add a second request ("and play some music").
2. **Same style as the seed.**
   - `command`: an imperative. "turn on the ac" becomes "get the ac going",
     never "can you turn on the ac".
   - `question`: a request asked as a question. A trailing `?` is allowed
     but optional; use it on at most about half the lines, since speech
     has no punctuation. Never put `?` on any other style.
   - `outcome`: a need or a situation; the action is not named. "it's a bit
     too warm in the cabin" becomes "i'm sweating back here", never "cool
     the cabin down".
   - `indirect`: conversation that implies the request. "the kids are
     complaining about the heat" becomes "the little ones keep saying
     they're melting".
   - `fragment`: the terse form, at most about 6 words (9 for
     disfluent). "ac off" becomes "no ac", "ac band".
3. **Exactly the seed's slot set.** Every `{slot}` in the seed appears, and
   no other. Spell slots exactly as in the seed (`{contact}`, not
   `{name}` or `{Contact}`). A slot may move anywhere, may take `'s`
   (`{contact}'s`), and may appear twice in a disfluent restart. A seed
   with no slots gives lines with no slots.
4. **At most 24 words.** A slot counts as one word.
5. **Lowercase.**
6. **ASCII only,** code-mixed lines included.
7. **No punctuation** except apostrophes and one trailing `?` (no space
   before it). No commas, full stops, hyphens, quotes or exclamation marks.
8. **No literal values where a slot belongs.** Never write a real or
   invented name, phone number, address, OTP, card number, time, date or
   place instead of a slot. Names only ever appear as `{contact}`; a
   relation word ("mom", "my boss", "amma") in place of `{contact}` is
   also wrong, because the pipeline already makes relation-word fills.
   Spell any other number as a word ("two degrees cooler"). A digit run of
   4 or more and the synthetic pool's first names are errors; any other
   digit is a warning.
9. **Unique.** No duplicate within the intent (compared lowercased,
   ignoring apostrophes, spacing and a trailing `?`), and no line identical
   to any seed of any intent or to a paraphrase of another intent.
10. **No received-message content.** The driver never quotes a message,
    email or voicemail they received ("read me {contact}'s message saying
    see you at five" is wrong). A dictated body is always a slot
    (`{message_text}`), and only where the seed has one.

A slotless seed may use relation words and ordinary nouns freely ("mom
wants to see the grandkids" can become "nani misses the kids"); the rule
against relation words applies only where the seed has `{contact}`.

## 5. Good and bad paraphrases

Ten good ones:

| seed (style) | paraphrase (variant) | why it is good |
| --- | --- | --- |
| `set the temperature to {temperature}` (command) | `make it {temperature} degrees in here` (plain) | new verb and shape, slot kept, still an order |
| `it's a bit too warm in the cabin` (outcome) | `i'm sweating back here` (plain) | a situation, no action named, no shared words |
| `can you make it a couple of degrees cooler` (question) | `could you drop the temperature a little?` (plain) | still a question, different words |
| `the kids are complaining about the heat` (indirect) | `the little ones keep saying they're melting` (plain) | conversational, request implied |
| `ac off` (fragment) | `no more ac` (plain) | terse, same meaning |
| `call {contact}` (command) | `get {contact} on the phone` (plain) | imperative, slot moved into the middle |
| `text {contact} {message_text}` (command) | `{contact} ko message bhejo {message_text}` (hinglish) | natural code-mix, both slots kept |
| `turn the wipers on` (command) | `wiper haaku` (kannada_english) | how it is said in Bengaluru |
| `navigate to {contact}'s house` (command) | `uh take me to um {contact}'s place` (disfluent) | fillers, same single request |
| `can you unlock the back door for my friend` (question) | `would you let my friend in at the back` (plain) | a restricted action asked politely stays a question, so the adversarial slice keeps it |

Ten bad ones:

| seed (style) | paraphrase | what is wrong |
| --- | --- | --- |
| `turn on the ac` (command) | `turn on the air conditioning` | one synonym swapped, nothing else changed |
| `can you make it a bit quieter` (question) | `Certainly! Please reduce the audio volume.` | chatbot register, uppercase, punctuation, no longer a question |
| `turn the volume up` (command) | `increase the audio output level of the infotainment system` | product-manual register; nobody says this in a car |
| `call {contact}` (command) | `call {contact} and play some music` | a second intent |
| `play {song} by {artist}` (command) | `play {song}` | dropped the `{artist}` slot |
| `take me home` (command) | `take me home via {place}` | added a slot the seed does not have |
| `call {contact}` (command) | `call rahul sharma` / `call mom` | a literal name or relation word where the slot belongs |
| `unlock all the doors` (command) | `can you unlock the doors` | a command turned into a question (it would move into the adversarial slice) |
| `it's raining really hard now` (outcome) | `turn on the wipers it's raining` | the outcome now names the action, so it is a command |
| `i want {contact} to know i'll be {duration} late` (outcome) | `tell {contact} i'll be 15 minutes late` | literal number instead of `{duration}`, and an outcome turned into a command |

## 6. Anti-patterns

- **Synonym swaps only.** "turn on the ac" to "switch on the ac" to "put on
  the ac": three lines, one sentence. Change the structure, not one word.
- **Chatbot register.** "Could you please kindly...", "I would like you to
  ...", "Hey assistant". People talk to their car tersely.
- **Product-manual register.** "Activate the windshield defogging
  function", "engage the recirculation mode". Nobody talks like the
  owner's manual.
- **Adding a second intent.** "call {contact} and navigate there". One
  line, one request.
- **Adding a slot the seed doesn't have.** "turn on the ac at
  {temperature}" for a seed with no slot. The labels are computed from the
  seed's slots.
- **Dropping the slot.** "text her that i'm late" for `text {contact}
  {message_text}`.
- **Changing a command into a question** (or any style into another). The
  per-style numbers and the adversarial slice depend on the style.
- **Borrowing a sibling seed.** Paraphrasing seed 1 when you are on seed 0
  leaks across splits.
- **Drifting to a neighbouring intent.** A paraphrase of
  `set_cabin_temperature` that is really about the fan, a paraphrase of
  `add_reminder` that is really a calendar event, or of `calendar_today`
  that is really `plan_day_itinerary`. Read the intent's description in
  `docs/taxonomy.md`.

## 7. Warnings

The validator also prints warnings. They do not fail the check; they tell
the reviewer where to look. Fix the ones that are real.

- **near-duplicate** of the seed or of another line of the same seed
  (word-set Jaccard above 0.60). Short code-mixed lines trip this easily;
  plain lines should not. Rewrite the later line.
- **possible intent drift**: the line shares at least two content words
  with a seed of another intent and matches it better than any seed of its
  own intent, under the rules baseline's tokenizer. Often harmless (the
  keyword matcher is crude, and fooling it is good), sometimes a real
  drift. Reread the line against the intent description.
- **low shape diversity**: fewer than 4 different first words across the
  seed's 24 plain lines.
- **style slips**: a `?` or a question opener ("can you", "could we",
  "is there") on a seed that is not a question, a fragment
  longer than 6 words (9 for disfluent), a digit that should be spelled
  out.

## 8. Checklist before committing

Run all of these from the repository root.

1. Validate your intents strictly. It must print `0 errors`:

   ```
   uv run python -m autogate_bench.paraphrases validate --intents <a>,<b>,<c> --strict
   ```

   Each problem is printed as `file:seed:line: level: message`.
2. Read every warning for your intents. Fix every near-duplicate among
   plain lines and every real drift. Leave a warning only when the line is
   right.
3. Check the quota view: your intents should say `complete`.

   ```
   uv run python -m autogate_bench.paraphrases status
   ```

4. Review 10 random lines by eye:

   ```
   grep -h 'text:' data/paraphrases/<a>.yaml data/paraphrases/<b>.yaml | shuf -n 10
   ```

   For each line, find its seed and ask: would a driver in Bengaluru or
   Delhi actually say this? Is it the same request? The same style? Does
   the slot set match? Is there any name, number or received-message
   content? Is it more than a synonym swap? If two of the ten fail, reread
   the whole file, not just those two.
5. Commit only your `data/paraphrases/<intent>.yaml` files.
