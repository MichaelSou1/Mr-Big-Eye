# Mr. Big-Eye eval report

## Global summary

| metric | value |
| --- | --- |
| Total | 60 |
| Passed | 44 |
| Pass rate | 0.733 |
| Retrieval pass | — |
| Answer pass | 0.733 |
| Agent loop pass | 1.000 |
| Recall@k mean | — |
| Timestamp dist mean | — |
| LLM judge score mean | 2.946 |
| LLM judge count | 56 |

## Per-group summary

| group | total | pass_rate | retrieval | answer | agent | recall@k | ts_dist | judge_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vme | 33 | 0.909 | — | 0.909 | 1.000 | — | — | 3.750 |
| worldsense | 27 | 0.519 | — | 0.519 | 1.000 | — | — | 1.875 |

## Failure tags

| tag | count |
| --- | --- |
| `wrong_fact_option` | 14 |
| `missing_expected_keyword` | 11 |
| `missing_citation_kind` | 10 |
| `missing_frame_or_slide` | 2 |
| `temporal_order_error` | 2 |
| `wrong_temporal_option` | 2 |

## Worst 10 failures

| case_id | failing section(s) | tags | question |
| --- | --- | --- | --- |
| `worldsense-rhiWCTTn-task0` | answer | missing_citation_kind, wrong_fact_option; selected=A; recommended=D | What is the most likely emotional tone of this piece of music?  Candidates: A) Calm and steady. B) Light and prolonge... |
| `worldsense-GtNxfwCV-task1` | answer | missing_citation_kind, wrong_fact_option; selected=D; recommended=A | How many times did the police car's siren sound in the video?  Candidates: A) One time. B) Three times. C) Zero times... |
| `worldsense-YFoSwjHZ-task0` | answer | missing_expected_keyword, wrong_fact_option; selected=C; recommended=D | Is the tuba on the left in the video louder than the one on the right?  Candidates: A) Yes, the sound from the tuba o... |
| `worldsense-WybMuwJr-task1` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, wrong_fact_option; selected=-; recommended=B | What celebratory actions are performed by the player wearing yellow number 21 and the player wearing yellow number 30... |
| `worldsense-pAZpYwBE-task1` | answer | missing_citation_kind, missing_expected_keyword, wrong_fact_option; selected=A; recommended=B | Who spells the word "business" in the video?  Candidates: A) A young man with light brown hair. B) A man with long bl... |
| `vme-381-3` | answer | missing_citation_kind, missing_frame_or_slide, wrong_fact_option; selected=A; recommended=B | Which year marked the debut of the typewriter introduced in the video?  Candidates: A) 1961. B) 1971. C) 1981. D) 1986. |
| `worldsense-dgrNkrce-task0` | answer | missing_citation_kind, missing_expected_keyword, wrong_fact_option; selected=A; recommended=C | Is there clapping sound in the video?  Candidates: A) No. B) I am not sure. C) Yes. |
| `worldsense-WybMuwJr-task0` | answer | missing_citation_kind, missing_expected_keyword, wrong_fact_option; selected=A; recommended=C | How many times does the whistle sound appear in the video?  Candidates: A) One. B) Six. C) Four. D) Two. |
| `worldsense-YFoSwjHZ-task2` | answer | missing_expected_keyword, wrong_fact_option; selected=C; recommended=B | How many loud voices can be heard in the video?  Candidates: A) Three. B) Two. C) One. D) None. |
| `vme-496-3` | answer | missing_citation_kind, wrong_fact_option; selected=A; recommended=C | What sentence best describes the performance?  Candidates: A) The magic is not wonderful enough to make the audiences... |
