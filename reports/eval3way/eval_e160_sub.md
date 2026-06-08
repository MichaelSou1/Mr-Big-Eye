# Mr. Big-Eye eval report

## Global summary

| metric | value |
| --- | --- |
| Total | 60 |
| Passed | 43 |
| Pass rate | 0.717 |
| Retrieval pass | — |
| Answer pass | 0.717 |
| Agent loop pass | 0.983 |
| Recall@k mean | — |
| Timestamp dist mean | — |
| LLM judge score mean | 3.017 |
| LLM judge count | 60 |

## Per-group summary

| group | total | pass_rate | retrieval | answer | agent | recall@k | ts_dist | judge_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vme | 33 | 0.788 | — | 0.788 | 0.970 | — | — | 3.273 |
| worldsense | 27 | 0.630 | — | 0.630 | 1.000 | — | — | 2.704 |

## Failure tags

| tag | count |
| --- | --- |
| `wrong_fact_option` | 13 |
| `missing_expected_keyword` | 12 |
| `missing_citation_kind` | 11 |
| `missing_frame_or_slide` | 6 |
| `temporal_order_error` | 4 |
| `wrong_temporal_option` | 4 |
| `missing_slide` | 2 |
| `no_final_answer` | 2 |

## Worst 10 failures

| case_id | failing section(s) | tags | question |
| --- | --- | --- | --- |
| `worldsense-GtNxfwCV-task1` | answer | missing_citation_kind, missing_expected_keyword, wrong_fact_option; selected=-; recommended=A | How many times did the police car's siren sound in the video?  Candidates: A) One time. B) Three times. C) Zero times... |
| `worldsense-zdcvGZEb-task0` | answer | missing_citation_kind, missing_frame_or_slide, temporal_order_error, wrong_temporal_option; selected=B; recommended=C | When does the man adjust the camera?  Candidates: A) At the beginning of the video. B) At the end of the video. C) In... |
| `worldsense-YFoSwjHZ-task0` | answer | missing_expected_keyword, wrong_fact_option; selected=C; recommended=D | Is the tuba on the left in the video louder than the one on the right?  Candidates: A) Yes, the sound from the tuba o... |
| `worldsense-WybMuwJr-task1` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, wrong_fact_option; selected=-; recommended=B | What celebratory actions are performed by the player wearing yellow number 21 and the player wearing yellow number 30... |
| `vme-382-1` | answer | missing_citation_kind, missing_frame_or_slide, wrong_fact_option; selected=D; recommended=D | According to the video, why is IMAX not suitable for filming movies?  Candidates: A) The motors make too much noise a... |
| `vme-313-2` | answer | missing_expected_keyword, wrong_fact_option; selected=C; recommended=B | What is the teacher's attitude toward the student when the student attempts to play middle C on the violin?  Candidat... |
| `worldsense-WybMuwJr-task0` | answer | missing_expected_keyword, wrong_fact_option; selected=A; recommended=C | How many times does the whistle sound appear in the video?  Candidates: A) One. B) Six. C) Four. D) Two. |
| `vme-568-1` | answer | missing_citation_kind, missing_expected_keyword, missing_slide, no_final_answer, wrong_fact_option; selected=-; recommended=B | How far does the main character in the video travel in total?  Candidates: A) 254,000 miles. B) 3,224 miles. C) 37 mi... |
| `worldsense-YFoSwjHZ-task2` | answer | missing_expected_keyword, wrong_fact_option; selected=C; recommended=B | How many loud voices can be heard in the video?  Candidates: A) Three. B) Two. C) One. D) None. |
| `worldsense-jHlldnZG-task1` | answer | missing_expected_keyword, temporal_order_error, wrong_temporal_option; selected=B; recommended=A | At which point in the video does the guitar's sound disappear?  Candidates: A) In the middle of the video. B) At the ... |
