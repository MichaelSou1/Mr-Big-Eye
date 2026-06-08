# Mr. Big-Eye eval report

## Global summary

| metric | value |
| --- | --- |
| Total | 61 |
| Passed | 39 |
| Pass rate | 0.639 |
| Retrieval pass | — |
| Answer pass | 0.639 |
| Agent loop pass | 1.000 |
| Recall@k mean | — |
| Timestamp dist mean | — |
| LLM judge score mean | 2.407 |
| LLM judge count | 59 |

## Per-group summary

| group | total | pass_rate | retrieval | answer | agent | recall@k | ts_dist | judge_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vme | 30 | 0.800 | — | 0.800 | 1.000 | — | — | 3.233 |
| worldsense | 31 | 0.484 | — | 0.484 | 1.000 | — | — | 1.552 |

## Failure tags

| tag | count |
| --- | --- |
| `wrong_fact_option` | 19 |
| `missing_citation_kind` | 16 |
| `missing_expected_keyword` | 14 |
| `missing_frame_or_slide` | 3 |
| `no_final_answer` | 3 |
| `temporal_order_error` | 3 |
| `wrong_temporal_option` | 3 |
| `missing_slide` | 1 |

## Worst 10 failures

| case_id | failing section(s) | tags | question |
| --- | --- | --- | --- |
| `vme-354-1` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, wrong_fact_option; selected=A; recommended=C | How does the speaker introduce the fact that the Mediterranean might disappear for a long while?  Candidates: A) MSC.... |
| `vme-424-1` | answer | missing_citation_kind, missing_frame_or_slide, wrong_fact_option; selected=A; recommended=A | According to the video, which of the following countries has the lowest birth rate?  Candidates: A) South Korea. B) U... |
| `vme-455-1` | answer | missing_expected_keyword, wrong_fact_option; selected=B; recommended=D | What are the players doing before the match formally begins, according to the video?  Candidates: A) Exchanging team ... |
| `vme-558-1` | answer | missing_expected_keyword, wrong_fact_option; selected=B; recommended=D | What is the purpose of the high table placed at the back of the classroom in the video?  Candidates: A) Used to displ... |
| `vme-564-1` | answer | missing_expected_keyword, wrong_fact_option; selected=C; recommended=A | What is the underlying reason why the main character in the video takes off his shoes when he gets into the car?  Can... |
| `vme-588-1` | answer | missing_citation_kind, missing_slide, wrong_fact_option; selected=B; recommended=D | What weight did the first girl in the video's female group attempt to bench press before failing?  Candidates: A) 150... |
| `worldsense-GtNxfwCV-task1` | answer | missing_citation_kind, wrong_fact_option; selected=C; recommended=A | How many times did the police car's siren sound in the video?  Candidates: A) One time. B) Three times. C) Zero times... |
| `worldsense-Pirddbfp-task0` | answer | missing_citation_kind, missing_expected_keyword, temporal_order_error, wrong_temporal_option; selected=C; recommended=A | At what point in the video does the woman in black start playing the violin?  Candidates: A) In the middle of the vid... |
| `worldsense-RlonNikG-task0` | answer | missing_expected_keyword, wrong_fact_option; selected=B; recommended=D | How many types of sounds are heard in the video?  Candidates: A) Two. B) One. C) Four. D) Three. |
| `worldsense-WBtKutyY-task0` | answer | missing_citation_kind, wrong_fact_option; selected=A; recommended=B | In the video, which instrument is the first to make a sound?  Candidates: A) A violin. B) A cello. C) A piano. D) An ... |
