# Mr. Big-Eye eval report

## Global summary

| metric | value |
| --- | --- |
| Total | 60 |
| Passed | 40 |
| Pass rate | 0.667 |
| Retrieval pass | — |
| Answer pass | 0.667 |
| Agent loop pass | 0.983 |
| Recall@k mean | — |
| Timestamp dist mean | — |
| LLM judge score mean | 2.583 |
| LLM judge count | 60 |

## Per-group summary

| group | total | pass_rate | retrieval | answer | agent | recall@k | ts_dist | judge_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vme | 33 | 0.758 | — | 0.758 | 0.970 | — | — | 3.485 |
| worldsense | 27 | 0.556 | — | 0.556 | 1.000 | — | — | 1.481 |

## Failure tags

| tag | count |
| --- | --- |
| `missing_expected_keyword` | 18 |
| `missing_citation_kind` | 13 |
| `wrong_fact_option` | 11 |
| `temporal_order_error` | 9 |
| `wrong_temporal_option` | 9 |
| `missing_frame_or_slide` | 3 |
| `no_final_answer` | 1 |

## Worst 10 failures

| case_id | failing section(s) | tags | question |
| --- | --- | --- | --- |
| `worldsense-zdcvGZEb-task0` | answer | missing_expected_keyword, temporal_order_error, wrong_temporal_option; selected=B; recommended=C | When does the man adjust the camera?  Candidates: A) At the beginning of the video. B) At the end of the video. C) In... |
| `vme-538-2` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, temporal_order_error, wrong_temporal_option; selected=-; recommended=B | When making the first dish in the video, what is the second food that is plated?  Candidates: A) Onion slice. B) Seaw... |
| `worldsense-YFoSwjHZ-task0` | answer | missing_citation_kind, wrong_fact_option; selected=C; recommended=D | Is the tuba on the left in the video louder than the one on the right?  Candidates: A) Yes, the sound from the tuba o... |
| `worldsense-ipBErquQ-task1` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, temporal_order_error, wrong_temporal_option; selected=B; recommended=D | In which part of the video do the catcher and pitcher change their strategies?  Candidates: A) At the beginning of th... |
| `vme-507-2` | answer | missing_citation_kind, missing_expected_keyword, missing_frame_or_slide, no_final_answer, temporal_order_error, wrong_temporal_option; selected=-; recommended=A | Based on the order of introduction in the video, which of the following options is introduced last?  Candidates: A) L... |
| `worldsense-yAvDdKyd-task0` | answer | missing_citation_kind, missing_expected_keyword, wrong_fact_option; selected=-; recommended=C | What instrument's sound can be heard in the video?  Candidates: A) French horn. B) Trumpet. C) Saxophone. D) Bassoon. |
| `vme-313-2` | answer | missing_citation_kind, missing_expected_keyword, wrong_fact_option; selected=C; recommended=B | What is the teacher's attitude toward the student when the student attempts to play middle C on the violin?  Candidat... |
| `worldsense-dgrNkrce-task0` | answer | missing_citation_kind, missing_expected_keyword, wrong_fact_option; selected=A; recommended=C | Is there clapping sound in the video?  Candidates: A) No. B) I am not sure. C) Yes. |
| `worldsense-WybMuwJr-task0` | answer | missing_expected_keyword, wrong_fact_option; selected=A; recommended=C | How many times does the whistle sound appear in the video?  Candidates: A) One. B) Six. C) Four. D) Two. |
| `vme-313-3` | answer | missing_citation_kind, missing_expected_keyword, wrong_fact_option; selected=D; recommended=A | Which of the following statements about the violin-learner in the video is true?  Candidates: A) He keeps improving h... |
