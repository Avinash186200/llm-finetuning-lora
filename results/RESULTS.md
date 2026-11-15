# Results

Model: `HuggingFaceTB/SmolLM2-135M-Instruct`. Hardware: CPU (16 threads). torch 2.15.0+cpu. Greedy decoding. Data seed 1234. Latency is batch wall time divided by batch size.

## Main comparison (percent)

| Run | Method | Retry | Test valid | Test exact | Hard valid | Hard exact | ms/example (test) | Prompt tok | Gen tok |
|---|---|---|---|---|---|---|---|---|---|
| few_shot | few_shot | no | 46.0 | 10.0 | 51.2 | 0.0 | 1396.4 | 746.3 | 52.2 |
| few_shot_retry | few_shot | yes | 52.0 | 10.0 | 55.0 | 0.0 | 2128.5 | 1193.7 | 79.8 |
| lora_r16_s0 | lora | no | 100.0 | 100.0 | 98.8 | 26.2 | 602.2 | 161.3 | 51.3 |
| lora_r4_s0 | lora | no | 100.0 | 100.0 | 98.8 | 30.0 | 589.2 | 161.3 | 51.3 |
| lora_r8_s0 | lora | no | 100.0 | 100.0 | 98.8 | 26.2 | 567.0 | 161.3 | 51.3 |
| lora_r8_s0_retry | lora | yes | 100.0 | 100.0 | 98.8 | 26.2 | 490.9 | 161.3 | 51.3 |
| lora_r8_s1 | lora | no | 100.0 | 100.0 | 100.0 | 30.0 | 611.1 | 161.3 | 51.3 |
| lora_r8_s2 | lora | no | 100.0 | 100.0 | 100.0 | 28.7 | 584.2 | 161.3 | 51.3 |
| zero_shot | zero_shot | no | 0.0 | 0.0 | 0.0 | 0.0 | 1063.7 | 161.3 | 108.4 |
| zero_shot_retry | zero_shot | yes | 0.0 | 0.0 | 0.0 | 0.0 | 2337.2 | 468.4 | 216.8 |

## Per-field accuracy, test set (percent)

| Run | entity | metric | value | unit | period | direction |
|---|---|---|---|---|---|---|
| few_shot | 84.0 | 95.0 | 46.0 | 18.0 | 64.0 | 82.0 |
| few_shot_retry | 77.0 | 88.0 | 46.0 | 18.5 | 60.0 | 77.5 |
| lora_r16_s0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| lora_r4_s0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| lora_r8_s0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| lora_r8_s0_retry | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| lora_r8_s1 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| lora_r8_s2 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| zero_shot | 7.5 | 2.5 | 4.0 | 3.5 | 5.5 | 6.0 |
| zero_shot_retry | 7.5 | 2.0 | 3.5 | 3.5 | 5.5 | 6.0 |

## Per-field accuracy, hard set (percent)

| Run | entity | metric | value | unit | period | direction |
|---|---|---|---|---|---|---|
| few_shot | 28.7 | 70.0 | 46.2 | 2.5 | 20.0 | 70.0 |
| few_shot_retry | 26.2 | 67.5 | 46.2 | 2.5 | 20.0 | 68.8 |
| lora_r16_s0 | 53.8 | 95.0 | 83.8 | 98.8 | 68.8 | 72.5 |
| lora_r4_s0 | 58.8 | 96.2 | 86.2 | 97.5 | 70.0 | 70.0 |
| lora_r8_s0 | 57.5 | 95.0 | 86.2 | 96.2 | 66.2 | 71.2 |
| lora_r8_s0_retry | 57.5 | 95.0 | 86.2 | 96.2 | 66.2 | 71.2 |
| lora_r8_s1 | 57.5 | 96.2 | 86.2 | 96.2 | 73.8 | 70.0 |
| lora_r8_s2 | 60.0 | 96.2 | 87.5 | 96.2 | 65.0 | 71.2 |
| zero_shot | 0.0 | 0.0 | 0.0 | 1.2 | 1.2 | 1.2 |
| zero_shot_retry | 0.0 | 0.0 | 0.0 | 1.2 | 1.2 | 1.2 |

## LoRA runs

| Run | r | alpha | lr | seed | Trainable | Total | Trainable % | Train s | Final val loss | Test exact | Hard exact |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lora_r16_s0 | 16 | 32 | 0.0005 | 0 | 1,843,200 | 136,358,208 | 1.3517 | 1184.9 | 0.0033 | 100.0 | 26.2 |
| lora_r4_s0 | 4 | 8 | 0.0005 | 0 | 460,800 | 134,975,808 | 0.3414 | 1132.1 | 0.0059 | 100.0 | 30.0 |
| lora_r8_s0 | 8 | 16 | 0.0005 | 0 | 921,600 | 135,436,608 | 0.6805 | 1112.8 | 0.0044 | 100.0 | 26.2 |
| lora_r8_s1 | 8 | 16 | 0.0005 | 1 | 921,600 | 135,436,608 | 0.6805 | 1162.5 | 0.0047 | 100.0 | 30.0 |
| lora_r8_s2 | 8 | 16 | 0.0005 | 2 | 921,600 | 135,436,608 | 0.6805 | 1146.0 | 0.0048 | 100.0 | 28.7 |

## LoRA by rank (mean ± std over seeds, percent)

| r | Seeds | Test exact | Hard exact | Hard valid |
|---|---|---|---|---|
| 4 | 0 | 100.0 | 30.0 | 98.8 |
| 8 | 0,1,2 | 100.0 ± 0.0 | 28.3 ± 1.9 | 99.6 ± 0.7 |
| 16 | 0 | 100.0 | 26.2 | 98.8 |

## Knowledge probes (percent)

| Run | Recall of training values (n=30) | World facts (n=10) |
|---|---|---|
| lora_r16_s0 | 0.0 | 90.0 |
| lora_r4_s0 | 0.0 | 90.0 |
| lora_r8_s0 | 0.0 | 90.0 |
| lora_r8_s1 | 0.0 | 90.0 |
| lora_r8_s2 | 0.0 | 90.0 |
| zero_shot | 0.0 | 70.0 |
