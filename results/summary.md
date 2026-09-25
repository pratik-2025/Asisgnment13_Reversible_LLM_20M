| run | mode | batch | steps | tokens seen | final train loss | final val loss | tokens/s (median) | peak mem (GiB) | wall time (min) | est. cost ($) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01_baseline | baseline | 104 | 939 | 50.0M | 2.1417 | 2.1885 | 62,963 | 13.32 | 14.2 | 0.08 |
| 02_hamiltonian | hamiltonian | 104 | 939 | 50.0M | 2.3914 | 2.4389 | 51,615 | 5.28 | 17.1 | 0.10 |
| 02_leapfrog | leapfrog | 104 | 939 | 50.0M | 2.0984 | 2.1478 | 47,812 | 5.28 | 18.5 | 0.11 |
| 02_midpoint | midpoint | 104 | 939 | 50.0M | 2.2606 | 2.3055 | 50,283 | 5.28 | 17.8 | 0.10 |
| 03_rev_maxbatch | leapfrog | 1056 | 92 | 49.7M | 4.9845 | 4.6963 | 44,572 | 13.27 | 19.8 | 0.12 |
| 03_rev_maxbatch_lr_unscaled | leapfrog | 1048 | 93 | 49.9M | 4.0121 | 3.9478 | 43,753 | 13.17 | 20.1 | 0.12 |
