# Long run `f19-long`: per window, first and last tenth (median), max, slope per 100 windows

- straggler/gather_ms: 120 windows; 0.89 -> 0.74; max 3.65; slope -0.128
- straggler/analyze_ms: 120 windows; 2.92 -> 2.93; max 5.68; slope +0.003
- straggler/latency/analyzed_ms: 120 windows; 11.67 -> 12.45; max 256.11; slope +3.973
- straggler/self_overhead_ms: 120 windows; 0.13 -> 0.13; max 0.15; slope +0.001
- straggler/dropped_events: 120 windows; 0.00 -> 0.00; max 0.00; slope +0.000
- straggler/unread_events: 120 windows; 0.00 -> 0.00; max 0.00; slope +0.000
- straggler/trace/lost: 120 windows; 0.00 -> 0.00; max 0.00; slope +0.000
- straggler/health/errors: 120 windows; 0.00 -> 0.00; max 0.00; slope +0.000
- straggler/flagged/count: 120 windows; 0.00 -> 0.00; max 0.00; slope +0.000
- straggler/uncertain/count: 120 windows; 0.00 -> 0.00; max 0.00; slope +0.000
- GPU memory (mean of 8, MiB): 123 samples; 63001 -> 57379; max 64778
- python RSS (sum, MiB): 10038 -> 10118; max 110201
- windows without a straggler: 120; alerts: 0 []
- profiler switched off: 0; ['training completed step 1199']
