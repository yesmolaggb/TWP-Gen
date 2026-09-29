# Baseline implementations

The comparison methods retain their official planning and generation workflows.
For the experiments in the paper, all methods use the same topic set, retrieval
backend and source pool, generation model, decoding configuration, and maximum
output length. Method-specific retrieval and reasoning calls remain unchanged.

| Method | Official implementation | Experimental note |
|---|---|---|
| Direct RAG | No separate official repository | Implemented as a standard retrieve-then-generate baseline under the shared experimental configuration. |
| STORM | https://github.com/stanford-oval/storm | Official implementation; model and retrieval settings were aligned with the shared configuration. |
| OmniThink | https://github.com/zjunlp/OmniThink | Official implementation; model and retrieval settings were aligned with the shared configuration. |
| ConvergeWriter | https://github.com/JiBinquan/ConvergeWriter | Official implementation; model and retrieval settings were aligned with the shared configuration. |

Third-party source code is not copied into this repository. Please consult each
project's license and installation instructions at the links above.
