# Physics-guided EV energy benchmark

The executable notebook is `notebooks/train_kaggle_ds.ipynb`. It audits the supplied
5,000-row CSV, creates only the paper-grounded physics proxies from Equations
52-56, partitions complete
`Vehicle_ID` groups into train/validation/test, and compares classical ML, an MLP,
and an inductive PyTorch Geometric GCN on identical partitions.

Run it from the repository root with the `llm-env` kernel. `QUICK=True` is the
reproducible smoke benchmark; set it to `False` for longer training. Generated
metrics, figures, split IDs, predictions, and failure logs are written under
`reports/`; fitted artifacts are written under `models/`.

The benchmark is retrospective because it uses realized telemetry. It is not an
exact reproduction of the cited paper and does not establish real-world BMW i3
accuracy.
