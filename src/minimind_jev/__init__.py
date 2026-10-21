"""minimind-jev v0.3: from-scratch tiny decision transformer.

Tokenizer (own BBPE) -> optional short MLM pretraining on the task corpus ->
binary-BCE decision fine-tuning -> temperature calibration. White-box by design:
every module is small enough to read in one sitting.
"""
