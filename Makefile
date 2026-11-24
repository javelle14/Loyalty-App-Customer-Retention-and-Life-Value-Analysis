.PHONY: appendix-retail all data features train clv decide uplift sensitivity playbook tableau appendix test survival
PY=python
all: data features train clv decide uplift sensitivity playbook tableau appendix survival test

data:      ; $(PY) -m src.generate_data
features:  ; $(PY) -m src.build_features
train:     ; $(PY) -m src.train_churn
clv:       ; $(PY) -m src.clv
decide:    ; $(PY) -m src.decision
uplift:    ; $(PY) -m src.uplift
sensitivity: ; $(PY) -m src.sensitivity
playbook:    ; $(PY) -m src.playbook
tableau:     ; $(PY) -m src.export_tableau
appendix:    ; $(PY) -m src.appendix_real_data
appendix-retail: ; $(PY) -m src.appendix_real_data --csv data/external/online_retail_ii_orders.csv --name online_retail_ii
test:        ; $(PY) -m pytest -q
survival:  ; Rscript r/survival.R
