# Stacking meta-model

Ovaj direktorij spaja tri postojeća podsustava u jedan završni model koji za svaki komentar vraća jednu vjerojatnost:

```text
P(AI | komentar) u [0, 1]
```

Ulazi meta-modela su tri bazne vjerojatnosti:

- `p_semantic_svm_ai`: izlaz semantičkog SVM-a
- `p_tree_based_ai`: izlaz stablastog modela
- `p_tf_logreg_ai`: izlaz TF-IDF + logističke regresije

Završni izlaz je:

- `p_meta_ai`: kombinirana vjerojatnost završnog stacking modela

## Kratki sažetak

Skripta `meta_model/src/main.py` radi sljedeće:

1. Učitava `train.csv`, `dev.csv` i `test.csv`.
2. Učitava već izračunate SVM embeddinge iz `semantic_svm/artifacts/embedding_manifest.json`.
3. Provjerava postojeći `tf-logreg/reddit_ai_detector.joblib`.
4. Generira out-of-fold bazne vjerojatnosti za train.
5. Na tim OOF vjerojatnostima trenira logistički meta-model.
6. Za dev/test koristi finalne bazne modele trenirane na cijelom train skupu.
7. Sprema bazne i meta-vjerojatnosti za train/dev/test.
8. Sprema metrike, manifest i naučeni meta-model.

Ovo je implementacija opcije B: OOF služi za učenje meta-modela, a finalne predikcije rade bazni modeli trenirani na 100% train podataka.

## Zašto se train dijeli na foldove?

Meta-model ne smije učiti na preoptimističnim predikcijama baznih modela.

Ako bazni model treniramo na cijelom `train.csv`, pa zatim na tom istom `train.csv` izračunamo njegove vjerojatnosti, dobivamo prelagan zadatak. Bazni model je već vidio te primjere. Meta-model bi tada učio na signalima koji izgledaju bolje nego što će izgledati na neviđenim podacima.

Zato koristimo out-of-fold postupak:

1. `train.csv` se podijeli na `K` stratificiranih foldova.
2. Za svaki fold treniraju se sva tri bazna modela na preostalim foldovima.
3. Na izostavljenom foldu izračunaju se tri vjerojatnosti.
4. Nakon svih foldova svaki train primjer ima bazne vjerojatnosti dobivene od modela koji taj primjer nije vidio.
5. Tek na tim OOF vjerojatnostima trenira se meta-model.

To je glavna zaštita od curenja informacije.

## Opcija B: finalni bazni modeli na 100% train skupa

OOF modeli iz foldova ne koriste se u produkciji i ne spremaju se kao ansambl. Oni postoje samo da meta-model dobije poštene train predikcije.

Nakon treniranja meta-modela koriste se finalni bazni modeli:

- semantički SVM iz `semantic_svm/artifacts/semantic_svm.joblib`
- novi tree-based model treniran na cijelom `train.csv`
- novi TF-IDF logistički model treniran na cijelom `train.csv`

Ti finalni modeli proizvode bazne vjerojatnosti za `dev.csv` i `test.csv`. Zatim se nad njima izračunavaju meta-značajke i dobiva `p_meta_ai`.

Drugim riječima:

```text
OOF foldovi -> treniranje meta-modela
full-train bazni modeli -> dev/test/produkcijske predikcije
```

## Meta-značajke

Meta-model ne koristi samo sirove vjerojatnosti. Prvo ih pretvara u logit-oblik:

```text
logit(p) = log(p / (1 - p))
```

U kodu se vjerojatnosti prije toga režu na interval `[1e-5, 1 - 1e-5]` da se izbjegne beskonačan logit za vrijednosti jako blizu 0 ili 1.

Ako su bazne vjerojatnosti:

```text
p_sem, p_tree, p_tf
```

onda su logiti:

```text
l_sem, l_tree, l_tf
```

Logistički meta-model trenira se nad 9 značajki:

- `l_sem`
- `l_tree`
- `l_tf`
- `l_sem * l_tree`
- `l_sem * l_tf`
- `l_tree * l_tf`
- `l_sem^2`
- `l_tree^2`
- `l_tf^2`

Ovo daje jednostavnu nelinearnost: model može naučiti ne samo koliko je svaki bazni model važan, nego i što znači kada se dva bazna modela međusobno slažu ili proturječe.

Nismo dodali trostruki produkt. To je namjerno, zbog jednostavnosti i lakšeg tumačenja.

## Meta-model

Završni model je regularna logistička regresija:

```text
LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
```

Njezina uloga je naučiti kako kombinirati tri bazne vjerojatnosti. Koeficijenti meta-modela spremaju se u:

```text
meta_model/artifacts/stacking_logreg_meta_model.joblib
```

U toj datoteci se spremaju:

- naučeni `sklearn` model
- imena meta-značajki
- nazivi baznih stupaca vjerojatnosti
- vrijednost `epsilon` korištena za logit-transformaciju

## Što je u postojećem tf-logreg `.joblib`?

Skripta provjerava `tf-logreg/reddit_ai_detector.joblib` i zapisuje sažetak u:

```text
meta_model/artifacts/tf_logreg_existing_joblib_summary.json
```

Sažetak pokazuje da postojeći `.joblib` sadrži spremljeni `sklearn` pipeline:

- TF-IDF vektorizatore
- logističku regresiju
- poredak klasa

Ne sadrži unaprijed izračunate vjerojatnosti za train/dev/test.

Zato ova skripta trenira novi TF-IDF logistički model na cijelom `train.csv` i sprema ga u:

```text
meta_model/artifacts/tf_logreg_full_train.joblib
```

Originalna datoteka iz `tf-logreg/` se ne mijenja.

## Pokretanje

Iz korijena repozitorija:

```bash
uv run python meta_model/src/main.py
```

Za brzi smoke test:

```bash
uv run python meta_model/src/main.py --max-train-rows 2000 --max-dev-rows 500 --max-test-rows 500 --folds 3 --output-dir meta_model/artifacts/smoke
```

## Izlazne datoteke

Glavni artefakti su:

- `train_oof_probabilities.csv`: OOF bazne vjerojatnosti za train i meta-vjerojatnost
- `train_full_base_probabilities.csv`: dijagnostičke train vjerojatnosti iz finalnih baznih modela treniranih na cijelom trainu
- `dev_probabilities.csv`: bazne i meta-vjerojatnosti za dev
- `test_probabilities.csv`: bazne i meta-vjerojatnosti za test
- `stacking_logreg_meta_model.joblib`: spremljeni meta-model
- `tree_based_full_train.joblib`: finalni stablasti model treniran na cijelom trainu
- `tf_logreg_full_train.joblib`: finalni TF-IDF logistički model treniran na cijelom trainu
- `tf_logreg_existing_joblib_summary.json`: pregled postojećeg tf-logreg modela
- `metrics.json`: metrike za bazne modele i završni meta-model
- `manifest.json`: popis korištenih ulaza i proizvedenih artefakata

`train_full_base_probabilities.csv` nije namijenjen treniranju meta-modela jer je leaky: bazni modeli su tu predviđali primjere koje su već vidjeli. Za treniranje meta-modela koristi se `train_oof_probabilities.csv`.

## Rezultati

Usporedba triju baznih modela i završnog meta-modela na `dev.csv`:

| Model         | Accuracy | Precision |  Recall |      F1 | ROC AUC | Log loss |   Brier |
| ------------- | -------: | --------: | ------: | ------: | ------: | -------: | ------: |
| Semantic SVM  |  0.90948 |   0.90341 | 0.91283 | 0.90810 | 0.96885 |  0.22646 | 0.06715 |
| Tree-based    |  0.90487 |   0.89214 | 0.91664 | 0.90423 | 0.96696 |  0.23115 | 0.07009 |
| TF-IDF logreg |  0.98085 |   0.98627 | 0.97447 | 0.98033 | 0.99780 |  0.06739 | 0.01626 |
| Meta-model    |  0.98406 |   0.98788 | 0.97949 | 0.98366 | 0.99840 |  0.04564 | 0.01240 |

Konfuzijske matrice na `dev.csv`:

| Model         |    TN |   FP |   FN |    TP |
| ------------- | ----: | ---: | ---: | ----: |
| Semantic SVM  | 19549 | 2022 | 1806 | 18912 |
| Tree-based    | 19275 | 2296 | 1727 | 18991 |
| TF-IDF logreg | 21290 |  281 |  529 | 20189 |
| Meta-model    | 21322 |  249 |  425 | 20293 |

Završni meta-model ima najbolji rezultat po svim navedenim metrikama na razvojnom skupu. Najjači pojedinačni bazni model je TF-IDF logistička regresija, ali meta-model dodatno smanjuje broj lažno pozitivnih i lažno negativnih primjera.

Ako `test.csv` sadrži oznake, skripta računa i test metrike. U trenutnom pokretanju završni meta-model na test skupu postiže:

| Accuracy | Precision |  Recall |      F1 | ROC AUC | Log loss |   Brier |
| -------: | --------: | ------: | ------: | ------: | -------: | ------: |
|  0.98406 |   0.98722 | 0.98016 | 0.98368 | 0.99842 |  0.04499 | 0.01226 |

## Kako čitati CSV izlaze

Svaki probability CSV sadrži stupce:

- `row_id`: redni broj retka u odgovarajućem splitu
- `p_semantic_svm_ai`: AI-vjerojatnost semantičkog SVM-a
- `p_tree_based_ai`: AI-vjerojatnost stablastog modela
- `p_tf_logreg_ai`: AI-vjerojatnost TF-IDF logističkog modela
- `p_meta_ai`: završna AI-vjerojatnost meta-modela
- `label`: oznaka, ako postoji u ulaznom CSV-u

Za produkcijski izlaz najvažniji stupac je `p_meta_ai`.
