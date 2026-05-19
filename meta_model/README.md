# Stacking meta-model

Ovaj direktorij spaja tri postojeća podsustava u jedan završni model koji vraća jednu vjerojatnost:

```text
P(AI | komentar) u [0, 1]
```

Ulazi meta-modela su tri bazne vjerojatnosti:

- `p_semantic_svm_ai`: izlaz semantičkog SVM-a
- `p_tree_based_ai`: izlaz stablastog modela
- `p_tf_logreg_ai`: izlaz TF-IDF + logističke regresije

## Zašto se train dijeli na foldove?

Meta-model ne smije učiti na preoptimističnim predikcijama baznih modela.

Ako bazni model treniramo na cijelom `train.csv`, a zatim na tom istom `train.csv` izračunamo njegove vjerojatnosti, dobivamo previše lagan zadatak. Bazni model je već vidio te primjere. Meta-model bi tada učio na signalima koji izgledaju bolje nego što će izgledati na neviđenim podacima.

Zato koristimo out-of-fold postupak:

1. `train.csv` se podijeli na `K` stratificiranih foldova.
2. Za svaki fold treniramo sva tri bazna modela na preostalim foldovima.
3. Na izostavljenom foldu izračunamo tri vjerojatnosti.
4. Nakon svih foldova svaki train primjer ima bazne vjerojatnosti dobivene od modela koji taj primjer nije vidio.
5. Tek na tim OOF vjerojatnostima trenira se meta-model.

To je glavna zaštita od curenja informacije.

## Meta-značajke

Meta-model ne koristi samo sirove vjerojatnosti. Prvo ih pretvara u logit-oblik:

```text
logit(p) = log(p / (1 - p))
```

Zatim trenira logističku regresiju nad ovim značajkama:

- tri logita: `l_sem`, `l_tree`, `l_tf`
- tri parne interakcije: `l_sem*l_tree`, `l_sem*l_tf`, `l_tree*l_tf`
- tri kvadrata: `l_sem^2`, `l_tree^2`, `l_tf^2`

To je dovoljno fleksibilno da model uhvati jednostavne nelinearnosti, ali ostaje puno čitljivije od male neuronske mreže.

## Što je u postojećem tf-logreg `.joblib`?

Skripta provjerava `tf-logreg/reddit_ai_detector.joblib` i zapisuje sažetak u:

```text
meta_model/artifacts/tf_logreg_existing_joblib_summary.json
```

Ta datoteka sadrži spremljeni `sklearn` pipeline: TF-IDF vektorizatore i logističku regresiju. Ne sadrži unaprijed izračunate vjerojatnosti za train/dev/test.

Za završne dev/test vjerojatnosti ova skripta zato trenira novu TF-IDF logističku regresiju na cijelom `train.csv` i sprema je u `meta_model/artifacts/tf_logreg_full_train.joblib`, bez mijenjanja originalne datoteke iz `tf-logreg/`.

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
- `metrics.json`: metrike za bazne modele i završni meta-model
- `manifest.json`: popis korištenih ulaza i proizvedenih artefakata

`train_full_base_probabilities.csv` nije namijenjen treniranju meta-modela jer je leaky: bazni modeli su tu predviđali primjere koje su već vidjeli. Za treniranje meta-modela koristi se `train_oof_probabilities.csv`.
