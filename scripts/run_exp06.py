from __future__ import annotations

import gc
import json
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.tree import export_text


EXP_DIR = Path(__file__).resolve().parents[1]
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from src.exp06_data import (  # noqa: E402
    TARGET_CODE, YEAR, annual_stability_summary, audit_and_load_model_frame,
    class_distribution, distributions_by_period, distributions_by_year, load_config,
)
from src.exp06_evaluation import (  # noqa: E402
    CLASS_NAMES, confusion_frame, evaluate_model, scenario_score,
)
from src.exp06_modeling import (  # noqa: E402
    fit_tree, prepare_bundle, root_and_level_variables, split_distribution,
)
from src.exp06_paths import (  # noqa: E402
    CONFIG_PATH, DATA_DIR, FIGURE_DIR, LOG_PATH, MODEL_DIR, REPORT_PATH,
    TABLE_DIR, ensure_dirs, project_path, relative,
)
from src.exp06_report import build_report  # noqa: E402
from src.exp06_rules import (  # noqa: E402
    compare_fatal_grave, extract_leaves, top_rules_for_class,
    variables_by_depth,
)
from src.exp06_visualization import (  # noqa: E402
    plot_balance_comparison, plot_class_distribution, plot_confusion,
    plot_distribution_periods, plot_f1_by_class, plot_full_tree,
    plot_metric_scenarios, plot_simplified_tree, plot_stability,
)


def setup_logging() -> None:
    ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def save_table(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep=";", index=False, encoding="utf-8-sig")
    logging.info("Tabela salva: %s (%s linhas)", relative(path), len(frame))


def save_json(value: dict, path: Path) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("JSON salvo: %s", relative(path))


def class_weight_label(value: str | None) -> str:
    return "balanced" if value == "balanced" else "None"


def predictor_check(predictors: list[str], forbidden: list[str]) -> pd.DataFrame:
    intersection = sorted(set(predictors) & set(forbidden))
    rows = [
        {"tipo": "preditor_final", "coluna": col, "incluida": True, "status": "APROVADA" if not intersection else "REPROVADA"}
        for col in predictors
    ]
    rows += [
        {"tipo": "coluna_proibida_excluida", "coluna": col, "incluida": False, "status": "APROVADA" if col not in intersection else "REPROVADA"}
        for col in forbidden
    ]
    if intersection:
        raise RuntimeError(f"Vazamento detectado nos preditores: {intersection}")
    logging.info("Preditores finais: %s", predictors)
    logging.info("Colunas proibidas/excluídas: %s", forbidden)
    logging.info("Checagem de vazamento APROVADA")
    return pd.DataFrame(rows)


def model_record(
    scenario: str,
    stage: str,
    period: str,
    target_kind: str,
    balance: str | None,
    criterion: str,
    depth: int,
    leaf_prop: float,
    leaf_abs: int,
    seed: int,
    predictors: list[str],
    model,
    bundle,
    split: str,
) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray]:
    metrics, per_class, cm_abs, cm_norm = evaluate_model(model, bundle, split)
    root_var, root_cat, level_vars = root_and_level_variables(model, bundle.feature_meta)
    row = {
        "cenario": scenario,
        "etapa": stage,
        "periodo": period,
        "alvo": target_kind,
        "balanceamento": class_weight_label(balance),
        "class_weight": balance,
        "criterion": criterion,
        "max_depth": depth,
        "min_samples_leaf_proportion": leaf_prop,
        "min_samples_leaf_absolute": leaf_abs,
        "random_state": seed,
        "split_avaliado": split,
        "n_preditores": len(predictors),
        "preditores": " | ".join(predictors),
        "raiz_variavel": root_var,
        "raiz_categoria": root_cat,
        "variaveis_primeiros_niveis": " | ".join(level_vars),
        "numero_folhas": int(model.get_n_leaves()),
        "profundidade_real": int(model.get_depth()),
        **metrics,
    }
    row["score_selecao"] = scenario_score(row, target_kind)
    per_class.insert(0, "cenario", scenario)
    per_class.insert(1, "split", split)
    return row, per_class, cm_abs, cm_norm


def period_frame(frame: pd.DataFrame, periods: dict, period: str) -> pd.DataFrame:
    start, end = periods[period]
    return frame.loc[frame[YEAR].between(start, end)].reset_index(drop=True)


def save_confusions(prefix: str, target_kind: str, cm_abs: np.ndarray, cm_norm: np.ndarray, title: str) -> None:
    save_table(confusion_frame(cm_abs, target_kind), TABLE_DIR / f"matriz_confusao_{prefix}_absoluta.csv")
    save_table(confusion_frame(cm_norm, target_kind), TABLE_DIR / f"matriz_confusao_{prefix}_normalizada.csv")
    plot_confusion(cm_abs, target_kind, FIGURE_DIR / f"matriz_confusao_{prefix}_absoluta.png", False, title + " — absoluta")
    plot_confusion(cm_norm, target_kind, FIGURE_DIR / f"matriz_confusao_{prefix}_normalizada.png", True, title + " — normalizada por classe real")


def top_rule_text(leaves: pd.DataFrame, class_name: str) -> str:
    slug = class_name.lower().replace(" ", "_").replace("í", "i").replace("ã", "a")
    column = f"lift_{slug}"
    if column not in leaves:
        return ""
    eligible = leaves[leaves["volume_teste"] >= max(100, int(leaves["volume_teste"].sum()*.001))]
    if eligible.empty:
        eligible = leaves
    return str(eligible.sort_values([column, "volume_teste"], ascending=False).iloc[0]["regra"])


def run() -> None:
    config = load_config(CONFIG_PATH)
    input_path = project_path(config["input_base"])
    predictors = list(config["predictors"])
    forbidden = list(config["forbidden_predictors"])
    logging.info("EXP-06 iniciado. Base: %s", relative(input_path))
    leak_audit = predictor_check(predictors, forbidden)

    frame, audit = audit_and_load_model_frame(input_path, config)
    general_distribution = class_distribution(frame)
    period_distribution = distributions_by_period(frame, config["periods"])
    annual_distribution = distributions_by_year(frame)
    annual_summary = annual_stability_summary(annual_distribution, config["periods"])
    save_table(audit["auditoria_base"], TABLE_DIR / "auditoria_base.csv")
    save_table(audit["auditoria_colunas"], TABLE_DIR / "auditoria_colunas.csv")
    save_table(audit["auditoria_alvos"], TABLE_DIR / "auditoria_alvos.csv")
    save_table(general_distribution, TABLE_DIR / "distribuicao_classes_geral.csv")
    save_table(period_distribution, TABLE_DIR / "distribuicao_classes_por_periodo.csv")
    save_table(annual_distribution, TABLE_DIR / "distribuicao_classes_por_ano.csv")
    save_table(annual_summary, TABLE_DIR / "estabilidade_anual_periodos.csv")
    save_table(leak_audit, TABLE_DIR / "checagem_vazamento.csv")
    save_table(pd.DataFrame({"ordem": range(1, len(predictors)+1), "preditor": predictors}), TABLE_DIR / "preditores_utilizados.csv")
    save_table(audit["mapeamento_tracado_macro"], DATA_DIR / "mapeamento_tracado_macro.csv")
    macro_counts = frame["tracado_macro"].value_counts(dropna=False).rename_axis("tracado_macro").reset_index(name="n")
    macro_counts["proporcao"] = macro_counts["n"] / len(frame)
    save_table(macro_counts, TABLE_DIR / "distribuicao_tracado_macro.csv")
    plot_class_distribution(general_distribution, FIGURE_DIR / "distribuicao_classes_geral.png", "Distribuição nacional da severidade registrada")
    plot_distribution_periods(period_distribution, FIGURE_DIR / "distribuicao_classes_por_periodo.png")

    catalog: list[dict] = []
    validation_rows: list[dict] = []
    split_rows: list[pd.DataFrame] = []

    # Etapa 1 — seis cenários fixos; o teste não é consultado.
    logging.info("Etapa 1 iniciada: triagem de período e balanceamento")
    for period in config["periods"]:
        subset = period_frame(frame, config["periods"], period)
        bundle = prepare_bundle(subset, predictors, "4_classes", int(config["random_state"]))
        split_rows.append(split_distribution(bundle, period))
        for class_weight in (None, "balanced"):
            scenario = f"E1_{period}_{class_weight_label(class_weight)}"
            model, leaf_abs = fit_tree(
                bundle, config["stage1"]["criterion"], int(config["stage1"]["max_depth"]),
                float(config["stage1"]["min_samples_leaf_proportion"]), class_weight,
                int(config["random_state"]),
            )
            row, _, _, _ = model_record(
                scenario, "Etapa 1", period, "4_classes", class_weight,
                config["stage1"]["criterion"], int(config["stage1"]["max_depth"]),
                float(config["stage1"]["min_samples_leaf_proportion"]), leaf_abs,
                int(config["random_state"]), predictors, model, bundle, "validation",
            )
            validation_rows.append(row)
            catalog.append({**row, "finalidade": "triagem", "teste_usado_na_selecao": False})
            logging.info("%s: macroF1=%.4f recall_fatal=%.4f precision_fatal=%.4f", scenario, row["macro_f1"], row["recall_fatal"], row["precision_fatal"])
        del model, bundle, subset
        gc.collect()

    stage1_df = pd.DataFrame(validation_rows)
    stage1_df = stage1_df[stage1_df["etapa"] == "Etapa 1"].copy()
    selected_stage1 = stage1_df.sort_values("score_selecao", ascending=False).head(2)
    logging.info("Combinações selecionadas para Etapa 2: %s", selected_stage1[["periodo", "balanceamento"]].to_dict("records"))

    # Etapa 2 — grade somente nas duas combinações selecionadas.
    logging.info("Etapa 2 iniciada: escolha de complexidade")
    stage2_rows: list[dict] = []
    for selected in selected_stage1.itertuples(index=False):
        subset = period_frame(frame, config["periods"], selected.periodo)
        bundle = prepare_bundle(subset, predictors, "4_classes", int(config["random_state"]))
        class_weight = "balanced" if selected.balanceamento == "balanced" else None
        for depth in config["stage2"]["max_depth"]:
            for criterion in config["stage2"]["criterion"]:
                for leaf_prop in config["stage2"]["min_samples_leaf_proportion"]:
                    scenario = f"E2_{selected.periodo}_{class_weight_label(class_weight)}_{criterion}_d{depth}_l{leaf_prop}"
                    model, leaf_abs = fit_tree(bundle, criterion, int(depth), float(leaf_prop), class_weight, int(config["random_state"]))
                    row, _, _, _ = model_record(
                        scenario, "Etapa 2", selected.periodo, "4_classes", class_weight,
                        criterion, int(depth), float(leaf_prop), leaf_abs, int(config["random_state"]),
                        predictors, model, bundle, "validation",
                    )
                    stage2_rows.append(row)
                    validation_rows.append(row)
                    catalog.append({**row, "finalidade": "complexidade", "teste_usado_na_selecao": False})
        del model, bundle, subset
        gc.collect()

    stage2_df = pd.DataFrame(stage2_rows)
    best_score = float(stage2_df["score_selecao"].max())
    near_best = stage2_df[stage2_df["score_selecao"] >= best_score - 0.01].copy()
    candidate = near_best.sort_values(
        ["max_depth", "min_samples_leaf_proportion", "score_selecao"],
        ascending=[True, False, False],
    ).iloc[0].to_dict()
    official_period = candidate["periodo"]
    official_weight = "balanced" if candidate["balanceamento"] == "balanced" else None
    official_predictors = predictors.copy()
    logging.info("Configuração congelada antes do teste: %s", candidate)

    # Etapa 3 — oficial congelado e sensibilidades.
    logging.info("Etapa 3 iniciada: teste oficial e sensibilidades")
    official_subset = period_frame(frame, config["periods"], official_period)
    official_bundle = prepare_bundle(official_subset, official_predictors, "4_classes", int(config["random_state"]))
    official_model, official_leaf_abs = fit_tree(
        official_bundle, candidate["criterion"], int(candidate["max_depth"]),
        float(candidate["min_samples_leaf_proportion"]), official_weight, int(config["random_state"]),
    )
    official_val_row, official_val_class, _, _ = model_record(
        "S1_oficial_4_classes", "Etapa 3", official_period, "4_classes", official_weight,
        candidate["criterion"], int(candidate["max_depth"]), float(candidate["min_samples_leaf_proportion"]),
        official_leaf_abs, int(config["random_state"]), official_predictors, official_model, official_bundle, "validation",
    )
    official_test_row, official_test_class, cm_off_abs, cm_off_norm = model_record(
        "S1_oficial_4_classes", "Etapa 3", official_period, "4_classes", official_weight,
        candidate["criterion"], int(candidate["max_depth"]), float(candidate["min_samples_leaf_proportion"]),
        official_leaf_abs, int(config["random_state"]), official_predictors, official_model, official_bundle, "test",
    )
    validation_rows.append(official_val_row)
    catalog.append({**official_test_row, "finalidade": "modelo oficial congelado", "teste_usado_na_selecao": False})

    no_type_predictors = [p for p in official_predictors if p != "tipo_acidente_analise"]
    predictor_check(no_type_predictors, forbidden)
    no_type_bundle = prepare_bundle(official_subset, no_type_predictors, "4_classes", int(config["random_state"]))
    no_type_model, no_type_leaf_abs = fit_tree(
        no_type_bundle, candidate["criterion"], int(candidate["max_depth"]),
        float(candidate["min_samples_leaf_proportion"]), official_weight, int(config["random_state"]),
    )
    no_type_val_row, no_type_val_class, _, _ = model_record(
        "S2_sem_tipo_acidente", "Etapa 3", official_period, "4_classes", official_weight,
        candidate["criterion"], int(candidate["max_depth"]), float(candidate["min_samples_leaf_proportion"]),
        no_type_leaf_abs, int(config["random_state"]), no_type_predictors, no_type_model, no_type_bundle, "validation",
    )
    no_type_test_row, no_type_test_class, cm_nt_abs, cm_nt_norm = model_record(
        "S2_sem_tipo_acidente", "Etapa 3", official_period, "4_classes", official_weight,
        candidate["criterion"], int(candidate["max_depth"]), float(candidate["min_samples_leaf_proportion"]),
        no_type_leaf_abs, int(config["random_state"]), no_type_predictors, no_type_model, no_type_bundle, "test",
    )
    validation_rows.append(no_type_val_row)
    catalog.append({**no_type_test_row, "finalidade": "sensibilidade contextual", "teste_usado_na_selecao": False})

    three_bundle = prepare_bundle(official_subset, official_predictors, "3_classes", int(config["random_state"]))
    three_model, three_leaf_abs = fit_tree(
        three_bundle, candidate["criterion"], int(candidate["max_depth"]),
        float(candidate["min_samples_leaf_proportion"]), "balanced", int(config["random_state"]),
    )
    three_val_row, three_val_class, _, _ = model_record(
        "S3_tres_classes_balanced", "Etapa 3", official_period, "3_classes", "balanced",
        candidate["criterion"], int(candidate["max_depth"]), float(candidate["min_samples_leaf_proportion"]),
        three_leaf_abs, int(config["random_state"]), official_predictors, three_model, three_bundle, "validation",
    )
    three_test_row, three_test_class, cm_3_abs, cm_3_norm = model_record(
        "S3_tres_classes_balanced", "Etapa 3", official_period, "3_classes", "balanced",
        candidate["criterion"], int(candidate["max_depth"]), float(candidate["min_samples_leaf_proportion"]),
        three_leaf_abs, int(config["random_state"]), official_predictors, three_model, three_bundle, "test",
    )
    validation_rows.append(three_val_row)
    catalog.append({**three_test_row, "finalidade": "sensibilidade três classes", "teste_usado_na_selecao": False})

    # Folhas e regras com distribuição real do teste.
    official_leaves = extract_leaves(official_model, official_bundle)
    no_type_leaves = extract_leaves(no_type_model, no_type_bundle)
    three_leaves = extract_leaves(three_model, three_bundle)
    top_fatal = top_rules_for_class(official_leaves, "Fatal")
    top_grave = top_rules_for_class(official_leaves, "Grave")
    top_leve = top_rules_for_class(official_leaves, "Leve")
    top_sem = top_rules_for_class(official_leaves, "Sem vítimas")
    fatal_grave = compare_fatal_grave(official_leaves)
    save_table(official_leaves, TABLE_DIR / "folhas_modelo_oficial.csv")
    save_table(no_type_leaves, TABLE_DIR / "folhas_modelo_sem_tipo.csv")
    save_table(three_leaves, TABLE_DIR / "folhas_modelo_tres_classes.csv")
    save_table(top_fatal, TABLE_DIR / "top_regras_fatal.csv")
    save_table(top_grave, TABLE_DIR / "top_regras_grave.csv")
    save_table(top_leve, TABLE_DIR / "top_regras_leve.csv")
    save_table(top_sem, TABLE_DIR / "top_regras_sem_vitimas.csv")
    save_table(fatal_grave, TABLE_DIR / "comparacao_regras_fatal_grave.csv")

    save_confusions("oficial", "4_classes", cm_off_abs, cm_off_norm, "Modelo oficial de quatro classes")
    save_confusions("sem_tipo", "4_classes", cm_nt_abs, cm_nt_norm, "Sensibilidade sem tipo de acidente")
    save_confusions("tres_classes", "3_classes", cm_3_abs, cm_3_norm, "Sensibilidade de três classes")

    plot_full_tree(official_model, official_bundle, FIGURE_DIR / "arvore_oficial_completa.png", "Árvore oficial — auditoria")
    plot_simplified_tree(official_model, official_bundle, FIGURE_DIR / "arvore_oficial_simplificada.png", "Árvore oficial — leitura científica")
    plot_full_tree(no_type_model, no_type_bundle, FIGURE_DIR / "arvore_sem_tipo_completa.png", "Árvore sem tipo de acidente — auditoria")
    plot_simplified_tree(no_type_model, no_type_bundle, FIGURE_DIR / "arvore_sem_tipo_simplificada.png", "Árvore sem tipo de acidente")
    plot_simplified_tree(three_model, three_bundle, FIGURE_DIR / "arvore_tres_classes_simplificada.png", "Árvore de três classes")
    tree_text = export_text(
        official_model,
        feature_names=[f"{var} = {cat}" for var, cat in official_bundle.feature_meta],
        decimals=4,
    )
    (FIGURE_DIR / "arvore_oficial.txt").write_text(tree_text, encoding="utf-8")

    # Etapa 4 — cinco sementes para a oficial e a alternativa de três classes.
    logging.info("Etapa 4 iniciada: estabilidade")
    stability_seed_rows: list[dict] = []
    stability_structure_rows: list[dict] = []
    for cfg_name, target_kind, weight in (
        ("oficial_4_classes", "4_classes", official_weight),
        ("alternativa_3_classes", "3_classes", "balanced"),
    ):
        for seed in config["stability_seeds"]:
            bundle = prepare_bundle(official_subset, official_predictors, target_kind, int(seed))
            model, leaf_abs = fit_tree(
                bundle, candidate["criterion"], int(candidate["max_depth"]),
                float(candidate["min_samples_leaf_proportion"]), weight, int(seed),
            )
            row, _, _, _ = model_record(
                f"E4_{cfg_name}_seed{seed}", "Etapa 4", official_period, target_kind, weight,
                candidate["criterion"], int(candidate["max_depth"]), float(candidate["min_samples_leaf_proportion"]),
                leaf_abs, int(seed), official_predictors, model, bundle, "test",
            )
            row["configuracao"] = cfg_name
            row["seed"] = int(seed)
            row["tipo_linha"] = "semente"
            stability_seed_rows.append(row)
            leaves = extract_leaves(model, bundle)
            root_var, root_cat, _ = root_and_level_variables(model, bundle.feature_meta)
            stability_structure_rows.append({
                "configuracao": cfg_name, "seed": int(seed), "raiz_variavel": root_var,
                "raiz_categoria": root_cat,
                "variaveis_primeiros_niveis": " | ".join(variables_by_depth(model, bundle.feature_meta, 2)),
                "numero_folhas": int(model.get_n_leaves()),
                "profundidade_real": int(model.get_depth()),
                "principal_regra_fatal": top_rule_text(leaves, "Fatal"),
                "principal_regra_grave": top_rule_text(leaves, "Grave") if target_kind == "4_classes" else "Não aplicável",
            })
            catalog.append({**row, "finalidade": "estabilidade", "teste_usado_na_selecao": False})
            del leaves, model, bundle
            gc.collect()

    stability_seeds = pd.DataFrame(stability_seed_rows)
    metric_cols = ["macro_f1", "balanced_accuracy", "precision_fatal", "recall_fatal", "f1_fatal", "mae_ordinal", "erro_distancia_ge_2_taxa", "numero_folhas"]
    summary_parts = []
    for cfg_name, group in stability_seeds.groupby("configuracao"):
        for statistic, func in (("media", "mean"), ("desvio_padrao", "std")):
            values = getattr(group[metric_cols], func)(numeric_only=True)
            summary_parts.append({"configuracao": cfg_name, "estatistica": statistic, **values.to_dict()})
    stability_summary = pd.DataFrame(summary_parts)
    stability_structure = pd.DataFrame(stability_structure_rows)
    root_frequency = (
        stability_structure.groupby(["configuracao", "raiz_variavel"], as_index=False).size()
        .rename(columns={"size": "frequencia_raiz"})
    )
    stability_structure = stability_structure.merge(root_frequency, on=["configuracao", "raiz_variavel"], how="left")

    # Sínteses comparativas.
    balance_comparison = stage1_df.copy()
    for metric in ("macro_f1", "balanced_accuracy", "precision_fatal", "recall_fatal", "f1_fatal", "f1_grave", "falsos_fatais"):
        balance_comparison[f"delta_{metric}_vs_sem_peso"] = balance_comparison[metric] - balance_comparison.groupby("periodo")[metric].transform("first")
    period_comparison = stage1_df.merge(
        period_distribution.pivot(index="periodo", columns="classe", values="proporcao").reset_index(),
        on="periodo", how="left",
    ).merge(
        period_distribution[period_distribution["classe"] == "Fatal"][["periodo", "n"]].rename(columns={"n": "volume_fatal"}),
        on="periodo", how="left",
    ).merge(annual_summary, on="periodo", how="left")
    compare_type = pd.DataFrame([official_test_row, no_type_test_row])
    compare_type["delta_macro_f1_vs_oficial"] = compare_type["macro_f1"] - official_test_row["macro_f1"]
    compare_type["delta_recall_fatal_vs_oficial"] = compare_type["recall_fatal"] - official_test_row["recall_fatal"]
    compare_three = pd.DataFrame([official_test_row, three_test_row])
    compare_three["tarefas_diferentes_nao_comparar_diretamente"] = True

    all_validation = pd.DataFrame(validation_rows)
    all_test_class = pd.concat([official_test_class, no_type_test_class, three_test_class], ignore_index=True)
    all_test_class["avaliacao"] = "teste"
    save_table(pd.concat(split_rows, ignore_index=True), TABLE_DIR / "distribuicao_classes_splits.csv")
    save_table(pd.DataFrame(catalog), TABLE_DIR / "catalogo_cenarios.csv")
    save_table(all_validation, TABLE_DIR / "metricas_validacao_todos_cenarios.csv")
    save_table(pd.DataFrame([official_test_row]), TABLE_DIR / "metricas_teste_modelo_oficial.csv")
    save_table(all_test_class, TABLE_DIR / "metricas_por_classe.csv")
    save_table(period_comparison, TABLE_DIR / "comparacao_periodos.csv")
    save_table(balance_comparison, TABLE_DIR / "comparacao_balanceamento.csv")
    save_table(compare_type, TABLE_DIR / "comparacao_com_sem_tipo_acidente.csv")
    save_table(compare_three, TABLE_DIR / "comparacao_quatro_tres_classes.csv")
    save_table(stability_seeds, TABLE_DIR / "estabilidade_sementes.csv")
    save_table(stability_summary, TABLE_DIR / "estabilidade_metricas_resumo.csv")
    save_table(stability_structure, TABLE_DIR / "estabilidade_estrutura.csv")

    # Figuras comparativas.
    plot_metric_scenarios(all_validation, "macro_f1", FIGURE_DIR / "comparacao_macro_f1_cenarios.png", "Macro F1 na validação")
    plot_metric_scenarios(all_validation, "recall_fatal", FIGURE_DIR / "comparacao_recall_fatal_cenarios.png", "Recall fatal na validação")
    plot_metric_scenarios(all_validation, "precision_fatal", FIGURE_DIR / "comparacao_precision_fatal_cenarios.png", "Precision fatal na validação")
    plot_f1_by_class(all_test_class, FIGURE_DIR / "comparacao_f1_por_classe.png")
    plot_balance_comparison(balance_comparison, FIGURE_DIR / "comparacao_balanceamento.png")
    plot_stability(stability_seeds, FIGURE_DIR / "estabilidade_metricas_sementes.png")

    # Artefatos técnicos oficiais.
    joblib.dump(official_model, MODEL_DIR / "modelo_arvore_oficial.joblib")
    joblib.dump(official_bundle.encoder, MODEL_DIR / "encoder_oficial.joblib")
    official_pipeline = Pipeline([("encoder", official_bundle.encoder), ("model", official_model)])
    joblib.dump(official_pipeline, MODEL_DIR / "pipeline_oficial.joblib")
    feature_frame = pd.DataFrame([
        {"feature_index": i, "feature_name": name, "variavel_original": var, "categoria": cat}
        for i, (name, (var, cat)) in enumerate(zip(official_bundle.feature_names, official_bundle.feature_meta))
    ])
    save_table(feature_frame, MODEL_DIR / "feature_names_oficial.csv")
    encoder_categories = {
        var: [str(value) for value in categories]
        for var, categories in zip(official_predictors, official_bundle.encoder.categories_)
    }
    save_json(encoder_categories, MODEL_DIR / "categorias_encoder_oficial.json")
    save_json(config["class_mapping_4"], MODEL_DIR / "mapeamento_classes.json")

    official_stability = stability_seeds[stability_seeds["configuracao"] == "oficial_4_classes"]
    recall_sd = float(official_stability["recall_fatal"].std(ddof=1))
    authorization = bool(
        official_test_row["recall_fatal"] >= 0.20
        and official_test_row["precision_fatal"] >= 0.10
        and official_test_row["f1_grave"] >= 0.20
        and official_test_row["macro_f1"] >= 0.25
        and official_test_row["erro_extremo_fatal_sem_vitimas_taxa"] < 0.10
        and recall_sd <= 0.05
    )
    status_protocol = "OFICIAL_AUTORIZADO" if authorization else "EXPLORATORIO_NAO_AUTORIZADO"
    official_config = {
        "status_protocolo": status_protocol,
        "numero_classes": 4,
        "mapeamento_classes": config["class_mapping_4"],
        "periodo_principal": official_period,
        "anos_periodo": config["periods"][official_period],
        "preditores": official_predictors,
        "balanceamento": class_weight_label(official_weight),
        "criterion": candidate["criterion"],
        "max_depth": int(candidate["max_depth"]),
        "min_samples_leaf_proporcional": float(candidate["min_samples_leaf_proportion"]),
        "min_samples_leaf_absoluto": int(official_leaf_abs),
        "random_state": int(config["random_state"]),
        "sementes_estabilidade": config["stability_seeds"],
        "metricas_oficiais_teste": {key: value for key, value in official_test_row.items() if isinstance(value, (int, float, np.integer, np.floating))},
        "versoes_bibliotecas": {
            "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
            "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__, "joblib": joblib.__version__,
        },
        "data_execucao": datetime.now().astimezone().isoformat(),
    }
    save_json(official_config, MODEL_DIR / "configuracao_oficial.json")

    protocol = pd.DataFrame([
        {"Componente": "Status", "Decisão oficial": status_protocol, "Justificativa": "Critérios mínimos multiclasse, ordinais e de estabilidade"},
        {"Componente": "Classes", "Decisão oficial": "4", "Justificativa": "Fatal isolada e informação Leve–Grave preservada"},
        {"Componente": "Período", "Decisão oficial": official_period, "Justificativa": "Seleção na validação combinando desempenho, atualidade e volume"},
        {"Componente": "Balanceamento", "Decisão oficial": class_weight_label(official_weight), "Justificativa": "Efeito explícito em Fatal e Grave avaliado na validação"},
        {"Componente": "Variáveis", "Decisão oficial": " | ".join(official_predictors), "Justificativa": "Conjunto pré-especificado sem vazamento"},
        {"Componente": "Criterion", "Decisão oficial": candidate["criterion"], "Justificativa": "Grade restrita e regra de parcimônia"},
        {"Componente": "Profundidade", "Decisão oficial": int(candidate["max_depth"]), "Justificativa": "Menor árvore dentro de 0,01 do melhor escore"},
        {"Componente": "Folha mínima", "Decisão oficial": f"{candidate['min_samples_leaf_proportion']:.1%} ({official_leaf_abs})", "Justificativa": "Regra proporcional replicável por território"},
        {"Componente": "Métricas", "Decisão oficial": "Macro F1; balanced accuracy; métricas Fatal/Grave; erros ordinais", "Justificativa": "Acurácia geral é secundária"},
        {"Componente": "Estabilidade", "Decisão oficial": "sementes 42, 52, 62, 72, 82", "Justificativa": f"DP recall fatal={recall_sd:.4f}"},
    ])
    save_table(protocol, TABLE_DIR / "protocolo_oficial.csv")

    trace_note = (
        "O EXP-05 referencia `tracado_macro` e registra sete categorias, mas não preserva a célula de criação. "
        "O EXP-06 aplica uma agregação explícita sobre `tracado_via_analise`, priorizando curva, interseção/retorno/rotatória, "
        "aclive/declive, obra de arte, obras/desvio, reta e outros; `MISSING` permanece separado. "
        "O mapeamento completo e as frequências foram salvos em `outputs/data/mapeamento_tracado_macro.csv`."
    )
    report = build_report(
        audit["auditoria_base"], period_distribution, stage1_df, stage2_df,
        official_test_row, official_test_class, {
            **candidate, "predictors": official_predictors, "min_samples_leaf_absolute": official_leaf_abs,
        },
        top_fatal, top_grave, top_leve, top_sem, fatal_grave, compare_type,
        compare_three, stability_summary, stability_structure, annual_summary,
        cm_off_abs, protocol, authorization, trace_note,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    logging.info("Relatório salvo: %s", relative(REPORT_PATH))

    required_tables = [
        "auditoria_base.csv", "auditoria_colunas.csv", "auditoria_alvos.csv",
        "distribuicao_classes_geral.csv", "distribuicao_classes_por_periodo.csv",
        "distribuicao_classes_por_ano.csv", "estabilidade_anual_periodos.csv",
        "distribuicao_classes_splits.csv", "checagem_vazamento.csv", "preditores_utilizados.csv",
        "catalogo_cenarios.csv", "metricas_validacao_todos_cenarios.csv",
        "metricas_teste_modelo_oficial.csv", "metricas_por_classe.csv", "comparacao_periodos.csv",
        "comparacao_balanceamento.csv", "comparacao_com_sem_tipo_acidente.csv",
        "comparacao_quatro_tres_classes.csv", "estabilidade_sementes.csv", "estabilidade_estrutura.csv",
        "folhas_modelo_oficial.csv", "folhas_modelo_sem_tipo.csv", "folhas_modelo_tres_classes.csv",
        "top_regras_fatal.csv", "top_regras_grave.csv", "top_regras_leve.csv",
        "top_regras_sem_vitimas.csv", "comparacao_regras_fatal_grave.csv",
    ]
    required_figures = [
        "distribuicao_classes_geral.png", "distribuicao_classes_por_periodo.png",
        "comparacao_macro_f1_cenarios.png", "comparacao_recall_fatal_cenarios.png",
        "comparacao_precision_fatal_cenarios.png", "comparacao_f1_por_classe.png",
        "comparacao_balanceamento.png", "estabilidade_metricas_sementes.png",
        "matriz_confusao_oficial_absoluta.png", "matriz_confusao_oficial_normalizada.png",
        "arvore_oficial_completa.png", "arvore_oficial_simplificada.png",
        "arvore_sem_tipo_completa.png", "arvore_sem_tipo_simplificada.png",
        "arvore_tres_classes_simplificada.png",
    ]
    required_models = [
        "modelo_arvore_oficial.joblib", "encoder_oficial.joblib", "pipeline_oficial.joblib",
        "configuracao_oficial.json", "feature_names_oficial.csv", "mapeamento_classes.json",
    ]
    missing = [str(TABLE_DIR / name) for name in required_tables if not (TABLE_DIR / name).exists()]
    missing += [str(FIGURE_DIR / name) for name in required_figures if not (FIGURE_DIR / name).exists()]
    missing += [str(MODEL_DIR / name) for name in required_models if not (MODEL_DIR / name).exists()]
    if not REPORT_PATH.exists():
        missing.append(str(REPORT_PATH))
    if missing:
        raise RuntimeError(f"Critérios de aceite falharam; artefatos ausentes: {missing}")
    logging.info("EXP-06 concluído com sucesso. Status do protocolo: %s", status_protocol)


def main() -> int:
    setup_logging()
    try:
        run()
    except Exception as exc:
        logging.exception("EXP-06 falhou: %s", exc)
        print(f"EXP-06 falhou. Consulte {LOG_PATH}")
        return 1
    print(f"EXP-06 concluído. Relatório: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
