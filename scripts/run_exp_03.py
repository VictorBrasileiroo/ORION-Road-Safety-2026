from __future__ import annotations

import logging
import sys
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parents[1]
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from src.audit import build_auditoria_geral, validate_acceptance, validate_required_columns  # noqa: E402
from src.bivariada import (  # noqa: E402
    build_comparacao_critico_fatal_grave,
    build_critico_por_periodo_variavel,
    build_critico_por_regiao_variavel,
    build_taxa_critico_por_variavel_categoria,
)
from src.data_io import load_config, read_base, save_base_csv, save_table  # noqa: E402
from src.paths import (  # noqa: E402
    ANALYSIS_VARIABLES,
    CONFIG_PATH,
    DATA_DIR,
    EXP_DIR as EXP03_DIR,
    FIGURE_DIR,
    LOG_PATH,
    PERIOD_ANALYSIS_VARIABLES,
    REGION_ANALYSIS_VARIABLES,
    REPORT_PATH,
    TABLE_DIR,
    ensure_dirs,
    project_path,
)
from src.plotting import (  # noqa: E402
    plot_alto_risco_alto_volume,
    plot_critico_fatal_grave_top_categorias,
    plot_delta_periodo_top_variaveis,
    plot_resumo_variaveis_critico,
    plot_top_lift_critico,
    plot_top_taxa_critico,
    plot_top_volume_critico,
)
from src.rankings import (  # noqa: E402
    build_ranking_alto_risco_alto_volume,
    build_ranking_alto_risco_baixo_volume,
    build_ranking_lift_critico,
    build_ranking_taxa_critico,
    build_ranking_volume_critico,
    build_recomendacao_variaveis_exp05,
    build_resumo_variaveis_critico,
)
from src.report import render_report  # noqa: E402


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


def run() -> None:
    logging.info("EXP-03 iniciado")
    config = load_config(CONFIG_PATH)
    input_path = project_path(config["input_base"])
    output_path = project_path(config["output_base"])
    target = config["target"]
    low_volume_threshold = int(config["low_volume_threshold"])
    top_n = int(config["top_n"])
    confidence_level = float(config["confidence_level"])

    if not input_path.exists():
        raise FileNotFoundError(f"Base oficial do EXP-02 nao encontrada: {input_path}")

    df = read_base(input_path)
    rows_before = len(df)
    cols_before = len(df.columns)
    logging.info("Base EXP-02 lida com %s linhas e %s colunas", rows_before, cols_before)

    validate_required_columns(df)
    logging.info("Colunas obrigatorias validadas")

    varcat = build_taxa_critico_por_variavel_categoria(
        df,
        ANALYSIS_VARIABLES,
        target=target,
        low_volume_threshold=low_volume_threshold,
        confidence_level=confidence_level,
    )
    global_critico_taxa = float(df[target].mean())

    ranking_taxa = build_ranking_taxa_critico(varcat)
    ranking_lift = build_ranking_lift_critico(varcat)
    ranking_volume = build_ranking_volume_critico(varcat)
    ranking_alto_volume = build_ranking_alto_risco_alto_volume(varcat, global_critico_taxa)
    ranking_baixo_volume = build_ranking_alto_risco_baixo_volume(varcat, global_critico_taxa)
    resumo = build_resumo_variaveis_critico(varcat)
    comparacao = build_comparacao_critico_fatal_grave(
        varcat,
        low_volume_threshold=low_volume_threshold,
    )
    periodo = build_critico_por_periodo_variavel(
        df,
        PERIOD_ANALYSIS_VARIABLES,
        target=target,
        low_volume_threshold=low_volume_threshold,
    )
    regiao = build_critico_por_regiao_variavel(
        df,
        REGION_ANALYSIS_VARIABLES,
        target=target,
        low_volume_threshold=low_volume_threshold,
    )
    recomendacao = build_recomendacao_variaveis_exp05(resumo)
    auditoria = build_auditoria_geral(
        config=config,
        input_base=config["input_base"],
        df_before_rows=rows_before,
        df_before_cols=cols_before,
        df=df,
        variables=ANALYSIS_VARIABLES,
    )
    logging.info("Tabelas analiticas calculadas")

    save_table(auditoria, TABLE_DIR / "auditoria_geral_exp03.csv")
    save_table(varcat, TABLE_DIR / "taxa_critico_por_variavel_categoria.csv")
    save_table(ranking_taxa, TABLE_DIR / "ranking_taxa_critico.csv")
    save_table(ranking_lift, TABLE_DIR / "ranking_lift_critico.csv")
    save_table(ranking_volume, TABLE_DIR / "ranking_volume_critico.csv")
    save_table(ranking_alto_volume, TABLE_DIR / "ranking_alto_risco_alto_volume.csv")
    save_table(ranking_baixo_volume, TABLE_DIR / "ranking_alto_risco_baixo_volume.csv")
    save_table(resumo, TABLE_DIR / "resumo_variaveis_critico.csv")
    save_table(comparacao, TABLE_DIR / "comparacao_critico_fatal_grave.csv")
    save_table(periodo, TABLE_DIR / "critico_por_periodo_variavel.csv")
    save_table(regiao, TABLE_DIR / "critico_por_regiao_variavel.csv")
    save_table(recomendacao, TABLE_DIR / "recomendacao_variaveis_exp05.csv")

    plot_top_taxa_critico(ranking_taxa, FIGURE_DIR / "grafico_top_taxa_critico.png", top_n=top_n)
    plot_top_lift_critico(ranking_lift, FIGURE_DIR / "grafico_top_lift_critico.png", top_n=top_n)
    plot_top_volume_critico(ranking_volume, FIGURE_DIR / "grafico_top_volume_critico.png", top_n=top_n)
    plot_alto_risco_alto_volume(
        ranking_alto_volume,
        FIGURE_DIR / "grafico_alto_risco_alto_volume.png",
        top_n=top_n,
    )
    plot_resumo_variaveis_critico(resumo, FIGURE_DIR / "grafico_resumo_variaveis_critico.png")
    plot_critico_fatal_grave_top_categorias(
        comparacao,
        ranking_taxa,
        FIGURE_DIR / "grafico_critico_fatal_grave_top_categorias.png",
    )
    plot_delta_periodo_top_variaveis(
        periodo,
        FIGURE_DIR / "grafico_delta_periodo_top_variaveis.png",
        top_n=top_n,
        low_volume_threshold=low_volume_threshold,
    )
    logging.info("Graficos salvos")

    save_base_csv(df, output_path)

    report_text = render_report(TABLE_DIR, FIGURE_DIR, DATA_DIR, LOG_PATH)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    logging.info("Relatorio salvo em %s", REPORT_PATH)

    errors = validate_acceptance(
        input_path=input_path,
        output_path=output_path,
        table_dir=TABLE_DIR,
        figure_dir=FIGURE_DIR,
        report_path=REPORT_PATH,
        log_path=LOG_PATH,
        artifact_dir=EXP03_DIR,
        df=df,
        rows_before=rows_before,
    )
    if errors:
        for error in errors:
            logging.error(error)
        raise RuntimeError("EXP-03 falhou nos criterios de aceite")

    logging.info("EXP-03 concluido com sucesso")


def main() -> int:
    setup_logging()
    try:
        run()
    except Exception as exc:
        logging.exception("EXP-03 falhou: %s", exc)
        print(f"EXP-03 falhou. Veja o log: {LOG_PATH}")
        return 1
    print(f"EXP-03 concluido com sucesso. Relatorio: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
