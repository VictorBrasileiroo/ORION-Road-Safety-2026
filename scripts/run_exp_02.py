from __future__ import annotations

import logging
import sys
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parents[1]
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from src.audit import build_auditoria_geral, validate_acceptance, validate_required_columns  # noqa: E402
from src.data_io import load_config, read_base, save_base_csv, save_table  # noqa: E402
from src.harmonizacao import apply_harmonization  # noqa: E402
from src.paths import (  # noqa: E402
    CONFIG_PATH,
    CONTEXT_COLUMNS,
    DATA_DIR,
    EXP_DIR as EXP02_DIR,
    FIGURE_DIR,
    LOG_PATH,
    REPORT_PATH,
    TABLE_DIR,
    ensure_dirs,
    project_path,
)
from src.plotting import (  # noqa: E402
    plot_distribuicao_severidade_geral,
    plot_extremos_barras,
    plot_resumo_variaveis_contextuais,
    plot_severidade_por_ano,
    plot_top_contextos,
)
from src.rankings import (  # noqa: E402
    build_ranking_contextos_criticos,
    build_ranking_contextos_sem_vitimas,
    build_ranking_contraste_extremos,
    build_recomendacao_variaveis_proximos_exps,
    build_resumo_variaveis_contextuais,
)
from src.report import render_report  # noqa: E402
from src.severidade import (  # noqa: E402
    build_distribuicao_severidade_geral,
    build_severidade_por_grupo,
    build_severidade_por_variavel_categoria,
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


def run() -> None:
    logging.info("EXP-02 iniciado")
    config = load_config(CONFIG_PATH)
    input_path = project_path(config["input_base"])
    output_path = project_path(config["output_base"])
    low_volume_threshold = int(config["low_volume_threshold"])
    top_n = int(config["top_n"])

    if not input_path.exists():
        raise FileNotFoundError(f"Base oficial do EXP-01 nao encontrada: {input_path}")

    df = read_base(input_path)
    rows_before = len(df)
    cols_before = len(df.columns)
    logging.info("Base EXP-01 lida com %s linhas e %s colunas", rows_before, cols_before)

    validate_required_columns(df)
    logging.info("Colunas obrigatorias validadas")

    auditoria_harmonizacao = apply_harmonization(df, low_volume_threshold)
    logging.info("Harmonizacao categorica controlada concluida")

    auditoria_geral = build_auditoria_geral(
        df_before_rows=rows_before,
        df_before_cols=cols_before,
        df=df,
        config=config,
        input_base=config["input_base"],
    )
    distribuicao = build_distribuicao_severidade_geral(df)
    severidade_por_ano = build_severidade_por_grupo(df, config["year_column"], output_col="ano_evento")
    severidade_por_periodo = build_severidade_por_grupo(df, config["period_column"], output_col="periodo")
    severidade_por_regiao = build_severidade_por_grupo(df, config["region_column"], output_col="regiao")
    severidade_por_uf = build_severidade_por_grupo(
        df,
        config["uf_column"],
        output_col="uf_limpa",
        low_volume_threshold=low_volume_threshold,
    )
    severidade_por_variavel = build_severidade_por_variavel_categoria(
        df,
        CONTEXT_COLUMNS,
        low_volume_threshold=low_volume_threshold,
    )
    ranking_critico = build_ranking_contextos_criticos(severidade_por_variavel)
    ranking_sem = build_ranking_contextos_sem_vitimas(severidade_por_variavel)
    ranking_contraste = build_ranking_contraste_extremos(severidade_por_variavel)
    resumo_variaveis = build_resumo_variaveis_contextuais(severidade_por_variavel)
    recomendacao = build_recomendacao_variaveis_proximos_exps(resumo_variaveis)
    logging.info("Tabelas analiticas calculadas")

    save_table(auditoria_geral, TABLE_DIR / "auditoria_geral_exp02.csv")
    save_table(auditoria_harmonizacao, TABLE_DIR / "auditoria_harmonizacao_categorias.csv")
    save_table(distribuicao, TABLE_DIR / "distribuicao_severidade_geral.csv")
    save_table(severidade_por_ano, TABLE_DIR / "severidade_por_ano.csv")
    save_table(severidade_por_periodo, TABLE_DIR / "severidade_por_periodo.csv")
    save_table(severidade_por_regiao, TABLE_DIR / "severidade_por_regiao.csv")
    save_table(severidade_por_uf, TABLE_DIR / "severidade_por_uf.csv")
    save_table(severidade_por_variavel, TABLE_DIR / "severidade_por_variavel_categoria.csv")
    save_table(ranking_critico, TABLE_DIR / "ranking_contextos_criticos.csv")
    save_table(ranking_sem, TABLE_DIR / "ranking_contextos_sem_vitimas.csv")
    save_table(ranking_contraste, TABLE_DIR / "ranking_contraste_extremos.csv")
    save_table(resumo_variaveis, TABLE_DIR / "resumo_variaveis_contextuais.csv")
    save_table(recomendacao, TABLE_DIR / "recomendacao_variaveis_proximos_exps.csv")
    logging.info("Tabelas salvas")

    plot_distribuicao_severidade_geral(
        distribuicao,
        FIGURE_DIR / "grafico_distribuicao_severidade_geral.png",
    )
    plot_severidade_por_ano(severidade_por_ano, FIGURE_DIR / "grafico_severidade_por_ano.png")
    plot_extremos_barras(
        severidade_por_periodo,
        "periodo",
        FIGURE_DIR / "grafico_critico_sem_vitimas_por_periodo.png",
        "Taxas de critico e sem vitimas por periodo",
    )
    plot_extremos_barras(
        severidade_por_regiao,
        "regiao",
        FIGURE_DIR / "grafico_critico_sem_vitimas_por_regiao.png",
        "Taxas de critico e sem vitimas por regiao",
    )
    plot_top_contextos(
        ranking_critico,
        FIGURE_DIR / "grafico_top_contextos_criticos.png",
        metric_col="critico_taxa",
        title="Top contextos associados ao extremo critico",
        top_n=top_n,
    )
    plot_top_contextos(
        ranking_sem,
        FIGURE_DIR / "grafico_top_contextos_sem_vitimas.png",
        metric_col="sem_vitimas_taxa",
        title="Top contextos associados a acidentes sem vitimas",
        top_n=top_n,
    )
    plot_resumo_variaveis_contextuais(
        resumo_variaveis,
        FIGURE_DIR / "grafico_resumo_variaveis_contextuais.png",
    )
    logging.info("Graficos salvos")

    save_base_csv(df, output_path)
    logging.info("Base EXP-02 salva em %s", output_path)

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
        artifact_dir=EXP02_DIR,
        df=df,
        rows_before=rows_before,
    )
    if errors:
        for error in errors:
            logging.error(error)
        raise RuntimeError("EXP-02 falhou nos criterios de aceite")

    logging.info("EXP-02 concluido com sucesso")


def main() -> int:
    setup_logging()
    try:
        run()
    except Exception as exc:
        logging.exception("EXP-02 falhou: %s", exc)
        print(f"EXP-02 falhou. Veja o log: {LOG_PATH}")
        return 1
    print(f"EXP-02 concluido com sucesso. Relatorio: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
