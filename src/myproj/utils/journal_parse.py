import polars as pl
from typing import Tuple, Optional, Mapping, Sequence, Union, Literal

# ------------------------------
# Reusable column normalization
# ------------------------------
NormalizeMethod = Literal[
    "utf8",          # cast to Utf8
    "strip",         # strip leading/trailing whitespace
    "upper",         # uppercase
    "lower",         # lowercase
    "zfill9",        # zero-fill to 9 chars (useful for IPP)
    "ascii",         # normalise les caractères ASCII
]

DFLike = Union[pl.DataFrame, pl.LazyFrame]


def _df_columns(df: DFLike) -> list[str]:
    """Return column names for both DataFrame and LazyFrame without collecting data."""
    if isinstance(df, pl.LazyFrame):
        # collect_schema is cheap (metadata only) and avoids full collect.
        return df.collect_schema().names()
    return df.columns


def normalize_columns(
    df: DFLike,
    rules: Mapping[str, Sequence[NormalizeMethod]],
    *,
    strict: bool = True,
) -> DFLike:
    """Normalize selected columns using configurable rules.

    Parameters
    ----------
    df:
        Polars DataFrame or LazyFrame.
    rules:
        Mapping {column_name: [methods...]}.
        Methods are applied in order.
    strict:
        If True, raise if a requested column is missing.
        If False, silently skip missing columns.

    Returns
    -------
    Same type as input (DataFrame or LazyFrame) with normalized columns.
    """
    cols = set(_df_columns(df))

    exprs: list[pl.Expr] = []
    for col, methods in rules.items():
        if col not in cols:
            if strict:
                raise ValueError(f"Colonne {col!r} absente du DataFrame")
            continue

        e = pl.col(col)
        for m in methods:
            if m == "utf8":
                e = e.cast(pl.Utf8)
            elif m == "strip":
                e = e.cast(pl.Utf8).str.strip_chars()
            elif m == "upper":
                e = e.cast(pl.Utf8).str.to_uppercase()
            elif m == "lower":
                e = e.cast(pl.Utf8).str.to_lowercase()
            elif m == "zfill9":
                e = e.cast(pl.Utf8).str.zfill(9)
            elif m == "ascii":
                e = (
                    e.cast(pl.Utf8)
                    .str.replace_all(r"\s+", " ")
                    .str.replace_all("/nl/", "\\n")
                    .str.replace_all("@047@", "/")
                    .str.replace_all("@059@", ";")
                    .str.replace_all("@092@", "\\")
                )
            else:
                raise ValueError(f"Méthode de normalisation inconnue: {m!r}")

        exprs.append(e.alias(col))

    if not exprs:
        return df

    return df.with_columns(exprs)


def parse_journal(
    df_journal: pl.DataFrame,
    drop_col: bool,
    normalize_keys: bool = False,
    ) -> pl.DataFrame:
    """
    Analyse et parse la colonne `LIBELLE` issue de la table JOURNAL.

    Chaque libellé suit en pratique le schéma :

      1. Bloc journal (datetime + USER_ID + '@' + POST_ID)
      2. Nature de l'événement (URQUAL_ACTION)
      3. Identifiant patient/dossier (IPP ou IPPDATE), parfois absent selon le type d'événement
      4. Détail de l'événement (URQUAL_DETAIL), éventuellement structuré en code interne + texte

    Le parsing applique les mêmes transformations que le reste du pipeline :
    normalisation des espaces, décodage des séquences ASCII spéciales
    (`/nl/`, `@047@`, `@059@`, `@092@`), puis découpe en blocs
    pour reconstruire les colonnes analytiques.

    Args:
        df_journal: DataFrame Polars contenant au minimum la colonne
            `LIBELLE` telle qu'exportée du journal applicatif.
        drop_col: Si True, supprime les colonnes brutes non nécessaires (`LIBELLE`, `URQUAL_DETAIL`).
        normalize_keys: Si True, normalise les colonnes clés pour éviter les doublons liés à la casse/espaces.
            Applique typiquement `strip()` + `uppercase()` sur `USER_ID`, `POST_ID`, `URQUAL_ACTION`, `ACTION_CODE`.

    Returns:
        Un nouveau DataFrame Polars avec les colonnes dérivées suivantes :

        - `date_YYYYMMDD`: date du journal au format AAAAMMJJ.
        - `heure_HHMM`: heure du journal au format HHMM.
        - `USER_ID`: identifiant de l'utilisateur (extrait de LIBELLE).
        - `POST_ID`: identifiant du poste / contexte (extrait de LIBELLE).
        - `IPPDATE`: IPP ou IPP-Visite, si présent dans le détail.
        - `URQUAL_ACTION`: type d'événement (LOGIN, ENR_RESULTAT, INFOS_CACHE, etc.).
        - `ACTION_CODE`: code d'action dérivé de `URQUAL_DETAIL` quand c'est possible.
        - `ACTION_DETAIL`: partie textuelle du détail (sans le code).
        - `URQUAL_DETAIL`: détail nettoyé (partie avant éventuel '::').
        - `ETAT`: éventuelle partie d'état (suffixe `::...` conservé).
        - `IPP`: IPP pur (partie gauche de `IPPDATE`).
        - `visit_id`: identifiant de visite (partie droite de `IPPDATE`, si présente).
        - `datetime_str`: chaîne de datetime au format ISO "YYYY-MM-DD HH:MM".
        - `dt`: datetime typé Polars.

    Notes:
        Cette fonction ne filtre pas les lignes techniques ou sans IPP :
        elles restent présentes, avec `IPPDATE` et dérivés à null lorsque
        aucun identifiant patient n'est détecté.
    """
    # Étapes de traitement :
        # Normalise les espaces et certains tokens ASCII
    LIB_COND   = pl.col('LIBELLE')\
        .str.replace_all(r'\s+', ' ')\
        .str.replace_all('/nl/', '\\n')\
        .str.replace_all('@047@', '/')\
        .str.replace_all('@059@', ';')\
        .str.replace_all('@092@', '\\')

    SPLIT_2    = LIB_COND.str.splitn(' ', n=2)                          # 1ère séparation : journal | (action + détails)
    F_0        = SPLIT_2.struct.field('field_0')                        # bloc "journal" (datetime, user_id, poste_id)
    F_1        = SPLIT_2.struct.field('field_1')                        # bloc "action + IPP + détails"

    URQ_ACT    = F_1.str.splitn(' ', n=2).struct.field('field_0')       # nature de l'action
    RESTE_0    = F_1.str.splitn(' ', n=2).struct.field('field_1')       # partie "détails de l'action"

    # En général, le détail se structure ainsi : "IPPDATE  CODE  TEXTE"
    SPLIT_RESTE = RESTE_0.str.splitn(' ', n=2)

    # IPP/IPPDATE attendu en première position (sauf cas particuliers)
    IPP_COND  = SPLIT_RESTE.struct.field('field_0').str.contains(r'^\d+')          # vérifie présence d'un IPP/IPPDATE
    IPP_COND2 = URQ_ACT.is_in(['LOGIN'])                                           # LOGIN : pas d'IPP/IPPDATE, mais un résumé des connexions

    # Colonne intermédiaire contenant (IPPDATE, reste) lorsque pertinent
    IPP_DATE_int = (pl.when(IPP_COND & ~IPP_COND2).then(SPLIT_RESTE).otherwise(None))

    # IPPDATE final (ou IPP) : ne remonter que la 1ère partie si elle existe et est valide
    IPP_DATE = (
        pl.when(IPP_DATE_int.is_not_null())
        .then(IPP_DATE_int.struct.field('field_0'))
        .otherwise(None)
    )

    # Cas spécifique INFOS_CACHE : le nom d'action contient un suffixe de 11 caractères à extraire,
    # et la partie "détail" contient un chemin (C:\...) à nettoyer.
    ACT_COND = URQ_ACT.str.contains('INFOS_CACHE')

    # URQUAL_ACTION final : nettoyage du suffixe INFOS_CACHE (11 derniers caractères)
    URQ_ACT2 = pl.when(ACT_COND).then(URQ_ACT.str.slice(-11)).otherwise(URQ_ACT)

    # URQUAL_DETAIL brut (avant nettoyage des tokens ASCII) :
    # - si INFOS_CACHE : enlever le suffixe et décoder les backslashes
    # - sinon si IPP_DATE_int existe : prendre la 2e partie (après l'IPP/IPPDATE)
    # - sinon : reprendre le détail initial
    URQ_DET = (
        pl.when(ACT_COND)
        .then(F_1.str.slice(0, F_1.str.len_chars() - 11))
        .when(IPP_DATE_int.is_not_null())
        .then(IPP_DATE_int.struct.field('field_1'))
        .when(IPP_DATE_int.is_null())
        .then(RESTE_0)
    )

    # URQUAL_DETAIL nettoyé : décodage des tokens ASCII / coupe avant '::'
    URQ_DET_CLEAN = (URQ_DET.str.split('::').list.get(0, null_on_oob=True))
    ETAT = (
        URQ_DET
        .str.splitn('::', n=2)
        .struct.field('field_1')
        .str.replace(r"^:+", "")
        .fill_null("")
        )
      
    # Découpage du détail pour dériver ACTION_CODE / ACTION_DETAIL :
    # Règles pour limiter le nombre de codes :
    # - Exclure certaines actions : ['LOGIN', 'MediQual', 'INFOS_CACHE', 'ENR_CONSIGNE']
    # - Si le détail contient un motif de code structuré `_[A-Z]?[0-9]{3,4}`, le split se fait sur '_'
    # - Sinon, s'il contient ':', le split se fait sur ':'
    EXCLUDED = ['LOGIN', 'MediQual', 'INFOS_CACHE', 'ENR_CONSIGNE']
    ACT_COND2 = URQ_DET_CLEAN.str.contains(r'_[A-Z]?[0-9]{3,4}') & (~URQ_ACT2.is_in(EXCLUDED))
    ACT_COND3 = URQ_DET_CLEAN.str.contains(':')                      & (~URQ_ACT2.is_in(EXCLUDED))

    ACT_SPLIT2 = URQ_DET_CLEAN.str.splitn('_', n=2)
    ACT_SPLIT3 = URQ_DET_CLEAN.str.splitn(':', n=2)

    # ACTION_CODE : partie gauche du split (selon motif)
    ACT_COD = (
        pl.when(ACT_COND3).then(ACT_SPLIT3.struct.field('field_0'))
        .when(ACT_COND2).then(ACT_SPLIT2.struct.field('field_0'))
        .otherwise(None)
    )

    # ACTION_DETAIL : partie droite du split si applicable, sinon le détail complet nettoyé
    ACT_DET = (
        pl.when(ACT_COND3).then(ACT_SPLIT3.struct.field('field_1'))
        .when(ACT_COND2).then(ACT_SPLIT2.struct.field('field_1'))
 
        .otherwise(URQ_DET_CLEAN)
    )

    # Materialisation des colonnes finales (normalisation optionnelle des clés)
    df1 = df_journal.with_columns(
        F_0.str.slice(0, 8).alias('date_YYYYMMDD'),
        F_0.str.slice(8, 4).alias('heure_HHMM'),
        F_0.str.split('@').list.get(0, null_on_oob=True).str.slice(12).alias('USER_ID'),
        F_0.str.split('@').list.get(1, null_on_oob=True).alias('POST_ID'),
        IPP_DATE.cast(pl.Utf8).alias('IPPDATE'),
        URQ_ACT2.alias('URQUAL_ACTION'),
        ACT_COD.alias('ACTION_CODE'),
        ACT_DET.alias('ACTION_DETAIL'),
        URQ_DET_CLEAN.alias('URQUAL_DETAIL'),
        ETAT.alias('ETAT'),
        )
    
    if normalize_keys:
        df1 = normalize_columns(
            df1,
            {
                'USER_ID':['utf8','strip','upper'],
                "POST_ID":['utf8','strip','upper'],
                'URQUAL_ACTION':['utf8','strip','upper'],
                'ACTION_CODE':['utf8','strip','upper'],
            },
            strict=True
        )

    df1 = (
        df1
        .with_columns([
            pl.col('IPPDATE').str.split("-").list.get(0,null_on_oob=True).alias('IPP'),
            pl.col('IPPDATE').str.split("-").list.get(1,null_on_oob=True).alias('visit_id'),
            # cast en string
            pl.col("date_YYYYMMDD").cast(pl.Utf8),
        # pad HHMM → toujours 4 caractères (ex: "915" -> "0915")
            pl.col("heure_HHMM").cast(pl.Utf8),
        ])
        .with_columns([
            # construire une chaîne complète YYYY-MM-DD HH:MM
            (
                pl.col("date_YYYYMMDD").str.slice(0, 4) + "-" +
                pl.col("date_YYYYMMDD").str.slice(4, 2) + "-" +
                pl.col("date_YYYYMMDD").str.slice(6, 2) + " " +
                pl.col("heure_HHMM").str.slice(0, 2) + ":" +
                pl.col("heure_HHMM").str.slice(2, 2)
            ).alias("datetime_str")
        ])
        # convertir en datetime
        .with_columns(
            pl.col("datetime_str").str.to_datetime(format="%Y-%m-%d %H:%M").alias("dt")
        )
    )
    
    if drop_col:
       return df1.drop(["LIBELLE","URQUAL_DETAIL"])
    else:
        return df1
    
    
def _add_ippdate_parts(
    df: DFLike,
    ipp_col: str = "IPP",
    ippdate_col: str = "IPPDATE",
    *,
    strict: bool = True,
) -> DFLike:
    """Normalise IPP/IPPDATE et dérive `IPP_ippdate` + `visit_idx`.

    - IPP est normalisé en string 9 caractères (zfill9).
    - IPPDATE est casté en Utf8 (pas de modification du contenu).
    - `IPP_ippdate` = partie gauche de IPPDATE (avant '-')
    - `visit_idx` = partie droite de IPPDATE (après '-') castée en UInt64

    Compatible DataFrame et LazyFrame.
    """
    # 1) Normaliser les colonnes clés via l'helper générique
    df = normalize_columns(
        df,
        {
            ipp_col: ["utf8", "strip", "zfill9"],
            ippdate_col: ["utf8", "strip"],
        },
        strict=strict,
    )

    # 2) Dérivations à partir de IPPDATE
    split_ippdate = pl.col(ippdate_col).str.split_exact("-", 1)

    return df.with_columns(
        split_ippdate.struct.field("field_0").alias("IPP_ippdate"),
        split_ippdate.struct.field("field_1").cast(pl.UInt64).alias("visit_idx"),
    )

def _reduce_JOURNAL(
    df_journal: pl.DataFrame,
    j_ipp_col: str,
    j_ippdate_col: str,
    j_actioncode_col: str,
    j_actiondetail_col: str,
    j_actioncode_dateheureentree: str,
):
    df_j_pre = (
        df_journal
        .select(j_ipp_col,j_ippdate_col,j_actioncode_col,j_actiondetail_col)
        .filter(pl.col(j_actioncode_col) == j_actioncode_dateheureentree)
        .drop(j_actioncode_col)
        .group_by([j_ipp_col,j_ippdate_col])
        .agg(pl.col(j_actiondetail_col).first().alias('dt_entree_str'))
        .with_columns(
            pl.col('dt_entree_str')
            .str.to_datetime(format="%Y%m%d%H%M%S")
            .dt.truncate('1m')
            .alias('dt_entree')
        )
    )
    
    return df_j_pre


def find_concordance_JOURNAL_MULTICOL(
    df_journal:pl.DataFrame,
    df_multicol:pl.DataFrame,
    j_ipp_col: str = "IPP",
    j_ippdate_col: str = "IPPDATE",
    j_actioncode_col: str = "ACTION_CODE",
    j_actiondetail_col: str = "ACTION_DETAIL",
    j_actioncode_dateheureentree:str ="DATERDV",
    m_ipp_col: str = "IPP",
    m_ippdate_col: str = "IPPDATE",
    m_ist_col: str = "IST",
    m_dateentree_col:str = "DATE_ENTREE",
    m_heureentree_col:str = "HEURE_ENTREE",
):
    """
        Construit une table de concordance JOURNAL ↔ MULTICOL pour stabiliser l'identifiant de séjour.

        Objectif
        --------
        Le JOURNAL peut contenir un `IPPDATE` parfois divergent ou incomplet. MULTICOL est utilisé
        comme référence de séjour. La concordance est construite en alignant :
            - l'IPP (normalisé)
            - la date/heure d'entrée (ancre temporelle), dérivée de l'événement `DATERDV` côté JOURNAL
            et de (DATE_ENTREE, HEURE_ENTREE) côté MULTICOL, toutes deux tronquées à la minute.

        Entrées
        -------
        df_journal:
            JOURNAL parsé (incluant au minimum IPP, IPPDATE, ACTION_CODE, ACTION_DETAIL).
            La colonne `ACTION_DETAIL` de l'action `DATERDV` est supposée contenir un datetime au format
            `%Y%m%d%H%M%S`.
        df_multicol:
            Table MULTICOL contenant IPPDATE, IST, DATE_ENTREE, HEURE_ENTREE.

        Colonnes de sortie (df_corr)
        ----------------------------
        Côté JOURNAL (suffixe _j implicite) :
        - IPP_j
        - IPPDATE_j
        - IPP_ippdate
        - visit_idx
        - dt_entree

        Côté MULTICOL (suffixe _m explicite) :
        - IPP (colonne MULTICOL conservée telle quelle si présente dans df_m)
        - IPP_m
        - IPPDATE_m
        - IST_m
        - IPP_ippdate_m (si collision de nom, via suffix Polars)
        - visit_idx_m   (si collision de nom, via suffix Polars)

        Colonnes dérivées :
        - IPPDATE_corr :
            * = IPPDATE_m si IPPDATE_m existe ET IPPDATE_m != IPPDATE_j
            * sinon = IPPDATE_j
        - is_different_IPPDATE (tri-état) :
            * True  si IPPDATE_m existe ET différent de IPPDATE_j
            * False si IPPDATE_m existe ET égal à IPPDATE_j
            * Null  si IPPDATE_m est absent (pas de match)

        Notes
        -----
        - Cette table sert à la fois de mapping ET de contrôle qualité (QC).
        - Pour l'enrichissement du JOURNAL, on sélectionne ensuite un sous-ensemble minimal
        (IPPDATE_j → IPPDATE_multicol, IST, is_different_multicol).
        """
    df_j_pre = _reduce_JOURNAL(
        df_journal,
        j_ipp_col=j_ipp_col,
        j_ippdate_col=j_ippdate_col,
        j_actioncode_col=j_actioncode_col,
        j_actiondetail_col=j_actiondetail_col,
        j_actioncode_dateheureentree=j_actioncode_dateheureentree
    )  

    # --- 2) Normalisation IPP/IPPDATE ---
    df_j = (
        _add_ippdate_parts(
            df=df_j_pre,
            ipp_col=j_ipp_col,
            ippdate_col=j_ippdate_col,
        )
        .select(
            pl.col(j_ipp_col).alias("IPP_j"),
            pl.col(j_ippdate_col).alias("IPPDATE_j"),
            "IPP_ippdate",
            "visit_idx",
            "dt_entree",
        )
    )

    df_m_pre = (
        df_multicol
        .select(
            m_ipp_col,
            m_ippdate_col,
            m_ist_col,
            m_dateentree_col,
            m_heureentree_col,
        )
    )
    
    df_m_clean = _add_ippdate_parts(
        df = df_m_pre,
        ipp_col = m_ipp_col,
        ippdate_col = m_ippdate_col
    )
    
    df_m = (
        df_m_clean
        .with_columns(
            (
                pl.col(m_dateentree_col) 
                + ' ' 
                + pl.col(m_heureentree_col)
            )
            .cast(pl.Utf8)
            .str.to_datetime(format="%d/%m/%Y %H:%M")
            .dt.truncate('1m')
            .alias("dt_entree")
        )
        .select(
            pl.col(m_ipp_col).alias("IPP_m"),
            pl.col(m_ippdate_col).alias("IPPDATE_m"),
            pl.col(m_ist_col).alias("IST_m"),
            "IPP_ippdate",
            "visit_idx",
            "dt_entree",
        )
    )

    df_corr = (
        df_j
        .join(
            df_m,
            left_on=["IPP_j", "dt_entree"],
            right_on=["IPP_m", "dt_entree"],
            how="left",
            suffix="_m",       
        )
        .with_columns(
            pl.when(
                pl.col('IPPDATE_m').is_not_null()
                & (pl.col("IPPDATE_j") != pl.col('IPPDATE_m'))
            )
            .then(pl.col('IPPDATE_m'))
            .otherwise(pl.col('IPPDATE_j'))
            .alias("IPPDATE_corr"),
            pl.when(pl.col('IPPDATE_m').is_null())
            .then(None)
            .otherwise(pl.col("IPPDATE_j") != pl.col('IPPDATE_m'))
            .alias('is_different_IPPDATE'),
        )
    )
    
    return df_corr


def check_unique_IPPDATE_DATERDV(
    df:pl.DataFrame,
    actioncode_col:str = "ACTION_CODE",
    actioncode_value:str = "DATERDV",
    ipp_col:str = "IPP",
    ippdate_col:str = "IPPDATE",
    actiondetail_col:str = "ACTION_DETAIL"
)-> bool:
    
    df_j_check = (
        df
        .filter(pl.col(actioncode_col) == actioncode_value)
        .group_by([ipp_col, ippdate_col])
        .agg(
            pl.col(actiondetail_col).n_unique().alias("n_dt"),
            pl.col(actiondetail_col).first().alias("example_DATERDV"),
        )
        .filter(pl.col("n_dt") > 1)
    )
    return df_j_check.is_empty(), df_j_check


def add_concordance_bt_JOURNAL_MULTICOL(
    df_journal:pl.DataFrame,
    df_multicol:pl.DataFrame,
    j_ipp_col: str = "IPP",
    j_ippdate_col: str = "IPPDATE",
    j_actioncode_col: str = "ACTION_CODE",
    j_actiondetail_col: str = "ACTION_DETAIL",
    j_actioncode_dateheureentree:str ="DATERDV",
    m_ipp_col: str = "IPP",
    m_ippdate_col: str = "IPPDATE",
    m_dateentree_col:str = "DATE_ENTREE",
    m_heureentree_col:str = "HEURE_ENTREE",
    drop_na_ippdate:bool = False,
)-> Tuple[pl.DataFrame,pl.DataFrame]:
    """
    Enrichit le JOURNAL avec l'identifiant de séjour de référence MULTICOL + clé IST.

    But
    ---
    - Produire un identifiant de séjour stable `IPPDATE_multicol` basé sur MULTICOL quand nécessaire.
    - Conserver un indicateur `is_different_multicol` indiquant si `IPPDATE` JOURNAL diffère de MULTICOL.
    - Ajouter `IST` (issu de MULTICOL) pour permettre les jointures vers les autres tables non-URQUAL.

    Paramètres clés
    ---------------
    drop_na_ippdate:
        - False (défaut): conserve toutes les lignes du JOURNAL, même sans concordance.
        - True : filtre les lignes où `IPPDATE_multicol` est null (recommandé pour les métriques dashboard).

    Sorties
    -------
    df_journal_new:
        df_journal enrichi avec 3 colonnes ajoutées :
        - `IPPDATE_multicol`
        - `is_different_multicol`
        - `IST`
    df_corr:
        Table de concordance détaillée (QC / vérification), telle que produite par
        `find_concordance_JOURNAL_MULTICOL`.
    """
    df_corr = find_concordance_JOURNAL_MULTICOL(
        df_journal, 
        df_multicol,
        j_ipp_col = j_ipp_col,
        j_ippdate_col= j_ippdate_col,
        j_actioncode_col = j_actioncode_col,
        j_actiondetail_col = j_actiondetail_col,
        j_actioncode_dateheureentree =j_actioncode_dateheureentree,
        m_ipp_col = m_ipp_col,
        m_ippdate_col = m_ippdate_col,
        m_dateentree_col = m_dateentree_col,
        m_heureentree_col = m_heureentree_col,
    )
    
    df_join = df_corr.select(
        'IPPDATE_j',
        pl.col('IST_m').alias('IST'),
        pl.col('IPPDATE_corr').alias('IPPDATE_multicol'),
        pl.col('is_different_IPPDATE').alias('is_different_multicol')
    )

    df_journal_new = df_journal.join(
        df_join,
        how="left",
        left_on=["IPPDATE"],
        right_on=["IPPDATE_j"],
        suffix="_corr"
    )
    
    if drop_na_ippdate:
        return df_journal_new.filter(pl.col('IPPDATE_multicol').is_not_null()),df_corr

    return df_journal_new,df_corr
    
    
def clean_actioncode_col(
    df:pl.DataFrame, 
    actioncode_col:str,
    )->None:
    
    
    if actioncode_col not in df.columns:
        raise ValueError(f"Colonne {actioncode_col!r} absente du DataFrame")
    
    AC = pl.col("ACTION_CODE")

    # 1) Expressions réutilisables pour la première étape (X ACTES ...)
    cond_actes = AC.str.contains(r"[A-Z]+\s+ACTES")
    split_space = AC.str.splitn(" ", 2)

    ref_expr = pl.when(cond_actes) \
        .then(split_space.struct.field("field_1")) \
        .otherwise(AC)

    etacte_expr = pl.when(cond_actes) \
        .then(split_space.struct.field("field_0")) \
        .otherwise(pl.lit(None))

    df = (
        df
        .with_columns(
            ref_expr.alias("ACTION_CODE"),
            etacte_expr.alias("ETACTE"),
        )
    )

    # 2) Expressions pour la partie codes type _A123 / _1234
    cond_ref_code = AC.str.contains(r"_[a-zA-Z]?\d{3,4}")
    split_ref = AC.str.splitn("_", 2)

    df = (
        df
        .with_columns(
            pl.when(cond_ref_code)
            .then(split_ref.struct.field("field_0"))
            .otherwise(AC)
            .alias("ACTION_CODE"),

            pl.when(cond_ref_code)
            .then(split_ref.struct.field("field_1"))
            .otherwise(pl.lit(None))
            .alias("ACTION_CODE_PART"),
        )
    )
    
    return df


def add_UF_name(
    df:pl.DataFrame,
    df_UF: pl.DataFrame,
    UF_col: str = "UF",
    actioncode_col: str = "ACTION_CODE",
    actioncode: str = "UFACTE",
    actiondetail_col: str = "ACTION_DETAIL",
    ippdate_col: str = "IPPDATE_multicol",
    uf_select: Optional[str] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    uf_select:
        - None  : garde toutes les combinaisons IPPDATE x UF (peut dupliquer les lignes au join final)
        - "first": choisit la première UF rencontrée pour chaque IPPDATE
        - "last" : choisit la dernière UF rencontrée pour chaque IPPDATE
        - "SAU", "UHCD", "PSY" ,"UGA" : privilégie le TYPE_UF demandé, sinon retombe sur la première UF du groupe.
    """

    # 1) Construire la table de correspondance IPPDATE x UF à partir des lignes UFACTE
    pl_ippdate_uf = (
        df.filter(
            (pl.col(actioncode_col) == actioncode)
            & (pl.col(ippdate_col).is_not_null())
        )
        .select(
            pl.col(ippdate_col),
            pl.col(actiondetail_col).alias("UF_code"),
        )
        .unique()
        .join(
            df_UF,
            left_on="UF_code",
            right_on=UF_col,
            how="left"
        )
    )
    
    
    # 2) Optionnel : réduire à une seule UF par IPPDATE selon la stratégie uf_select
    if uf_select is not None:
        valid_strategies = {"first", "last", "SAU", "UHCD", "PSY", "UGA"}
        if uf_select not in valid_strategies:
            raise ValueError(
                f"uf_select doit être l'un de {sorted(valid_strategies)} ou None, reçu: {uf_select!r}"
            )

        if uf_select in ("first", "last"):
            keep = "first" if uf_select == "first" else "last"
            pl_ippdate_uf = (
                pl_ippdate_uf
                .sort(ippdate_col)
                .unique(subset=[ippdate_col], keep=keep)
            )
        else:
            # Priorité sur un TYPE_UF donné (ex: "SAU", "UHCD", "PSY")
            pl_ippdate_uf = (
                pl_ippdate_uf
                .with_columns(
                    pl.when(pl.col("TYPE_UF") == uf_select)
                    .then(1)
                    .otherwise(0)
                    .alias("_pref")
                )
                .sort([ippdate_col, "_pref"], descending=[False, True])
                .unique(subset=[ippdate_col], keep="first")
                .drop("_pref")
            )

    # 3) Join final : enrichir toutes les lignes de df partageant cet IPPDATE
    df_enrichi = df.join(
        pl_ippdate_uf,
        how="left",
        on=ippdate_col,
    )
    
    return df_enrichi, pl_ippdate_uf
    
    