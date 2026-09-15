"""
EmotionLens — Registro de Procedencia Científica

Fuente única de verdad sobre el respaldo (o la ausencia de respaldo) de cada
constante numérica del sistema. El PDF de bibliografía se genera a partir de
este módulo, de modo que la documentación no puede desincronizarse del código.

Cada constante se clasifica en uno de tres niveles:

  VALIDADO   — El valor proviene de una fuente publicada y revisada por pares.
  CALIBRADO  — El valor fue ajustado empíricamente sobre un dataset público
               con codificación FACS certificada, con métricas de error reportadas.
  HEURISTICO — No existe evidencia publicada para este valor. Fue elegido por
               prueba y error. Debe mostrarse como tal en la interfaz y reportes.

Regla del proyecto: ninguna constante nueva entra al código sin una entrada aquí.
"""

from dataclasses import dataclass
from enum import Enum


class Tier(str, Enum):
    VALIDATED = "VALIDADO"
    CALIBRATED = "CALIBRADO"
    HEURISTIC = "HEURISTICO"


@dataclass(frozen=True)
class Reference:
    key: str
    authors: str
    year: int
    title: str
    venue: str
    identifier: str  # DOI, PMID o URL estable


@dataclass(frozen=True)
class ConstantEvidence:
    """Vincula una constante del código con su respaldo (o falta de él)."""
    name: str             # Identificador en el código
    location: str         # archivo:linea aproximada
    current_value: str    # Valor vigente hoy
    tier: Tier
    refs: tuple[str, ...] # Claves de BIBLIOGRAPHY
    note: str             # Qué respalda la cita, o por qué no hay respaldo


# ═══════════════════════════════════════════════════════════════════════
# BIBLIOGRAFÍA
# ═══════════════════════════════════════════════════════════════════════

BIBLIOGRAPHY: dict[str, Reference] = {
    # ── Base anatómica: FACS ──────────────────────────────────────────
    "ekman1978facs": Reference(
        key="ekman1978facs",
        authors="Ekman, P., & Friesen, W. V.",
        year=1978,
        title="Facial Action Coding System: A Technique for the Measurement of Facial Movement",
        venue="Consulting Psychologists Press, Palo Alto, CA",
        identifier="https://www.paulekman.com/facial-action-coding-system/",
    ),
    "ekman2002facs": Reference(
        key="ekman2002facs",
        authors="Ekman, P., Friesen, W. V., & Hager, J. C.",
        year=2002,
        title="Facial Action Coding System: The Manual (2nd ed.)",
        venue="Research Nexus, Salt Lake City, UT",
        identifier="https://www.paulekman.com/facial-action-coding-system/",
    ),

    # ── Micro-expresiones: temporalidad ───────────────────────────────
    "yan2013duration": Reference(
        key="yan2013duration",
        authors="Yan, W.-J., Wu, Q., Liang, J., Chen, Y.-H., & Fu, X.",
        year=2013,
        title="How Fast are the Leaked Facial Expressions: The Duration of Micro-Expressions",
        venue="Journal of Nonverbal Behavior, 37(4), 217–230",
        identifier="10.1007/s10919-013-0159-8",
    ),

    # ── Datasets con codificación FACS certificada ────────────────────
    "yan2014casme2": Reference(
        key="yan2014casme2",
        authors="Yan, W.-J., Li, X., Wang, S.-J., Zhao, G., Liu, Y.-J., Chen, Y.-H., & Fu, X.",
        year=2014,
        title="CASME II: An Improved Spontaneous Micro-Expression Database and the Baseline Evaluation",
        venue="PLoS ONE, 9(1), e86041",
        identifier="10.1371/journal.pone.0086041",
    ),
    "davison2018samm": Reference(
        key="davison2018samm",
        authors="Davison, A. K., Lansley, C., Costen, N., Tan, K., & Yap, M. H.",
        year=2018,
        title="SAMM: A Spontaneous Micro-Facial Movement Dataset",
        venue="IEEE Transactions on Affective Computing, 9(1), 116–129",
        identifier="10.1109/TAFFC.2016.2573832",
    ),
    "mavadati2013disfa": Reference(
        key="mavadati2013disfa",
        authors="Mavadati, S. M., Mahoor, M. H., Bartlett, K., Trinh, P., & Cohn, J. F.",
        year=2013,
        title="DISFA: A Spontaneous Facial Action Intensity Database",
        venue="IEEE Transactions on Affective Computing, 4(2), 151–160",
        identifier="10.1109/T-AFFC.2013.4",
    ),

    # ── Parpadeo ──────────────────────────────────────────────────────
    "bentivoglio1997blink": Reference(
        key="bentivoglio1997blink",
        authors="Bentivoglio, A. R., Bressman, S. B., Cassetta, E., Carretta, D., Tonali, P., & Albanese, A.",
        year=1997,
        title="Analysis of blink rate patterns in normal subjects",
        venue="Movement Disorders, 12(6), 1028–1034",
        identifier="10.1002/mds.870120629",
    ),
    "soukupova2016ear": Reference(
        key="soukupova2016ear",
        authors="Soukupová, T., & Čech, J.",
        year=2016,
        title="Real-Time Eye Blink Detection using Facial Landmarks",
        venue="21st Computer Vision Winter Workshop, Rimske Toplice, Slovenia",
        identifier="https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf",
    ),
    "plos2025blink": Reference(
        key="plos2025blink",
        authors="Maffei, A., et al.",
        year=2025,
        title="The eye, a spy hole on human mind: Spontaneous blink rate and amplitude, "
              "and their variability, as new psychobiological markers of anxiety",
        venue="PLOS ONE, 20",
        identifier="10.1371/journal.pone.0338262",
    ),

    # ── Ritmo cardíaco remoto (rPPG) ──────────────────────────────────
    "dehaan2013chrom": Reference(
        key="dehaan2013chrom",
        authors="de Haan, G., & Jeanne, V.",
        year=2013,
        title="Robust Pulse Rate from Chrominance-Based rPPG",
        venue="IEEE Transactions on Biomedical Engineering, 60(10), 2878–2886",
        identifier="10.1109/TBME.2013.2266196",
    ),
    "wu2012evm": Reference(
        key="wu2012evm",
        authors="Wu, H.-Y., Rubinstein, M., Shih, E., Guttag, J., Durand, F., & Freeman, W. T.",
        year=2012,
        title="Eulerian Video Magnification for Revealing Subtle Changes in the World",
        venue="ACM Transactions on Graphics (SIGGRAPH), 31(4), 65",
        identifier="10.1145/2185520.2185561",
    ),

    # ── Límites: qué NO se puede inferir del rostro ───────────────────
    "barrett2019reconsidered": Reference(
        key="barrett2019reconsidered",
        authors="Barrett, L. F., Adolphs, R., Marsella, S., Martinez, A. M., & Pollak, S. D.",
        year=2019,
        title="Emotional Expressions Reconsidered: Challenges to Inferring Emotion "
              "From Human Facial Movements",
        venue="Psychological Science in the Public Interest, 20(1), 1–68",
        identifier="10.1177/1529100619832930",
    ),
    "bond2006accuracy": Reference(
        key="bond2006accuracy",
        authors="Bond, C. F., & DePaulo, B. M.",
        year=2006,
        title="Accuracy of Deception Judgments",
        venue="Personality and Social Psychology Review, 10(3), 214–234",
        identifier="10.1207/s15327957pspr1003_2",
    ),
    "nrc2003polygraph": Reference(
        key="nrc2003polygraph",
        authors="National Research Council, Committee to Review the Scientific "
                "Evidence on the Polygraph",
        year=2003,
        title="The Polygraph and Lie Detection",
        venue="The National Academies Press, Washington, DC",
        identifier="10.17226/10420",
    ),
    "matsumoto2018microintent": Reference(
        key="matsumoto2018microintent",
        authors="Matsumoto, D., & Hwang, H. C.",
        year=2018,
        title="Microexpressions Differentiate Truths From Lies About Future Malicious Intent",
        venue="Frontiers in Psychology, 9, 2545",
        identifier="10.3389/fpsyg.2018.02545",
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# INVENTARIO DE CONSTANTES
# ═══════════════════════════════════════════════════════════════════════
#
# Este inventario refleja el ESTADO ACTUAL del código, incluyendo las
# constantes que hoy no tienen respaldo. Se actualiza conforme cada
# constante migra de HEURISTICO a VALIDADO o CALIBRADO.

CONSTANTS: tuple[ConstantEvidence, ...] = (

    # ── Micro-expresiones: ventana temporal ───────────────────────────
    ConstantEvidence(
        name="micro_expr_min_duration_ms / max_duration_ms",
        location="config.py:57-58",
        current_value="40 ms / 500 ms",
        tier=Tier.VALIDATED,
        refs=("yan2013duration",),
        note="La literatura sitúa la duración total de una micro-expresión por debajo "
             "de 500 ms. La cota inferior de 40 ms es conservadora y compatible.",
    ),
    ConstantEvidence(
        name="Bonificación por duración 'óptima'",
        location="micro_expressions.py:535",
        current_value="80–200 ms",
        tier=Tier.VALIDATED,
        refs=("yan2013duration",),
        note="Corregido (Fase 1.5): Yan et al. sitúan la moda de duración en 80–200 ms; "
             "el código usaba 100–250 ms. Ya coincide con la fuente.",
    ),

    # ── Patrones AU → emoción ─────────────────────────────────────────
    ConstantEvidence(
        name="MICRO_EXPR_PATTERNS (6 prototipos EMFACS)",
        location="micro_expressions.py:54-97",
        current_value="miedo, ira, asco, tristeza, sorpresa, desprecio",
        tier=Tier.VALIDATED,
        refs=("ekman2002facs",),
        note="Corregido (Fase 1.3): se agregó AU5 a miedo/ira/sorpresa y AU1 a sorpresa "
             "(EMFACS los incluye y el código los omitía, sesgando esas tres emociones "
             "hacia falsos negativos); se agregó AU4 a tristeza; se quitó AU25 de asco "
             "(no pertenece a su prototipo EMFACS). Los 6 patrones ahora coinciden con "
             "los prototipos publicados. La categoría 'stress' es un componente aparte, "
             "ver la entrada siguiente.",
    ),
    ConstantEvidence(
        name="Patrón 'stress' (no-EMFACS)",
        location="micro_expressions.py:94-99",
        current_value="AU23+AU24 (req.), AU4+AU7 (sup.)",
        tier=Tier.HEURISTIC,
        refs=(),
        note="No es un prototipo EMFACS publicado: es invención del proyecto. Se decidió "
             "mantenerlo (código externo depende de la etiqueta 'stress') pero marcarlo "
             "explícitamente como heurístico en el propio código, en vez de eliminarlo "
             "o presentarlo como si tuviera el mismo respaldo que los otros 6.",
    ),
    ConstantEvidence(
        name="AU5 (Upper Lid Raiser)",
        location="action_units.py:249-267",
        current_value="clip((ratio − 0.10) × 10.0, 0, 1)",
        tier=Tier.HEURISTIC,
        refs=("ekman2002facs",),
        note="Implementado (Fase 1.4): AU5 aparece en 3 de los 6 prototipos EMFACS de "
             "emoción básica y antes no se calculaba, sesgando miedo/ira/sorpresa hacia "
             "falsos negativos. Que el AU exista y sea geométricamente medible está "
             "respaldado; el umbral numérico concreto (0.10, ×10.0) es heurístico, igual "
             "que el resto de las ~30 constantes geométricas de este archivo — pendiente "
             "de calibración contra DISFA (Fase 3).",
    ),

    # ── Parpadeo ──────────────────────────────────────────────────────
    ConstantEvidence(
        name="Umbral EAR de parpadeo",
        location="action_units.py:88, 428-452",
        current_value="0.20, promedio de 2 pares verticales por ojo",
        tier=Tier.VALIDATED,
        refs=("soukupova2016ear",),
        note="Corregido (Fase 1.1): el umbral canónico publicado es 0.20 (antes 0.15), y "
             "la fórmula EAR ahora promedia dos pares verticales de landmarks por ojo "
             "como especifica Soukupová & Čech, en vez de un solo par.",
    ),
    ConstantEvidence(
        name="Normalización de blink_rate",
        location="action_units.py:442-483",
        current_value="clip((bpm - ancla) / 25, 0, 1); ancla = basal del sujeto o 26 ppm",
        tier=Tier.VALIDATED,
        refs=("bentivoglio1997blink", "plos2025blink"),
        note="Corregido (Fase 1.2): el ancla ya no asume tasa de reposo (15-17 ppm). Usa "
             "26 ppm (norma conversacional, Bentivoglio et al.) como respaldo antes de "
             "calibrar, y el basal propio del sujeto (medido durante los 30s de "
             "calibración) en cuanto está disponible — la normalización intra-sujeto que "
             "el rango individual amplio (4-48 ppm) exige.",
    ),
    ConstantEvidence(
        name="Cortes de blink_score",
        location="congruence.py:262-269",
        current_value="0.3 / 0.6 / 0.8 → 80/90/60/30",
        tier=Tier.HEURISTIC,
        refs=("bentivoglio1997blink",),
        note="La dirección del efecto (más ansiedad → más parpadeo) está respaldada, "
             "pero ni los puntos de corte ni los puntajes asignados provienen de "
             "ninguna fuente. Además, el rango de referencia individual es muy amplio "
             "(4–48 ppm), lo que exige normalización intra-sujeto, no cortes absolutos.",
    ),

    # ── rPPG ──────────────────────────────────────────────────────────
    ConstantEvidence(
        name="Coeficientes CHROM (3R−2G, 1.5R+G−1.5B)",
        location="heart_rate.py:547-548",
        current_value="3.0 / 2.0 / 1.5",
        tier=Tier.VALIDATED,
        refs=("dehaan2013chrom",),
        note="Reproducen exactamente las ecuaciones publicadas. Sin cambios pendientes.",
    ),
    ConstantEvidence(
        name="Banda de frecuencia EVM",
        location="config.py:81-85",
        current_value="0.7–3.0 Hz (42–180 BPM)",
        tier=Tier.VALIDATED,
        refs=("wu2012evm", "dehaan2013chrom"),
        note="Ampliada (Fase 1.6) de 0.8–2.0 Hz (48–120 BPM) a 0.7–3.0 Hz para cubrir "
             "taquicardia situacional en entrevistas, dentro del rango habitual de la "
             "literatura rPPG.",
    ),
    ConstantEvidence(
        name="Umbral de movimiento (motion_threshold)",
        location="heart_rate.py:61",
        current_value="15.0 (píxeles normalizados por diagonal de frame)",
        tier=Tier.HEURISTIC,
        refs=(),
        note="Corregido de 5.0 a 15.0 tras diagnosticar que el valor anterior descartaba "
             "casi todos los frames: a 640×480 el umbral raw era de solo ~4 px, por "
             "debajo del jitter propio de los landmarks de MediaPipe en reposo (~2-5 px), "
             "así que el buffer nunca se llenaba y el BPM quedaba fijo en 0. No existe un "
             "valor publicado para este umbral — 15.0 es una estimación empírica con "
             "margen sobre el jitter conocido, pendiente de validar contra grabaciones "
             "con movimiento real etiquetado.",
    ),
    ConstantEvidence(
        name="EMA de deriva del ROI de referencia nasal",
        location="heart_rate.py:174-186",
        current_value="alpha = 0.01",
        tier=Tier.HEURISTIC,
        refs=(),
        note="Corrige un bug distinto del umbral de movimiento: el código restaba el "
             "valor RGB crudo del puente nasal al de la frente para cancelar ruido de "
             "iluminación, lo que con frecuencia daba medias negativas y disparaba la "
             "guarda de _compute_chrom_signal (BPM atascado en 0). Ahora se resta solo "
             "la deriva del ROI de referencia respecto a su propio basal (EMA con "
             "alpha=0.01), preservando el nivel absoluto de brillo de la frente que "
             "CHROM necesita. El valor de alpha es una elección empírica sin fuente "
             "publicada.",
    ),
    ConstantEvidence(
        name="Mezcla de estrés fisiológico",
        location="heart_rate.py:692",
        current_value="hr_stress*0.6 + variability_stress*0.4",
        tier=Tier.HEURISTIC,
        refs=(),
        note="Sin fuente. Los pesos 0.6/0.4 fueron elegidos por prueba y error.",
    ),

    # ── Congruencia: el núcleo sin respaldo ───────────────────────────
    ConstantEvidence(
        name="Pesos del score de congruencia",
        location="config.py:64-67",
        current_value="0.30 / 0.35 / 0.20 / 0.15",
        tier=Tier.HEURISTIC,
        refs=("barrett2019reconsidered", "bond2006accuracy"),
        note="No existe ni puede existir una fuente para estos pesos: no hay ningún "
             "dataset público etiquetado con 'congruencia verdadera' contra el cual "
             "ajustarlos. Sólo pueden derivarse contra un proxy medible (p. ej. ansiedad "
             "auto-reportada con STAI) o declararse explícitamente como heurísticos.",
    ),
    ConstantEvidence(
        name="Cortes alto/moderado/bajo",
        location="congruence.py:131-136",
        current_value="≥80 alto, ≥50 moderado, <50 bajo",
        tier=Tier.HEURISTIC,
        refs=(),
        note="Sin fuente. Definen la semántica visible (verde/amarillo/rojo) que el "
             "entrevistador interpreta, por lo que son las constantes de mayor "
             "consecuencia práctica de todo el sistema.",
    ),
    ConstantEvidence(
        name="Penalización por desviación del baseline",
        location="congruence.py:247",
        current_value="score = 100 − desviación × 160",
        tier=Tier.HEURISTIC,
        refs=("nrc2003polygraph",),
        note="El multiplicador 160 no tiene fuente. Más grave: la premisa de que "
             "desviarse del basal indica ocultamiento es la del Comparison Question "
             "Technique del polígrafo, que el NRC evaluó como de base científica débil "
             "y tasa de error desconocida.",
    ),
    ConstantEvidence(
        name="Penalización por micro-expresiones contradictorias",
        location="congruence.py:221",
        current_value="100 − ratio × 80 × relevancia",
        tier=Tier.HEURISTIC,
        refs=("barrett2019reconsidered",),
        note="Sin fuente para el factor 80. La inferencia de 'enmascaramiento social' a "
             "partir de incongruencia AU↔emoción no está respaldada: Barrett et al. "
             "documentan que la relación entre configuración facial y estado interno es "
             "altamente variable entre personas, contextos y culturas.",
    ),
    ConstantEvidence(
        name="Pesos del relevance score",
        location="micro_expressions.py:524-551",
        current_value="25 / 25 / 30 / 20 / 15",
        tier=Tier.HEURISTIC,
        refs=(),
        note="Sin fuente. El umbral de corte 60 tampoco la tiene.",
    ),
    ConstantEvidence(
        name="Umbrales de fricción del timeline",
        location="interview_analyzer.py:92-93",
        current_value="nerviosismo +0.15, congruencia −10.0",
        tier=Tier.HEURISTIC,
        refs=(),
        note="Sin fuente. Disparan 'red flags' visibles en el reporte del candidato.",
    ),

    # ── Geometría de AUs ──────────────────────────────────────────────
    ConstantEvidence(
        name="Constantes de normalización geométrica de AUs",
        location="action_units.py:249-441 (~31 constantes, incl. AU5)",
        current_value="p. ej. (ratio − 0.12) × 8.0",
        tier=Tier.HEURISTIC,
        refs=("mavadati2013disfa", "yan2014casme2"),
        note="Ninguna proviene de la literatura: son factores de ajuste específicos de "
             "la malla de MediaPipe. No son 'ciencia' sino calibración, y sólo pueden "
             "fijarse ajustándolos contra intensidades AU codificadas por humanos "
             "certificados en FACS (DISFA aporta intensidad 0–5 para 12 AUs).",
    ),
)


def constants_by_tier(tier: Tier) -> list[ConstantEvidence]:
    return [c for c in CONSTANTS if c.tier is tier]


def coverage() -> dict[str, int]:
    """Cuántas constantes hay en cada nivel de evidencia."""
    return {t.value: len(constants_by_tier(t)) for t in Tier}
