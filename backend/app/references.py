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
    # ── Dinámica de estados y marcadores de frustración ───────────────
    "dmello2012dynamics": Reference(
        key="dmello2012dynamics",
        authors="D'Mello, S., & Graesser, A.",
        year=2012,
        title="Dynamics of affective states during complex learning",
        venue="Learning and Instruction, 22(2), 145-157",
        identifier="https://doi.org/10.1016/j.learninstruc.2011.10.001",
    ),
    "ihme2018frustration": Reference(
        key="ihme2018frustration",
        authors="Ihme, K., Dömeland, C., Freese, M., & Jipp, M.",
        year=2018,
        title="Recognizing Frustration of Drivers From Face Video Recordings and "
              "Brain Activation Measurements With Functional Near-Infrared Spectroscopy",
        venue="Frontiers in Human Neuroscience, 12:327",
        identifier="https://doi.org/10.3389/fnhum.2018.00327",
    ),
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

    # ── Calibración guiada por pose ───────────────────────────────────
    ConstantEvidence(
        name="Anclas de pose de calibración",
        location="pose_calibration.py:POSE_ANCHORS",
        current_value="centro (0,0); izq/der ±22° yaw; arriba/abajo ∓18° pitch",
        tier=Tier.HEURISTIC,
        refs=(),
        note="Objetivos de captura, no medidas de un sujeto. Se piden giros cómodos "
             "y no extremos porque MediaPipe pierde precisión en los landmarks más "
             "allá de ~35°, y una línea base tomada ahí no sería fiable. Los grados "
             "son los de _estimate_head_pose (websocket.py), que es una aproximación "
             "por desplazamiento de la nariz respecto a los ojos y el eje "
             "frente-mentón, no una estimación de pose 3D calibrada: la escala es "
             "consistente consigo misma, no con grados reales. Sin validar contra "
             "ground truth de pose.",
    ),
    ConstantEvidence(
        name="Tolerancia de pose para aceptar muestras",
        location="pose_calibration.py:POSE_TOLERANCE_DEG, CENTER_TOLERANCE_DEG",
        current_value="12° (poses giradas), 8° (centro)",
        tier=Tier.HEURISTIC,
        refs=(),
        note="El objetivo es cubrir una región de ángulos, no clavar un valor exacto, "
             "y un margen estrecho deja al sujeto atrapado intentando acertar un "
             "número que no ve. El centro se exige más estricto porque define el "
             "reposo frontal del que se resta todo lo demás. Sin ajustar con datos "
             "de uso real.",
    ),
    ConstantEvidence(
        name="Interpolación de línea base entre poses",
        location="pose_calibration.py:INTERPOLATION_POWER, CORRECTION_FULL_DEG, CORRECTION_ZERO_DEG",
        current_value="Shepard IDW p=2; corrección completa ≤18°, nula ≥40° del ancla más cercana",
        tier=Tier.HEURISTIC,
        refs=(),
        note="Se eligió un interpolador (Shepard) y no un promedio ponderado con "
             "núcleo gaussiano porque el promedio no reproduce el valor del ancla al "
             "consultarlo en el ancla: corregía frames frontales contra una línea "
             "base contaminada por las poses giradas. La continuidad importa porque "
             "un salto de línea base se detecta aguas abajo como una microexpresión "
             "falsa. El desvanecimiento fuera de la región calibrada evita "
             "extrapolar una corrección grande sin datos que la respalden. "
             "Pendiente: medir la tasa de falsos positivos de microexpresión con y "
             "sin corrección, con la cabeza girada.",
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
        name="Banda del estimador rPPG en vivo",
        location="heart_rate.py:58-59",
        current_value="0.7–2.5 Hz (42–150 BPM)",
        tier=Tier.HEURISTIC,
        refs=("dehaan2013chrom",),
        note="Ojo: la entrada anterior ('Banda de frecuencia EVM') es la del "
             "renderizador offline en config.py; ésta es la del estimador que "
             "produce el BPM que ve el entrevistador, que usaba sus propios "
             "valores por defecto (0.7–4.0 Hz = 42–240 BPM) y no estaba "
             "registrada. Se estrechó el techo a 2.5 Hz tras medir que, con "
             "ruido de webcam, el pico del pulso deja de dominar el espectro y "
             "el argmax se iba a picos de ruido repartidos por toda la banda, "
             "produciendo lecturas de 190–220 BPM (inalcanzables sentado) y un "
             "rango mostrado de hasta 169 BPM para un pulso real de 72. El piso "
             "de 0.7 Hz es el habitual en la literatura rPPG; el techo de 2.5 Hz "
             "es una decisión de dominio (sujeto sentado en entrevista), ajustada "
             "en simulación sintética, no contra ground truth humano — de ahí el "
             "nivel heurístico. Ensancharla reintroduce el síntoma.",
    ),
    ConstantEvidence(
        name="Ventana mínima de estimación rPPG",
        location="heart_rate.py:62, 734-754",
        current_value="8 s de span temporal; FFT con ventana Hann y zero-padding 4×",
        tier=Tier.HEURISTIC,
        refs=("dehaan2013chrom",),
        note="La necesidad de una ventana larga no es opinable: la resolución del "
             "FFT es fs/n, así que con los 3 s que se exigían antes (a 20 FPS) el "
             "BPM reportado sólo podía caer en 60, 80, 100 … — saltos de 20 BPM "
             "medidos, y con pulso real de 72 devolvía 80. Ahora se exige el span "
             "medido por timestamps (no un conteo de frames, para que valga a "
             "cualquier FPS real), y el zero-padding 4× interpola la rejilla "
             "espectral para quitar esa cuantización. El valor 8 s es un "
             "compromiso entre resolución y latencia de la primera lectura; la "
             "literatura rPPG usa ventanas de 10–30 s. Además el FFT ya no se "
             "aplica sobre la señal filtrada con Butterworth: la respuesta del "
             "propio filtro tiene un máximo dentro de la banda y sesgaba los "
             "espectros de ruido hacia un pico repetible de ~105 BPM, "
             "indistinguible de un pulso real. El enmascaramiento de banda del "
             "FFT cumple la función sin introducir ese sesgo.",
    ),
    ConstantEvidence(
        name="Umbral de confianza espectral (min_confidence)",
        location="heart_rate.py:63, 783",
        current_value="0.40 (fracción de potencia en banda dentro de ±0.2 Hz del pico)",
        tier=Tier.HEURISTIC,
        refs=("dehaan2013chrom",),
        note="Antes no había ningún filtro: cualquier estimación con bpm > 0 "
             "entraba al historial con el mismo peso que una lectura buena, que "
             "es lo que permitía que un pico de ruido se mostrara como pulso. La "
             "forma de la métrica sigue la SNR de de Haan & Jeanne (potencia del "
             "fundamental frente al resto de la banda) y se define sobre una "
             "vecindad en Hz, no en bins, para que no cambie con el zero-padding. "
             "El umbral 0.40 se eligió midiendo la distribución en simulación: "
             "señal buena da 0.65–0.94, ruido sin pulso da ~0.36. No está "
             "calibrado contra registros humanos con ECG de referencia, que es lo "
             "que haría falta para subirlo de nivel.",
    ),
    ConstantEvidence(
        name="Consistencia del pico entre ventanas",
        location="heart_rate.py:66-67, 349-364",
        current_value="IQR ≤ 12 BPM sobre 20 estimaciones espaciadas 0.5 s",
        tier=Tier.HEURISTIC,
        refs=(),
        note="Necesario porque el umbral de confianza solo no separa un pulso "
             "débil del ruido estructurado: con ruido alto ambos puntúan ~0.38. "
             "Lo que sí los separa es la repetición — un pulso real pone el pico "
             "en la misma frecuencia ventana tras ventana. Sin este criterio, el "
             "ruido fuerte producía una lectura estable pero equivocada (~105 BPM "
             "para un pulso real de 72), que es peor que una oscilante porque "
             "parece confiable. Las estimaciones se espacian 0.5 s a propósito: "
             "calculadas por frame, ventanas consecutivas comparten ~95% de las "
             "muestras y su acuerdo mide el solape, no el pulso (medido: con "
             "estimación por frame el criterio dejaba pasar el 35% de las "
             "lecturas falsas; espaciadas, baja al 1–29%). Ni el IQR de 12 BPM ni "
             "el intervalo de 0.5 s provienen de ninguna fuente.",
    ),
    ConstantEvidence(
        name="Límite de cambio fisiológico del BPM",
        location="heart_rate.py:64, 366-381",
        current_value="8 BPM por segundo",
        tier=Tier.HEURISTIC,
        refs=(),
        note="La dirección tiene respaldo fisiológico obvio (el ritmo cardíaco no "
             "se mueve 60 BPM en un segundo, así que un salto así es artefacto de "
             "estimación, no medición), pero el valor concreto de 8 BPM/s no "
             "proviene de ninguna fuente. Actúa junto a la mediana de las últimas "
             "~10 estimaciones: una lectura espuria aislada que supere el filtro "
             "de confianza queda amortiguada en vez de desplazar el valor "
             "mostrado.",
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
        name="Indicador de estrés fisiológico",
        location="heart_rate.py:856",
        current_value="clip((BPM medio − 70) / 50, 0, 1)",
        tier=Tier.HEURISTIC,
        refs=(),
        note="Antes era hr_stress*0.6 + variability_stress*0.4, con pesos elegidos "
             "por prueba y error. Se eliminó el término de variabilidad, que "
             "puntuaba la desviación estándar de este mismo historial de BPM como "
             "si fuera HRV: la HRV real exige intervalos R-R latido a latido, "
             "mientras que este historial contiene estimaciones espectrales de "
             "ventanas solapadas, así que su dispersión mide ruido del estimador. "
             "Con la regla 'poca variabilidad = estrés', eso premiaba con "
             "'relajado' justamente a la señal de cámara mala (que produce "
             "estimaciones erráticas) y, tras suavizar la salida de BPM, habría "
             "quedado clavado en estrés máximo. Queda sólo el nivel de HR, cuya "
             "dirección (HR elevado acompaña activación simpática) sí tiene "
             "respaldo; los cortes 70/120 siguen sin fuente.",
    ),

    ConstantEvidence(
        name="'nervousness' / 'confidence' fuera de la emoción dominante",
        location="emotion_classifier.py:_apply_hysteresis, _merge_cnn_with_derived",
        current_value="se calculan y se muestran, pero no pueden ganar el argmax",
        tier=Tier.HEURISTIC,
        refs=("barrett2019reconsidered",),
        note="Fase 2: estas dos etiquetas no existen en ningún dataset FER público "
             "ni tienen prototipo EMFACS; son una combinación propia de blendshapes "
             "elegida por prueba y error. Antes competían de igual a igual con las 7 "
             "clases del CNN entrenado y podían reservar hasta el 50% de la masa de "
             "probabilidad, de modo que la interfaz presentaba una heurística sin "
             "fuente y una predicción validada como si tuvieran el mismo respaldo. "
             "Ahora siguen reportándose —eliminarlas rompería código que depende de "
             "esas etiquetas— pero no pueden ser la emoción dominante. Medido contra "
             "el vídeo etiquetado de RAVDESS, el cambio subió el acierto de 50% a "
             "64%: 'confidence' se estaba llevando un segmento de ira y otro de "
             "miedo. La corrección era exigible por honestidad y además resultó "
             "más exacta.",
    ),

    ConstantEvidence(
        name="Alcance real del reconocimiento de emoción",
        location="emotion_classifier.py (todo el clasificador)",
        current_value="36% sobre 20 actores; solo 'happy' se sostiene (95%)",
        tier=Tier.HEURISTIC,
        refs=("barrett2019reconsidered",),
        note="Medición del 2026-09-16 sobre 220 clips de CREMA-D (20 actores, "
             "ODbL). Con un único actor de RAVDESS el sistema puntuaba 46-54%; "
             "con 20 caras distintas cae al 36%. Por clase el deterioro es "
             "mayor que esa cifra: 'disgust' pasa de 88% a 15% y 'angry' de 62% "
             "a 22%, es decir que estaban ajustados a una cara concreta, no "
             "funcionando en general. El dato más relevante es la dispersión "
             "entre sujetos: del 9% en el peor actor al 73% en el mejor. La "
             "herramienta funciona bien con algunas personas y es inútil con "
             "otras, algo que una evaluación de un solo sujeto no puede "
             "detectar. Sólo 'happy' aguanta (95% aquí, 100% en RAVDESS). "
             "Diferencia por sexo: 40% en mujeres frente a 30% en hombres, no "
             "concluyente con esta muestra pero a vigilar, porque el sesgo "
             "demográfico en reconocimiento facial está documentado y esta "
             "herramienta interviene en contrataciones. La diferencia de "
             "resolución entre ambos conjuntos (480x360 frente a 1280x720) se "
             "descartó como explicación: evaluar los mismos clips de RAVDESS "
             "reducidos a 480x360 da 64% frente al 59% original, o sea que "
             "bajar resolución no perjudica —coherente con que el modelo se "
             "entrenó sobre imágenes de 48x48— y la caída al 36% se debe "
             "íntegramente a cambiar de persona. Queda un matiz a favor del "
             "sistema: los clips son demasiado cortos para calibrar el basal "
             "por sujeto que sí existe en una sesión real, de modo que el uso "
             "en vivo debería quedar por encima de este 36%. Reproducible con "
             "data/test_videos/evaluate_cremad.py y evaluate_resolution.py.",
    ),

    ConstantEvidence(
        name="Retiro de 'fear' del conjunto reportable",
        location="emotion_classifier.py:DOMINANT_LABELS, config.py:emotion_labels",
        current_value="8 etiquetas expuestas; 'fear' calculada pero no reportable",
        tier=Tier.HEURISTIC,
        refs=("barrett2019reconsidered",),
        note="Sobre 220 clips etiquetados de 20 personas, 'fear' no gano ni un "
             "solo segmento, y el CNN le asigna alrededor del 2% de los frames "
             "incluso dentro de clips de miedo actuado. Es tambien su peor clase "
             "en el propio FER2013 (0.454). Se mantiene en FER7_LABELS porque esa "
             "lista mapea los indices de salida del modelo y alterarla "
             "desalinearia las predicciones, pero queda fuera de la seleccion de "
             "emocion dominante y de las etiquetas que la interfaz ofrece. "
             "Exponer una categoria que solo aparece por azar es peor que no "
             "ofrecerla.",
    ),
    ConstantEvidence(
        name="'surprise' como nota de revision, no como estado",
        location="emotion_classifier.py:BRIEF_LABELS, _detect_peak",
        current_value="pico >= 0.60 genera nota para revision humana",
        tier=Tier.HEURISTIC,
        refs=("yan2013duration", "barrett2019reconsidered"),
        note="La sorpresa alcanza 0.94 de probabilidad dentro de sus propios "
             "segmentos pero solo gana el 21% de sus frames: dura menos de lo que "
             "el promedio movil puede sostener, asi que como estado dominante "
             "marcaba 0 de 8. En vez de forzarla, se emite como observacion de que "
             "hubo un cambio facial compatible con sorpresa y conviene revisar ese "
             "punto de la grabacion. El umbral de 0.60 no tiene fuente: captura 5 "
             "de los 8 segmentos etiquetados y por debajo empieza a disparar con "
             "fluctuacion normal.",
    ),
    ConstantEvidence(
        name="Lecturas posibles de transicion emocional",
        location="interview_analyzer.py:TRANSITION_INTERPRETATIONS",
        current_value="16 secuencias con lectura en condicional y base citada",
        tier=Tier.HEURISTIC,
        refs=("dmello2012dynamics", "ihme2018frustration", "barrett2019reconsidered"),
        note="Antes el reporte afirmaba causas ('posible reaccion a una pregunta "
             "incomoda'). Ahora cada cambio separa lo observado de la lectura "
             "posible, y cada lectura declara de donde sale. La logica de "
             "secuencias sigue el modelo de D'Mello y Graesser: el desconcierto "
             "surge ante un obstaculo y es productivo mientras se resuelve; si no "
             "se resuelve deriva en frustracion y luego en desconexion. Los "
             "marcadores de tension facial se apoyan en Ihme et al., que reportan "
             "62% de acierto discriminando intervalos frustrados, cifra que "
             "conviene tener presente al leer estas notas. El encuadre condicional "
             "responde a Barrett et al.: una configuracion facial no es "
             "diagnostica de un estado interno. Ninguna de estas lecturas esta "
             "validada sobre datos de este proyecto; son hipotesis para orientar "
             "a quien revisa.",
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

    # ── Índice de desempeño en vivo ───────────────────────────────────
    ConstantEvidence(
        name="EMOTION_VALENCE (valencia por emoción)",
        location="live_score.py:24-33",
        current_value="happy +1.0, confidence +0.8, surprise +0.2, neutral 0.0, "
                      "nervousness −0.5, disgust −0.7, sad −0.8, angry −1.0",
        tier=Tier.HEURISTIC,
        refs=("barrett2019reconsidered",),
        note="El orden de los signos responde a un pedido de producto (tristeza e ira "
             "restan, alegría suma); las magnitudes intermedias no salen de ninguna "
             "escala publicada. Anclarlas exigiría normas de valencia medidas en la "
             "propia población, no elegidas a ojo. A 'surprise' se la deja casi neutra "
             "a propósito, porque el proyecto ya la trata como ambigua y la reporta "
             "como nota de revisión en vez de como emoción dominante.",
    ),
    ConstantEvidence(
        name="Pesos del índice en vivo (valencia / congruencia)",
        location="live_score.py:36-37",
        current_value="0.60 valencia, 0.40 congruencia",
        tier=Tier.HEURISTIC,
        refs=(),
        note="Reparto elegido sin medición. No es la misma métrica que "
             "interview_analyzer.overall_score, que pondera cinco dimensiones sobre la "
             "sesión completa; este índice sólo usa lo que existe frame a frame y no "
             "debe leerse como una predicción de aquél.",
    ),
    ConstantEvidence(
        name="Ventana de promediado del índice en vivo",
        location="live_score.py:43,47",
        current_value="3.0 s de ventana; 2.0 s antes de mostrar número",
        tier=Tier.HEURISTIC,
        refs=(),
        note="Se promedia por tiempo y no por número de muestras porque la entrega de "
             "frames del navegador es irregular — el mismo problema que sesga la FFT "
             "del rPPG. Ambos valores se fijaron para que el número no saltara por "
             "ruido, no contra ningún criterio externo.",
    ),
)


def constants_by_tier(tier: Tier) -> list[ConstantEvidence]:
    return [c for c in CONSTANTS if c.tier is tier]


def coverage() -> dict[str, int]:
    """Cuántas constantes hay en cada nivel de evidencia."""
    return {t.value: len(constants_by_tier(t)) for t in Tier}
