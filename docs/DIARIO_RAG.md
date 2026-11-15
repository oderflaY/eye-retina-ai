# Diario de salud · semáforo · RAG de primeros auxilios

Módulo de AstroPulse (app OjoScan). Todo corre **en el teléfono, sin conexión**.

## Cómo funciona

1. La persona escribe en el diario cómo se siente (y opcionalmente su ánimo 1–5).
2. **Semáforo** (`astro/Semaforo.kt`): reglas fijas y revisables, **no** un modelo.
   - 🔴 ROJO: señales de alarma (dolor de pecho, no respira, desmayo, derrame, convulsión, sangrado que no para,
     anafilaxia, envenenamiento, químico en el ojo, descarga eléctrica, golpe en la cabeza con vómito, fiebre ≥ 41 °C
     o en bebé, crisis emocional → Línea de la Vida 800 911 2000), o pulso/SpO₂ críticos de la pulsera.
   - 🟡 AMARILLO: síntomas a atender hoy, ánimo «muy mal», o el mismo síntoma ≥ 3 días en la semana (→ médico).
   - 🟢 VERDE: sin síntomas. Las negaciones («no tengo fiebre») no cuentan.
3. **RAG extractivo** (`astro/Rag.kt`): BM25 con raíces en español + sinónimos coloquiales mexicanos. Muestra
   2 procedimientos de primeros auxilios + 1 resumen de MedlinePlus (por coincidencia de título), **tal cual vienen de
   la fuente** y con su origen. No genera texto: no puede inventar indicaciones.
4. El color de las últimas 12 h entra al triage general de AstroPulse. El chat CDSS usa la base cuando no reconoce el síntoma.

## Base de conocimiento

`python src/rag/construir_primeros_auxilios.py` (venv `.venv-rag`: ctranslate2, sentencepiece, pandas, pyarrow)
→ `mobile/androidApp/src/main/assets/primeros_auxilios.json` (≈ 9 900 documentos, 5.4 MB).

| Fuente | Licencia | Uso | Docs |
|---|---|---|---|
| [First-Aid-Dataset](https://huggingface.co/datasets/belvisk/First-Aid-Dataset) (belvisk) | MIT | Q&A curadas a mano, traducidas | 312 |
| [FirstAidQA](https://huggingface.co/datasets/i-am-mushfiq/FirstAidQA) (Muna et al., arXiv 2511.01289) | CC BY 4.0 | Q&A **sintéticas**, traducidas y filtradas | 5 485 |
| [MedlinePlus en español](https://medlineplus.gov/spanish/) — Biblioteca Nacional de Medicina de EE. UU. | Dominio público (resúmenes de temas) | 1 014 temas en español nativo, en fragmentos | 4 108 |
| [HUMADEX spanish_ner_dataset](https://huggingface.co/datasets/HUMADEX/spanish_ner_dataset) | Apache-2.0 | Solo léxico de síntomas (1 518 palabras PROBLEM) | — |
| AstroPulse (propio) | — | Picadura de alacrán (hueco de las fuentes) | 1 |

- **Traducción** inglés→español local con CTranslate2 + modelo OPUS de Argos (CPU int8), caché en
  `data/processed/primeros_auxilios/traducciones.jsonl`; glosario de correcciones (seizure → convulsión, no «embargo»;
  choking → atragantamiento; CPR → RCP; first aider → quien da los primeros auxilios…).
- **Filtro de seguridad**: descarta pares que recomiendan, sin negación, mantequilla en quemaduras, provocar el vómito,
  succionar veneno, objeto en la boca en convulsiones, etc. (3 descartados; el README de FirstAidQA advierte de respuestas inseguras).
- MedlinePlus: solo `<full-summary>`. **No** se usa la enciclopedia A.D.A.M. ni las fichas de fármacos (tienen copyright).
  Atribución en la app: «Fuente: MedlinePlus, Biblioteca Nacional de Medicina de EE. UU.».
- `preguntasmedicas` (fhirfly) se revisó y se descartó: solo trae preguntas, sin respuestas.

## Evaluación (tests)

- `DiarioTest` (commonTest): frases reales de diario → color esperado (rojas, amarillas, verdes/negaciones, persistencia, signos).
- `BaseRealTest` (androidHostTest, con la base real): 20 situaciones de primeros auxilios → tema correcto
  en los 2 procedimientos: **20/20** (primero 18/20); 12 síntomas → resumen de MedlinePlus correcto: **12/12**.

## Limitaciones

- Traducción automática: hay frases torpes. FirstAidQA es sintético y solo pasó filtros automáticos, no revisión médica.
- Orientación general, no diagnóstico. En iOS falta leer el asset (el semáforo sí funciona).
