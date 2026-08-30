import type { IdentificationParameter } from "@/lib/services/analysisPresentation";

export interface VisualizationInsightInput {
  isAI: boolean;
  confidence: number;
  model: string;
  forensicParameters: IdentificationParameter[];
}

export interface InsightModelOption {
  id: string;
  label: string;
  note: string;
}

const INSIGHT_MODELS: InsightModelOption[] = [
  {
    id: "local-nlp",
    label: "Local NLP",
    note: "Fully free local explanation used by default",
  },
  {
    id: "google/flan-t5-large",
    label: "FLAN-T5 Large",
    note: "Optional remote free model (token required)",
  },
];

const DEFAULT_MODEL_ID = INSIGHT_MODELS[0].id;

export function getInsightModelOptions(): InsightModelOption[] {
  return INSIGHT_MODELS;
}

function getEndpoint(modelId: string): string {
  return `https://api-inference.huggingface.co/models/${modelId}`;
}

function buildPrompt(input: VisualizationInsightInput): string {
  const topSignals = [...input.forensicParameters]
    .sort((a, b) => b.score - a.score)
    .slice(0, 5)
    .map((s) => `${s.label}: ${s.value}`)
    .join("; ");

  return [
    "You are a forensic image analyst.",
    "Write a concise explanation (3 bullet points) for a detection dashboard user.",
    "Include: confidence interpretation, likely generation mechanism, and caution/risk.",
    "Use plain language and do not hallucinate hidden evidence.",
    `Prediction: ${input.isAI ? "AI-generated" : "Authentic"}`,
    `Confidence: ${input.confidence.toFixed(1)}%`,
    `Detected model candidate: ${input.model}`,
    `Signals: ${topSignals}`,
  ].join("\n");
}

function fallbackInsight(input: VisualizationInsightInput): string {
  const sorted = [...input.forensicParameters].sort((a, b) => b.score - a.score);
  const strongest = sorted.slice(0, 3);
  const weakest = [...sorted].reverse().slice(0, 2);

  const confidenceBand =
    input.confidence >= 85 ? "high confidence" : input.confidence >= 65 ? "moderate confidence" : "low confidence";

  return [
    `- The detector reports ${confidenceBand} (${input.confidence.toFixed(1)}%) for ${input.isAI ? "AI-generated" : "authentic"} content.`,
    `- The strongest signals are ${strongest.map((s) => `${s.label} (${s.value})`).join(", ")}, which align with ${input.model}.`,
    `- Lower-impact signals (${weakest.map((s) => `${s.label} (${s.value})`).join(", ")}) suggest reviewing source quality (compression/resizing) before final action.`,
  ].join("\n");
}

function cleanOutput(text: string): string {
  return text
    .replace(/^\s+|\s+$/g, "")
    .replace(/\n{3,}/g, "\n\n");
}

function localInsight(input: VisualizationInsightInput): string {
  return fallbackInsight(input);
}

export async function generateVisualizationInsight(input: VisualizationInsightInput): Promise<string> {
  return generateVisualizationInsightWithModel(input, DEFAULT_MODEL_ID);
}

export async function generateVisualizationInsightWithModel(
  input: VisualizationInsightInput,
  modelId: string
): Promise<string> {
  if (modelId === "local-nlp") {
    return localInsight(input);
  }

  const token = import.meta.env.VITE_HF_API_TOKEN;
  if (!token) {
    return localInsight(input);
  }

  const prompt = buildPrompt(input);
  const endpoint = getEndpoint(modelId || "google/flan-t5-large");

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        inputs: prompt,
        parameters: {
          max_new_tokens: 180,
          temperature: 0.3,
          return_full_text: false,
        },
      }),
    });

    if (!response.ok) {
      return localInsight(input);
    }

    const data = (await response.json()) as Array<{ generated_text?: string }> | { generated_text?: string };

    if (Array.isArray(data) && data[0]?.generated_text) {
      return cleanOutput(data[0].generated_text);
    }
    if (!Array.isArray(data) && data.generated_text) {
      return cleanOutput(data.generated_text);
    }

    return localInsight(input);
  } catch {
    return localInsight(input);
  }
}
