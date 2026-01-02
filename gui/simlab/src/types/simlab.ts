/* -------------------------------------------------------
 * Primitivos e auxiliares
 * ----------------------------------------------------- */

export type ID = string;

// No front, datas chegam geralmente como ISO string.
// Se você preferir trabalhar com Date, converta no client.
export type ISODateTime = string;

// Mapas de métricas/objetivos calculados (key -> value)
export type FloatMap = Record<string, number>;

// JSON genérico para parâmetros dinâmicos (evita `any` solto)
export type JsonValue =
  | string
  | number
  | boolean
  | null
  | { [k: string]: JsonValue }
  | JsonValue[];

export type JsonObject = Record<string, JsonValue>;

/* -------------------------------------------------------
 * Status (sugestão: alinhar com enums reais do back-end)
 * ----------------------------------------------------- */

export type ExperimentStatus = "Building" | "Waiting" | "Running" | "Finished" | "Failed" | "Canceled";
export type GenerationStatus = "Waiting" | "Running" | "Finished" | "Failed";
export type SimulationStatus = "Queued" | "Running" | "Finished" | "Failed";

/* -------------------------------------------------------
 * TransformConfig (usado no front para configurar análise)
 * ----------------------------------------------------- */

export type ObjectiveKind = string; // opcional: trocar por union literal quando estabilizar
export type MetricKind = string;

export type ObjectiveGoal = "min" | "max" | string;

export interface ObjectiveItemDto {
  name: string;
  kind: ObjectiveKind;
  column: string;
  goal: ObjectiveGoal;
  q?: number;      // requerido dependendo do kind
  scale?: number;  // requerido dependendo do kind
}

export interface MetricItemDto {
  name: string;
  kind: MetricKind;
  column: string;
  q?: number; // requerido dependendo do kind
}

export interface TransformConfigDto {
  node_col: string;
  time_col: string;
  objectives: ObjectiveItemDto[];
  metrics: MetricItemDto[];
}

/* -------------------------------------------------------
 * Problemas (para exibição/validação leve no front)
 * Mantemos o essencial para UI (forms, detalhes, etc.)
 * ----------------------------------------------------- */

export type Position = [number, number];

export interface MobileNodeDto {
  path_segments: [string, string][];
  is_closed: boolean;
  is_round_trip: boolean;
  speed: number;
  time_step: number;
}

export interface SojournLocationDto {
  id: number;
  position: Position;
  adjacency: number[];
  visibleNodes: number[];
}

export interface HomogeneousProblemDto {
  name: string;
  radius_of_reach: number;
  radius_of_inter: number;
  region: number[]; // ex.: [xmin,ymin,xmax,ymax] ou polígono; manter flexível
}

export interface ProblemP1Dto extends HomogeneousProblemDto {
  sink: Position;
  mobile_nodes: MobileNodeDto[];
  number_of_relays: number;
}

export interface ProblemP2Dto extends HomogeneousProblemDto {
  sink: Position;
  mobile_nodes: MobileNodeDto[];
  candidates: Position[];
}

export interface ProblemP3Dto extends HomogeneousProblemDto {
  sink: Position;
  targets: Position[];
  candidates: Position[];
  radius_of_cover: number;
  k_required: number;
}

export interface ProblemP4Dto extends HomogeneousProblemDto {
  nodes: Position[];
  sink_base: Position;
  initial_energy: number;
  buffer_capacity: number;
  data_rate: number;
  sojourns: SojournLocationDto[];
  speed: number;
  time_step: number;
}

// Se o back-end envia problem como objeto heterogêneo,
// no front você pode manter como JsonObject.
// Se quiser tipar melhor, use union discriminada com `name`/`type`.
export type ProblemDto = JsonObject | HomogeneousProblemDto | ProblemP1Dto | ProblemP2Dto | ProblemP3Dto | ProblemP4Dto;

/* -------------------------------------------------------
 * SimulationConfig (front: exibir/inspecionar, não editar source code)
 * ----------------------------------------------------- */

// Versão enxuta: posição e parâmetros essenciais para renderização.
export interface FixedMoteDto {
  name: string;
  position: number[]; // [x,y]
}

export interface MobileMoteDto {
  name: string;
  functionPath: [string, string][];
  isClosed: boolean;
  isRoundTrip: boolean;
  speed: number;
  timeStep: number;
}

export interface SimulationElementsDto {
  fixedMotes: FixedMoteDto[];
  mobileMotes: MobileMoteDto[];
}

export interface SimulationConfigDto {
  name: string;
  duration: number;
  radiusOfReach: number;
  radiusOfInter: number;
  region: [number, number, number, number]; // [xmin,ymin,xmax,ymax]
  simulationElements: SimulationElementsDto;
}

/* -------------------------------------------------------
 * Configs de algoritmo (para forms e inspeção)
 * ----------------------------------------------------- */

export interface GeneticAlgorithmConfigDto {
  population_size: number;
  number_of_generations: number;
  prob_cx: number;
  prob_mt: number;
  selection_method: string;
  crossover_method: string;
  mutation_method: string;
  per_gene_prob: number;
  eta_cx: number;
  eta_mt: number;
}

export interface NsgaIIIConfigDto extends GeneticAlgorithmConfigDto {
  divisions: number;
}

/* -------------------------------------------------------
 * API DTOs principais (o que o front consome)
 * ----------------------------------------------------- */

export interface ParametersDto {
  algorithm: JsonObject;  // ou GeneticAlgorithmConfigDto | NsgaIIIConfigDto | ...
  simulation: JsonObject; // ou SimulationConfigDto se for sempre esse shape
  problem: JsonObject;    // ou ProblemDto quando estabilizar
}

export interface ExperimentDto {
  id?: ID;
  name: string;

  status?: ExperimentStatus;

  created_time?: ISODateTime | null;
  start_time?: ISODateTime | null;
  end_time?: ISODateTime | null;

  parameters: ParametersDto;

  // O front tipicamente consome IDs para navegar e lazy-load
  generations: ID[];

  // Repositório de fontes referenciado (detalhes via endpoint específico)
  source_repository_id: ID;

  transform_config: TransformConfigDto;

  // Front normalmente trata como dataset/estrutura serializada
  pareto_front?: JsonValue | null;
}

export interface GenerationDto {
  id: ID;
  index: number;
  experiment_id: ID;

  status: GenerationStatus;

  start_time?: ISODateTime | null;
  end_time?: ISODateTime | null;

  simulations_ids: ID[];
}

export interface SimulationDto {
  id: ID;
  experiment_id: ID;
  generation_id: ID;

  status: SimulationStatus;

  start_time?: ISODateTime | null;
  end_time?: ISODateTime | null;

  // No front, manter como JsonObject é suficiente e robusto.
  // Se você realmente usa SimulationConfig sempre, troque para SimulationConfigDto.
  parameters: JsonObject | SimulationConfigDto;

  objectives: FloatMap;
  metrics: FloatMap;
}

/* -------------------------------------------------------
 * Source repository (apenas para UI de inspeção)
 * ----------------------------------------------------- */

export interface SourceFileDto {
  id: ID;
  file_name: string;
}

export interface SourceRepositoryDto {
  id: ID;
  name: string;
  description: string;
  source_files: SourceFileDto[];
}
