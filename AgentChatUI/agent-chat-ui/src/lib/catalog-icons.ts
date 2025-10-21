import type { LucideIcon } from "lucide-react";
import {
  Globe2,
  BookOpen,
  GraduationCap,
  MessagesSquare,
  Landmark,
  Brain,
  Workflow,
  Code,
  Sparkles,
  Database,
  FolderSearch,
  UserRound,
  ClipboardList,
  Rocket,
  Shield,
  FlaskConical,
} from "lucide-react";

const iconNameMap: Record<string, LucideIcon> = {
  globe: Globe2,
  web: Globe2,
  knowledge: BookOpen,
  research: GraduationCap,
  academic: GraduationCap,
  social: MessagesSquare,
  finance: Landmark,
  analysis: Brain,
  workflow: Workflow,
  code: Code,
  data: Database,
  search: FolderSearch,
  user: UserRound,
  tasks: ClipboardList,
  launch: Rocket,
  security: Shield,
  experiment: FlaskConical,
  default: Sparkles,
};

const capabilityIconMap: Record<string, LucideIcon> = {
  web_search: Globe2,
  knowledge_search: BookOpen,
  academic_search: GraduationCap,
  social_search: MessagesSquare,
  finance_search: Landmark,
  analysis_mode: Brain,
  workflow_orchestration: Workflow,
  code_execution: Code,
  data_index: Database,
  document_search: FolderSearch,
  user_profile: UserRound,
  task_queue: ClipboardList,
  deployment: Rocket,
  security_audit: Shield,
  lab_mode: FlaskConical,
};

const agentIconMap: Record<string, LucideIcon> = {};

function resolveIconName(name: unknown, fallback: LucideIcon): LucideIcon {
  if (typeof name === "string") {
    const key = name.toLowerCase();
    if (iconNameMap[key]) {
      return iconNameMap[key];
    }
  }
  return fallback;
}

export function resolveCapabilityIcon(name: string, override?: unknown): LucideIcon {
  if (override) {
    return resolveIconName(override, Sparkles);
  }
  if (capabilityIconMap[name]) {
    return capabilityIconMap[name];
  }
  return Sparkles;
}

export function resolveAgentIcon(agentId: string, options: { icon?: unknown; category?: string | null }): LucideIcon {
  if (options.icon) {
    return resolveIconName(options.icon, Sparkles);
  }
  if (agentIconMap[agentId]) {
    return agentIconMap[agentId];
  }
  if (options.category) {
    return resolveIconName(options.category, Sparkles);
  }
  return Sparkles;
}
