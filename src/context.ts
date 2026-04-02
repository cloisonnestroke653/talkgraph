import type { ConversationContext, NodeResult } from "./types.js";
import type { StateManager } from "./state.js";
import type { AdapterRegistry } from "./llm/registry.js";
import type { SystemPromptBuilder } from "./llm/prompts.js";

interface ContextInit<S extends Record<string, unknown>> {
  sessionId: string;
  state: S;
  stateManager: StateManager<S>;
  turn: number;
  adapterRegistry?: AdapterRegistry;
  systemPromptBuilder?: SystemPromptBuilder;
}

export class ConversationContextImpl<S extends Record<string, unknown>>
  implements ConversationContext<S>
{
  readonly sessionId: string;
  readonly state: Readonly<S>;
  readonly turn: number;
  private readonly stateManager: StateManager<S>;
  private readonly adapterRegistry?: AdapterRegistry;
  private readonly systemPromptBuilder?: SystemPromptBuilder;

  constructor(init: ContextInit<S>) {
    this.sessionId = init.sessionId;
    this.state = init.stateManager.freeze(init.state);
    this.turn = init.turn;
    this.stateManager = init.stateManager;
    this.adapterRegistry = init.adapterRegistry;
    this.systemPromptBuilder = init.systemPromptBuilder;
  }

  update(partial: Partial<S>): ConversationContext<S> {
    const newState = this.stateManager.apply(
      this.state as S,
      partial,
    );
    return new ConversationContextImpl({
      sessionId: this.sessionId,
      state: newState,
      stateManager: this.stateManager,
      turn: this.turn,
      adapterRegistry: this.adapterRegistry,
      systemPromptBuilder: this.systemPromptBuilder,
    });
  }

  reply(text: string): NodeResult {
    return { type: "reply", text };
  }

  goto(nodeName: string): NodeResult {
    return { type: "goto", gotoNode: nodeName };
  }

  replyAndGoto(text: string, nodeName: string): NodeResult {
    return { type: "reply_goto", text, gotoNode: nodeName };
  }

  async generate(prompt: string, opts?: { model?: string; systemPrompt?: string }): Promise<string> {
    if (!this.adapterRegistry) throw new Error("No LLM adapter configured");

    const { adapter, model } = opts?.model
      ? this.adapterRegistry.resolveWithModel(opts.model)
      : this.adapterRegistry.resolveDefault();

    const systemPrompt = opts?.systemPrompt
      ?? this.systemPromptBuilder?.build(this.state as Record<string, unknown>)
      ?? undefined;

    const response = await adapter.complete({
      model,
      messages: [{ role: "user", content: prompt }],
      systemPrompt,
    });

    return response.content;
  }

  async classify(intents: string[], opts?: { model?: string }): Promise<{ label: string; confidence: number }> {
    if (!this.adapterRegistry) throw new Error("No LLM adapter configured");

    const { adapter, model } = opts?.model
      ? this.adapterRegistry.resolveWithModel(opts.model)
      : this.adapterRegistry.resolveDefault();

    const intentList = intents.join(", ");
    const prompt = `Classify the user's intent into exactly one of these categories: ${intentList}. Respond with JSON: {"label": "<category>", "confidence": <0-1>}`;

    const systemPrompt = this.systemPromptBuilder?.build(this.state as Record<string, unknown>) ?? undefined;

    const response = await adapter.complete({
      model,
      messages: [{ role: "user", content: prompt }],
      systemPrompt,
    });

    try {
      const parsed = JSON.parse(response.content);
      return { label: parsed.label, confidence: parsed.confidence ?? 0.5 };
    } catch {
      // If LLM doesn't return valid JSON, try to extract intent
      const lowerContent = response.content.toLowerCase();
      for (const intent of intents) {
        if (lowerContent.includes(intent.toLowerCase())) {
          return { label: intent, confidence: 0.5 };
        }
      }
      return { label: intents[0], confidence: 0.1 };
    }
  }
}
