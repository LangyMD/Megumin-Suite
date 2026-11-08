// ────────────────────────────────────────────────────────────────────────────
// Context fit: keeps the prompt that is actually sent inside the context size.
//
// SillyTavern decides how much chat history fits BEFORE Megumin runs, while the
// preset's [[placeholders]] are still short tags. Megumin then expands them into
// the engine, the rules, memory, World State and the NPC list, so the request
// can end up thousands of tokens over the context size. Cloud models with huge
// contexts never notice. A local model with an 8–32k context does: the server
// errors out or cuts the prompt from the top, losing the rules.
//
// This runs after the expansion, measures the real payload and, if it is too
// big, drops the oldest chat messages until it fits, the way SillyTavern would
// have if it had known the final size. Only messages inside the preset's
// <history> … </history> wrapper are dropped (all Megumin presets have it), and
// never the player's latest message, so the rules, card, memory and this turn's
// notes always stay. A preset without the wrapper is left alone, with a warning.
//
// Token counts: SillyTavern's own counter guesses the tokenizer and can be ~10%
// off for a local model, so a wide margin is kept. When the connection is a
// Custom (OpenAI-compatible) server that can count tokens itself (llama.cpp's
// /tokenize, vLLM's /tokenize, KoboldCpp's /api/extra/tokencount) that exact
// count is used and the margin shrinks.
//
// Ported from Abdulhanan535's fork (branch websocket-image-delivery), changed
// for V11: the V11 preset sends its own user-role messages before and after the
// history, which the original would have dropped first.
// ────────────────────────────────────────────────────────────────────────────

// Tokens a chat template adds around each message (Gemma's turn markers are
// ~8–12), and what an image part is counted as, since text counters can't see it.
export const FIT_ROLE_OVERHEAD = 12;
const FIT_IMAGE_STANDIN_CHARS = 400;

export function fitContentToText(content) {
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
        let text = "";
        for (const part of content) {
            if (part?.type === "text") text += part.text || "";
            else if (part?.type === "image_url") text += " ".repeat(FIT_IMAGE_STANDIN_CHARS);
        }
        return text;
    }
    return "";
}

/**
 * The messages that may be dropped: the ones between the message holding
 * "<history>" and the last one holding "</history>", except system messages
 * (author's notes and lore placed at a depth) and the player's latest message.
 * Oldest first. Empty when the wrapper isn't there.
 */
export function evictableIndexes(messages) {
    const texts = messages.map(m => fitContentToText(m?.content));
    const open = texts.findIndex(t => t.includes("<history>"));
    let close = -1;
    for (let i = texts.length - 1; i > open; i--) {
        if (texts[i].includes("</history>")) { close = i; break; }
    }
    if (open < 0 || close < 0) return [];
    let lastUser = -1;
    for (let i = close - 1; i > open; i--) {
        if (messages[i]?.role === "user") { lastUser = i; break; }
    }
    const out = [];
    for (let i = open + 1; i < close; i++) {
        if (i === lastUser) continue;
        if (messages[i]?.role === "system") continue;
        out.push(i);
    }
    return out;
}

/**
 * Drops the oldest history messages, in place, until the payload fits.
 * `countTokens(text)` counts one string (it may be a network call, so the whole
 * payload is counted at once and per-message costs are estimated from that
 * count). `targetFor(exact)` gives the token target for the counter that
 * answered. Returns { removed, total, target, fits, wrapped }.
 */
export async function fitMessages(messages, { countTokens, targetFor }) {
    const overhead = () => FIT_ROLE_OVERHEAD * messages.length;
    const join = () => messages.map(m => fitContentToText(m?.content)).join("\n");

    let { tokens, exact } = await countTokens(join());
    let total = tokens + overhead();
    const target = targetFor(exact);
    const result = { removed: 0, total, target, fits: total <= target, wrapped: true };
    if (result.fits) return result;

    if (!evictableIndexes(messages).length) {
        result.wrapped = messages.some(m => fitContentToText(m?.content).includes("<history>"));
        return result;
    }

    const chars = join().length;
    // Cost per character, nudged up so the estimate drops a little too much
    // rather than landing just over the line.
    const ratio = Math.max(tokens / Math.max(1, chars), 0.25) * 1.15;

    // Pick oldest-first, by estimate, then remove in one go (highest index
    // first, so the indexes stay right). Re-count after; a second round only
    // when the estimate came up short.
    let removed = 0;
    for (let round = 0; round < 3 && total > target; round++) {
        const pool = evictableIndexes(messages);
        if (!pool.length) break;
        const drop = [];
        let est = total;
        for (const i of pool) {
            if (est <= target) break;
            est -= fitContentToText(messages[i]?.content).length * ratio + FIT_ROLE_OVERHEAD;
            drop.push(i);
        }
        for (const i of drop.sort((a, b) => b - a)) messages.splice(i, 1);
        removed += drop.length;
        ({ tokens } = await countTokens(join()));
        total = tokens + overhead();
    }

    return { removed, total, target, fits: total <= target, wrapped: true };
}

// ── Exact counts from the local server ─────────────────────────────────────
// Each server's base URL remembers which endpoint answered (or that none did,
// for 10 minutes), so a server without one costs nothing after the first turn.

const PROBE_TIMEOUT_MS = 3000;
const RETRY_FAILED_MS = 10 * 60 * 1000;
const endpointCache = new Map(); // base -> { probe } | { failedAt }

const PROBES = [
    // llama.cpp server
    { path: "/tokenize", body: t => ({ content: t }), read: d => Array.isArray(d?.tokens) ? d.tokens.length : null },
    // vLLM
    { path: "/tokenize", body: t => ({ prompt: t }), read: d => Number.isFinite(d?.count) ? d.count : (Array.isArray(d?.tokens) ? d.tokens.length : null) },
    // KoboldCpp
    { path: "/api/extra/tokencount", body: t => ({ prompt: t }), read: d => Number.isFinite(d?.value) ? d.value : null },
];

/** The Custom connection's server root ("http://127.0.0.1:8080"), or null. */
export function customServerBase(cs) {
    if (!cs || cs.chat_completion_source !== "custom" || !cs.custom_url) return null;
    let url = String(cs.custom_url).trim().replace(/\/+$/, "").replace(/\/v1$/, "");
    if (!/^https?:\/\//i.test(url)) url = "http://" + url;
    return url;
}

async function tryProbe(base, probe, text, fetchFn) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), PROBE_TIMEOUT_MS);
    try {
        const res = await fetchFn(base + probe.path, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(probe.body(text)),
            signal: controller.signal,
        });
        if (!res.ok) return null;
        return probe.read(await res.json());
    } catch {
        return null;
    } finally {
        clearTimeout(timer);
    }
}

/** An exact token count from the local server, or null when it can't give one. */
export async function serverTokenCount(base, text, fetchFn = globalThis.fetch) {
    if (!base || typeof fetchFn !== "function") return null;
    const known = endpointCache.get(base);
    if (known?.probe) {
        const n = await tryProbe(base, known.probe, text, fetchFn);
        if (n != null) return n;
        endpointCache.delete(base); // the server changed; probe again next time
        return null;
    }
    if (known?.failedAt && Date.now() - known.failedAt < RETRY_FAILED_MS) return null;
    for (const probe of PROBES) {
        const n = await tryProbe(base, probe, text, fetchFn);
        if (n != null) {
            endpointCache.set(base, { probe });
            return n;
        }
    }
    endpointCache.set(base, { failedAt: Date.now() });
    return null;
}

export function _resetEndpointCache() { endpointCache.clear(); }

// ── The SillyTavern side ───────────────────────────────────────────────────

let lastToast = 0;

/**
 * Fits a Chat Completion payload to the connection's context size, in place.
 * `ctx` is SillyTavern's getContext(). Does nothing on a dry run, when the
 * setting is off, or when the context size can't be read.
 */
export async function fitPromptToContext(messages, ctx, { enabled = true } = {}) {
    if (!enabled || !Array.isArray(messages) || !messages.length || !ctx) return;
    const cs = ctx.chatCompletionSettings;
    const maxContext = Number(cs?.openai_max_context);
    const response = Number(cs?.openai_max_tokens) || 0;
    const budget = maxContext - response;
    if (!Number.isFinite(budget) || budget <= 0) return;

    const wideTarget = budget - Math.max(150, Math.round(budget * 0.09));
    if (wideTarget <= 0) return;
    // Even at one token per character, a payload this short fits.
    const chars = messages.reduce((n, m) => n + fitContentToText(m?.content).length, 0);
    if (chars < wideTarget) return;

    const base = customServerBase(cs);
    const countTokens = async text => {
        const exact = await serverTokenCount(base, text);
        if (exact != null) return { tokens: exact, exact: true };
        const n = typeof ctx.getTokenCountAsync === "function" ? await ctx.getTokenCountAsync(text) : Math.ceil(text.length / 3);
        return { tokens: n, exact: false };
    };
    const targetFor = exact => exact ? budget - Math.max(250, Math.round(budget * 0.01)) : wideTarget;

    let r;
    try {
        r = await fitMessages(messages, { countTokens, targetFor });
    } catch (e) {
        console.warn("[Megumin Suite] Context fit skipped:", e);
        return;
    }

    const now = Date.now();
    const toastOk = now - lastToast > 5 * 60 * 1000 && typeof toastr !== "undefined";
    if (r.removed > 0) {
        console.info(`[Megumin Suite] Expanded prompt was over the ${budget}-token budget; dropped ${r.removed} oldest chat message(s) so it fits (${r.total} tokens now).`);
        if (toastOk) {
            lastToast = now;
            toastr.info(`Dropped ${r.removed} old message(s) so the prompt fits your ${maxContext}-token context.`, "Megumin Suite");
        }
    }
    if (!r.fits) {
        console.warn(`[Megumin Suite] Prompt is still ${r.total} tokens, over the ${r.target}-token target, with no chat history left to drop. Turn off some blocks or raise the context size.`);
        if (toastOk) {
            lastToast = now;
            toastr.warning("The prompt is bigger than your context size even without chat history. Turn off some blocks or raise the context size.", "Megumin Suite");
        }
    }
}
