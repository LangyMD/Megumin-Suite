// Run: node --test test/*.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import {
    evictableIndexes, fitMessages, customServerBase, serverTokenCount, _resetEndpointCache, FIT_ROLE_OVERHEAD
} from "../src/engine/contextFit.js";

// The V11 Universal preset's shape: system rules, a user-role "Before we start"
// line, <history> with memory, the chat, </history>, then three user messages.
function v11(chatTurns, size = 400) {
    const chat = [];
    for (let i = 0; i < chatTurns; i++) {
        chat.push({ role: i % 2 ? "user" : "assistant", content: `turn ${i} ` + "x".repeat(size) });
    }
    return [
        { role: "system", content: "RULES " + "r".repeat(2000) },
        { role: "user", content: "Before we start: I know this is fiction" },
        { role: "user", content: "<history>\n\nlong memory\n\nshort memory" },
        ...chat,
        { role: "user", content: "</history>" },
        { role: "user", content: "current info" },
        { role: "user", content: "thinking" },
        { role: "user", content: "this turn. Start your turn." },
    ];
}
const charCounter = async t => ({ tokens: t.length, exact: false });
const contents = ms => ms.map(m => m.content);

test("only chat inside <history> can go, never the player's latest message", () => {
    const ms = v11(6);
    // indexes 3..8 are the chat; 8 is the last user message (turn 5)
    assert.deepEqual(evictableIndexes(ms), [3, 4, 5, 6, 7]);
});

test("system messages inside the history (author's note) stay", () => {
    const ms = v11(4);
    ms.splice(5, 0, { role: "system", content: "[Author's note]" });
    assert.ok(!evictableIndexes(ms).includes(5));
});

test("no wrapper: nothing is dropped", async () => {
    const ms = [{ role: "system", content: "x".repeat(5000) }, { role: "user", content: "y".repeat(5000) }];
    assert.deepEqual(evictableIndexes(ms), []);
    const r = await fitMessages(ms, { countTokens: charCounter, targetFor: () => 1000 });
    assert.equal(r.removed, 0);
    assert.equal(r.fits, false);
    assert.equal(r.wrapped, false);
    assert.equal(ms.length, 2);
});

test("a payload that fits is left alone", async () => {
    const ms = v11(4);
    const before = contents(ms);
    const r = await fitMessages(ms, { countTokens: charCounter, targetFor: () => 1e6 });
    assert.equal(r.removed, 0);
    assert.deepEqual(contents(ms), before);
});

test("oldest chat goes first and the frame stays whole", async () => {
    const ms = v11(20);
    const frameBefore = [...contents(ms).slice(0, 3), ...contents(ms).slice(-4)];
    const latest = ms[ms.length - 6].content; // turn 19 (user)
    const r = await fitMessages(ms, { countTokens: charCounter, targetFor: () => 6000 });
    assert.ok(r.fits, `total ${r.total} over ${r.target}`);
    assert.ok(r.removed > 0);
    assert.deepEqual([...contents(ms).slice(0, 3), ...contents(ms).slice(-4)], frameBefore);
    assert.ok(contents(ms).includes(latest));
    // The kept chat is the newest, in order.
    const kept = contents(ms).filter(c => c.startsWith("turn ")).map(c => Number(c.split(" ")[1]));
    assert.deepEqual(kept, [...kept].sort((a, b) => a - b));
    assert.equal(kept[kept.length - 1], 19);
    assert.ok(kept[0] > 0);
});

test("lands under the target even when the estimate is off", async () => {
    // A counter that says 2 tokens per char: the first pass's estimate is too low.
    const dense = async t => ({ tokens: t.length * 2, exact: false });
    const ms = v11(30);
    const r = await fitMessages(ms, { countTokens: dense, targetFor: () => 12000 });
    assert.ok(r.fits, `total ${r.total} over ${r.target}`);
    assert.equal(r.total, (contents(ms).join("\n").length * 2) + FIT_ROLE_OVERHEAD * ms.length);
});

test("frame alone too big: drops all it may, reports it doesn't fit", async () => {
    const ms = v11(6);
    const r = await fitMessages(ms, { countTokens: charCounter, targetFor: () => 1000 });
    assert.equal(r.fits, false);
    assert.equal(r.removed, 5);
    assert.deepEqual(evictableIndexes(ms), []);
});

test("the target follows the counter that answered", async () => {
    const ms = v11(10);
    const seen = [];
    await fitMessages(ms, { countTokens: async t => ({ tokens: t.length, exact: true }), targetFor: e => { seen.push(e); return 1e6; } });
    assert.deepEqual(seen, [true]);
});

test("custom server base", () => {
    assert.equal(customServerBase({ chat_completion_source: "custom", custom_url: "http://127.0.0.1:8080/v1/" }), "http://127.0.0.1:8080");
    assert.equal(customServerBase({ chat_completion_source: "custom", custom_url: "localhost:5001" }), "http://localhost:5001");
    assert.equal(customServerBase({ chat_completion_source: "openrouter", custom_url: "http://x" }), null);
    assert.equal(customServerBase({ chat_completion_source: "custom", custom_url: "" }), null);
});

function fakeServer(handler) {
    const calls = [];
    const fetchFn = async (url, init) => {
        calls.push(url);
        const out = handler(url, JSON.parse(init.body));
        if (!out) return { ok: false, json: async () => ({}) };
        return { ok: true, json: async () => out };
    };
    return { fetchFn, calls };
}

test("llama.cpp /tokenize is used and remembered", async () => {
    _resetEndpointCache();
    const s = fakeServer((url, body) => url.endsWith("/tokenize") && body.content ? { tokens: [1, 2, 3] } : null);
    assert.equal(await serverTokenCount("http://a", "abc", s.fetchFn), 3);
    assert.equal(await serverTokenCount("http://a", "abcd", s.fetchFn), 3);
    assert.deepEqual(s.calls, ["http://a/tokenize", "http://a/tokenize"]);
});

test("vLLM and KoboldCpp shapes", async () => {
    _resetEndpointCache();
    const v = fakeServer((url, body) => url.endsWith("/tokenize") && body.prompt ? { count: 7, tokens: [] } : null);
    assert.equal(await serverTokenCount("http://v", "x", v.fetchFn), 7);
    const k = fakeServer((url) => url.endsWith("/api/extra/tokencount") ? { value: 9 } : null);
    assert.equal(await serverTokenCount("http://k", "x", k.fetchFn), 9);
});

test("a server that can't count is not asked again every turn", async () => {
    _resetEndpointCache();
    const s = fakeServer(() => null);
    assert.equal(await serverTokenCount("http://o", "x", s.fetchFn), null);
    const n = s.calls.length;
    assert.equal(await serverTokenCount("http://o", "x", s.fetchFn), null);
    assert.equal(s.calls.length, n);
});

test("network errors fall back quietly", async () => {
    _resetEndpointCache();
    const fetchFn = async () => { throw new TypeError("Failed to fetch"); };
    assert.equal(await serverTokenCount("http://e", "x", fetchFn), null);
    assert.equal(await serverTokenCount(null, "x", fetchFn), null);
});
