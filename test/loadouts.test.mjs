// Run: node --test test/*.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { LOADOUTS, COMING_SOON, loadoutById, applyLoadout, changedFromLoadout, settingFor } from "../src/features/modelsetup/loadouts.js";

const CTX = { lengthLabel: v => v, lengthValue: l => l };

test("the Local loadout is offered and no longer 'coming soon'", () => {
    assert.ok(loadoutById("local"));
    assert.ok(!COMING_SOON.some(l => l.name === "Local"));
});

test("every loadout setting is a known row", () => {
    for (const l of LOADOUTS) for (const key of Object.keys(l.settings)) assert.ok(settingFor(key), `${l.id}.${key}`);
});

test("using Local sets V11 with the Thinking Cap, short ban list, no HTML, no Chomusuke", () => {
    const p = { mode: "v10-ukiyo", model: "cot-v10-ukiyo-english", addons: ["html"], storyConfig: {} };
    const l = loadoutById("local");
    applyLoadout(p, l, CTX);
    assert.equal(p.mode, "v11-voice");
    assert.equal(p.model, "cot-v11-voice-cap-english");
    assert.equal(p.cotEnabled, true);
    assert.equal(p.banListShort, true);
    assert.ok(!p.addons.includes("html"));
    assert.equal(p.chomusuke.features.blocks, false);
    assert.equal(p.loadout, "local");
    assert.deepEqual(changedFromLoadout(p, l, CTX), []);
});
