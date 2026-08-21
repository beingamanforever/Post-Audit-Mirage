#!/usr/bin/env node

"use strict";

const fs = require("node:fs");

const MAX_INPUT_BYTES = 65536;
const MAX_FRACTION_DIGITS = 40;
const GROUP_IDS = ["common", "rare"];
const POLICY_IDS = ["candidate", "incumbent"];
const PRIORITIES = new Set(["common_first", "earliest_deadline", "random"]);

function gcd(left, right) {
  left = left < 0n ? -left : left;
  right = right < 0n ? -right : right;
  while (right !== 0n) [left, right] = [right, left % right];
  return left;
}

function rational(numerator, denominator = 1n) {
  if (denominator === 0n) throw new Error("zero rational denominator");
  if (denominator < 0n) {
    numerator = -numerator;
    denominator = -denominator;
  }
  const divisor = gcd(numerator, denominator);
  return { n: numerator / divisor, d: denominator / divisor };
}

const ZERO = rational(0n);
const ONE = rational(1n);

function add(left, right) {
  return rational(left.n * right.d + right.n * left.d, left.d * right.d);
}

function multiply(left, right) {
  return rational(left.n * right.n, left.d * right.d);
}

function divide(left, right) {
  if (right.n === 0n) throw new Error("division by zero");
  return rational(left.n * right.d, left.d * right.n);
}

function equal(left, right) {
  return left.n === right.n && left.d === right.d;
}

function fractionText(value) {
  return `${value.n}/${value.d}`;
}

function requireObject(value, name) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${name} must be an object`);
  }
  return value;
}

function exactKeys(value, keys, name) {
  requireObject(value, name);
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`${name} has unexpected fields`);
  }
}

function boundedInteger(value, name, minimum, maximum) {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new Error(`${name} must be an integer from ${minimum} through ${maximum}`);
  }
  return value;
}

function exactFraction(value, name, maximum = ONE) {
  if (typeof value !== "string" || !/^(0|[1-9][0-9]*)\/(0|[1-9][0-9]*)$/.test(value)) {
    throw new Error(`${name} must be a non-negative exact fraction string`);
  }
  const [numeratorText, denominatorText] = value.split("/");
  if (numeratorText.length > MAX_FRACTION_DIGITS || denominatorText.length > MAX_FRACTION_DIGITS) {
    throw new Error(`${name} exceeds the fraction size bound`);
  }
  const result = rational(BigInt(numeratorText), BigInt(denominatorText));
  if (fractionText(result) !== value) {
    throw new Error(`${name} must be a reduced exact fraction string`);
  }
  if (result.n < 0n || (maximum && result.n * maximum.d > maximum.n * result.d)) {
    throw new Error(`${name} is outside its allowed range`);
  }
  return result;
}

function countIncrement(counts, groupId) {
  return { ...counts, [groupId]: counts[groupId] + 1 };
}

function stateKey(state) {
  return [
    state.queue.join(","),
    state.completed.common,
    state.completed.rare,
    state.harmed.common,
    state.harmed.rare,
    state.firstHarmTime === null ? "null" : state.firstHarmTime,
  ].join("|");
}

function addState(distribution, state, mass) {
  if (mass.n === 0n) return;
  const key = stateKey(state);
  const current = distribution.get(key);
  if (current) current.mass = add(current.mass, mass);
  else distribution.set(key, { mass, state });
}

function serviceOptions(queue, priority, groups) {
  if (priority === "random") return queue.map((_, index) => index);
  if (priority === "common_first") {
    const commonIndex = queue.indexOf("common");
    return [commonIndex === -1 ? 0 : commonIndex];
  }
  const earliest = Math.min(...queue.map((groupId) => groups.get(groupId).deadline));
  return queue
    .map((groupId, index) => ({ groupId, index }))
    .filter(({ groupId }) => groups.get(groupId).deadline === earliest)
    .map(({ index }) => index);
}

function serviceState(state, mass, capacity, priority, groups) {
  let distribution = new Map();
  addState(distribution, state, mass);
  for (let slot = 0; slot < capacity; slot += 1) {
    const following = new Map();
    for (const { state: current, mass: currentMass } of distribution.values()) {
      if (current.queue.length === 0) {
        addState(following, current, currentMass);
        continue;
      }
      const options = serviceOptions(current.queue, priority, groups);
      const choiceMass = multiply(currentMass, rational(1n, BigInt(options.length)));
      for (const index of options) {
        const groupId = current.queue[index];
        addState(following, {
          ...current,
          completed: countIncrement(current.completed, groupId),
          queue: current.queue.filter((_, queueIndex) => queueIndex !== index),
        }, choiceMass);
      }
    }
    distribution = following;
  }
  return distribution;
}

function accountLateStarts(state, tick, groups, harmDelay) {
  let harmed = state.harmed;
  let firstHarmTime = state.firstHarmTime;
  for (const groupId of state.queue) {
    const deadline = groups.get(groupId).deadline;
    if (tick < deadline || harmed[groupId] > 0) continue;
    harmed = countIncrement(harmed, groupId);
    const harmTime = deadline + harmDelay;
    firstHarmTime = firstHarmTime === null ? harmTime : Math.min(firstHarmTime, harmTime);
  }
  return { ...state, firstHarmTime, harmed };
}

function initialDistribution(support) {
  const distribution = new Map();
  for (const outcome of support) {
    const queue = GROUP_IDS.filter((groupId) => outcome[groupId]);
    addState(distribution, {
      completed: { common: 0, rare: 0 },
      firstHarmTime: null,
      harmed: { common: 0, rare: 0 },
      queue,
    }, outcome.probability);
  }
  return distribution;
}

function evaluateQueue(parsed, policy, support) {
  let distribution = initialDistribution(support);
  let backlogProbability = ZERO;
  for (let tick = 0; tick < parsed.horizon; tick += 1) {
    const following = new Map();
    for (const { state, mass } of distribution.values()) {
      const accounted = accountLateStarts(state, tick, parsed.groups, parsed.harmDelay);
      const serviced = serviceState(accounted, mass, parsed.capacity, policy.priority, parsed.groups);
      for (const entry of serviced.values()) addState(following, entry.state, entry.mass);
    }
    distribution = following;
    if (tick === 0) {
      for (const { state, mass } of distribution.values()) {
        if (state.queue.length > 0) backlogProbability = add(backlogProbability, mass);
      }
    }
  }

  const completed = { common: ZERO, rare: ZERO };
  const harmed = { common: ZERO, rare: ZERO };
  let firstHarmTime = null;
  for (const { state, mass } of distribution.values()) {
    if (state.queue.length > 0) throw new Error("queue remained active beyond the horizon");
    for (const groupId of GROUP_IDS) {
      completed[groupId] = add(completed[groupId], multiply(mass, rational(BigInt(state.completed[groupId]))));
      harmed[groupId] = add(harmed[groupId], multiply(mass, rational(BigInt(state.harmed[groupId]))));
    }
    if (state.firstHarmTime !== null && (firstHarmTime === null || state.firstHarmTime < firstHarmTime)) {
      firstHarmTime = state.firstHarmTime;
    }
  }
  return { backlogProbability, completed, firstHarmTime, harmed };
}

function parseGroup(group) {
  exactKeys(group, ["arrival_probability", "deadline", "group_id"], "group");
  if (!GROUP_IDS.includes(group.group_id)) throw new Error("unknown group ID");
  return {
    arrivalProbability: exactFraction(group.arrival_probability, "group arrival probability"),
    deadline: boundedInteger(group.deadline, "group deadline", 1, 32),
    groupId: group.group_id,
  };
}

function parsePolicy(policy) {
  exactKeys(policy, ["policy_id", "priority", "service_cost"], "policy");
  if (!POLICY_IDS.includes(policy.policy_id)) throw new Error("unknown policy ID");
  if (!PRIORITIES.has(policy.priority)) throw new Error("unknown policy priority");
  const serviceCost = exactFraction(policy.service_cost, "policy service cost", rational(1000n));
  if (serviceCost.n === 0n) throw new Error("policy service cost must be positive");
  return {
    policyId: policy.policy_id,
    priority: policy.priority,
    serviceCost,
  };
}

function parseSupport(value, world) {
  if (!Array.isArray(value) || value.length !== 4) {
    throw new Error(`${world} support must contain all four arrival outcomes`);
  }
  const outcomes = new Set();
  const support = value.map((outcome) => {
    exactKeys(outcome, ["common", "probability", "rare"], `${world} support row`);
    if (typeof outcome.common !== "boolean" || typeof outcome.rare !== "boolean") {
      throw new Error(`${world} support arrivals must be booleans`);
    }
    const key = `${Number(outcome.common)}${Number(outcome.rare)}`;
    if (outcomes.has(key)) throw new Error(`${world} support contains duplicate outcomes`);
    outcomes.add(key);
    return {
      common: outcome.common,
      probability: exactFraction(outcome.probability, `${world} support probability`),
      rare: outcome.rare,
    };
  });
  const total = support.reduce((sum, outcome) => add(sum, outcome.probability), ZERO);
  if (!equal(total, ONE)) throw new Error(`${world} support probabilities must sum to 1/1`);
  return support;
}

function supportMarginals(support) {
  return Object.fromEntries(GROUP_IDS.map((groupId) => [
    groupId,
    support.reduce((sum, outcome) => outcome[groupId] ? add(sum, outcome.probability) : sum, ZERO),
  ]));
}

function parseInstance(instance) {
  exactKeys(instance, ["capacity", "family", "groups", "harm_delay", "horizon", "policies", "worlds"], "evaluator input");
  if (instance.family !== "batch_triage_v0") throw new Error("unknown evaluator family");
  const capacity = boundedInteger(instance.capacity, "capacity", 1, 2);
  const harmDelay = boundedInteger(instance.harm_delay, "harm delay", 1, 32);
  const horizon = boundedInteger(instance.horizon, "horizon", 2, 96);

  if (!Array.isArray(instance.groups) || instance.groups.length !== GROUP_IDS.length) {
    throw new Error("groups must contain common and rare exactly once");
  }
  const groups = new Map();
  for (const value of instance.groups.map(parseGroup)) {
    if (groups.has(value.groupId)) throw new Error("group IDs must be unique");
    groups.set(value.groupId, value);
  }
  if (GROUP_IDS.some((groupId) => !groups.has(groupId))) throw new Error("groups must contain common and rare");
  if (groups.get("rare").deadline > groups.get("common").deadline) {
    throw new Error("rare deadline must not exceed common deadline");
  }
  const latestHarm = Math.max(...[...groups.values()].map((group) => group.deadline)) + harmDelay;
  if (horizon < latestHarm) throw new Error("horizon must cover every deadline and harm delay");

  if (!Array.isArray(instance.policies) || instance.policies.length !== POLICY_IDS.length) {
    throw new Error("policies must contain candidate and incumbent exactly once");
  }
  const policies = new Map();
  for (const value of instance.policies.map(parsePolicy)) {
    if (policies.has(value.policyId)) throw new Error("policy IDs must be unique");
    policies.set(value.policyId, value);
  }
  if (POLICY_IDS.some((policyId) => !policies.has(policyId))) {
    throw new Error("policies must contain candidate and incumbent");
  }

  exactKeys(instance.worlds, ["harmful", "safe"], "worlds");
  const worlds = {
    harmful: parseSupport(instance.worlds.harmful, "harmful"),
    safe: parseSupport(instance.worlds.safe, "safe"),
  };
  const marginals = Object.fromEntries(Object.entries(worlds).map(([world, support]) => [world, supportMarginals(support)]));
  for (const groupId of GROUP_IDS) {
    const declared = groups.get(groupId).arrivalProbability;
    for (const world of ["harmful", "safe"]) {
      if (!equal(marginals[world][groupId], declared)) {
        throw new Error(`${world} ${groupId} marginal does not match its declaration`);
      }
    }
    if (!equal(marginals.harmful[groupId], marginals.safe[groupId])) {
      throw new Error(`${groupId} marginals differ across worlds`);
    }
    if (declared.n === 0n) throw new Error("group arrival probabilities must be positive");
  }
  return { capacity, groups, harmDelay, horizon, marginals, policies, worlds };
}

function resultRow(parsed, policy, world) {
  const evaluation = evaluateQueue(parsed, policy, parsed.worlds[world]);
  const totalArrivals = GROUP_IDS.reduce(
    (sum, groupId) => add(sum, parsed.groups.get(groupId).arrivalProbability),
    ZERO,
  );
  const completedPerBatch = GROUP_IDS.reduce((sum, groupId) => add(sum, evaluation.completed[groupId]), ZERO);
  const unconditionalHarm = GROUP_IDS.reduce((sum, groupId) => add(sum, evaluation.harmed[groupId]), ZERO);
  const groupHarm = Object.fromEntries(GROUP_IDS.map((groupId) => [
    groupId,
    fractionText(divide(evaluation.harmed[groupId], parsed.groups.get(groupId).arrivalProbability)),
  ]));
  return {
    audit: {
      completion_rate: fractionText(ONE),
      deadlines: Object.fromEntries(GROUP_IDS.map((groupId) => [
        groupId,
        parsed.groups.get(groupId).deadline,
      ])),
      marginals: Object.fromEntries(GROUP_IDS.map((groupId) => [
        groupId,
        fractionText(parsed.groups.get(groupId).arrivalProbability),
      ])),
      service_cost: fractionText(policy.serviceCost),
    },
    backlog_probability: fractionText(evaluation.backlogProbability),
    completed_per_batch: fractionText(completedPerBatch),
    first_harm_time: unconditionalHarm.n === 0n ? null : evaluation.firstHarmTime,
    group_harm: groupHarm,
    overall_harm: fractionText(divide(unconditionalHarm, totalArrivals)),
    policy_id: policy.policyId,
    unconditional_harm: fractionText(unconditionalHarm),
    world,
  };
}

function sortedValue(value) {
  if (Array.isArray(value)) return value.map(sortedValue);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(Object.keys(value).sort().map((key) => [key, sortedValue(value[key])]));
}

function handleRequest(request) {
  const parsed = parseInstance(request);
  const results = [];
  for (const policy of [...parsed.policies.values()].sort((left, right) => left.policyId.localeCompare(right.policyId))) {
    for (const world of ["harmful", "safe"]) results.push(resultRow(parsed, policy, world));
  }
  return { results };
}

try {
  const input = fs.readFileSync(0, "utf8");
  if (Buffer.byteLength(input, "utf8") > MAX_INPUT_BYTES) throw new Error("evaluator input exceeds the size bound");
  process.stdout.write(`${JSON.stringify(sortedValue(handleRequest(JSON.parse(input))))}\n`);
} catch (error) {
  process.stderr.write(`error: ${error.message}\n`);
  process.exitCode = 2;
}
