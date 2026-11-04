import { api } from './api.js'
import { i18n } from "../i18n/i18n.svelte.js";

const STORAGE_KEY = 'awb-participant-id';

function loadSavedId() {
    try {
        return localStorage.getItem(STORAGE_KEY);
    } catch { return null; }
}
function saveId(id) {
    try {
        localStorage.setItem(STORAGE_KEY, id);
    } catch { /*? safe to ignore */}
}

function clearSavedId() {
    try {
        localStorage.removeItem(STORAGE_KEY);
    } catch { /* safe to ignore: no key or localstorage was present so it's already clear */}
}

export const session = $state({
    status: 'loading',  // loading | landing | trial | done | error
    participantId: null,
    plan: [],           // questions prepared by backend
    answered: [],       // [{ trial_index, question_type }]
    manifest: null,
    error: null,
});

// --- helpers ---

export function allSteps() {
    return session.plan.flatMap(question =>
        question.question_order.map(qType => ({question: question, question_type: qType})),
    );
}

function isAnswered(trialIndex, type) {
    return session.answered.some(
        a => a.question_index === trialIndex && a.question_type === type
    );
}

export function currentStep() {
    return allSteps().find(step => !isAnswered(step.question.question_index, step.question_type)) ?? null
}

export function progress() {
    return { done: session.answered.length, total: allSteps().length }
}

export function variantUrl(sampleId, variantId) {
    const sample = session.manifest.samples.find(sample => sample.id === sampleId)
    const variant = sample.variants.find(variant => variant.id === variantId)
    return `/assets/${variant.path}`
}


// --- actions ---
function applyState(state) {
    session.participantId = state.participant_id;
    session.plan = state.plan;
    session.answered = state.answered;
    saveId(state.participant_id);
    session.status = currentStep() ? 'trial' : 'done'
}

/**
 * Initializes participant status using localStorage or from scratch
 */
export async function init() {
    try {
        session.manifest = await api.getManifest()
        const savedId = loadSavedId()
        if (savedId) {
            try {
                applyState(await api.getParticipant(savedId))
                return;
            } catch (e) {
                if (e.status !== 404) throw e;
                clearSavedId(); // ghost id: client knows but server doesn't -> clear localStorage ghost value
            }
        }
        session.status = 'landing';
    } catch (e) {
        session.error = e.message;
        session.status = 'error';
    }
}

export async function start() {
    const state = await api.createParticipant({
        language: i18n.lang,
        user_agent: navigator.userAgent,
        screen_width: window.screen.width,
        screen_height: window.screen.height,
    });
    // apply new state received from backend to local client
    applyState(state);
}

/** save answer [picked variant + slide adjustments] */
export async function submitStep({ questionIndex, questionType, chosen, responseTimeMs, adjustment }){
    const id = session.participantId;

    await api.submitTrial({
        participant_id: id,
        question_index: questionIndex,
        question_type: questionType,
        chosen_variant: chosen,
        response_time_ms: responseTimeMs,
    });
    await api.submitAdjustment({
        participant_id: id,
        question_index: questionIndex,
        question_type: questionType,
        base_variant: chosen,
        final_t: adjustment.t,
        final_q: adjustment.q,
        confirmed_as_is: adjustment.confirmedAsIs,
    });

    session.answered.push({question_index: questionIndex, question_type: questionType});

    if (!currentStep()) {
        await api.finish(id)
        session.status = 'done'
    }
}