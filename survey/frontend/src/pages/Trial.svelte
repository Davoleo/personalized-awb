<script>
    import { t } from '../i18n/i18n.svelte.js';
    import { session, currentStep, progress, variantUrl } from "../api/Session.svelte.js";
    import Question from "../components/Question.svelte";

    const step = $derived(currentStep());
    const counts = $derived(progress())

    //preload next trial
    $effect(() => {
        if (!step) return;
        const next = session.plan[step.question.question_index + 1];
        if (!next) return;
        for (const id of next.variant_order) {
            new Image().src = variantUrl(next.sample_id, id)
        }
    });
</script>

{#if step}
    <p>{t('trial.progress', { n: counts.done + 1, total: counts.total})}</p>
    <progress value={counts.done} max="{counts.total}"></progress>

    <!-- {#key} - resets Question local state for every step and forces re-rendering. -->
    {#key `${step.question.question_index}-${step.question_type}`}
        <Question {step} />
    {/key}
{/if}

<style>
    progress { width: 100%; margin-bottom: 1rem}
</style>