<script>
    import { t } from '../i18n/i18n.svelte.js'
    import { variantUrl, submitStep } from "../api/Session.svelte.js";
    import VariantGrid from "./VariantGrid.svelte";
    import SliderPanel from "./SliderPanel.svelte";
    import Preview from "./Preview.svelte";

    let { step } = $props();

    let selected = $state(null);
    let valueT = $state(0);
    let valueQ = $state(0);
    let submitting = $state(false);
    let error = $state(null);

    const shownAt = performance.now();
    let pickTimeMs = 0

    // A,B,C in shuffled order via plan
    const variants = $derived(
        step.question.variant_order.map((id, i) => ({
            id,
            label: String.fromCharCode(65+i),
            url: variantUrl(step.question.sample_id, id)
        })),
    );

    const adjusted = $derived(valueT !== 0 || valueQ !== 0)

    function pick(id) {
        if (id === selected) return;
        selected = id;
        pickTimeMs = Math.round(performance.now() - shownAt)
        valueT = 0;
        valueQ = 0;
    }

    async function confirm() {
        submitting = true;
        error = null;
        try {
            await submitStep({
                questionIndex: step.question.question_index,
                questionType: step.question_type,
                chosen: selected,
                responseTimeMs: pickTimeMs,
                adjustment: { t: valueT, q: valueQ, confirmedAsIs: !adjusted},
            });
        } catch (e) {
            error = e.message + ' : ' + (e.detail ?? '');
            submitting = false;
        }
    }
</script>

<section class="card">
    <h2>{t(`question.${step.question_type}`)}</h2>
    <p class="hint">{t(`question.${step.question_type}.hint`)}</p>

    <VariantGrid variants={variants} {selected} onselect={pick} />

    {#if selected}
        <Preview variantUrl={variantUrl(step.question.sample_id, selected)} bind:t={valueT} bind:q={valueQ} />
    {/if}

    <SliderPanel bind:valueT bind:valueQ disabled={!selected} />

    <p>
        <button type="button" class="primary" onclick={confirm} disabled={!selected || submitting}>
            {adjusted ? t('trial.confirm.adjusted') : t('trial.confirm.asis')}
        </button>
    </p>
    {#if !selected}<p class="hint">{t('trial.pick_first')}</p>{/if}
    {#if error}<p class="error">{error}</p>{/if}
</section>