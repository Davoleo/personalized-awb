<script>
    import { t } from '../i18n/i18n.svelte.js'

    //TODO apply sliders to the image
    let { valueT = $bindable(0), valueQ = $bindable(0), disabled = false } = $props()

    function reset() {
        valueT = 0;
        valueQ = 0;
    }
</script>

<div class="sliders">
    <p class="title">{t('slider.title')}</p>

    <label>
        <span>{t('slider.t.left')}</span>
        <input type="range" min="-1" max="1" step="0.01" bind:value={valueT} {disabled} />
        <span>{t('slider.t.right')}</span>
    </label>
    <label>
        <span>{t('slider.q.left')}</span>
        <input type="range" min="-1" max="1" step="0.01" bind:value={valueQ} {disabled} />
        <span>{t('slider.q.right')}</span>
    </label>

    <button type="button" onclick={reset} {disabled}>{t('slider.reset')}</button>
</div>

<style>
    .sliders { margin: 1rem 0; }
    .title { font-weight: 600; margin: 0 0 0.5rem;}
    label {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.5rem;
    }
    label span { flex: 0 0 7rem; }
    label span:last-child { text-align: right; }
    label input {flex: 1; min-width: 0}

    /* column orientation on small screen */
    @media (max-width: 560px) {
        label { flex-direction: column; align-items: stretch; gap: 0.15rem}
        label span { flex: none }
        label span:last-child {text-align: left;}
    }
</style>

