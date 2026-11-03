<script>
    import { t } from '../i18n/i18n.svelte.js'

    //variant: {id, label, url}
    let { variants, selected, onselect } = $props();
</script>

<div class="grid">
    {#each variants as v (v.id)}
        <button type="button" class="cell" class:selected={v.id === selected} aria-pressed={v.id === selected}
                onclick={() => onselect(v.id)}>
            <img src={v.url} alt={t('grid.option', { label: v.label })} draggable="false" />
            <span class="label">{v.label}</span>
        </button>
    {/each}
</div>

<style>
    .grid {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    .cell {
        flex: 0 0 100%;
        padding: 0.4rem;
        background: #8A8A8A;
        border: 3px solid transparent;
        border-radius: 8px;
    }
    @media (min-width: 600px) {
        .cell { flex-basis: calc(50% - 0.375rem); }
    }
    @media (min-width: 1000px) {
        .cell { flex-basis: calc(33.333% - 0.5rem); }
    }
    .cell.selected { border-color: #111111 }
    img { display: block; width: 100%; margin-top: 0.3rem; font-weight: 700; }
    .label { display: block; margin-top: 0.3rem; font-weight: 700; }
</style>