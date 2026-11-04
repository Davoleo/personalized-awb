<script>
    import {onMount } from "svelte";
    import { i18n, setLang, t } from "./i18n/i18n.svelte.js";
    import { init, session} from "./api/Session.svelte.js";
    import Landing from "./pages/Landing.svelte";
    import Trial from "./pages/Trial.svelte";
    import Done from "./pages/Done.svelte";

    onMount(init)
</script>

<header>
    <h1>{t('app.title')}</h1>
    <div class="lang">
        {#each ['en', 'it'] as lang}
            <button type="button" class:active={i18n.lang === lang} onclick={() => setLang(lang)}>
                {lang.toUpperCase()}
            </button>
        {/each}
    </div>
</header>

<main>
    {#if session.status === 'loading'}
        <p>{t('common.leading')}</p>
    {:else if session.status === 'landing'}
        <Landing />
    {:else if session.status === 'trial' }
        <Trial />
    {:else if session.status === 'done' }
        <Done />
    {:else}
        <p class="error">{session.error}</p>
    {/if}
</main>

<style>
    header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        max-width: 1100px;
        margin: 0 auto;
        padding: 0.75rem 1rem 0;
    }
    h1 { font-size: 1.1rem; margin: 0; }
    .lang { display: flex; gap: 0.25rem; }
    .lang button {padding: 0.25rem 0.6rem; }
    .lang button.active { background-color: #222222; color: #FFFFFF}
</style>