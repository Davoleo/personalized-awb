<script>
    import { t } from '../i18n/i18n.svelte.js'
    import { start } from '../api/Session.svelte.js'
    
    let agreed = $state(false)
    let starting = $state(false)
    let error = $state(null)
    
    async function begin() {
        starting = true;
        error = null;
        try {
            await start();
        }
        catch (e) {
            
        }
    }
</script>

<section class="card">
    <h2>{t('landing.heading')}</h2>
    <p>{t('landing.intro')}</p>
    <p>{t('landing.tips')}</p>

    <!--? demographics form maybe -->

    <label>
        <input type="checkbox" bind:checked={agreed} />
        {t('landing.consent')}
    </label>
    <p>
        <button type="button" class="primary" onclick={begin} disabled={!agreed || starting}>
            {t('landing.start')}
        </button>
    </p>
    {#if error}<p class="error">{error}</p>{/if}
</section>