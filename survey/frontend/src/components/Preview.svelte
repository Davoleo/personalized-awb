<script>
    import { userGains } from "../lib/gains.js";

    const { variantUrl, t, q } = $props();

    const gains = $derived(userGains(t, q));
</script>

<!-- SVG FILTER -->
<!-- zero-size rather than display:none correct behaviour on most browsers [otherwise might ignore filter] -->
<svg width="0" height="0" aria-hidden="true" style="position: absolute">
    <filter id="p-awb" color-interpolation-filters="linearRGB"> <!-- ! must test on Safari -->
        <feComponentTransfer>
            <feFuncR type="linear" slope={gains.r} />
            <feFuncG type="linear" slope={gains.g} />
            <feFuncB type="linear" slope={gains.b} />
        </feComponentTransfer>
    </filter>
</svg>

<img src={variantUrl} alt="" style:filter="url(#p-awb)" />

