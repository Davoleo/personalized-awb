export const ALPHA = 0.20;
export const BETA = 0.15;


/**
 * Function to return the user gains in RGB [normalized with G fixed at 1].
 * note that t and q are between [-1,1] and correspond respectively to the temperature slider and green-magenta slider.
 * ALPHA and BETA are constants that affect how strong the effect of t and q with sane defaults.
 */
export function userGains(t, q, alpha=ALPHA, beta=BETA) {
    return {
        r: Math.exp(alpha * t + beta * q),
        g: 1,
        b: Math.exp(-alpha * t + beta * q)
    };
}