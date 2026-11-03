import en from './en.json'
import it from './it.json'

const dictionaries = { en, it };
const STORAGE_KEY = 'awb-lang';

function initialLang() {
    try {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved in dictionaries) return saved;
    }
    catch {
        // storage unavailable -> fallback to browser language
    }
    return navigator.language?.toLowerCase().startsWith('it') ? 'it' : 'en';
}

// Reactivity: any component that calls t() re-renders on language change
export const i18n = $state({ lang: initialLang() });
document.documentElement.lang = i18n.lang;

export function setLang(lang) {
    i18n.lang = lang;
    document.documentElement.lang = lang;
    try {
        localStorage.setItem(STORAGE_KEY, lang);
    }
    catch {
        //non-critical: will fall back to browser lang on render.
    }
}

/**
 * Translation function, will re-trigger for every component if i18n.lang changes
 * @param key translation key
 * @param params object map of all parameter names and values
 * @returns {*} translated text with injected parameters
 */
export function t(key, params = {}) {
    const text = dictionaries[i18n.lang][key] ?? dictionaries.en[key] ?? key;
    return text.replace(/\{(\w+)\}/g, (_, name) => params[name] ?? `{${name}}`);
}