import { defineConfig } from "vite";
import { svelte } from '@sveltejs/vite-plugin-svelte'

export default defineConfig({
    plugins: [svelte()],
    server: {
        // dev -> forward API calls  to FastAPI backend, browser only talks to that domain.
        proxy: {
            '/api': "http://localhost:8000",
            '/assets': "http://localhost:8000"
        }
    },
    build: {
        //Change vite's output folder from assets to static [to avoid collisions]
        assetsDir: 'static'
    }
})