import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';

// https://astro.build/config
export default defineConfig({
  // SUSTITUYE 'WichoArenas' por tu usuario exacto de GitHub
  site: 'https://WichoArenas.github.io', 
  
  // SUSTITUYE 'portafolio-luis' por el nombre EXACTO de tu repositorio
  base: '/portafolio-luis', 
  
  vite: {
    plugins: [tailwindcss()]
  }
});