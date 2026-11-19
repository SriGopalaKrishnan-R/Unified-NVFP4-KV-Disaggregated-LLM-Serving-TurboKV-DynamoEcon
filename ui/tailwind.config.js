/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        nvgreen: '#76b900',
        nvcyan: '#00d4ff',
        nvbg: '#070a0f',
        nvpanel: '#0e141f',
        nvborder: '#1e293b',
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
};
