import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    domains: ['i.pinimg.com', 'i.ytimg.com', 'gamemag.ru', 'm.media-amazon.com', 'static.life.ru'],
  },
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
  },
  // Опционально: добавьте эту настройку, если хотите использовать переменные окружения в браузере
  publicRuntimeConfig: {
    apiUrl: process.env.NEXT_PUBLIC_API_URL,
  },
};

export default nextConfig;
