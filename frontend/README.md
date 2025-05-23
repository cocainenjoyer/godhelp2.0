# KursachFE

Фронтенд проект на Next.js с аутентификацией и базой данных PostgreSQL.

## Технологии

- [Next.js 15](https://nextjs.org/) - React фреймворк с серверными компонентами
- [React 19](https://react.dev/) - JavaScript библиотека для создания пользовательских интерфейсов
- [TypeScript](https://www.typescriptlang.org/) - Типизированный JavaScript
- [Next Auth](https://next-auth.js.org/) - Аутентификация для Next.js
- [Drizzle ORM](https://orm.drizzle.team/) - ORM для работы с базой данных
- [PostgreSQL](https://www.postgresql.org/) - Реляционная база данных
- [TailwindCSS 4](https://tailwindcss.com/) - Utility-first CSS фреймворк

## Начало работы

### Установка зависимостей

```bash
npm install
```

### Настройка переменных окружения

Создайте файл `.env` в корне проекта со следующими переменными:

```
# База данных
DATABASE_URL=postgres://postgres:postgres@localhost:5432/kin4ik

# NextAuth
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=ваш_секретный_ключ

# Другие переменные окружения при необходимости
```

### Настройка базы данных

1. Установите PostgreSQL, если он еще не установлен
2. Создайте базу данных:
```bash
createdb kin4ik
```
3. База данных автоматически подключится с использованием строки подключения из переменной окружения `DATABASE_URL`

### Запуск в режиме разработки

```bash
npm run dev
```

### Сборка для продакшена

```bash
npm run build
```

### Запуск продакшен версии

```bash
npm run start
```

## Структура проекта

- `/app` - Основной код приложения (Next.js App Router)
  - `/api` - API роуты
  - `/components` - Компоненты приложения
  - `/about` - Страница "О нас"
  - `/profile` - Страница профиля пользователя
  - `/login` - Страница входа
  - `/register` - Страница регистрации
- `/lib` - Вспомогательные функции и утилиты
  - `/db` - Конфигурация базы данных и схемы Drizzle ORM
  - `/auth` - Настройки аутентификации NextAuth
- `/public` - Статические файлы
- `/types` - TypeScript типы 