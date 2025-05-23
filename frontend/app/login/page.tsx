//TODO: move to unauthenticated page, implement auth, add middleware
"use client"
import Link from "next/link"
import { ArrowRight, Loader2 } from "lucide-react"
import { useState } from "react"
import { useRouter } from "next/navigation"
import { signIn, useSession } from "next-auth/react"
import { useEffect } from "react"

export default function LandingPage() {
  const router = useRouter()
  const { data: session, status } = useSession()
  const [loginIdentifier, setLoginIdentifier] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState("")
  const [success, setSuccess] = useState(false)
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    if (status === "authenticated" && session) {
      router.push("/")
    }
  }, [session, status, router])

  const handleSignIn = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    setSuccess(false)

    if (!loginIdentifier || !password) {
      setError("Пожалуйста, заполните все поля")
      return
    }

    if (password.length < 6) {
      setError("Пароль должен содержать минимум 6 символов")
      return
    }

    try {
      setIsLoading(true)
      
      const result = await signIn("credentials", {
        loginIdentifier,
        password,
        redirect: false
      })
      
      if (result?.ok) {
        setSuccess(true)
        setTimeout(() => {
          router.push("/")
        }, 1500)
      } else {
        setError("Неверный логин или пароль")
      }
    } catch (err) {
      console.error("Login error:", err)
      setError("Произошла непредвиденная ошибка")
    } finally {
      setIsLoading(false)
    }
  }

  if (status === "loading") {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-gray-900 to-gray-800">
        <div className="flex items-center text-white text-xl">
          <Loader2 className="mr-2 h-6 w-6 animate-spin" />
          Загрузка...
        </div>
      </div>
    )
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-gray-900 to-gray-800 p-4 text-white relative overflow-hidden">
      <div className="absolute inset-0 z-0">
        <div className="absolute inset-0 bg-[url('/movies-bg.jpg')] bg-cover bg-center opacity-40"></div>
        <div className="absolute inset-0 bg-gradient-to-br from-gray-900/90 to-gray-800/90"></div>
      </div>

      <div className="w-full max-w-md transform rounded-xl bg-gray-800/50 p-8 shadow-2xl backdrop-blur-sm transition-all relative z-10">
        <div className="space-y-6">
          <div className="space-y-2 text-center">
            <h1 className="text-4xl font-bold tracking-tighter sm:text-5xl">Добро пожаловать в Kin4ik</h1>
            <p className="text-gray-400">Пожалуйста, войдите, чтобы продолжить</p>
          </div>

          <form onSubmit={handleSignIn} className="space-y-4">
            <div className="space-y-2">
              <input
                type="text"
                placeholder="Email или имя пользователя"
                value={loginIdentifier}
                onChange={(e) => setLoginIdentifier(e.target.value)}
                className="w-full rounded-lg bg-gray-700/50 px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>
            <div className="space-y-2">
              <input
                type="password"
                placeholder="Пароль"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full rounded-lg bg-gray-700/50 px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>

            {error && (
              <div className="text-red-400 text-sm text-center">{error}</div>
            )}
            
            {success && (
              <div className="text-green-400 text-sm text-center">Вход выполнен успешно!</div>
            )}

            <button
              type="submit"
              disabled={isLoading}
              className="group flex w-full items-center justify-center rounded-lg bg-purple-600 px-4 py-3 font-medium text-white transition-colors hover:bg-purple-500 disabled:opacity-70"
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Выполняется вход...
                </>
              ) : (
                <>
                  Войти
                  <ArrowRight className="ml-2 h-4 w-4 opacity-70 transition-transform group-hover:translate-x-1" />
                </>
              )}
            </button>
          </form>

          <div className="text-center">
            <Link
              href="/register"
              className="text-sm text-gray-400 hover:text-white transition-colors"
            >
              Нет аккаунта? Зарегистрируйтесь здесь
            </Link>
          </div>

          <div className="mt-8 text-center text-sm text-gray-500">
            <p>© {new Date().getFullYear()} Kin4ik. Все права защищены.</p>
          </div>
        </div>
      </div>
    </div>
  )
}
