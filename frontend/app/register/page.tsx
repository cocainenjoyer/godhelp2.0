//TODO: improve design, do right questions, add validation, add backend
"use client"
import Link from "next/link"
import { ArrowRight, ArrowLeft, Check, Loader2 } from "lucide-react"
import { useState, useEffect } from "react"
import { registerUser, checkEmailExists, checkUsernameExists } from "@/lib/actions/auth"
import { useRouter } from "next/navigation"

export default function RegisterPage() {
  const router = useRouter()
  const [step, setStep] = useState(1)
  const [formData, setFormData] = useState({
    email: "",
    username: "",
    password: "",
    genres: [] as string[],
    timePeriod: "",
    episodeDuration: "",
  })
  const [error, setError] = useState("")
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [selectedGenres, setSelectedGenres] = useState<string[]>([])
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isCheckingEmail, setIsCheckingEmail] = useState(false)
  const [isCheckingUsername, setIsCheckingUsername] = useState(false)
  const [registrationComplete, setRegistrationComplete] = useState(false)

  const questions = [
    {
      type: "multi",
      question: "Какие жанры сериалов вы предпочитаете?",
      options: [
        { value: "documentation", label: "Документальные" },
        { value: "comedy", label: "Комедии" },
        { value: "european", label: "Европейские" },
        { value: "animation", label: "Анимация" },
        { value: "family", label: "Семейные" },
        { value: "fantasy", label: "Фэнтези" },
        { value: "music", label: "Музыкальные" },
        { value: "drama", label: "Драмы" },
        { value: "action", label: "Боевики" },
        { value: "war", label: "Военные" },
        { value: "crime", label: "Криминальные" },
        { value: "scifi", label: "Научная фантастика" },
        { value: "reality", label: "Реалити-шоу" },
        { value: "western", label: "Вестерны" },
        { value: "thriller", label: "Триллеры" },
        { value: "romance", label: "Романтические" },
        { value: "horror", label: "Ужасы" },
        { value: "sport", label: "Спортивные" },
        { value: "history", label: "Исторические" }
      ],
      name: "genres",
      required: true
    },
    {
      type: "single",
      question: "Какой временной период в мире сериалов вам наиболее интересен?",
      options: [
        { value: "1920s", label: "1920-е" },
        { value: "1930s", label: "1930-е" },
        { value: "1940s", label: "1940-е" },
        { value: "1950s", label: "1950-е" },
        { value: "1960s", label: "1960-е" },
        { value: "1970s", label: "1970-е" },
        { value: "1980s", label: "1980-е" },
        { value: "1990s", label: "1990-е" },
        { value: "2000s", label: "2000-е" },
        { value: "2010s", label: "2010-е" },
        { value: "2020s", label: "2020-е" },
      ],
      name: "timePeriod",
      required: true
    },
    {
      type: "single",
      question: "Какова средняя продолжительность одной серии в ваших любимых сериалах? (в минутах)",
      options: [
        { value: "under60", label: "до 60" },
        { value: "60to120", label: "60-120" },
        { value: "120to180", label: "120-180" },
        { value: "180to210", label: "180-210" },
      ],
      name: "episodeDuration",
      required: true
    }
  ]

  useEffect(() => {
    // Redirect to login page after successful registration and a short delay
    if (registrationComplete) {
      const timer = setTimeout(() => {
        router.push("/login");
      }, 3000);
      
      return () => clearTimeout(timer);
    }
  }, [registrationComplete, router]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
    
    // Clear error when user starts typing again
    if (error) {
      setError("")
    }
  }

  const handleMultiSelect = (value: string) => {
    setSelectedGenres(prev => {
      const newSelection = prev.includes(value)
        ? prev.filter(v => v !== value)
        : [...prev, value]
      
      setFormData(prev => ({
        ...prev,
        genres: newSelection
      }))
      
      return newSelection
    })
  }

  const handleQuestionAnswer = (value: string) => {
    const currentQ = questions[currentQuestion]
    
    if (currentQ.type === "single") {
      setFormData(prev => ({
        ...prev,
        [currentQ.name as keyof typeof formData]: value
      }))
    }
  }

  const validateStep1 = () => {
    if (!formData.email || !formData.username || !formData.password) {
      setError("Пожалуйста, заполните все поля")
      return false
    }
    if (!formData.email.includes("@")) {
      setError("Пожалуйста, введите корректный email")
      return false
    }
    if (formData.password.length < 6) {
      setError("Пароль должен содержать минимум 6 символов")
      return false
    }
    if (formData.username.length < 3) {
      setError("Имя пользователя должно содержать минимум 3 символа")
      return false
    }
    return true
  }

  const validateCurrentQuestion = () => {
    const currentQ = questions[currentQuestion]
    
    if (currentQ.required) {
      if (currentQ.type === "multi" && formData.genres.length === 0) {
        setError("Пожалуйста, выберите хотя бы один жанр")
        return false
      }
      
      if (currentQ.type === "single" && !formData[currentQ.name as keyof typeof formData]) {
        setError("Пожалуйста, выберите один из вариантов")
        return false
      }
    }
    
    return true
  }

  const checkEmail = async () => {
    if (!formData.email.includes("@")) {
      setError("Пожалуйста, введите корректный email")
      return false
    }
    
    setIsCheckingEmail(true)
    try {
      const result = await checkEmailExists(formData.email)
      if (result.exists) {
        setError("Этот email уже используется")
        setIsCheckingEmail(false)
        return false
      }
      setIsCheckingEmail(false)
      return true
    } catch (err) {
      console.error("Error checking email:", err)
      setError("Ошибка при проверке email")
      setIsCheckingEmail(false)
      return false
    }
  }

  const checkUsername = async () => {
    if (formData.username.length < 3) {
      setError("Имя пользователя должно содержать минимум 3 символа")
      return false
    }
    
    setIsCheckingUsername(true)
    try {
      const result = await checkUsernameExists(formData.username)
      if (result.exists) {
        setError("Это имя пользователя уже занято")
        setIsCheckingUsername(false)
        return false
      }
      setIsCheckingUsername(false)
      return true
    } catch (err) {
      console.error("Error checking username:", err)
      setError("Ошибка при проверке имени пользователя")
      setIsCheckingUsername(false)
      return false
    }
  }

  const handleNext = async () => {
    setError("")
    if (step === 1 && validateStep1()) {
      const emailAvailable = await checkEmail()
      if (!emailAvailable) return;
      
      const usernameAvailable = await checkUsername()
      if (usernameAvailable) {
        setStep(2)
      }
    }
  }

  const handleBackQ = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(prev => prev - 1)
    } else {
      setStep(1)
    }
  }

  const handleNextQ = () => {
    setError("")
    
    if (!validateCurrentQuestion()) {
      return
    }
    
    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(prev => prev + 1)
    } else {
      handleSubmitRegistration()
    }
  }

  const handleSubmitRegistration = async () => {
    try {
      setIsSubmitting(true)
      
      const initialData = {
        genres: formData.genres,
        timePeriod: formData.timePeriod,
        episodeDuration: formData.episodeDuration
      }

      const result = await registerUser({
        email: formData.email,
        username: formData.username,
        password: formData.password,
        initial: initialData
      })

      if (result.success) {
        setStep(3)
        setRegistrationComplete(true)
      } else {
        setError(result.error || "Ошибка регистрации")
        setIsSubmitting(false)
      }
    } catch (err) {
      console.error("Registration error:", err)
      setError("Произошла непредвиденная ошибка")
      setIsSubmitting(false)
    }
  }

  const progress = ((currentQuestion + 1) / questions.length) * 100

  const renderQuestion = () => {
    const currentQ = questions[currentQuestion]

    switch (currentQ.type) {
      case "single":
        if (currentQ.name === "timePeriod") {
          return (
            <div className="space-y-3 max-h-80 overflow-y-auto pr-2 custom-scrollbar">
              {currentQ.options?.map((option) => (
                <button
                  key={option.value}
                  onClick={() => handleQuestionAnswer(option.value)}
                  className={`w-full flex items-center justify-between px-4 py-3 rounded-lg transition-colors ${
                    formData.timePeriod === option.value
                      ? "bg-purple-600/70 hover:bg-purple-600"
                      : "bg-gray-600/50 hover:bg-gray-600"
                  }`}
                >
                  <span>{option.label}</span>
                  {formData.timePeriod === option.value && (
                    <Check className="h-4 w-4" />
                  )}
                </button>
              ))}
            </div>
          )
        }
        return (
          <div className="space-y-3">
            {currentQ.options?.map((option) => (
              <button
                key={option.value}
                onClick={() => handleQuestionAnswer(option.value)}
                className={`w-full flex items-center justify-between px-4 py-3 rounded-lg transition-colors ${
                  formData[currentQ.name as keyof typeof formData] === option.value
                    ? "bg-purple-600/70 hover:bg-purple-600"
                    : "bg-gray-600/50 hover:bg-gray-600"
                }`}
              >
                <span>{option.label}</span>
                {formData[currentQ.name as keyof typeof formData] === option.value && (
                  <Check className="h-4 w-4" />
                )}
              </button>
            ))}
          </div>
        )

      case "multi":
        return (
          <div className="space-y-3 max-h-80 overflow-y-auto pr-2 custom-scrollbar">
            {currentQ.options?.map((option) => (
              <button
                key={option.value}
                onClick={() => handleMultiSelect(option.value)}
                className={`w-full flex items-center justify-between px-4 py-3 rounded-lg transition-colors ${
                  selectedGenres.includes(option.value)
                    ? "bg-purple-600/70 hover:bg-purple-600"
                    : "bg-gray-600/50 hover:bg-gray-600"
                }`}
              >
                <span>{option.label}</span>
                {selectedGenres.includes(option.value) && (
                  <Check className="h-4 w-4" />
                )}
              </button>
            ))}
          </div>
        )
    }
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-gray-900 to-gray-800 p-4 text-white relative overflow-hidden">
      <div className="absolute inset-0 z-0">
        <div className="absolute inset-0 bg-[url('/movies-bg.jpg')] bg-cover bg-center opacity-40"></div>
        <div className="absolute inset-0 bg-gradient-to-br from-gray-900/90 to-gray-800/90"></div>
      </div>

      <style jsx global>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(75, 85, 99, 0.2);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(139, 92, 246, 0.5);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(139, 92, 246, 0.7);
        }
      `}</style>

      <div className="w-full max-w-md transform rounded-xl bg-gray-800/50 p-8 shadow-2xl backdrop-blur-sm transition-all relative z-10">
        <div className="space-y-6">
          <div className="space-y-2 text-center">
            <h1 className="text-4xl font-bold tracking-tighter sm:text-5xl">
              {step === 3 ? "Успех!" : "Создание аккаунта"}
            </h1>
            <p className="text-gray-400">
              {step === 1 
                ? "Заполните ваши данные" 
                : step === 2 
                  ? "Расскажите о ваших предпочтениях"
                  : "Регистрация завершена успешно"}
            </p>
          </div>

          {step === 1 ? (
            <form onSubmit={(e) => { e.preventDefault(); handleNext(); }} className="space-y-4">
              <div className="space-y-2">
                <input
                  type="email"
                  name="email"
                  placeholder="Email"
                  value={formData.email}
                  onChange={handleInputChange}
                  className="w-full rounded-lg bg-gray-700/50 px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  disabled={isCheckingEmail || isCheckingUsername}
                />
              </div>
              <div className="space-y-2">
                <input
                  type="text"
                  name="username"
                  placeholder="Имя пользователя"
                  value={formData.username}
                  onChange={handleInputChange}
                  className="w-full rounded-lg bg-gray-700/50 px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  disabled={isCheckingEmail || isCheckingUsername}
                />
              </div>
              <div className="space-y-2">
                <input
                  type="password"
                  name="password"
                  placeholder="Пароль"
                  value={formData.password}
                  onChange={handleInputChange}
                  className="w-full rounded-lg bg-gray-700/50 px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  disabled={isCheckingEmail || isCheckingUsername}
                />
              </div>

              {error && (
                <div className="text-red-400 text-sm text-center">{error}</div>
              )}

              <button
                type="submit"
                className="group flex w-full items-center justify-center rounded-lg bg-purple-600 px-4 py-3 font-medium text-white transition-colors hover:bg-purple-500 disabled:opacity-70"
                disabled={isCheckingEmail || isCheckingUsername}
              >
                {isCheckingEmail || isCheckingUsername ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Проверка...
                  </>
                ) : (
                  <>
                    Далее
                    <ArrowRight className="ml-2 h-4 w-4 opacity-70 transition-transform group-hover:translate-x-1" />
                  </>
                )}
              </button>
            </form>
          ) : step === 2 ? (
            <div className="space-y-6">
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                ></div>
              </div>

              <div className="bg-gray-700/50 rounded-xl p-6 space-y-6">
                <h3 className="text-xl font-medium text-center">
                  {questions[currentQuestion].question}
                </h3>
                {renderQuestion()}
                
                {error && (
                  <div className="text-red-400 text-sm text-center mt-3">{error}</div>
                )}
                
                {currentQuestion === 0 && (
                  <div className="text-sm text-gray-300 mt-2 h-5">
                    {formData.genres.length > 0 && `Выбрано жанров: ${formData.genres.length}`}
                  </div>
                )}
              </div>

              <div className="flex justify-between items-center">
                <button
                  onClick={handleBackQ}
                  className="flex items-center px-4 py-2 text-white bg-purple-600 rounded-lg hover:bg-purple-500 transition-colors"
                  disabled={isSubmitting}
                >
                  <ArrowLeft className="mr-2 h-4 w-4" />
                  Назад
                </button>
                <span className="text-sm text-gray-400">
                  Вопрос {currentQuestion + 1} из {questions.length}
                </span>
                <button
                  onClick={handleNextQ}
                  className="flex items-center px-4 py-2 text-white bg-purple-600 rounded-lg hover:bg-purple-500 transition-colors"
                  disabled={isSubmitting}
                >
                  {currentQuestion === questions.length - 1 ? 
                    (isSubmitting ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Отправка...
                      </>
                    ) : "Завершить") 
                    : "Далее"}
                  {!isSubmitting && <ArrowRight className="ml-2 h-4 w-4" />}
                </button>
              </div>
            </div>
          ) : (
            <div className="text-center space-y-4">
              <div className="flex justify-center mb-6">
                <div className="w-20 h-20 bg-green-500 rounded-full flex items-center justify-center">
                  <Check className="h-10 w-10" />
                </div>
              </div>
              <h2 className="text-2xl font-bold">Регистрация завершена!</h2>
              <p className="text-gray-400">Спасибо за регистрацию. Сейчас вы будете перенаправлены на страницу входа.</p>
              <div className="flex justify-center mt-2">
                <Loader2 className="h-6 w-6 text-purple-500 animate-spin" />
              </div>
              <Link
                href="/login"
                className="inline-flex items-center justify-center rounded-lg bg-purple-600 px-4 py-3 font-medium text-white transition-colors hover:bg-purple-500"
              >
                Перейти к входу
                <ArrowRight className="ml-2 h-4 w-4 opacity-70" />
              </Link>
            </div>
          )}

          {step === 1 && (
            <div className="text-center">
              <Link
                href="/login"
                className="text-sm text-gray-400 hover:text-white transition-colors"
              >
                Уже есть аккаунт? Войдите здесь
              </Link>
            </div>
          )}

          <div className="mt-8 text-center text-sm text-gray-500">
            <p>© {new Date().getFullYear()} Kin4ik. Все права защищены.</p>
          </div>
        </div>
      </div>
    </div>
  )
} 