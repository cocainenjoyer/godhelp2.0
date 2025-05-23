"use client"

import { useState, useEffect } from "react"
import { User, Film, Clock, ChevronLeft, ChevronRight } from "lucide-react"
import { useSession } from "next-auth/react"
import Navbar from "../components/navbar"
import Footer from "../components/footer"
import { redirect } from "next/navigation"
import { getUserPreferences, UserPreferences } from "@/lib/actions/user"

export default function ProfilePage() {
  const { data: session, status } = useSession()
  const [currentPage, setCurrentPage] = useState(0)
  const [userPrefs, setUserPrefs] = useState<UserPreferences | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  
  const mockRecommendations = [
    { id: 1, title: "Интерстеллар", date: "12.05.2023", posterUrl: "/interstellar.jpg" },
    { id: 2, title: "Начало", date: "10.05.2023", posterUrl: "/inception.jpg" },
    { id: 3, title: "Матрица", date: "05.05.2023", posterUrl: "/matrix.jpg" },
    { id: 4, title: "Матрица", date: "05.05.2023", posterUrl: "/matrix.jpg" },
    { id: 5, title: "Матрица", date: "05.05.2023", posterUrl: "/matrix.jpg" },
  ]
  
  const genreLabels: Record<string, string> = {
    documentation: "Документальные",
    comedy: "Комедии",
    european: "Европейские",
    animation: "Анимация",
    family: "Семейные",
    fantasy: "Фэнтези",
    music: "Музыкальные",
    drama: "Драмы",
    action: "Боевики",
    war: "Военные",
    crime: "Криминальные",
    scifi: "Научная фантастика",
    reality: "Реалити-шоу",
    western: "Вестерны",
    thriller: "Триллеры",
    romance: "Романтические",
    horror: "Ужасы",
    sport: "Спортивные",
    history: "Исторические"
  }

  const timePeriodLabels: Record<string, string> = {
    "1920s": "1920-е",
    "1930s": "1930-е",
    "1940s": "1940-е",
    "1950s": "1950-е",
    "1960s": "1960-е",
    "1970s": "1970-е",
    "1980s": "1980-е",
    "1990s": "1990-е",
    "2000s": "2000-е",
    "2010s": "2010-е",
    "2020s": "2020-е"
  }

  const durationLabels: Record<string, string> = {
    "under60": "до 60 минут",
    "60to120": "60-120 минут",
    "120to180": "120-180 минут",
    "180to210": "180-210 минут"
  }
  
  useEffect(() => {
    async function loadUserPreferences() {
      try {
        setIsLoading(true)
        const result = await getUserPreferences()
        
        if (result.success && result.data) {
          setUserPrefs(result.data)
        } else if (result.error) {
          setError(result.error)
        }
      } catch (err) {
        console.error("Failed to load user preferences:", err)
        setError("Не удалось загрузить предпочтения пользователя")
      } finally {
        setIsLoading(false)
      }
    }

    if (status === "authenticated") {
      loadUserPreferences()
    }
  }, [status])

  const userQuestions = userPrefs ? [
    { 
      id: 1, 
      question: "Ваш любимый жанр фильмов?", 
      answer: userPrefs.genres?.map(genre => genreLabels[genre] || genre).join(", ") || "Не указано" 
    },
    { 
      id: 2, 
      question: "Предпочитаемый год выпуска?", 
      answer: timePeriodLabels[userPrefs.timePeriod || ""] || userPrefs.timePeriod || "Не указано" 
    },
    { 
      id: 3, 
      question: "Продолжительность серии?", 
      answer: durationLabels[userPrefs.episodeDuration || ""] || userPrefs.episodeDuration || "Не указано" 
    },
  ] : []

  if (status === "unauthenticated") {
    redirect("/login")
  }

  if (status === "loading" || isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500"></div>
      </div>
    )
  }

  const itemsPerPage = 3
  const totalPages = Math.ceil(mockRecommendations.length / itemsPerPage)
  const paginatedRecommendations = mockRecommendations.slice(
    currentPage * itemsPerPage, 
    (currentPage + 1) * itemsPerPage
  )

  const goToNextPage = () => {
    setCurrentPage((prev) => (prev < totalPages - 1 ? prev + 1 : prev))
  }

  const goToPrevPage = () => {
    setCurrentPage((prev) => (prev > 0 ? prev - 1 : prev))
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white relative overflow-hidden">
      <div className="absolute inset-0 z-0">
        <div className="absolute inset-0 bg-[url('/movies-bg.jpg')] bg-cover bg-center opacity-40"></div>
        <div className="absolute inset-0 bg-gradient-to-br from-gray-900/90 to-gray-800/90"></div>
      </div>

      <div className="relative z-10">
        <Navbar session={session} />

        <main className="container mx-auto px-4 py-16">
          <div className="max-w-3xl mx-auto">

            
            <div className="bg-gray-800/80 backdrop-blur-sm rounded-xl shadow-lg overflow-hidden mb-6">
              <div className="p-8">
                <div className="flex items-center space-x-4 mb-6">
                  <div className="h-20 w-20 rounded-full bg-purple-600 flex items-center justify-center">
                    <User className="h-10 w-10" />
                  </div>
                  <h1 className="text-3xl font-bold">Профиль пользователя</h1>
                </div>
                
                <div className="space-y-6">
                  <div className="space-y-1">
                    <div className="flex justify-between items-center">
                      <label className="block text-sm font-medium text-gray-400">Имя пользователя</label>
                    </div>
                    
                    <div className="mt-1 block w-full rounded-md bg-gray-700 border-gray-600 px-3 py-2">
                      {session?.user?.name || "Не указано"}
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-gray-800/80 backdrop-blur-sm rounded-xl shadow-lg overflow-hidden mb-6">
              <div className="p-6">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="h-8 w-8 rounded-full bg-purple-600 flex items-center justify-center">
                    <User className="h-4 w-4" />
                  </div>
                  <h2 className="text-xl font-bold">Выбранные предпочтения</h2>
                </div>
                
                {error ? (
                  <div className="bg-red-500/20 text-red-300 p-4 rounded-lg">
                    {error}
                  </div>
                ) : (
                  <div className="space-y-3">
                    {userQuestions.map((item) => (
                      <div key={item.id} className="bg-gray-700/50 rounded-lg p-4">
                        <p className="text-gray-300 text-sm">{item.question}</p>
                        <p className="font-medium text-white">{item.answer}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
            
            {/* Recommendation History with Pagination */}
            <div className="bg-gray-800/80 backdrop-blur-sm rounded-xl shadow-lg overflow-hidden mb-6">
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className="h-8 w-8 rounded-full bg-purple-600 flex items-center justify-center">
                      <Clock className="h-4 w-4" />
                    </div>
                    <h2 className="text-xl font-bold">История рекомендаций</h2>
                  </div>
                  
                  {/* Pagination Controls */}
                  <div className="flex items-center space-x-2">
                    <button 
                      onClick={goToPrevPage}
                      disabled={currentPage === 0}
                      className="p-1 rounded-full bg-gray-700 hover:bg-gray-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <ChevronLeft className="h-5 w-5" />
                    </button>
                    <span className="text-sm">{currentPage + 1} / {totalPages}</span>
                    <button 
                      onClick={goToNextPage}
                      disabled={currentPage === totalPages - 1}
                      className="p-1 rounded-full bg-gray-700 hover:bg-gray-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <ChevronRight className="h-5 w-5" />
                    </button>
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {paginatedRecommendations.map((movie) => (
                    <div key={movie.id} className="bg-gray-700/50 rounded-lg overflow-hidden">
                      <div className="h-40 bg-gray-600 flex items-center justify-center">
                        <Film className="h-16 w-16 text-gray-500" />
                      </div>
                      <div className="p-3">
                        <h3 className="font-medium">{movie.title}</h3>
                        <p className="text-gray-400 text-sm">{movie.date}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </main>

        <Footer />
      </div>
    </div>
  )
}
