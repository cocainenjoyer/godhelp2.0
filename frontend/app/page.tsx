"use client"

import { useState } from "react"
import HeroSlider from "./components/hero-slider"
import MovieQuiz from "./components/movie-quiz"
import SimilarMovies from "./components/similar-movies"
import Navbar from "./components/navbar"
import Footer from "./components/footer"
import { useSession } from "next-auth/react"

export default function HomePage() {
  const [showSurveyModal, setShowSurveyModal] = useState(false)
  const [surveyResults, setSurveyResults] = useState<null | any[]>(null)
  const { data: session, status } = useSession()

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white relative overflow-hidden">
      <div className="absolute inset-0 z-0">
        <div className="absolute inset-0 bg-[url('/movies-bg.jpg')] bg-cover bg-center opacity-40"></div>
        <div className="absolute inset-0 bg-gradient-to-br from-gray-900/90 to-gray-800/90"></div>
      </div>

      <div className="relative z-10">
        <Navbar session={session} />

        <main>
          <HeroSlider />
          {status === "authenticated" ? (
            <div className="container mx-auto px-4 py-16 grid md:grid-cols-2 gap-8">
              <MovieQuiz
                onOpenSurvey={() => setShowSurveyModal(true)}
                surveyResults={surveyResults}
                setSurveyResults={setSurveyResults}
                showSurveyModal={showSurveyModal}
                setShowSurveyModal={setShowSurveyModal}
              />
              <SimilarMovies />
            </div>
          ) : (
            <div className="container mx-auto px-4 py-16 flex justify-center items-center">
              <h1 className="text-2xl font-bold">Пожалуйста, войдите в систему, чтобы воспользоваться сервисом</h1>
            </div>
          )}
        </main>

        <Footer />
      </div>
    </div>
  )
}
