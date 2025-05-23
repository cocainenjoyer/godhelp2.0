"use client"

import { useState, useEffect, useCallback } from "react"
import { ChevronLeft, ChevronRight, Star } from "lucide-react"
import Image from "next/image"
import { getMedianReccomendations } from "@/lib/actions/reccomendations"
import type { Movie } from "@/lib/types/movie"

export default function HeroSlider() {
  const [currentSlide, setCurrentSlide] = useState(0)
  const [movies, setMovies] = useState<Movie[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const fetchMovies = async () => {
      setIsLoading(true)
      const response = await getMedianReccomendations()
      if (response.success && response.recommendations) {
        setMovies(response.recommendations)
      }
      setIsLoading(false)
    }
    fetchMovies()
  }, [])

  const nextSlide = useCallback(() => {
    setCurrentSlide((prev) => (prev === movies.length - 1 ? 0 : prev + 1))
  }, [movies.length])

  const prevSlide = useCallback(() => {
    setCurrentSlide((prev) => (prev === 0 ? movies.length - 1 : prev - 1))
  }, [movies.length])

  useEffect(() => {
    if (movies.length === 0) return

    const timeout = setTimeout(() => {
      nextSlide()
    }, 6000)

    return () => clearTimeout(timeout)
  }, [currentSlide, movies.length, nextSlide])

  if (isLoading) {
    return (
      <section className="relative w-full" style={{ height: "min(80vh, 700px)" }}>
        <div className="absolute inset-0 bg-gray-900 animate-pulse">
          <div className="absolute inset-0 flex items-end">
            <div className="container mx-auto px-4 pb-12 md:pb-16 lg:pb-24">
              <div className="max-w-xl md:max-w-2xl space-y-4">
                <div className="h-6 w-24 bg-gray-800 rounded animate-pulse"></div>
                <div className="h-12 w-3/4 bg-gray-800 rounded animate-pulse"></div>
                <div className="flex gap-2">
                  <div className="h-6 w-16 bg-gray-800 rounded-full animate-pulse"></div>
                  <div className="h-6 w-16 bg-gray-800 rounded-full animate-pulse"></div>
                  <div className="h-6 w-16 bg-gray-800 rounded-full animate-pulse"></div>
                </div>
                <div className="h-20 w-full bg-gray-800 rounded animate-pulse"></div>
              </div>
            </div>
          </div>
        </div>
      </section>
    )
  }

  if (movies.length === 0) {
    return null
  }

  return (
    <section className="relative w-full" style={{ height: "min(80vh, 700px)" }}>
      <div className="relative h-full w-full">
        {movies.map((movie, index) => (
          <div
            key={movie.id}
            className={`absolute inset-0 transition-opacity duration-1000 ${
              index === currentSlide ? "opacity-100" : "opacity-0 pointer-events-none"
            }`}
          >
            <div className="absolute inset-0">
              <Image
                src={movie.image || "/placeholder.svg"}
                alt={movie.title}
                fill
                className="object-fill object-center"
                priority={index === currentSlide}
              />
              <div className="absolute inset-0 bg-gradient-to-t from-gray-900 via-gray-900/70 to-transparent" />
            </div>

            <div className="absolute inset-0 flex items-end">
              <div className="container mx-auto px-4 pb-12 md:pb-16 lg:pb-24">
                <div className="max-w-xl md:max-w-2xl">
                  <div className="flex items-center space-x-2 mb-2 md:mb-3">
                    <div className="flex items-center space-x-1 bg-purple-600/80 px-2 py-1 rounded text-xs font-medium">
                      <Star className="h-3 w-3 fill-current" />
                      <span>{movie.rating}</span>
                    </div>
                    <div className="text-xs font-medium text-gray-300">{movie.year}</div>
                  </div>

                  <h1 className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl xl:text-6xl font-bold mb-2 md:mb-3">
                    {movie.title}
                  </h1>

                  <div className="flex flex-wrap gap-1 sm:gap-2 mb-3 md:mb-4">
                    {movie.genres.map((genre) => (
                      <span
                        key={genre}
                        className="px-2 sm:px-3 py-0.5 sm:py-1 bg-gray-800/60 backdrop-blur-sm rounded-full text-xs font-medium"
                      >
                        {genre}
                      </span>
                    ))}
                  </div>

                  <p className="text-sm sm:text-base text-gray-300 mb-4 md:mb-6 line-clamp-2 sm:line-clamp-3">
                    {movie.description}
                  </p>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="absolute bottom-4 sm:bottom-auto sm:top-1/2 left-0 right-0 sm:-translate-y-1/2 flex justify-between px-2 sm:px-4 md:px-6 z-10">
        <button
          onClick={prevSlide}
          className="p-1.5 sm:p-2 rounded-full bg-gray-800/60 backdrop-blur-sm hover:bg-gray-700/60 transition-colors"
          aria-label="Previous slide"
        >
          <ChevronLeft className="h-4 w-4 sm:h-5 sm:w-5 md:h-6 md:w-6" />
        </button>
        <button
          onClick={nextSlide}
          className="p-1.5 sm:p-2 rounded-full bg-gray-800/60 backdrop-blur-sm hover:bg-gray-700/60 transition-colors"
          aria-label="Next slide"
        >
          <ChevronRight className="h-4 w-4 sm:h-5 sm:w-5 md:h-6 md:w-6" />
        </button>
      </div>

      <div className="absolute bottom-2 sm:bottom-4 left-1/2 -translate-x-1/2 flex space-x-1.5 sm:space-x-2 z-10">
        {movies.map((_, index) => (
          <button
            key={index}
            onClick={() => setCurrentSlide(index)}
            className={`h-1.5 sm:h-2 rounded-full transition-all ${
              index === currentSlide ? "w-6 sm:w-8 bg-purple-500" : "w-1.5 sm:w-2 bg-gray-400/50 hover:bg-gray-400"
            }`}
            aria-label={`Go to slide ${index + 1}`}
          />
        ))}
      </div>
    </section>
  )
}
