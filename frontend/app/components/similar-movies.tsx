"use client"

import { useEffect, useState } from "react"
import { Search, X } from "lucide-react"
import { getAllMoviesMap, getSimilarMovie } from "@/lib/actions/reccomendations"

const similarMoviesResults = [
  {
    id: 301,
    title: "Казино",
    image: "https://i.pinimg.com/originals/61/81/52/618152b971ff5b62749da0fb08d8de37.jpg",
    year: 1995,
    rating: 8.2,
    genres: ["Криминал", "Драма"],
  },
]

export default function SimilarMovies() {
  const [searchTerm, setSearchTerm] = useState("")
  const [selectedMovie, setSelectedMovie] = useState<any | null>(null)
  const [showResults, setShowResults] = useState(false)
  const [similarMovies, setSimilarMovies] = useState<any[]>([])
  const [showDropdown, setShowDropdown] = useState(false)
  const [allMovies, setAllMovies] = useState<any[]>([])

  useEffect(() => {
    const fetchAllMovies = async () => {
      const result = await getAllMoviesMap();
      if (result.success && result.map) {
        setAllMovies(result.map);
      }
    }
    fetchAllMovies();
  }, [])

  const filteredMovies = allMovies.filter((movie) => movie.title.toLowerCase().includes(searchTerm.toLowerCase()))

  const handleSelectMovie = (movie: any) => {
    setSelectedMovie(movie)
    setSearchTerm("")
    setShowDropdown(false)
  }

  const handleFindSimilar = async () => {
    if (!selectedMovie) return;
    
    const result = await getSimilarMovie(selectedMovie.id);
    if (result.success && result.recommendations) {
      setSimilarMovies([result.recommendations]);
      setShowResults(true);
    } else {
      setSimilarMovies([]);
      setShowResults(true);
    }
  }

  const handleClearSelection = () => {
    setSelectedMovie(null)
    setShowResults(false)
    setSimilarMovies([])
  }

  return (
    <>
    <div className="transform rounded-xl bg-gray-800/50 p-8 shadow-2xl backdrop-blur-sm transition-all h-full">
      <div className="flex items-center space-x-3 mb-6">
        <Search className="h-8 w-8 text-purple-500" />
        <h2 className="text-2xl font-bold">Найти похожие сериалы</h2>
      </div>

      <p className="text-gray-400 mb-6">Введите сериал, который вам понравился, и мы подберём похожие.</p>

      <div className="relative mb-6">
        <div className="flex items-center space-x-2">
          {selectedMovie && (
            <div className="flex items-center space-x-2 bg-gray-700 px-3 py-2 rounded-lg">
              <span>{selectedMovie.title}</span>
              <button onClick={handleClearSelection} className="p-1 rounded-full hover:bg-gray-600 transition-colors">
                <X className="h-4 w-4" />
              </button>
            </div>
          )}

          {!selectedMovie && (
            <div className="relative flex-1">
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => {
                  setSearchTerm(e.target.value)
                  setShowDropdown(e.target.value.length > 0)
                }}
                onFocus={() => setShowDropdown(searchTerm.length > 0)}
                placeholder="Поиск сериала..."
                className="w-full rounded-lg bg-gray-700/80 px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
              />

              {showDropdown && (
                <div className="absolute z-10 mt-1 w-full rounded-lg bg-gray-700 shadow-lg max-h-60 overflow-auto">
                  {filteredMovies.length > 0 ? (
                    filteredMovies.map((movie) => (
                      <button
                        key={movie.id}
                        onClick={() => handleSelectMovie(movie)}
                        className="w-full text-left px-4 py-2 hover:bg-gray-600 transition-colors"
                      >
                        {movie.title}
                      </button>
                    ))
                  ) : (
                    <div className="px-4 py-2 text-gray-400">Сериалы не найдены</div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <button
        onClick={handleFindSimilar}
        disabled={!selectedMovie}
        className="group flex w-full items-center justify-center rounded-lg bg-purple-600 px-4 py-3 font-medium text-white transition-colors hover:bg-purple-500 disabled:bg-gray-700 disabled:text-gray-500 disabled:cursor-not-allowed"
      >
        Найти похожие сериалы
      </button>
</div>
      {similarMovies && showResults && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70">
          <div className="w-full max-w-xl rounded-xl bg-gray-800 p-6 shadow-2xl overflow-y-auto max-h-[90vh]">
            <button
              onClick={() => setShowResults(false)}
              className="absolute right-4 top-4 p-1 rounded-full hover:bg-gray-700 transition-colors"
            >
              <X className="h-5 w-5" />
            </button>

            {similarMovies.length > 0 ? (
              <>
                <div className="mb-4">
                  <h3 className="text-xl font-semibold mb-2">Идеальный выбор для вас</h3>
                  <p className="text-gray-400">Основываясь на ваших предпочтениях, мы рекомендуем:</p>
                </div>
                
                <div className="flex flex-col">
                  <h2 className="text-2xl font-bold mb-2">{similarMovies[0].title}</h2>
                  
                  <div className="relative w-full mb-4 rounded-lg overflow-hidden">
                    <img 
                      src={similarMovies[0].image} 
                      alt={similarMovies[0].title}
                      className="w-full h-auto object-cover"
                    />
                    <div className="absolute top-2 right-2 bg-black/70 text-yellow-400 font-bold px-2 py-1 rounded text-sm">
                      ★ {similarMovies[0].rating}
                    </div>
                  </div>
                  
                  <div className="flex flex-wrap gap-2 mb-4">
                    {similarMovies[0].genres.map((genre: string, index: number) => (
                      <span key={index} className="bg-purple-600/30 text-purple-200 px-3 py-1 rounded-full text-sm">
                        {genre}
                      </span>
                    ))}
                  </div>
                  
                  <div className="flex items-center text-gray-400 mb-6">
                    <span className="mr-4">Год: {similarMovies[0].year}</span>
                    <span>ID: {similarMovies[0].id}</span>
                  </div>
                  
                  <p className="text-gray-300 mb-6">
                    Этот захватывающий сериал отлично соответствует вашим предпочтениям. 
                    Насладитесь просмотром качественного кино, которое мы специально подобрали для вас.
                  </p>
                </div>
              </>
            ) : (
              <p className="text-gray-400">К сожалению, мы не смогли найти подходящих рекомендаций. Пожалуйста, попробуйте другие варианты.</p>
            )}

            <button
              onClick={handleClearSelection}
              className="mt-2 w-full flex items-center justify-center rounded-lg border border-purple-500 px-4 py-3 font-medium text-purple-400 transition-colors hover:bg-purple-500/10"
            >
              Закрыть
            </button>
          </div>
        </div>
      )}
    </>
  )
}
