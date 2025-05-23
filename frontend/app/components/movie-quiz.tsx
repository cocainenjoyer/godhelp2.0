"use client"

import { useState, type Dispatch, type SetStateAction } from "react"
import { X, Brain, ArrowRight, ArrowLeft, Search } from "lucide-react"
import { getRecommendationQuiz } from "@/lib/actions/reccomendations"

const surveyQuestions = [
  {
    id: "country",
    question: "Какая из стран-производителей сериалов занимает особое место в Вашем сердце?",
    options: [
      { value: "AE", label: "Объединённые Арабские Эмираты" },
      { value: "AR", label: "Аргентина" },
      { value: "AT", label: "Австрия" },
      { value: "AU", label: "Австралия" },
      { value: "BE", label: "Бельгия" },
      { value: "BR", label: "Бразилия" },
      { value: "BY", label: "Беларусь" },
      { value: "CA", label: "Канада" },
      { value: "CH", label: "Швейцария" },
      { value: "CL", label: "Чили" },
      { value: "CN", label: "Китай" },
      { value: "CO", label: "Колумбия" },
      { value: "CZ", label: "Чехия" },
      { value: "DE", label: "Германия" },
      { value: "DK", label: "Дания" },
      { value: "EG", label: "Египет" },
      { value: "ES", label: "Испания" },
      { value: "FI", label: "Финляндия" },
      { value: "FR", label: "Франция" },
      { value: "GB", label: "Великобритания" },
      { value: "GT", label: "Гватемала" },
      { value: "HK", label: "Гонконг" },
      { value: "HU", label: "Венгрия" },
      { value: "ID", label: "Индонезия" },
      { value: "IE", label: "Ирландия" },
      { value: "IL", label: "Израиль" },
      { value: "IN", label: "Индия" },
      { value: "IO", label: "Британская Территория в Индийском Океане" },
      { value: "IS", label: "Исландия" },
      { value: "IT", label: "Италия" },
      { value: "JO", label: "Иордания" },
      { value: "JP", label: "Япония" },
      { value: "KE", label: "Кения" },
      { value: "KN", label: "Сент-Китс и Невис" },
      { value: "KR", label: "Республика Корея" },
      { value: "KW", label: "Кувейт" },
      { value: "LB", label: "Ливан" },
      { value: "LU", label: "Люксембург" },
      { value: "MA", label: "Марокко" },
      { value: "MX", label: "Мексика" },
      { value: "MY", label: "Малайзия" },
      { value: "NC", label: "Новая Каледония" },
      { value: "NG", label: "Нигерия" },
      { value: "NL", label: "Нидерланды" },
      { value: "NO", label: "Норвегия" },
      { value: "NZ", label: "Новая Зеландия" },
      { value: "PE", label: "Перу" },
      { value: "PH", label: "Филиппины" },
      { value: "PL", label: "Польша" },
      { value: "PR", label: "Пуэрто-Рико" },
      { value: "PT", label: "Португалия" },
      { value: "RO", label: "Румыния" },
      { value: "RU", label: "Россия" },
      { value: "SA", label: "Саудовская Аравия" },
      { value: "SE", label: "Швеция" },
      { value: "SG", label: "Сингапур" },
      { value: "SN", label: "Сенегал" },
      { value: "SY", label: "Сирия" },
      { value: "TH", label: "Таиланд" },
      { value: "TN", label: "Тунис" },
      { value: "TR", label: "Турция" },
      { value: "TW", label: "Китайская Республика" },
      { value: "UA", label: "Украина" },
      { value: "US", label: "Соединённые Штаты Америки" },
      { value: "UY", label: "Уругвай" },
      { value: "VN", label: "Вьетнам" },
      { value: "ZA", label: "Южно-Африканская Республика" },
      { value: "ZM", label: "Замбия" },
    ],
  },
  {
    id: "mood",
    question: "Сериал с каким настроением сейчас Вам по душе?",
    options: [
      { value: "happy", label: "😊 веселое" },
      { value: "sad", label: "😢 грустное" },
      { value: "melancholic", label: "😔 меланхоличное" },
      { value: "positive", label: "🌞 позитивное" },
      { value: "touching", label: "💖 трогательное" },
      { value: "joyful", label: "🎉 радостное" },
      { value: "relaxing", label: "😌 расслабляющее" },
      { value: "exciting", label: "🤩 захватывающее" },
    ],
  },
  {
    id: "company",
    question: "С какой компанией Вы хотите посмотреть сейчас сериал?",
    options: [
      { value: "alone", label: "👤 один" },
      { value: "friends", label: "👥 с друзьями" },
      { value: "family", label: "👨‍👩‍👧‍👦 с семьёй" },
      { value: "children", label: "👶 с детьми" },
    ],
  },
]

interface MovieQuizProps {
  onOpenSurvey: () => void
  surveyResults: any[] | null
  setSurveyResults: Dispatch<SetStateAction<any[] | null>>
  showSurveyModal: boolean
  setShowSurveyModal: Dispatch<SetStateAction<boolean>>
}

export default function MovieQuiz({
  onOpenSurvey,
  surveyResults,
  setSurveyResults,
  showSurveyModal,
  setShowSurveyModal,
}: MovieQuizProps) {
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [answers, setAnswers] = useState<Record<string, string>>({})
  const [showResults, setShowResults] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")

  const handleAnswer = (questionId: string, answer: string) => {
    setAnswers((prev) => ({
      ...prev,
      [questionId]: answer,
    }))
  }

  const nextQuestion = async () => {
    if (currentQuestion < surveyQuestions.length - 1) {
      setCurrentQuestion((prev) => prev + 1)
    } else {
      try {
        const quizData = Object.entries(answers).map(([id, option]) => ({
          id,
          option
        }));

        const result = await getRecommendationQuiz(quizData);
        
        if (result.success && result.recommendation) {
          setSurveyResults([result.recommendation]);
        }
        setShowSurveyModal(false);
        setShowResults(true);
        setCurrentQuestion(0);
      } catch (error) {
        console.error("Error getting recommendation:", error);
        setShowSurveyModal(false);
        setShowResults(true);
        setCurrentQuestion(0);
      }
    }
  }

  const prevQuestion = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion((prev) => prev - 1)
    }
  }

  const resetSurvey = () => {
    setSurveyResults(null)
    setAnswers({})
    setShowResults(false)
    setCurrentQuestion(0)
  }

  return (
    <>
      <div className="rounded-xl bg-gray-800/50 p-8 shadow-2xl backdrop-blur-sm">
        <div className="flex items-center space-x-3 mb-6">
          <Brain className="h-8 w-8 text-purple-500" />
          <h2 className="text-2xl font-bold">Подбор сериала по вкусу</h2>
        </div>

        <div>
          <p className="text-gray-400 mb-6">
            Не знаете, что посмотреть? Ответьте на несколько вопросов, и мы порекомендуем сериалы, подходящие вашему настроению и предпочтениям.
          </p>

          <button
            onClick={onOpenSurvey}
            className="group flex w-full items-center justify-center rounded-lg bg-purple-600 px-4 py-3 font-medium text-white transition-colors hover:bg-purple-500"
          >
            Пройти опрос
            <ArrowRight className="ml-2 h-4 w-4 opacity-70 transition-transform group-hover:translate-x-1" />
          </button>
        </div>
      </div>

      {showSurveyModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70">
          <div className="w-full max-w-xl rounded-xl bg-gray-800 p-6 shadow-2xl overflow-y-auto max-h-[90vh]">
            <button
              onClick={() => {setShowSurveyModal(false)
                resetSurvey()
              }}
              className="absolute right-4 top-4 p-1 rounded-full hover:bg-gray-700 transition-colors"
            >
              <X className="h-5 w-5" />
            </button>

            <div className="mb-6">
              <h3 className="text-xl font-bold mb-1">Подбор сериала</h3>
              <p className="text-gray-400 text-sm">
                Вопрос {currentQuestion + 1} из {surveyQuestions.length}
              </p>
            </div>

            <div className="mb-6">
              <h4 className="text-lg font-medium mb-4">{surveyQuestions[currentQuestion].question}</h4>

              {surveyQuestions[currentQuestion].id === "country" ? (
                <div className="space-y-3">
                  <div className="relative mb-4">
                    <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
                      <Search className="h-4 w-4 text-gray-400" />
                    </div>
                    <input
                      type="text"
                      className="bg-gray-700 text-white placeholder-gray-400 border-none rounded-lg block w-full pl-10 p-2.5 focus:ring-2 focus:ring-purple-500 focus:outline-none"
                      placeholder="Поиск страны..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                    />
                  </div>
                  
                  <div className="max-h-64 overflow-y-auto pr-1 space-y-2">
                    {surveyQuestions[currentQuestion].options
                      .filter(option => 
                        searchQuery === "" || 
                        option.label.toLowerCase().includes(searchQuery.toLowerCase())
                      )
                      .map((option) => (
                        <button
                          key={option.value}
                          onClick={() => handleAnswer(surveyQuestions[currentQuestion].id, option.value)}
                          className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${
                            answers[surveyQuestions[currentQuestion].id] === option.value
                              ? "bg-purple-600 text-white"
                              : "bg-gray-700 hover:bg-gray-600 text-gray-200"
                          }`}
                        >
                          {option.label}
                        </button>
                      ))}
                  </div>
                </div>
              ) : (
                <div className="space-y-3">
                  {surveyQuestions[currentQuestion].options.map((option) => (
                    <button
                      key={option.value}
                      onClick={() => handleAnswer(surveyQuestions[currentQuestion].id, option.value)}
                      className={`w-full text-left px-4 py-3 rounded-lg transition-colors ${
                        answers[surveyQuestions[currentQuestion].id] === option.value
                          ? "bg-purple-600 text-white"
                          : "bg-gray-700 hover:bg-gray-600 text-gray-200"
                      }`}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div className="flex justify-between">
              {currentQuestion > 0 && (
                <button
                  onClick={prevQuestion}
                  className="flex items-center space-x-2 px-6 py-2 rounded-lg bg-gray-700 hover:bg-gray-600 transition-colors"
                >
                  <ArrowLeft className="h-4 w-4 mr-2" />
                  <span>Назад</span>
                </button>
              )}
              <div className={currentQuestion === 0 ? "w-full flex justify-end" : ""}>
                <button
                  onClick={nextQuestion}
                  disabled={!answers[surveyQuestions[currentQuestion].id]}
                  className="flex items-center space-x-2 px-6 py-2 rounded-lg bg-purple-600 hover:bg-purple-500 disabled:bg-gray-700 disabled:text-gray-500 disabled:cursor-not-allowed transition-colors"
                >
                  <span>{currentQuestion === surveyQuestions.length - 1 ? "Готово" : "Дальше"}</span>
                  <ArrowRight className="h-4 w-4" />
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {surveyResults && showResults && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70">
          <div className="w-full max-w-xl rounded-xl bg-gray-800 p-6 shadow-2xl overflow-y-auto max-h-[90vh]">
            <button
              onClick={() => setShowResults(false)}
              className="absolute right-4 top-4 p-1 rounded-full hover:bg-gray-700 transition-colors"
            >
              <X className="h-5 w-5" />
            </button>

            {surveyResults.length > 0 ? (
              <>
                <div className="mb-4">
                  <h3 className="text-xl font-semibold mb-2">Идеальный выбор для вас</h3>
                  <p className="text-gray-400">Основываясь на ваших предпочтениях, мы рекомендуем:</p>
                </div>
                
                <div className="flex flex-col">
                  <h2 className="text-2xl font-bold mb-2">{surveyResults[0].title}</h2>
                  
                  <div className="relative w-full mb-4 rounded-lg overflow-hidden">
                    <img 
                      src={surveyResults[0].image} 
                      alt={surveyResults[0].title}
                      className="w-full h-auto object-cover"
                    />
                    <div className="absolute top-2 right-2 bg-black/70 text-yellow-400 font-bold px-2 py-1 rounded text-sm">
                      ★ {surveyResults[0].rating}
                    </div>
                  </div>
                  
                  <div className="flex flex-wrap gap-2 mb-4">
                    {surveyResults[0].genres.map((genre: string, index: number) => (
                      <span key={index} className="bg-purple-600/30 text-purple-200 px-3 py-1 rounded-full text-sm">
                        {genre}
                      </span>
                    ))}
                  </div>
                  
                  <div className="flex items-center text-gray-400 mb-6">
                    <span className="mr-4">Год: {surveyResults[0].year}</span>
                    <span>ID: {surveyResults[0].id}</span>
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
              onClick={resetSurvey}
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