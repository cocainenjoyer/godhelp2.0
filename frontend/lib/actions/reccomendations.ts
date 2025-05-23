"use server";

import { getServerSession } from "next-auth/next";
import { authOptions } from "@/lib/auth/options";
import { Movie } from "@/lib/types/movie";
const mockMap = [
    { "id": 1, "title": "Breaking Bad" },  { "id": 2, "title": "The Office" }
]
const mockRecommendations: Movie[] = [
  {
    id: 1,
    title: "The Shawshank Redemption",
    description:
      "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
    rating: 9.3,
    image:
      "https://m.media-amazon.com/images/S/aplus-media-library-service-media/384b4101-a7f0-4792-a9e4-6b5cb0f22ca2.__CR0,0,970,600_PT0_SX970_V1___.jpg",
    year: 1994,
    genres: ["Drama"],
  },
  {
    id: 2,
    title: "The Godfather",
    description:
      "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
    rating: 9.2,
    image:
      "https://static.life.ru/posts/2016/09/900620/411cee02ed83833671a9636f4309144b.jpg",
    year: 1972,
    genres: ["Crime", "Drama"],
  },
];

export async function getMedianReccomendations(): Promise<{
  success: boolean;
  error?: string;
  recommendations?: Movie[];
}> {
  const session = await getServerSession(authOptions);

  if (!session || !session.user?.email) {
    return { success: false, error: "Вы не авторизованы" };
  }

  try {
    const data = await fetch(
      process.env.NEXT_PUBLIC_API_URL + "/recommendations/profile?id=" + session.user.id
    );
    if (!data.ok) {
      console.log("API request failed, returning mock data");
      return { success: true, recommendations: mockRecommendations };
    }

    const recommendations = await data.json();
    return { success: true, recommendations };
  } catch (error) {
    console.error("Error fetching recommendations:", error);
    return { success: true, recommendations: mockRecommendations };
  }
}

interface QuizData {
  id: string;
  option: string;
}

export async function getRecommendationQuiz(
  quizData: QuizData[]
): Promise<{ success: boolean; error?: string; recommendation?: Movie }> {
  const session = await getServerSession(authOptions);

  if (!session || !session.user?.email) {
    return { success: false, error: "Вы не авторизованы" };
  }
  try {
    const data = await fetch(process.env.NEXT_PUBLIC_API_URL + "/recommendations/by-quiz", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        data: quizData,
      }),
    });
    if (!data.ok) {
      console.log("API request failed, returning mock data");
      return { success: true, recommendation: mockRecommendations[0] };
    }
    const recommendations = await data.json();
    // Проверяем, является ли ответ массивом, и если да, берем первый элемент
    if (Array.isArray(recommendations) && recommendations.length > 0) {
      return { success: true, recommendation: recommendations[0] };
    }
    return { success: true, recommendation: recommendations };
  } catch (error) {
    console.error("Error fetching recommendations:", error);
    return { success: true, recommendation: mockRecommendations[0] };
  }
}

export async function getAllMoviesMap(){
    const session = await getServerSession(authOptions);

  if (!session || !session.user?.email) {
    return { success: false, error: "Вы не авторизованы" };
  }

  try {
    const data = await fetch(process.env.NEXT_PUBLIC_API_URL + "/titles/map");
    if (!data.ok) {
      console.log("API request failed, returning mock data");
      return { success: true, map: mockMap };
    }
    const map = await data.json();
    return { success: true, map };
  } catch (error) {
    console.error("Error fetching recommendations:", error);
    return { success: true, map: mockMap };
  }  
}

export async function getSimilarMovie(movieId: number): Promise<{ success: boolean; error?: string; recommendations?: Movie }>{
    const session = await getServerSession(authOptions);

  if (!session || !session.user?.email) {
    return { success: false, error: "Вы не авторизованы" };
  }

  try {
    const data = await fetch(process.env.NEXT_PUBLIC_API_URL + "/recommendations/similar-by-id?id=" + movieId);
    if (!data.ok) {
      console.log("API request failed, returning mock data");
      return { success: true, recommendations: mockRecommendations[0] };
    }
    const recommendations = await data.json();
    // Проверяем, является ли ответ массивом, и если да, берем первый элемент
    if (Array.isArray(recommendations) && recommendations.length > 0) {
      return { success: true, recommendations: recommendations[0] };
    }
    return { success: true, recommendations };
  } catch (error) {
    console.error("Error fetching recommendations:", error);
    return { success: true, recommendations: mockRecommendations[0] };
  }
}
