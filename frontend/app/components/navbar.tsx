"use client"

import Link from "next/link"
import { useState } from "react"
import { Menu, X, Film, User, LogOut } from "lucide-react"
import { signOut } from "next-auth/react"

export default function Navbar({ session }: { session: any }) {
  const [isMenuOpen, setIsMenuOpen] = useState(false)

  const handleSignOut = () => {
    signOut();
  }

  return (
    <header className="w-full py-4 px-4 md:px-6">
      <div className="container mx-auto flex items-center justify-between">
        <Link href="/" className="flex items-center space-x-2">
          <Film className="h-8 w-8 text-purple-500" />
          <span className="text-2xl font-bold">Kin4ik</span>
        </Link>

        <div className="hidden md:flex items-center space-x-4">
            {!session ? (
                <Link
                href="/login"
                className="flex items-center space-x-1 px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-500 transition-colors"
          >
            <User className="h-4 w-4" />
            <span>Войти</span>
          </Link>
          ) : (
            <div className="relative group">
              <div className="flex items-center space-x-1 px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-500 transition-colors cursor-pointer">
                <User className="h-4 w-4" />
                <span>Профиль</span>
              </div>
              
              <div className="absolute right-0 mt-1 w-48 rounded-md shadow-lg bg-gray-800 ring-1 ring-black ring-opacity-5 z-10 invisible group-hover:visible opacity-0 group-hover:opacity-100 transition-all duration-150 ease-in-out">
                <div className="py-1">
                  <Link 
                    href="/profile" 
                    className="block px-4 py-2 text-sm text-white hover:bg-purple-600 transition-colors"
                  >
                    Профиль
                  </Link>
                  <button 
                    onClick={handleSignOut}
                    className="flex w-full items-center px-4 py-2 text-sm text-white hover:bg-purple-600 transition-colors cursor-pointer"
                  >
                    <LogOut className="h-4 w-4 mr-2" />
                    Выйти
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        <button
          className="md:hidden p-2 rounded-full hover:bg-gray-700/50 transition-colors"
          onClick={() => setIsMenuOpen(!isMenuOpen)}
        >
          {isMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
        </button>
      </div>

      {/* Mobile menu */}
      {isMenuOpen && (
        <div className="md:hidden absolute top-16 left-0 right-0 z-50 bg-gray-800/95 backdrop-blur-sm p-4 border-t border-gray-700">
          <nav className="flex flex-col space-y-4">
            <div className="pt-2 border-t border-gray-700">
              {!session ? (
                <Link
                  href="/login"
                  className="flex items-center justify-center space-x-1 w-full px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-500 transition-colors"
                >
                  <User className="h-4 w-4" />
                  <span>Войти</span>
                </Link>
              ) : (
                <div className="flex flex-col space-y-2">
                  <Link
                    href="/profile"
                    className="flex items-center justify-center space-x-1 w-full px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-500 transition-colors"
                  >
                    <User className="h-4 w-4" />
                    <span>Профиль</span>
                  </Link>
                  <button 
                    onClick={handleSignOut}
                    className="flex items-center justify-center space-x-1 w-full px-4 py-2 rounded-lg bg-red-600 hover:bg-red-500 transition-colors"
                  >
                    <LogOut className="h-4 w-4" />
                    <span>Выйти</span>
                  </button>
                </div>
              )}
            </div>
          </nav>
        </div>
      )}
    </header>
  )
}
