import Link from "next/link"
import { Film} from "lucide-react"

export default function Footer() {
  return (
    <footer className="w-full py-8 px-4 md:px-6 border-t border-gray-800">
      <div className="container mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="space-y-4">
            <Link href="/" className="flex items-center space-x-2">
              <Film className="h-6 w-6 text-purple-500" />
              <span className="text-xl font-bold">Kin4ik</span>
            </Link>
            <p className="text-gray-400 text-sm">
              Ваш личный помощник в подборе сериалов.
            </p>
            
          </div>

          <div>
            <h3 className="font-bold mb-4">Меню</h3>
            <ul className="space-y-2">
              <li>
                <Link href="/" className="text-gray-400 hover:text-white transition-colors">
                  Главная
                </Link>
              </li>
              <li>
                <Link href="/profile" className="text-gray-400 hover:text-white transition-colors">
                  Профиль
                </Link>
              </li>
              
            </ul>
          </div>

          <div>
            <h3 className="font-bold mb-4">Компания</h3>
            <ul className="space-y-2">
              <li>
                <Link href="/about" className="text-gray-400 hover:text-white transition-colors">
                  О нас
                </Link>
              </li>
            </ul>
          </div>

          
        </div>

        <div className="mt-8 pt-8 border-t border-gray-800 text-center text-sm text-gray-500">
          <p>© {new Date().getFullYear()} Kin4ik. Все права защищены.</p>
        </div>
      </div>
    </footer>
  )
}
