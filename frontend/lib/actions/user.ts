'use server';

import { db } from '@/lib/db';
import { users } from '@/lib/db/schema';
import { eq } from 'drizzle-orm';
import { getServerSession } from 'next-auth/next';
import { authOptions } from '@/lib/auth/options';

export type UserPreferences = {
  genres?: string[];
  timePeriod?: string;
  episodeDuration?: string;
};

export async function getUserPreferences(): Promise<{ 
  success: boolean; 
  data?: UserPreferences; 
  error?: string 
}> {
  try {
    const session = await getServerSession(authOptions);
    
    if (!session || !session.user?.email) {
      return { success: false, error: 'Вы не авторизованы' };
    }

    const user = await db.query.users.findFirst({
      where: (user, { eq }) => eq(user.email, session.user.email)
    });

    if (!user) {
      return { success: false, error: 'Пользователь не найден' };
    }

    // Получаем предпочтения из бэкенда, так как в таблице users нет поля initial
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/users/${user.id}/preferences`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        return {
          success: true,
          data: data.preferences as UserPreferences
        };
      }
    } catch (error) {
      console.error('Error fetching preferences from backend:', error);
    }

    // Возвращаем пустые предпочтения, если не удалось получить с бэкенда
    return {
      success: true,
      data: {} as UserPreferences
    };
  } catch (error) {
    console.error('Error fetching user preferences:', error);
    return { success: false, error: 'Не удалось получить предпочтения пользователя' };
  }
}
