import CredentialsProvider from "next-auth/providers/credentials";
import { loginUser } from "@/lib/actions/auth";
import type { JWT } from "next-auth/jwt";
import type { Session, AuthOptions, User } from "next-auth";

export const authOptions: AuthOptions = {
  providers: [
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        loginIdentifier: { label: "Имя пользователя или Email", type: "text" },
        password: { label: "Пароль", type: "password" }
      },
      
      async authorize(credentials) {
        if (!credentials?.loginIdentifier || !credentials?.password) {
          return null;
        }

        try {
          const isEmail = credentials.loginIdentifier.includes('@');
          
          const result = await loginUser({
            email: isEmail ? credentials.loginIdentifier : undefined,
            username: !isEmail ? credentials.loginIdentifier : undefined,
            password: credentials.password
          });

          if (result.success && result.data) {
            return {
              id: result.data.id.toString(),
              email: result.data.email,
              name: result.data.username,
            };
          }
          return null;
        } catch (error) {
          console.error("Auth error:", error);
          return null;
        }
      }
    })
  ],
  pages: {
    signIn: "/login",
    error: "/login"
  },
  callbacks: {
    async jwt({ token, user }: { token: JWT; user?: User }) {
      if (user) {
        token.id = user.id;
        token.email = user.email;
        token.name = user.name;
      }
      return token;
    },
    async session({ session, token }: { session: Session; token: JWT }) {
      if (token) {
        session.user.id = token.id;
        session.user.email = token.email;
        session.user.name = token.name;
      }
      return session;
    }
  },
  session: {
    strategy: "jwt" as const,
    maxAge: 30 * 24 * 60 * 60, 
  },
  secret: process.env.NEXTAUTH_SECRET || "your-secret-key-change-in-production",
}; 