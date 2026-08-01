"use client";

import { useState, type FormEvent } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { authClient } from "@/lib/auth-client";
import {
  ArrowRight,
  CheckCircle2,
  Eye,
  EyeOff,
  LoaderCircle,
  Lock,
  Mail,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

export default function RegisterPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [passwordTouched, setPasswordTouched] = useState(false);

  const passwordChecks = [
    {
      label: "At least 6 characters",
      met: password.length >= 6,
    },
    {
      label: "At least one uppercase letter",
      met: /[A-Z]/.test(password),
    },
    {
      label: "At least one number",
      met: /\d/.test(password),
    },
  ];

  const isPasswordValid = passwordChecks.every((check) => check.met);
  const showPasswordError = passwordTouched && !isPasswordValid;

  const handleRegister = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setPasswordTouched(true);

    if (!isPasswordValid) {
      return;
    }

    setLoading(true);
    try {
      const { data, error } = await authClient.signUp.email({
        email,
        password,
        name: email.split("@")[0],
      });

      if (error) {
        alert(error.message || "Registration failed");
        return;
      }

      router.push("/dashboard");
    } catch (err) {
      console.error(err);
      alert("An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-canvas text-ink">
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col lg:flex-row">
        <section className="relative flex items-center justify-center overflow-hidden bg-[radial-gradient(circle_at_top_left,_rgba(193,95,60,0.16),_transparent_35%),linear-gradient(135deg,_rgba(26,26,24,1),_rgba(33,33,32,1))] px-6 py-12 sm:px-8 lg:w-1/2 lg:px-12 lg:py-16">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_80%_20%,_rgba(217,119,87,0.12),_transparent_30%)]" />
          <div className="auth-panel-graphic" aria-hidden="true">
            <div className="auth-aurora auth-aurora-a" />
            <div className="auth-aurora auth-aurora-b" />
            <div className="auth-aurora auth-aurora-c" />
          </div>

          <div className="auth-panel-content relative z-10 max-w-xl">
            <div className="mb-6 flex items-center gap-3 text-accent-ink">
              <div className="rounded-full border border-accent/30 bg-accent-soft p-2">
                <ShieldCheck className="h-5 w-5" />
              </div>
              <span className="text-sm font-semibold uppercase tracking-[0.3em] text-ink">
                Secure legal AI
              </span>
            </div>

            <h1 className="text-4xl font-semibold tracking-tight text-ink sm:text-5xl">
              Legal AI Platform
            </h1>
            <p className="mt-4 max-w-lg text-lg leading-8 text-ink-muted">
              Structured legal analysis. Case references. Conflict detection. Confidence scoring.
            </p>

            <div className="mt-10 rounded-2xl border border-line bg-hover p-5 backdrop-blur-sm">
              <div className="flex items-start gap-3">
                <div className="rounded-xl bg-accent-soft p-2 text-accent-ink">
                  <Sparkles className="h-5 w-5" />
                </div>
                <div>
                  <p className="text-sm font-medium text-ink">
                    Start your legal AI workspace
                  </p>
                  <p className="mt-1 text-sm leading-6 text-ink-muted">
                    Create an account to begin structured case review and AI-assisted analysis.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="flex items-center justify-center px-4 py-10 sm:px-6 lg:w-1/2 lg:px-8 lg:py-16">
          <div className="w-full max-w-md rounded-3xl border border-line bg-raised/90 p-8 shadow-2xl shadow-black/40 backdrop-blur transition-all duration-300">
            <div className="mb-8">
              <p className="text-sm font-medium uppercase tracking-[0.3em] text-accent">
                Create account
              </p>
              <h2 className="mt-2 text-3xl font-semibold text-ink">
                Get started today
              </h2>
              <p className="mt-2 text-sm leading-6 text-ink-muted">
                Start using the AI Legal Research Assistant with a secure account.
              </p>
            </div>

            <form className="space-y-5" onSubmit={handleRegister}>
              <div>
                <label htmlFor="email" className="mb-2 block text-sm font-medium text-ink-muted">
                  Email
                </label>
                <div className="flex items-center gap-3 rounded-2xl border border-line bg-hover/80 px-4 py-3 transition-all duration-200 focus-within:border-accent focus-within:ring-2 focus-within:ring-accent/20">
                  <Mail className="h-4 w-4 text-ink-faint" />
                  <input
                    id="email"
                    value={email}
                    className="w-full bg-transparent text-ink outline-none placeholder:text-ink-faint"
                    placeholder="name@example.com"
                    onChange={(e) => setEmail(e.target.value)}
                  />
                </div>
              </div>

              <div>
                <label htmlFor="password" className="mb-2 block text-sm font-medium text-ink-muted">
                  Password
                </label>
                <div className="flex items-center gap-3 rounded-2xl border border-line bg-hover/80 px-4 py-3 transition-all duration-200 focus-within:border-accent focus-within:ring-2 focus-within:ring-accent/20">
                  <Lock className="h-4 w-4 text-ink-faint" />
                  <input
                    id="password"
                    value={password}
                    className="w-full bg-transparent text-ink outline-none placeholder:text-ink-faint"
                    placeholder="Create a password"
                    type={showPassword ? "text" : "password"}
                    onChange={(e) => {
                      setPassword(e.target.value);
                      if (!passwordTouched) {
                        setPasswordTouched(true);
                      }
                    }}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword((value) => !value)}
                    className="rounded-full p-1 text-ink-muted transition hover:text-ink"
                    aria-label={showPassword ? "Hide password" : "Show password"}
                  >
                    {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
              </div>

              {showPasswordError && (
                <p className="text-sm leading-6 text-rose-400">
                  Password must be at least 6 characters, include one uppercase letter and one number.
                </p>
              )}

              <div className="rounded-2xl border border-line bg-surface/60 p-3">
                <div className="space-y-2 text-sm">
                  {passwordChecks.map((check) => (
                    <div key={check.label} className="flex items-center gap-2">
                      <CheckCircle2 className={`h-4 w-4 ${check.met ? "text-accent" : "text-ink-faint"}`} />
                      <span className={check.met ? "text-ink" : "text-ink-faint"}>{check.label}</span>
                    </div>
                  ))}
                </div>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="flex w-full items-center justify-center gap-2 rounded-2xl bg-accent px-4 py-3 font-medium text-ink transition-all duration-200 hover:-translate-y-0.5 hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-70"
              >
                {loading ? (
                  <>
                    <LoaderCircle className="h-4 w-4 animate-spin" />
                    Creating account...
                  </>
                ) : (
                  <>
                    Register
                    <ArrowRight className="h-4 w-4" />
                  </>
                )}
              </button>
            </form>

            <p className="mt-6 text-center text-sm text-ink-muted">
              Already have an account? {" "}
              <Link href="/login" className="font-medium text-accent transition hover:text-accent-ink">
                Sign in
              </Link>
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}