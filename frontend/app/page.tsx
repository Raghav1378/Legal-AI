"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ArrowRight,
  BrainCircuit,
  CheckCircle2,
  ChevronDown,
  FileSearch,
  MessageSquareText,
  Scale,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

function Reveal({ children }: { children: React.ReactNode }) {
  const [visible, setVisible] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { threshold: 0.15 }
    );

    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      className={`transition-all duration-700 ${visible ? "translate-y-0 opacity-100" : "translate-y-6 opacity-0"}`}
    >
      {children}
    </div>
  );
}

export default function LandingPage() {
  const router = useRouter();
  const [mounted, setMounted] = useState(false);
  const [openFaq, setOpenFaq] = useState<number | null>(0);

  useEffect(() => {
    setMounted(true);
  }, []);

  const featureCards = [
    {
      icon: FileSearch,
      title: "Structured legal analysis",
      description: "Turn broad legal questions into clear issue summaries and action-oriented insights.",
      className: "md:col-span-2",
    },
    {
      icon: Scale,
      title: "Case references",
      description: "Surface precedent-driven guidance with supporting authorities and citations.",
      className: "md:col-span-1",
    },
    {
      icon: ShieldCheck,
      title: "Conflict detection",
      description: "Flag competing authorities and risky areas before they become problems.",
      className: "md:col-span-1",
    },
    {
      icon: BrainCircuit,
      title: "Confidence scoring",
      description: "Understand how strongly the platform supports each finding and recommendation.",
      className: "md:col-span-2",
    },
  ];

  const teamCards = [
    {
      icon: FileSearch,
      title: "Less time hunting for authorities",
      description: "It helps surface the most relevant legal material and keep the review path readable instead of scattered across tabs and notes.",
    },
    {
      icon: ShieldCheck,
      title: "Safer review of competing positions",
      description: "Conflict signals are surfaced early so teams can assess whether the answer is consistent, incomplete, or likely to need extra scrutiny.",
    },
    {
      icon: BrainCircuit,
      title: "More confidence in the conclusion",
      description: "Confidence cues and structured reasoning make it easier to understand where the platform is strong and where human judgment still matters.",
    },
  ];

  const roadmapStages = [
    {
      tag: "Now",
      title: "Structured legal analysis & conflict detection",
      description: "The current experience focuses on clear summaries, relevant authorities, and early conflict signals.",
    },
    {
      tag: "Next",
      title: "Multi-jurisdiction case law support",
      description: "Broader coverage for comparing rules and precedents across jurisdictions in a more consistent way.",
    },
    {
      tag: "Later",
      title: "Team workspaces & citation export",
      description: "Shared review spaces and cleaner export flows for internal drafting and case preparation.",
    },
    {
      tag: "Future",
      title: "Integrations with legal databases",
      description: "A longer-term path toward connecting the workspace with external legal research sources.",
    },
  ];

  const faqs = [
    {
      question: "Is this a replacement for a lawyer?",
      answer: "No. This product is designed to support legal research and drafting workflows, not replace professional judgment. It can help organize information and highlight areas that deserve closer review, but final legal decisions still need human expertise.",
    },
    {
      question: "What jurisdictions are supported?",
      answer: "Support is currently focused on the workflow and research structure rather than a full jurisdiction-specific database. As the product grows, broader jurisdiction coverage and better case-law handling will be added.",
    },
    {
      question: "How is confidence scoring calculated?",
      answer: "Confidence is a qualitative signal based on how strongly the available material supports a finding, the presence of conflict signals, and the clarity of the evidence presented. It should be treated as a guide, not a guarantee.",
    },
    {
      question: "Is my data kept private?",
      answer: "The current product is built with privacy in mind, but this is still an early-stage project. Sensitive data should be handled carefully, and any deployment should be reviewed against the privacy and retention needs of the user or organization.",
    },
    {
      question: "Who is this built for?",
      answer: "It is best suited to legal professionals, researchers, and small teams who want a more structured way to explore case law, compare arguments, and keep track of uncertainty while reviewing material.",
    },
  ];

  return (
    <div className="min-h-screen bg-canvas text-ink">
      <main className="relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,_rgba(193,95,60,0.14),_transparent_28%),radial-gradient(circle_at_80%_20%,_rgba(217,119,87,0.10),_transparent_30%)]" />
        <div className="absolute inset-0 opacity-30 [background-image:linear-gradient(rgba(255,255,255,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.05)_1px,transparent_1px)] [background-size:64px_64px]" />

        <section className="relative mx-auto flex min-h-screen max-w-7xl flex-col justify-center px-6 py-20 sm:px-8 lg:px-12">
          <div className={`max-w-4xl transition-all duration-700 ${mounted ? "translate-y-0 opacity-100" : "translate-y-6 opacity-0"}`}>
            <div className="inline-flex items-center gap-2 rounded-full border border-accent/20 bg-accent/10 px-3 py-1 text-sm font-medium text-accent-ink shadow-sm shadow-accent/10">
              <Sparkles className="h-4 w-4" />
              AI-Powered · For Legal Professionals
            </div>

            <h1 className="mt-8 text-4xl font-semibold tracking-tight text-ink sm:text-6xl lg:text-7xl">
              AI Powered Legal Intelligence
            </h1>

            <p className="mt-6 max-w-2xl text-lg leading-8 text-ink-muted sm:text-xl">
              Structured legal analysis. Case references. Conflict detection. Confidence scoring.
            </p>

            <div className="mt-10 flex flex-col items-start gap-4 sm:flex-row">
              <button
                onClick={() => router.push("/login")}
                className="flex items-center gap-2 rounded-full bg-accent px-6 py-3 font-medium text-ink transition duration-200 hover:-translate-y-0.5 hover:bg-accent-hover hover:shadow-lg hover:shadow-accent/20"
              >
                Login
                <ArrowRight className="h-4 w-4" />
              </button>

              <button
                onClick={() => router.push("/register")}
                className="rounded-full border border-line-strong bg-hover px-6 py-3 font-medium text-ink transition duration-200 hover:-translate-y-0.5 hover:border-accent/40 hover:bg-hover"
              >
                Register
              </button>
            </div>
          </div>
        </section>

        <section id="features" className="relative mx-auto max-w-7xl px-6 pb-20 sm:px-8 lg:px-12">
          <Reveal>
            <div className="rounded-[32px] border border-line bg-raised/70 p-6 shadow-2xl shadow-black/20 backdrop-blur sm:p-8 lg:p-10">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
                <div>
                  <p className="text-sm font-semibold uppercase tracking-[0.3em] text-accent">
                    How it works
                  </p>
                  <h2 className="mt-2 text-2xl font-semibold text-ink sm:text-3xl">
                    Move from messy facts to structured legal insight.
                  </h2>
                </div>
                <p className="max-w-xl text-sm leading-7 text-ink-muted sm:text-base">
                  The platform brings together legal research, precedent handling, and confidence-aware summaries in one experience.
                </p>
              </div>

              <div className="mt-8 grid gap-4 md:grid-cols-3">
                {featureCards.map((card, index) => {
                  const Icon = card.icon;
                  return (
                    <Reveal key={card.title}>
                      <div className={`rounded-[24px] border border-line bg-hover p-5 transition duration-300 hover:-translate-y-1 hover:border-accent/30 hover:bg-hover ${card.className}`}>
                        <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-accent/10 text-accent-ink">
                          <Icon className="h-5 w-5" />
                        </div>
                        <h3 className="mt-4 text-lg font-semibold text-ink">{card.title}</h3>
                        <p className="mt-2 text-sm leading-7 text-ink-muted">{card.description}</p>
                      </div>
                    </Reveal>
                  );
                })}
              </div>
            </div>
          </Reveal>
        </section>

        <section className="relative mx-auto max-w-7xl px-6 pb-24 sm:px-8 lg:px-12">
          <Reveal>
            <div className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
              <div className="rounded-[32px] border border-line bg-raised/70 p-6 shadow-2xl shadow-black/20 backdrop-blur sm:p-8">
                <div className="flex items-center gap-2 text-sm font-semibold uppercase tracking-[0.3em] text-accent">
                  <MessageSquareText className="h-4 w-4" />
                  Product preview
                </div>
                <h3 className="mt-4 text-2xl font-semibold text-ink">
                  A focused legal workspace, designed for clarity.
                </h3>
                <p className="mt-3 text-sm leading-7 text-ink-muted sm:text-base">
                  The interface is built to make legal reasoning feel structured, readable, and easy to act on.
                </p>

                <div className="mt-8 rounded-[28px] border border-line bg-canvas/80 p-4 sm:p-6">
                  <div className="flex items-center gap-2 text-sm text-ink-muted">
                    <div className="h-2.5 w-2.5 rounded-full bg-accent" />
                    <div className="h-2.5 w-2.5 rounded-full bg-hover" />
                    <div className="h-2.5 w-2.5 rounded-full bg-hover" />
                  </div>

                  <div className="mt-5 space-y-3">
                    <div className="rounded-2xl border border-line bg-raised/80 p-4">
                      <div className="flex items-center gap-2 text-sm font-medium text-ink">
                        <BrainCircuit className="h-4 w-4 text-accent" />
                        Legal AI Assistant
                      </div>
                      <p className="mt-2 text-sm leading-7 text-ink-muted">
                        “Here is a structured summary of the issue, the relevant provisions, and the likely precedent pathway.”
                      </p>
                    </div>
                    <div className="ml-auto max-w-[80%] rounded-2xl bg-accent px-4 py-3 text-sm font-medium text-ink">
                      Summarize the relevant case law for this dispute.
                    </div>
                    <div className="rounded-2xl border border-line bg-raised/80 p-4">
                      <div className="flex items-center gap-3">
                        <CheckCircle2 className="h-4 w-4 text-accent" />
                        <span className="text-sm font-medium text-ink">Confidence score: 87%</span>
                      </div>
                      <p className="mt-2 text-sm leading-7 text-ink-muted">
                        Conflict signals and citations are surfaced clearly for review.
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="rounded-[32px] border border-line bg-raised/70 p-6 shadow-2xl shadow-black/20 backdrop-blur sm:p-8">
                <div className="flex items-center gap-2 text-sm font-semibold uppercase tracking-[0.3em] text-accent">
                  <Sparkles className="h-4 w-4" />
                  Why teams use it
                </div>

                <div className="mt-6 grid gap-4">
                  {teamCards.map((card) => {
                    const Icon = card.icon;
                    return (
                      <div key={card.title} className="rounded-[24px] border border-line bg-hover p-4">
                        <div className="flex items-start gap-3">
                          <div className="mt-0.5 rounded-full bg-accent/10 p-2 text-accent-ink">
                            <Icon className="h-4 w-4" />
                          </div>
                          <div>
                            <h4 className="text-sm font-semibold text-ink">{card.title}</h4>
                            <p className="mt-1 text-sm leading-7 text-ink-muted">{card.description}</p>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </Reveal>
        </section>

        <section className="relative mx-auto max-w-7xl px-6 pb-20 sm:px-8 lg:px-12">
          <Reveal>
            <div className="rounded-[32px] border border-line bg-raised/70 p-6 shadow-2xl shadow-black/20 backdrop-blur sm:p-8 lg:p-10">
              <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                <div className="max-w-2xl">
                  <p className="text-sm font-semibold uppercase tracking-[0.3em] text-accent">
                    Why this exists
                  </p>
                  <h3 className="mt-2 text-2xl font-semibold text-ink sm:text-3xl">
                    Legal research is still too manual, and the hard part is often deciding what to trust.
                  </h3>
                </div>
                <div className="rounded-[24px] border border-line bg-hover p-4 text-sm leading-7 text-ink-muted">
                  <p className="font-medium text-ink">The goal is not to replace judgment.</p>
                  <p className="mt-2">It is to reduce the cost of finding the right authorities and seeing where the answer becomes uncertain.</p>
                </div>
              </div>

              <div className="mt-8 flex flex-wrap gap-3">
                {[
                  { label: "Manual research", icon: FileSearch },
                  { label: "Scattered precedent", icon: Scale },
                  { label: "Confidence gaps", icon: ShieldCheck },
                ].map((item) => {
                  const Icon = item.icon;
                  return (
                    <div key={item.label} className="flex items-center gap-2 rounded-full border border-line bg-canvas/70 px-3 py-2 text-sm text-ink-muted">
                      <Icon className="h-4 w-4 text-accent-ink" />
                      {item.label}
                    </div>
                  );
                })}
              </div>

              <p className="mt-6 max-w-3xl text-sm leading-8 text-ink-muted sm:text-base">
                In practice, legal work often means moving between case law, notes, and conflicting interpretations. The product tries to make that process more structured so the important question is not just “what did I find?” but “what should I review next?”
              </p>
            </div>
          </Reveal>
        </section>

        <section className="relative mx-auto max-w-7xl px-6 pb-20 sm:px-8 lg:px-12">
          <Reveal>
            <div className="rounded-[32px] border border-line bg-raised/70 p-6 shadow-2xl shadow-black/20 backdrop-blur sm:p-8 lg:p-10">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
                <div>
                  <p className="text-sm font-semibold uppercase tracking-[0.3em] text-accent">
                    Roadmap
                  </p>
                  <h3 className="mt-2 text-2xl font-semibold text-ink sm:text-3xl">
                    What comes next, in plain terms.
                  </h3>
                </div>
                <p className="max-w-xl text-sm leading-7 text-ink-muted sm:text-base">
                  This is an early project, so the roadmap stays focused on the core workflow before expanding too quickly.
                </p>
              </div>

              <div className="relative mt-8 space-y-4 before:absolute before:left-5 before:top-0 before:h-full before:w-px before:bg-line">
                {roadmapStages.map((stage, index) => (
                  <div key={stage.title} className="relative pl-12">
                    <div className="absolute left-2 top-5 h-3.5 w-3.5 rounded-full border-2 border-accent bg-canvas" />
                    <div className="rounded-[24px] border border-line bg-canvas/70 p-4">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className={`rounded-full px-2.5 py-1 text-xs font-semibold uppercase tracking-[0.25em] ${stage.tag === "Now" ? "bg-accent/15 text-accent-ink" : "bg-hover text-ink-muted"}`}>
                          {stage.tag}
                        </span>
                        <span className="text-sm font-medium text-ink">{stage.title}</span>
                      </div>
                      <p className="mt-3 text-sm leading-7 text-ink-muted">{stage.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </Reveal>
        </section>

        <section className="relative mx-auto max-w-7xl px-6 pb-20 sm:px-8 lg:px-12">
          <Reveal>
            <div className="rounded-[32px] border border-line bg-raised/70 p-6 shadow-2xl shadow-black/20 backdrop-blur sm:p-8 lg:p-10">
              <div className="max-w-2xl">
                <p className="text-sm font-semibold uppercase tracking-[0.3em] text-accent">
                  FAQ
                </p>
                <h3 className="mt-2 text-2xl font-semibold text-ink sm:text-3xl">
                  Questions legal professionals usually ask first.
                </h3>
              </div>

              <div className="mt-8 space-y-3">
                {faqs.map((item, index) => {
                  const isOpen = openFaq === index;
                  return (
                    <div key={item.question} className="rounded-[24px] border border-line bg-canvas/70">
                      <button
                        type="button"
                        className="flex w-full items-center justify-between px-4 py-4 text-left"
                        onClick={() => setOpenFaq(isOpen ? null : index)}
                        aria-expanded={isOpen}
                      >
                        <span className="text-sm font-medium text-ink">{item.question}</span>
                        <ChevronDown className={`h-4 w-4 text-ink-muted transition ${isOpen ? "rotate-180" : "rotate-0"}`} />
                      </button>
                      <div className={`grid transition-all duration-300 ${isOpen ? "grid-rows-[1fr]" : "grid-rows-[0fr]"}`}>
                        <div className="overflow-hidden">
                          <p className="px-4 pb-4 text-sm leading-7 text-ink-muted">{item.answer}</p>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </Reveal>
        </section>

        <section className="relative mx-auto max-w-7xl px-6 pb-24 sm:px-8 lg:px-12">
          <Reveal>
            <div className="rounded-[32px] border border-accent/20 bg-[linear-gradient(135deg,rgba(193,95,60,0.16),rgba(26,26,24,0.9))] p-8 shadow-2xl shadow-accent/10 sm:p-10">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
                <div>
                  <p className="text-sm font-semibold uppercase tracking-[0.3em] text-accent-ink">
                    Get started
                  </p>
                  <h3 className="mt-2 text-2xl font-semibold text-ink sm:text-3xl">
                    Turn scattered legal research into a more structured review workflow.
                  </h3>
                </div>
                <button
                  onClick={() => router.push("/register")}
                  className="inline-flex items-center justify-center gap-2 rounded-full bg-accent px-6 py-3 font-medium text-ink transition duration-200 hover:-translate-y-0.5 hover:bg-accent-hover"
                >
                  Register
                  <ArrowRight className="h-4 w-4" />
                </button>
              </div>
            </div>
          </Reveal>
        </section>
      </main>

      <footer className="border-t border-line bg-canvas/90 px-6 py-8 text-center text-sm text-ink-faint sm:px-8 lg:px-12">
        <div className="mx-auto flex max-w-7xl flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <p>© 2026 Legal AI Platform</p>
          <div className="flex items-center justify-center gap-4">
            <a href="#features" className="transition hover:text-accent">
              Features
            </a>
            <a href="#" className="transition hover:text-accent">
              Privacy
            </a>
            <a href="#" className="transition hover:text-accent">
              Contact
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}