"use client";

import { apiFetch } from "@/components/services/api";
import { useState, useEffect } from "react";
import { ChevronDown } from "lucide-react";

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(true);

  return (
    <div className="rounded-2xl bg-raised">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between rounded-t-2xl px-5 py-4 text-left text-ink transition hover:bg-hover"
      >
        <span className="font-semibold">{title}</span>
        <ChevronDown
          size={18}
          className={`transition-transform ${open ? "rotate-180" : "rotate-0"}`}
        />
      </button>

      {open && (
        <div className="px-5 pb-5 text-sm leading-relaxed text-ink-muted">
          {children}
        </div>
      )}
    </div>
  );
}

function MarkdownText({ content }: { content?: string }) {
  if (!content) return null;

  const lines = content.split(/\n/);

  return (
    <div className="space-y-2 text-ink-muted">
      {lines.map((line, index) => {
        const trimmed = line.trim();
        if (!trimmed) {
          return <div key={index} className="h-1" />;
        }

        if (/^###\s/.test(trimmed)) {
          return (
            <h3 key={index} className="text-base font-semibold text-ink">
              {trimmed.replace(/^###\s/, "")}
            </h3>
          );
        }

        if (/^##\s/.test(trimmed)) {
          return (
            <h2 key={index} className="text-lg font-semibold text-ink">
              {trimmed.replace(/^##\s/, "")}
            </h2>
          );
        }

        if (/^#\s/.test(trimmed)) {
          return (
            <h1 key={index} className="text-xl font-semibold text-ink">
              {trimmed.replace(/^#\s/, "")}
            </h1>
          );
        }

        if (/^[-*]\s/.test(trimmed)) {
          return (
            <li key={index} className="ml-4 list-disc text-ink-muted">
              {trimmed.replace(/^[-*]\s/, "")}
            </li>
          );
        }

        const parts = trimmed.split(/(\*\*[^*]+\*\*)/g);
        return (
          <p key={index} className="leading-7">
            {parts.map((part, partIndex) => {
              if (part.startsWith("**") && part.endsWith("**")) {
                return (
                  <strong key={`${index}-${partIndex}`} className="font-semibold text-ink">
                    {part.slice(2, -2)}
                  </strong>
                );
              }
              return <span key={`${index}-${partIndex}`}>{part}</span>;
            })}
          </p>
        );
      })}
    </div>
  );
}

export default function StructuredResponse({ data }: { data: any }) {
  const rawConfidence = data?.confidence_score ?? 0;
  const confidence =
    rawConfidence > 1 ? rawConfidence : Math.round(rawConfidence * 100);

  const [bookmarked, setBookmarked] = useState(data?.isBookmarked || false);

  useEffect(() => {
    setBookmarked(data?.isBookmarked || false);
  }, [data]);

  const confidenceColor =
    confidence >= 80 ? "bg-teal-500" : confidence >= 50 ? "bg-amber-500" : "bg-rose-500";

  const toggleBookmark = async () => {
    try {
      await apiFetch(`/chat/bookmark/${data.messageId}`, {
        method: "PATCH",
      });
      setBookmarked(!bookmarked);
    } catch (err) {
      console.error(err);
    }
  };

  const downloadPDF = () => {
    window.print();
  };

  return (
    <div className="space-y-5 print-area">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex gap-3">
          <button
            onClick={toggleBookmark}
            className="rounded-full border border-line bg-raised px-3 py-1.5 text-sm text-ink transition hover:bg-press"
          >
            {bookmarked ? "★ Bookmarked" : "☆ Bookmark"}
          </button>

          <button
            onClick={downloadPDF}
            className="rounded-full border border-line bg-raised px-3 py-1.5 text-sm text-ink transition hover:bg-press"
          >
            Download PDF
          </button>
        </div>
      </div>

      {data?.conflicts_detected && (
        <div className="rounded-2xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
          ⚠ Conflict detected in legal sources. Review carefully.
        </div>
      )}

      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm text-ink-muted">
          <span>Confidence Score</span>
          <span className="font-semibold text-ink">{confidence}%</span>
        </div>

        <div className="h-2 overflow-hidden rounded-full bg-hover">
          <div
            className={`h-full ${confidenceColor} transition-all duration-500`}
            style={{ width: `${confidence}%` }}
          />
        </div>
      </div>

      <Section title="Issue Summary">
        <MarkdownText content={data?.issue_summary} />
      </Section>

      <Section title="Relevant Legal Provisions">
        {data?.relevant_legal_provisions?.map((item: any, index: number) => (
          <div key={index} className="mb-4 rounded-2xl bg-surface p-4">
            <p className="font-semibold text-ink">
              {item.act_name ?? "Unknown Act"} {item.section && `– ${item.section}`}
            </p>
            <p className="mt-1 text-ink-muted">
              {item.explanation || item.description || "No explanation available."}
            </p>
          </div>
        ))}
      </Section>

      <Section title="Applicable Sections">
        <div className="space-y-4">
          {data?.applicable_sections?.map((sec: any, index: number) => {
            if (typeof sec === "string") {
              return (
                <p key={index} className="text-ink-muted">
                  {sec}
                </p>
              );
            }

            return (
              <div key={index} className="rounded-2xl bg-surface p-4">
                <p className="font-semibold text-ink">
                  {sec.section_number ?? "N/A"} – {sec.section_title ?? "Untitled"}
                </p>
                <p className="mt-1 text-sm text-ink-muted">
                  {sec.section_summary ?? "No summary available."}
                </p>
              </div>
            );
          })}
        </div>
      </Section>

      <Section title="Case References">
        {data?.case_references?.map((c: any, index: number) => (
          <div key={index} className="mb-4 rounded-2xl bg-surface p-4">
            <p className="font-semibold text-ink">
              {c.case_name || c.case_title || "Unknown Case"}
            </p>

            {c.court && (
              <p className="mt-1 text-sm text-ink-faint">
                {c.court} {c.year && `(${c.year})`}
              </p>
            )}

            <p className="mt-2 text-ink-muted">
              {c.citation_reference || c.holding_summary || ""}
            </p>
          </div>
        ))}
      </Section>

      <Section title="Key Observations">
        <ul className="list-disc space-y-1 pl-5 text-ink-muted">
          {data?.key_observations?.map((obs: string, index: number) => (
            <li key={index}>{obs}</li>
          ))}
        </ul>
      </Section>

      <Section title="Legal Interpretation">
        <MarkdownText content={data?.legal_interpretation} />
      </Section>

      <Section title="Precedents">
        {data?.precedents?.map((p: any, index: number) => {
          if (typeof p === "string") {
            return (
              <p key={index} className="text-ink-muted">
                {p}
              </p>
            );
          }

          return (
            <div key={index} className="mb-3 rounded-2xl bg-surface p-4">
              <p className="font-semibold text-ink">{p.case_title || "Unknown Case"}</p>
              <p className="mt-1 text-sm text-ink-muted">
                {p.principle_established || ""}
              </p>
            </div>
          );
        })}
      </Section>

      <Section title="Conclusion">
        <MarkdownText content={data?.conclusion} />
      </Section>

      <Section title="Citations">
        {data?.citations?.map((cit: any, index: number) => (
          <div key={index} className="mb-4 rounded-2xl bg-surface p-4">
            <p className="font-semibold text-ink">
              {cit.title || cit.citation_reference || "Unknown Citation"}
            </p>

            {cit.court && (
              <p className="mt-1 text-sm text-ink-faint">
                {cit.court} {cit.year && `(${cit.year})`}
              </p>
            )}

            {cit.source && <p className="mt-1 text-sm text-ink-muted">{cit.source}</p>}

            {(cit.url || cit.source_url) && (
              <a
                href={cit.url || cit.source_url}
                target="_blank"
                className="mt-2 inline-block text-sm font-medium text-accent underline transition hover:text-accent-hover"
              >
                View Source →
              </a>
            )}
          </div>
        ))}
      </Section>
    </div>
  );
}