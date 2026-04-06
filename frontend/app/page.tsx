"use client";

import { useCallback, useEffect, useState } from "react";
import Image from "next/image";
import { motion } from "framer-motion";
import { Frame, Loader2, SearchCheck, Stamp } from "lucide-react";
import { FileUpload } from "@/components/ui/file-upload";
import { ArtistMatchRow } from "@/components/ArtistMatchRow";
import { DistanceMeter } from "@/components/DistanceMeter";
import { ReferenceWorkCard } from "@/components/ReferenceWorkCard";
import { RiskBadge } from "@/components/RiskBadge";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { analyzeArtwork, fetchArtists } from "@/lib/api";
import { formatArtistName } from "@/lib/utils";
import type { AnalyzeResponse, ArtistsResponse } from "@/types/analysis";

const sectionReveal = {
  initial: { opacity: 0, y: 24 },
  whileInView: { opacity: 1, y: 0 },
  transition: { duration: 0.55, ease: [0.22, 1, 0.36, 1] },
  viewport: { once: true, amount: 0.2 },
} as const;

export default function LandingPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [selectedArtist, setSelectedArtist] = useState<string>("");
  const [artistsState, setArtistsState] = useState<{ loading: boolean; data: ArtistsResponse | null; error: string | null }>(
    { loading: true, data: null, error: null },
  );
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);

  const loadInitial = useCallback(async () => {
    setArtistsState((s) => ({ ...s, loading: true, error: null }));
    try {
      const artists = await fetchArtists();
      setArtistsState({ loading: false, data: artists, error: null });
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to load artists.";
      setArtistsState({ loading: false, data: null, error: msg });
    }
  }, []);

  useEffect(() => {
    void loadInitial();
  }, [loadInitial]);

  const handleRun = async () => {
    const file = files[0];
    if (!file || !selectedArtist) return;
    setAnalyzeError(null);
    setAnalyzing(true);
    setResult(null);
    try {
      const data = await analyzeArtwork(file, selectedArtist);
      setResult(data);
    } catch (e) {
      setAnalyzeError(e instanceof Error ? e.message : "Analysis failed.");
    } finally {
      setAnalyzing(false);
    }
  };

  const topMatches = result?.top_matches.slice(0, 3) ?? [];
  const maxDistance = topMatches.length ? Math.max(...topMatches.map((m) => m.distance), 0.001) : 1;
  const references = result?.nearest_reference_works.slice(0, 3) ?? [];

  return (
    <main className="bg-[#0a0a0a] text-neutral-200">
      <header className="sticky top-0 z-50 border-b border-amber-500/20 bg-[#0a0a0a]/95 backdrop-blur-md">
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 md:px-8">
          <span className="font-serif text-2xl text-white">ArtShield AI</span>
          <nav className="flex items-center gap-5 font-mono text-[11px] uppercase tracking-widest text-neutral-400">
            <a href="#how-it-works" className="transition-colors hover:text-amber-400">
              How It Works
            </a>
            <a href="#analyze" className="transition-colors hover:text-amber-400">
              Analyze
            </a>
            <a href="#about" className="transition-colors hover:text-amber-400">
              About
            </a>
          </nav>
        </div>
      </header>

      <section className="relative flex min-h-screen items-center overflow-hidden px-4 py-20 md:px-8">
        <div className="pointer-events-none absolute right-[6%] top-1/2 h-72 w-72 -translate-y-1/2 rounded-full bg-amber-500/20 blur-[120px]" />
        <div className="mx-auto grid w-full max-w-6xl gap-12 lg:grid-cols-5 lg:items-center">
          <motion.div className="lg:col-span-3" {...sectionReveal}>
            <p className="font-mono text-[10px] uppercase tracking-[0.32em] text-amber-500">
              Forensic Intelligence for the Art World
            </p>
            <h1 className="mt-6 max-w-3xl font-serif text-5xl leading-tight text-white md:text-7xl">
              Is This Painting Really What It Claims to Be?
            </h1>
            <p className="mt-6 max-w-2xl font-sans text-base leading-relaxed text-neutral-400 md:text-lg">
              ArtShield AI uses a fine-tuned deep learning model to analyze artwork attribution and return a calibrated
              forensic risk report - in seconds.
            </p>
            <a
              href="#analyze"
              className="mt-10 inline-flex border border-amber-500 px-8 py-3 font-sans text-sm font-semibold uppercase tracking-[0.18em] text-amber-500 transition-colors hover:bg-amber-500/15"
            >
              Analyze an Artwork
            </a>
          </motion.div>

          <motion.div className="relative mx-auto w-full max-w-sm lg:col-span-2 lg:max-w-none" {...sectionReveal}>
            <div className="relative mx-auto w-[85%] rounded-sm border border-neutral-700 bg-[#111111] p-3 shadow-[0_24px_80px_rgba(245,158,11,0.17)]">
              <div className="-rotate-2 overflow-hidden border border-neutral-800 bg-black">
                <Image
                  src="/hero-painting.jpg"
                  alt="Van Gogh artwork preview"
                  width={420}
                  height={560}
                  className="h-auto w-full object-cover"
                  priority
                />
              </div>
              <div className="absolute -left-6 top-6 rounded-md border border-neutral-800 bg-[#111111]/95 px-4 py-3 shadow-xl backdrop-blur-sm">
                <p className="font-mono text-[10px] uppercase tracking-widest text-neutral-500">Risk Preview</p>
                <div className="mt-2 flex items-center gap-2">
                  <span className="relative flex h-2.5 w-2.5">
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-500/70" />
                    <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-green-500" />
                  </span>
                  <span className="font-mono text-xs uppercase tracking-wider text-green-500">Low Risk</span>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      <motion.section id="how-it-works" className="px-4 py-24 md:px-8" {...sectionReveal}>
        <div className="mx-auto max-w-6xl">
          <p className="font-mono text-[10px] uppercase tracking-[0.34em] text-amber-500">How It Works</p>
          <h2 className="mt-5 font-serif text-4xl text-white md:text-5xl">Three Steps to a Forensic Risk Report</h2>

          <div className="relative mt-12 grid gap-5 md:grid-cols-3">
            <div className="pointer-events-none absolute left-1/2 top-1/2 hidden h-px w-[68%] -translate-x-1/2 border-t border-dashed border-amber-500/35 md:block" />

            {[
              {
                step: "01",
                title: "Upload the Artwork",
                body: "Provide a clear image of the painting you want to evaluate. The system inspects visual patterns learned from historical works.",
                icon: Frame,
              },
              {
                step: "02",
                title: "Claim the Artist",
                body: "Select the attribution being claimed for the piece. ArtShield compares your image embedding against artist-specific thresholds.",
                icon: SearchCheck,
              },
              {
                step: "03",
                title: "Receive the Report",
                body: "Get a calibrated LOW, MEDIUM, or HIGH risk outcome. The report includes nearest matches and interpretation guidance for experts.",
                icon: Stamp,
              },
            ].map((item) => {
              const Icon = item.icon;
              return (
                <article key={item.step} className="relative border border-neutral-800 bg-[#111111] p-7">
                  <Icon className="h-9 w-9 text-amber-500" strokeWidth={1.5} />
                  <p className="mt-5 font-mono text-xs tracking-widest text-amber-500">{item.step}</p>
                  <h3 className="mt-2 font-sans text-lg font-semibold text-white">{item.title}</h3>
                  <p className="mt-3 font-sans text-sm leading-relaxed text-neutral-400">{item.body}</p>
                </article>
              );
            })}
          </div>
        </div>
      </motion.section>

      <motion.section className="border-y border-neutral-800 bg-[#111111] px-4 py-4 md:px-8" {...sectionReveal}>
        <div className="mx-auto flex max-w-6xl flex-col divide-y divide-amber-500/25 font-mono text-xs text-neutral-400 md:flex-row md:divide-x md:divide-y-0">
          {["8 Artists Supported", "TPR 80.21%", "Fine-tuned DINOv2 ViT-B/14", "Risk Signal — Not a Verdict"].map((item) => (
            <p key={item} className="px-4 py-2 text-center tracking-wide md:flex-1">
              {item}
            </p>
          ))}
        </div>
      </motion.section>

      <motion.section id="analyze" className="px-4 py-24 md:px-8" {...sectionReveal}>
        <div className="mx-auto max-w-6xl">
          <p className="font-mono text-[10px] uppercase tracking-[0.34em] text-amber-500">Artwork Analysis</p>
          <h2 className="mt-5 font-serif text-4xl text-white md:text-5xl">Submit an Artwork for Risk Assessment</h2>

          {artistsState.error ? (
            <div role="alert" className="mt-8 border border-red-500/50 bg-red-500/10 px-4 py-3 font-sans text-sm text-red-300">
              {artistsState.error}
            </div>
          ) : null}

          <section className="mt-12 space-y-4">
            <label className="block font-sans text-sm font-medium text-neutral-300">Upload Artwork for Analysis</label>
            <div className="mx-auto min-h-96 w-full max-w-4xl rounded-lg border border-dashed border-neutral-200 bg-white dark:border-neutral-800 dark:bg-black">
              <FileUpload onChange={setFiles} isAnalyzing={analyzing} />
            </div>

            <div className="mx-auto max-w-4xl space-y-3 pt-6">
              <label htmlFor="artist-select" className="block font-sans text-sm text-neutral-400">
                Claimed artist
              </label>
              {artistsState.loading ? (
                <div className="flex h-12 items-center gap-2 border border-dashed border-neutral-700 bg-[#111111] px-4 font-mono text-xs text-neutral-500">
                  <Loader2 className="h-4 w-4 animate-spin text-amber-500" />
                  Loading supported artists...
                </div>
              ) : (
                <Select value={selectedArtist || undefined} onValueChange={setSelectedArtist}>
                  <SelectTrigger id="artist-select" className="w-full">
                    <SelectValue placeholder="Select claimed artist" />
                  </SelectTrigger>
                  <SelectContent>
                    {artistsState.data?.supported_artists.map((id) => (
                      <SelectItem key={id} value={id}>
                        {formatArtistName(id)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}

              <div className="pt-4">
                <Button
                  type="button"
                  size="lg"
                  className="w-full sm:w-auto"
                  disabled={!files[0] || !selectedArtist || analyzing || artistsState.loading || !!artistsState.error}
                  onClick={() => void handleRun()}
                >
                  {analyzing ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    "Run Analysis"
                  )}
                </Button>
              </div>
            </div>

            {analyzeError && !analyzeError.includes("Backend offline") ? (
              <p className="font-sans text-sm text-red-300" role="alert">
                {analyzeError}
              </p>
            ) : null}
          </section>

          {result ? (
            <>
              <motion.section
                key={`report-${result.target_distance}-${result.claimed_artist}`}
                className="mt-20 space-y-10 border-t border-dashed border-neutral-800 pt-16"
                initial={{ opacity: 0, y: 28 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
              >
                <h3 className="font-mono text-[11px] uppercase tracking-[0.35em] text-amber-500">Risk Report</h3>
                <RiskBadge level={result.risk_level} />
                <p className="text-center font-sans text-lg text-neutral-200">
                  Model Prediction: <span className="font-semibold text-white">{formatArtistName(result.predicted_artist)}</span>
                </p>
                <p className="mx-auto max-w-2xl text-center font-sans text-sm italic text-neutral-500">{result.confidence_note}</p>
                <div className="mx-auto max-w-3xl pt-4">
                  <DistanceMeter
                    targetDistance={result.target_distance}
                    low={result.per_artist_thresholds.low}
                    high={result.per_artist_thresholds.high}
                    riskLevel={result.risk_level}
                  />
                </div>
                <div className="mx-auto max-w-3xl border border-amber-500/40 bg-amber-500/[0.04] px-6 py-5">
                  <p className="font-mono text-[10px] uppercase tracking-widest text-amber-500">Recommendation</p>
                  <p className="mt-3 font-sans text-sm leading-relaxed text-neutral-300">{result.recommendation}</p>
                </div>
                <p className="text-center font-sans text-xs text-neutral-600">{result.disclaimer}</p>
              </motion.section>

              <motion.section
                key={`matches-${result.target_distance}-${result.claimed_artist}`}
                className="mt-20 space-y-6 border-t border-dashed border-neutral-800 pt-16"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.45, delay: 0.08 }}
              >
                <h3 className="font-mono text-[11px] uppercase tracking-[0.35em] text-amber-500">Top Artist Matches</h3>
                <div className="divide-y divide-neutral-800 border border-neutral-800 bg-[#111111]">
                  {topMatches.map((m) => (
                    <ArtistMatchRow
                      key={m.artist}
                      artist={m.artist}
                      distance={m.distance}
                      maxDistance={maxDistance}
                      isClaimed={m.artist === result.claimed_artist}
                    />
                  ))}
                </div>
              </motion.section>

              <motion.section
                key={`refs-${result.target_distance}-${result.claimed_artist}`}
                className="mt-20 space-y-6 border-t border-dashed border-neutral-800 pt-16"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.45, delay: 0.12 }}
              >
                <h3 className="font-mono text-[11px] uppercase tracking-[0.35em] text-amber-500">Reference Works</h3>
                <div className="flex flex-col items-center gap-8 md:flex-row md:flex-nowrap md:justify-center md:gap-10">
                  {references.map((ref) => (
                    <ReferenceWorkCard key={ref.filename} filename={ref.filename} similarityScore={ref.similarity_score} />
                  ))}
                </div>
              </motion.section>
            </>
          ) : null}
        </div>
      </motion.section>

      <motion.section id="about" className="border-t border-neutral-800 px-4 py-24 md:px-8" {...sectionReveal}>
        <div className="mx-auto grid max-w-6xl gap-10 md:grid-cols-2 md:items-start">
          <div>
            <p className="font-mono text-[10px] uppercase tracking-[0.34em] text-amber-500">About</p>
            <h2 className="mt-5 font-serif text-4xl text-white md:text-5xl">What ArtShield AI Is - And What It Isn't</h2>
            <p className="mt-6 max-w-xl font-sans text-sm leading-relaxed text-neutral-400">
              ArtShield AI is a risk signal platform designed for expert workflows. It helps collectors, galleries, and
              researchers triage attribution concerns with model-based evidence.
            </p>
            <p className="mt-4 max-w-xl font-sans text-sm leading-relaxed text-neutral-400">
              It is not an authenticity oracle and does not replace provenance research, connoisseurship, or lab
              testing. Results are intended for expert review.
            </p>
            <p className="mt-4 max-w-xl font-sans text-sm leading-relaxed text-neutral-400">
              The model was trained on WikiArt data across 8 major artists and reports calibrated forensic-style risk
              levels.
            </p>
          </div>

          <div className="border border-neutral-800 bg-[#111111] p-6">
            <p className="font-mono text-[10px] uppercase tracking-[0.24em] text-amber-500">Model Card</p>
            <dl className="mt-4 space-y-3 font-mono text-xs">
              <div className="flex items-center justify-between gap-4 border-b border-neutral-800 pb-3">
                <dt className="text-neutral-500">Model</dt>
                <dd className="text-neutral-300">DINOv2 ViT-B/14</dd>
              </div>
              <div className="flex items-center justify-between gap-4 border-b border-neutral-800 pb-3">
                <dt className="text-neutral-500">Training</dt>
                <dd className="text-neutral-300">Fine-tuned, 20 epochs</dd>
              </div>
              <div className="flex items-center justify-between gap-4 border-b border-neutral-800 pb-3">
                <dt className="text-neutral-500">Dataset</dt>
                <dd className="text-neutral-300">WikiArt, 8 artists</dd>
              </div>
              <div className="flex items-center justify-between gap-4">
                <dt className="text-neutral-500">Accuracy</dt>
                <dd className="text-neutral-300">TPR 80.21% / TNR 68.45%</dd>
              </div>
            </dl>
          </div>
        </div>
      </motion.section>

      <footer className="border-t border-amber-500/20 px-4 py-6 md:px-8">
        <div className="mx-auto grid max-w-6xl gap-3 font-mono text-[11px] text-neutral-500 md:grid-cols-3 md:items-center">
          <p className="text-left">© 2026 ArtShield AI — Built by Hafiz Hamza Ahmed</p>
          <p className="text-left md:text-center">This output is a risk signal for expert review, not an authenticity verdict.</p>
          <p className="text-left md:text-right">Powered by DINOv2 + FastAPI</p>
        </div>
      </footer>
    </main>
  );
}
