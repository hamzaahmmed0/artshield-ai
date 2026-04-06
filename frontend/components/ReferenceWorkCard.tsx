"use client";

import { useState } from "react";
import { CometCard } from "@/components/ui/comet-card";
import { formatSimilarityPercent } from "@/lib/utils";
import { referenceImageUrl } from "@/lib/api";

type Props = {
  filename: string;
  similarityScore: number;
};

export function ReferenceWorkCard({ filename, similarityScore }: Props) {
  const [broken, setBroken] = useState(false);
  const src = referenceImageUrl(filename);
  const pct = formatSimilarityPercent(similarityScore);

  return (
    <CometCard className="w-full max-w-sm shrink-0 md:w-80">
      <button
        type="button"
        className="my-10 flex w-full cursor-default flex-col items-stretch rounded-[16px] border-0 bg-[#1F2121] p-2 saturate-0 md:my-20 md:w-80 md:p-4"
        aria-label={`Reference work ${filename}, similarity ${pct}`}
        style={{ transformStyle: "preserve-3d", transform: "none", opacity: 1 }}
      >
        <div className="mx-2 flex-1">
          <div className="relative mt-2 aspect-[3/4] w-full">
            {broken ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center rounded-[16px] border border-dashed border-lab-border bg-lab-bg px-3 text-center">
                <span className="font-mono text-[10px] uppercase tracking-widest text-neutral-500">Reference</span>
                <span className="mt-2 break-all font-mono text-xs text-neutral-400">{filename}</span>
                <span className="mt-2 font-mono text-[10px] text-neutral-600">Thumbnail not served by API</span>
              </div>
            ) : (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                loading="lazy"
                className="absolute inset-0 h-full w-full rounded-[16px] bg-[#000000] object-cover contrast-75"
                alt={`Reference work ${filename}`}
                src={src}
                onError={() => setBroken(true)}
                style={{ boxShadow: "rgba(0, 0, 0, 0.05) 0px 5px 6px 0px", opacity: 1 }}
              />
            )}
          </div>
        </div>
        <div className="mt-2 flex flex-shrink-0 items-center justify-between p-4 font-mono text-white">
          <div className="max-w-[55%] truncate text-left text-xs text-neutral-300" title={filename}>
            {filename}
          </div>
          <div className="text-xs text-lab-accent opacity-90">{pct}</div>
        </div>
      </button>
    </CometCard>
  );
}
