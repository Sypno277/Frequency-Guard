/** Unit tests for the dependency-free batch PDF report generator. */

import { describe, expect, it } from "vitest";
import type { BatchJobResponse } from "@/lib/api/client";
import { buildBatchReportPdf } from "./pdfReport";

function sampleJob(): BatchJobResponse {
  return {
    job_id: "testtid",
    status: "completed",
    total_images: 3,
    completed: 3,
    results: [
      { index: 0, filename: "fake.png", status: "done", is_ai: true, fake_probability: 1.0, family: "gan", latency_ms: 292.7 },
      { index: 1, filename: "real (copy).png", status: "done", is_ai: false, fake_probability: 0.0, family: "real", latency_ms: 218.9 },
      { index: 2, filename: "bad.txt", status: "error", error: "Could not decode image payload" },
    ],
  };
}

async function blobText(blob: Blob): Promise<string> {
  return blob.text();
}

describe("buildBatchReportPdf", () => {
  it("produces a valid PDF 1.4 byte string", async () => {
    const blob = buildBatchReportPdf(sampleJob());
    expect(blob.type).toBe("application/pdf");
    const text = await blobText(blob);
    expect(text.startsWith("%PDF-1.4")).toBe(true);
    expect(text.trimEnd().endsWith("%%EOF")).toBe(true);
  });

  it("computes accurate stream Length for every content stream", async () => {
    const text = await blobText(buildBatchReportPdf(sampleJob()));
    const re = /<< \/Length (\d+) >>\nstream\n([\s\S]*?)\nendstream/g;
    let m: RegExpExecArray | null;
    let found = 0;
    while ((m = re.exec(text)) !== null) {
      expect(Number(m[1])).toBe(m[2].length);
      found++;
    }
    expect(found).toBeGreaterThanOrEqual(1);
  });

  it("emits catalog, page tree, media box, and both fonts", async () => {
    const text = await blobText(buildBatchReportPdf(sampleJob()));
    expect(text).toContain("/Type /Catalog");
    expect(text).toContain("/Count 1");
    expect(text).toContain("[0 0 792 612]");
    expect(text).toContain("/Helvetica-Bold");
    expect(text).toContain("/Courier");
  });

  it("writes valid xref offsets for the first objects", async () => {
    const text = await blobText(buildBatchReportPdf(sampleJob()));
    const seg = text.slice(text.indexOf("xref")).split("\n").slice(2);
    for (const n of [1, 2, 3, 4]) {
      const off = parseInt(seg[n].slice(0, 10), 10);
      // object header is followed by a newline before its dictionary
      expect(text.slice(off).startsWith(`${n} 0 obj`)).toBe(true);
    }
  });

  it("escapes literal parentheses and backslashes in filename cells", async () => {
    const text = await blobText(buildBatchReportPdf(sampleJob()));
    // "real (copy).png" must appear with escaped parens inside the stream
    expect(text).toContain("real \\(copy\\).png");
  });

  it("pages when more rows than one page fits", async () => {
    const job = sampleJob();
    const many = Array.from({ length: 30 }, (_, i) => ({
      index: i,
      filename: `img_${i}.png`,
      status: "done" as const,
      is_ai: i % 2 === 0,
      fake_probability: i % 2 === 0 ? 1.0 : 0.0,
      family: i % 2 === 0 ? "gan" : "real",
      latency_ms: 100 + i,
    }));
    const big = { ...job, total_images: 30, completed: 30, results: many };
    const text = await blobText(buildBatchReportPdf(big));
    expect(text).toContain("/Count 2");
  });
});
