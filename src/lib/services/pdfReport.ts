/**
 * Dependency-free PDF export for batch analysis reports (Masterplan §5.2).
 *
 * Emits a valid PDF 1.4 document using the Helvetica/Courier Type1 base
 * fonts (nothing to embed), containing a header block and a paginated
 * results table. Kept intentionally small: the report is tabular text, so
 * a full PDF library would violate the lightweight-stack mandate.
 */

import type { BatchJobResponse } from "@/lib/api/client";

const PAGE_W = 792; // US Letter landscape, points
const PAGE_H = 612;
const MARGIN = 48;
const ROW_H = 18;
// Reserve space for title block + table header + footer.
const ROWS_PER_PAGE = Math.floor((PAGE_H - 2 * MARGIN - 100) / ROW_H);

// Newline constant avoids embedding raw control characters in this source.
const NL = String.fromCharCode(10);
const BACKSLASH = String.fromCharCode(92);

interface Row {
  index: number;
  name: string;
  verdict: string;
  prob: string;
  family: string;
  ms: string;
}

/** Escape characters that are special inside PDF literal strings. */
function escapePdfText(text: string): string {
  return text
    .replace(new RegExp("[^" + BACKSLASH + "x20-" + BACKSLASH + "x7E]", "g"), "?")
    .split(BACKSLASH)
    .join(BACKSLASH + BACKSLASH)
    .split("(")
    .join(BACKSLASH + "(")
    .split(")")
    .join(BACKSLASH + ")");
}

function truncate(text: string, maxChars: number): string {
  const safe = escapePdfText(text);
  return safe.length <= maxChars ? safe : `${safe.slice(0, maxChars - 1)}~`;
}

/** One BT/ET text draw at absolute coordinates in default user space. */
function textCmd(x: number, y: number, size: number, font: "F1" | "F2", content: string): string {
  return `BT /${font} ${size} Tf ${x.toFixed(1)} ${y.toFixed(1)} Td (${content}) Tj ET`;
}

function lineCmd(x1: number, y1: number, x2: number, y2: number): string {
  return `${x1.toFixed(1)} ${y1.toFixed(1)} m ${x2.toFixed(1)} ${y2.toFixed(1)} l S`;
}

/** Build paginated per-page content streams for the results table. */
function buildPages(job: BatchJobResponse): { contents: string[]; totalRows: number } {
  const sorted = [...job.results].sort((a, b) => a.index - b.index);
  const rows: Row[] = sorted.map((r) => ({
    index: r.index,
    name: r.filename,
    verdict:
      r.status === "error" ? `ERROR: ${r.error ?? "unknown"}` : r.is_ai ? "AI Generated" : "Authentic",
    prob: r.fake_probability != null ? `${(r.fake_probability * 100).toFixed(1)}%` : "-",
    family: r.family ?? "-",
    ms: r.latency_ms != null ? r.latency_ms.toFixed(0) : "-",
  }));

  const pages: string[] = [];
  for (let start = 0; start < Math.max(1, rows.length); start += ROWS_PER_PAGE) {
    const slice = rows.slice(start, start + ROWS_PER_PAGE);
    const cmds: string[] = [];
    let y = PAGE_H - MARGIN;

    if (start === 0) {
      cmds.push(textCmd(MARGIN, y, 18, "F1", escapePdfText("Frequency Guard - Batch Analysis Report")));
      y -= 22;
      cmds.push(
        textCmd(
          MARGIN,
          y,
          10,
          "F2",
          escapePdfText(`Job ${job.job_id} | ${new Date().toISOString()} | ${rows.length} images`)
        )
      );
      y -= 30;
    } else {
      cmds.push(
        textCmd(MARGIN, y, 12, "F1", escapePdfText(`Frequency Guard Batch Report - Job ${job.job_id}`))
      );
      y -= 26;
    }

    // Table header row.
    cmds.push(
      textCmd(
        MARGIN,
        y,
        9,
        "F2",
        "#".padEnd(5) +
          escapePdfText("Filename").padEnd(34) +
          escapePdfText("Verdict").padEnd(14) +
          "FakeProb".padEnd(11) +
          escapePdfText("Family").padEnd(13) +
          "Lat(ms)"
      )
    );
    y -= 4;
    cmds.push(lineCmd(MARGIN, y, PAGE_W - MARGIN, y));
    y -= ROW_H;

    for (const r of slice) {
      cmds.push(
        textCmd(
          MARGIN,
          y,
          9,
          "F2",
          String(r.index).padEnd(5) +
            truncate(r.name, 32).padEnd(34) +
            truncate(r.verdict, 40).padEnd(42) +
            r.prob.padEnd(11) +
            truncate(r.family, 11).padEnd(13) +
            r.ms
        )
      );
      y -= ROW_H;
    }

    cmds.push(lineCmd(MARGIN, y + ROW_H - 8, PAGE_W - MARGIN, y + ROW_H - 8));
    cmds.push(
      textCmd(
        MARGIN,
        MARGIN / 2,
        8,
        "F2",
        escapePdfText(`Completed ${job.completed}/${job.total_images} | Status: ${job.status}`)
      )
    );

    pages.push(cmds.join(NL));
  }

  return { contents: pages, totalRows: rows.length };
}

/** Assemble the full PDF byte string with xref table and trailer. */
export function buildBatchReportPdf(job: BatchJobResponse): Blob {
  const { contents } = buildPages(job);
  const objects: string[] = [];

  // 1: catalog, 2: pages tree; then per page: page obj + content obj.
  const pageCount = contents.length;
  const firstPageObj = 3;

  objects[1] = "<< /Type /Catalog /Pages 2 0 R >>";
  const kids = Array.from({ length: pageCount }, (_, i) => `${firstPageObj + i * 2} 0 R`).join(" ");
  objects[2] = `<< /Type /Pages /Kids [${kids}] /Count ${pageCount} >>`;

  contents.forEach((stream, i) => {
    const pageNum = firstPageObj + i * 2;
    const contentNum = pageNum + 1;
    objects[pageNum] =
      `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${PAGE_W} ${PAGE_H}] ` +
      `/Resources << /Font << /F1 100 0 R /F2 101 0 R >> >> /Contents ${contentNum} 0 R >>`;
    objects[contentNum] =
      `<< /Length ${stream.length} >>` + NL + "stream" + NL + stream + NL + "endstream";
  });

  objects[100] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>";
  objects[101] = "<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>";

  let pdf = "%PDF-1.4" + NL;
  const offsets: number[] = [];
  for (let num = 1; num < objects.length; num++) {
    if (!objects[num]) continue;
    offsets[num] = pdf.length;
    pdf += `${num} 0 obj` + NL + `${objects[num]}` + NL + "endobj" + NL;
  }

  const xrefStart = pdf.length;
  const maxNum = objects.length;
  pdf += "xref" + NL + `0 ${maxNum}` + NL + "0000000000 65535 f " + NL;
  for (let num = 1; num < maxNum; num++) {
    pdf += offsets[num]
      ? `${String(offsets[num]).padStart(10, "0")} 00000 n ` + NL
      : "0000000000 65535 f " + NL;
  }
  pdf +=
    "trailer" + NL + `<< /Size ${maxNum} /Root 1 0 R >>` + NL + "startxref" + NL + `${xrefStart}` + NL + "%%EOF";

  return new Blob([pdf], { type: "application/pdf" });
}

/** Trigger a client-side download of the batch report as PDF. */
export function downloadBatchReportPdf(job: BatchJobResponse): void {
  const blob = buildBatchReportPdf(job);
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `batch_${job.job_id}.pdf`;
  a.click();
  URL.revokeObjectURL(url);
}
