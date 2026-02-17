import { lazy, Suspense } from "react";

const HighlightedCode = lazy(() => import("./HighlightedCode.jsx"));

function CodeFallback({ code }) {
  return (
    <pre className="bg-well rounded-2xl p-5 overflow-x-auto text-[13px] leading-[1.7] font-mono text-ink-code">
      <code>{code}</code>
    </pre>
  );
}

export function CodeBlock({ code }) {
  return (
    <Suspense fallback={<CodeFallback code={code} />}>
      <HighlightedCode code={code} />
    </Suspense>
  );
}
