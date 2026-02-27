import { AnimatePresence, motion } from "framer-motion";
import { DEPTH_CONFIG } from "../../data/sections";
import { CodeBlock } from "./CodeBlock";

function renderFormattedText(text) {
  const lines = text.split("\n");
  const elements = [];
  let blockquoteLines = [];

  const flushBlockquote = () => {
    if (blockquoteLines.length === 0) return;
    elements.push(
      <blockquote
        key={`bq-${elements.length}`}
        className="border-l-[3px] border-accent-fill pl-5 my-6 italic"
      >
        <p className="font-serif text-[16px] leading-[1.85] text-ink-secondary">
          {renderInlineFormatting(blockquoteLines.join("\n"))}
        </p>
      </blockquote>
    );
    blockquoteLines = [];
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith("> ")) {
      blockquoteLines.push(line.slice(2));
    } else {
      flushBlockquote();
      if (line.trim() === "") {
        // skip empty lines — whitespace-pre-wrap handles paragraph spacing
      } else {
        elements.push(
          <span key={`line-${i}`}>
            {i > 0 && lines[i - 1]?.trim() !== "" && !lines[i - 1]?.startsWith("> ") ? "" : ""}
            {renderInlineFormatting(line)}
            {"\n"}
          </span>
        );
      }
    }
  }
  flushBlockquote();

  return elements;
}

function renderInlineFormatting(text) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    return part;
  });
}

export function DepthContent({ section, activeDepth, direction = 1, onSwipe }) {
  const depthKey = DEPTH_CONFIG[activeDepth].key;
  const content = section[depthKey];
  const isPicture = depthKey === "picture";
  const isSource = depthKey === "source";

  const handleDragEnd = (_e, info) => {
    if (info.offset.x < -50) onSwipe?.('left');
    else if (info.offset.x > 50) onSwipe?.('right');
  };

  return (
    <div role="tabpanel" aria-label={`${DEPTH_CONFIG[activeDepth].label} content`}>
      <AnimatePresence mode="wait" initial={false}>
        <motion.div
          key={depthKey}
          initial={{ opacity: 0, x: direction * 40 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: direction * -40 }}
          transition={{ duration: 0.25 }}
          drag="x"
          dragConstraints={{ left: 0, right: 0 }}
          dragElastic={0.3}
          onDragEnd={handleDragEnd}
        >
          {isPicture ? (
            <img
              src={section.image}
              alt={section.title}
              width={1120}
              height={747}
              loading="lazy"
              decoding="async"
              className="w-full max-w-[560px] h-auto rounded-lg"
            />
          ) : isSource ? (
            <CodeBlock code={content} />
          ) : (
            <div className="font-serif text-[16px] leading-[1.9] text-ink whitespace-pre-wrap [&_strong]:font-semibold [&_strong]:text-[15px]">
              {depthKey === "simple"
                ? renderFormattedText(content)
                : content}
            </div>
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
