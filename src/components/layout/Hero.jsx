import { motion } from "framer-motion";

export function Hero({ onStartReading }) {
  return (
    <div className="min-h-[100dvh] flex flex-col items-center justify-center px-7 pb-16 relative">
      <div className="w-full max-w-[640px] md:max-w-[1040px] md:flex md:items-center md:gap-4">
        <div className="max-w-[640px] w-full md:flex-shrink-0">
        {/* Title */}
        <motion.h1
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          className="font-serif font-[200] text-[40px] md:text-[56px] leading-tight text-ink tracking-[-0.5px] mb-8 md:mb-10"
        >
          How GPTs Actually Work
        </motion.h1>

        {/* Overview */}
        <motion.p
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: "easeOut", delay: 0.15 }}
          className="font-serif text-[17px] md:text-[18px] leading-[1.9] text-ink-secondary mb-6 md:mb-8"
        >
          A GPT turns text into numbers, predicts the next token one at a time,
          and learns from its mistakes — that's it. Everything else is just
          scale.
        </motion.p>

        {/* Karpathy intro */}
        <motion.p
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: "easeOut", delay: 0.3 }}
          className="font-serif text-[17px] md:text-[18px] leading-[1.9] text-ink-secondary mb-6 md:mb-8"
        >
          Andrej Karpathy's{" "}
          <a
            href="https://karpathy.github.io/2026/02/12/microgpt/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-ink underline decoration-ink-tertiary underline-offset-2 hover:decoration-ink transition-colors"
          >
            microgpt.py
          </a>{" "}
          is the entire algorithm in 140 lines of Python, no dependencies. It
          intrigued me and gave me the cleanest understanding of how GPTs
          actually work under the hood.
        </motion.p>

        {/* What I built */}
        <motion.p
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: "easeOut", delay: 0.45 }}
          className="font-serif text-[17px] md:text-[18px] leading-[1.9] text-ink-secondary mb-10 md:mb-14"
        >
          I built this to help you understand it too, with the use of a lemonade
          stand analogy and some generated images to accompany them. Each section
          has four levels of depth — start simple and go deeper if you want to.
        </motion.p>

        {/* CTA row — on mobile, Sam sits to the right of the button */}
        <div className="flex items-end justify-between md:block">
          <motion.button
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.65 }}
            onClick={onStartReading}
            className="font-serif text-[16px] font-semibold text-accent-deep cursor-pointer bg-transparent border-none hover:underline underline-offset-2 transition-all"
          >
            Start reading ↓
          </motion.button>

          {/* Sam — mobile only (bottom-right of CTA row) */}
          <motion.img
            src="/sam-waving.webp"
            alt="Sam waving"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: "easeOut", delay: 0.5 }}
            className="md:hidden w-[208px] flex-shrink-0 -mb-4"
          />
        </div>
        </div>

        {/* Sam — desktop only (right side, anchoring the text) */}
        <motion.img
          src="/sam-waving.webp"
          alt="Sam waving"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: "easeOut", delay: 0.1 }}
          className="hidden md:block w-[546px] flex-shrink-0 -ml-12"
        />
      </div>

      {/* Footer credit */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.8 }}
        className="absolute bottom-8 left-0 right-0 text-center font-serif text-[14px] text-ink-tertiary"
      >
        Built around{" "}
        <a
          href="https://karpathy.github.io/2026/02/12/microgpt/"
          target="_blank"
          rel="noopener noreferrer"
          className="text-ink-tertiary hover:text-ink-secondary underline underline-offset-2 transition-colors"
        >
          microgpt.py
        </a>{" "}
        by Andrej Karpathy
      </motion.div>
    </div>
  );
}
