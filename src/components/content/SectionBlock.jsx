import { useState, useCallback } from "react";
import { motion } from "framer-motion";
import { DepthTabs } from "./DepthTabs";
import { DepthContent } from "./DepthContent";
import { DEPTH_CONFIG } from "../../data/sections";

export function SectionBlock({ section, sectionRef }) {
  const [activeDepth, setActiveDepth] = useState(0);
  const titleId = `section-title-${section.id}`;

  const handleDepthChange = useCallback((depthIndex) => {
    setActiveDepth(depthIndex);
  }, []);

  return (
    <section
      ref={sectionRef}
      aria-labelledby={titleId}
      className="scroll-mt-[60px]"
    >
      {/* Title — sticky below TopBar */}
      <h2
        id={titleId}
        className="sticky top-[60px] z-10 bg-[#faf9f6] py-4 font-serif font-[200] text-[40px] leading-tight text-ink tracking-[-0.5px]"
      >
        {section.title}
      </h2>

      {/* Lemonade Stand analogy — always visible above tabs */}
      {section.analogy && (
        <blockquote className="border-l-[3px] border-accent-fill pl-5 my-4 italic">
          <p className="font-serif text-[15px] leading-[1.85] text-ink-secondary">
            <strong className="not-italic font-semibold text-ink">
              Sam&rsquo;s Lemonade Stand
            </strong>
            {" \u2014 "}
            {section.analogy}
          </p>
        </blockquote>
      )}

      <motion.div
        initial={{ opacity: 0, y: 12 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-60px" }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="py-7"
      >
        {/* Depth Tabs */}
        <DepthTabs
          activeDepth={activeDepth}
          onDepthChange={handleDepthChange}
        />

        {/* Depth Content */}
        <DepthContent section={section} activeDepth={activeDepth} />
      </motion.div>
    </section>
  );
}
