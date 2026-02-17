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
      className="scroll-mt-20"
      style={{ contentVisibility: "auto" }}
    >
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-60px" }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="py-7"
      >
        {/* Title */}
        <h2
          id={titleId}
          className="font-serif font-[200] text-[40px] leading-tight text-ink mb-8 tracking-[-0.5px]"
        >
          {section.title}
        </h2>

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
