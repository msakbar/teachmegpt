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
      className="scroll-mt-[60px] md:scroll-mt-0"
    >
      {/* Title — sticky below TopBar */}
      <h2
        id={titleId}
        className="sticky top-[60px] md:top-0 z-10 bg-[#faf9f6] py-4 font-serif font-[200] text-[40px] leading-tight text-ink tracking-[-0.5px]"
      >
        {section.title}
      </h2>

      <div className="md:flex md:gap-12 md:items-start">
        {/* Lemonade Stand analogy — left column on desktop, centered with content */}
        {section.analogy && (
          <blockquote className="border-l-[3px] border-accent-fill pl-5 my-4 italic md:w-[380px] md:shrink-0 md:mt-[90px] md:mb-0 md:sticky md:top-[80px] md:py-3 md:pr-6">
            <p className="font-serif text-[15px] leading-[1.85] text-ink-secondary md:text-[14px] md:leading-[2]">
              <strong className="not-italic font-semibold text-ink block mb-2">
                Sam&rsquo;s Lemonade Stand
              </strong>
              {section.analogy}
            </p>
          </blockquote>
        )}

        {/* Tabs + content — right column on desktop */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-60px" }}
          transition={{ duration: 0.4, ease: "easeOut" }}
          className="py-7 md:flex-1 md:min-w-0"
        >
          <DepthTabs
            activeDepth={activeDepth}
            onDepthChange={handleDepthChange}
          />
          <DepthContent section={section} activeDepth={activeDepth} />
        </motion.div>
      </div>
    </section>
  );
}
