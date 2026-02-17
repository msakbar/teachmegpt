import { useCallback } from "react";

export function Sidebar({ sections, activeIndex, onNavigate }) {
  const handleClick = useCallback(
    (index) => {
      onNavigate(index);
    },
    [onNavigate]
  );

  return (
    <nav
      aria-label="Sections"
      className="hidden md:flex flex-col w-[260px] min-w-[260px] h-screen sticky top-0 bg-page border-r border-[rgba(0,0,0,0.06)] overflow-y-auto"
    >
      {/* Header */}
      <div className="px-5 pt-7 pb-5 border-b border-[rgba(0,0,0,0.06)]">
        <div className="font-serif text-[15px] font-semibold text-ink mb-0.5">
          microgpt.py
        </div>
        <div className="font-serif text-[13px] text-ink-secondary">
          Karpathy&rsquo;s 140-line GPT
        </div>
      </div>

      {/* Nav items */}
      <div className="flex-1 py-3 px-2.5">
        {sections.map((section, i) => {
          const isActive = i === activeIndex;
          return (
            <button
              key={section.id}
              onClick={() => handleClick(i)}
              className={`w-full text-left px-3 py-2.5 rounded-lg border-none cursor-pointer mb-0.5 transition-colors duration-150 font-serif text-[14px] leading-snug ${
                isActive
                  ? "font-bold text-ink border-l-[3px] border-accent-fill bg-transparent"
                  : "font-normal text-ink-secondary bg-transparent hover:bg-[rgba(0,0,0,0.02)]"
              }`}
            >
              {section.title}
            </button>
          );
        })}
      </div>
    </nav>
  );
}
