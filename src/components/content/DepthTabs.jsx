import { DEPTH_CONFIG } from "../../data/sections";

export function DepthTabs({ activeDepth, onDepthChange }) {
  return (
    <div
      role="tablist"
      aria-label="Content depth"
      className="flex gap-6 border-b border-[rgba(0,0,0,0.08)] mb-6"
    >
      {DEPTH_CONFIG.map((depth, i) => {
        const isActive = i === activeDepth;
        return (
          <button
            key={depth.key}
            role="tab"
            aria-selected={isActive}
            onClick={() => onDepthChange(i)}
            className={`relative pb-2.5 font-serif text-[14px] font-medium cursor-pointer border-none bg-transparent transition-colors duration-150 ${
              isActive
                ? "text-ink"
                : "text-ink-tertiary hover:text-ink-secondary"
            }`}
          >
            {depth.label}
            {isActive ? (
              <span className="absolute bottom-0 left-0 right-0 h-[2px] bg-accent-fill" />
            ) : null}
          </button>
        );
      })}
    </div>
  );
}
