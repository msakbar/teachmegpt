export function TopBar({
  activeIndex,
  totalSections,
  onPrev,
  onNext,
  onMenuOpen,
  isMobile,
}) {
  const inHero = activeIndex === -1;
  const hasPrev = activeIndex > 0;
  const hasNext = activeIndex < totalSections - 1;

  return (
    <div className={`sticky top-0 z-50 bg-[#faf9f6]/92 backdrop-blur-[12px] border-b border-[rgba(0,0,0,0.06)] px-3 md:px-10 py-2 md:py-3 flex items-center gap-1.5 transition-opacity duration-300${inHero && !isMobile ? " opacity-0 pointer-events-none" : ""}`}>
      {isMobile ? (
        <button
          onClick={onMenuOpen}
          aria-label="Open navigation"
          className="bg-[rgba(0,0,0,0.03)] border-none text-ink-secondary p-2.5 rounded-lg cursor-pointer text-sm leading-none min-w-[44px] min-h-[44px] flex items-center justify-center"
        >
          &#9776;
        </button>
      ) : null}

      {!inHero ? (
        <>
          <button
            onClick={onPrev}
            disabled={!hasPrev}
            aria-label="Previous section"
            className={`bg-[rgba(0,0,0,0.03)] border-none rounded-lg cursor-pointer font-serif text-sm min-w-[44px] min-h-[44px] flex items-center justify-center ${
              hasPrev ? "text-ink-secondary" : "text-ink-tertiary/40 cursor-default"
            }`}
          >
            &#8249;
          </button>

          <span className="font-serif text-[13px] text-ink-tertiary min-w-[40px] text-center">
            {activeIndex + 1} / {totalSections}
          </span>

          <button
            onClick={onNext}
            disabled={!hasNext}
            aria-label="Next section"
            className={`bg-[rgba(0,0,0,0.03)] border-none rounded-lg cursor-pointer font-serif text-sm min-w-[44px] min-h-[44px] flex items-center justify-center ${
              hasNext ? "text-ink-secondary" : "text-ink-tertiary/40 cursor-default"
            }`}
          >
            &#8250;
          </button>
        </>
      ) : null}
    </div>
  );
}
